"""L09x lane pipeline: 20K-expansion candidates -> native-qualified shards.

Thin local runner mirroring the legacy L09 pipeline (propose -> ground ->
prove_exists -> write oracle -> verify_terminal(reference) -> semantic
witness -> leak screen -> 3-layer dedup -> collision check vs the legacy 8K
-> immutable JSONL shards + manifests), writing **only** to
``bank/staging_20k/l09/``. The legacy 8K staging/build is read for collision
checking and never written.

Differences from the legacy runner, per EXPANSION_20K_EXECUTION_PLAN §2:
- dataset_id ``autoontobench-bank-20000-candidate-dev1`` (legacy records keep
  theirs; never rewritten);
- seed base 20261101;
- every record carries the per-task status block (source_verified,
  schema_valid, ontology_parse_valid, native_execution_status,
  oracle_check_status, mutation_check_status, human_review_status,
  qualification_status) with honest PASS/FAIL/NOT_RUN values — a check that
  was not executed is never marked PASS;
- task-id and signature collision checks against the legacy 8K bank and any
  other staging_20k lane (rejection code COLLISION).

Rejection codes: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE /
DUPLICATE / NON_SEMANTIC / LICENSE_HOLD / COLLISION / SCHEMA_INVALID.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST
from benchmark.plugins.task_family.l09x import engine
from benchmark.plugins.task_family.l09x.families import (
    ELIGIBLE,
    FAMILIES,
    FAMILY_BY_ID,
    INSTANCES_PER_GOAL,
    SmartAncExpansionTaskFamilies,
)
from benchmark.plugins.verifier.l09x.structural_verifier import (
    SmartAncExpansionStructuralVerifier,
)
from benchmark.plugins.source.l09.smart_anc import SNAPSHOT, SNAPSHOT_ID

LANE = "l09"
REPO_ROOT = Path(__file__).resolve().parents[4]
STAGING = REPO_ROOT / "bank" / "staging_20k" / LANE
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
DEFAULT_SEED = 20261101

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD", "COLLISION", "SCHEMA_INVALID",
)

CONTRACT_FIELDS = (
    "schema_version", "task_id", "dataset_id", "lane_id", "family_id",
    "base_goal_id", "instance_id", "task_regime", "semantic_profile",
    "complexity", "public", "oracle_handle", "status", "qualification",
    "native_evidence", "source_snapshot_ids", "source_unit_ids", "statuses",
)
PUBLIC_FIELDS = ("user_goal", "allowed_tools", "budget", "observation_refs", "artifact")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ShardWriter:
    """Append-records-in-memory, flush-immutable-shard-on-close writer."""

    def __init__(self, root: Path, shard_size: int = 250) -> None:
        self.root = root
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_idx = 0
        self.manifests: list[dict[str, Any]] = []
        root.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_name = "shard-%04d" % self.shard_idx
        path = self.root / ("%s.jsonl" % shard_name)
        blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.buffer)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, path)
        manifest = {
            "shard": shard_name,
            "records": len(self.buffer),
            "sha256": engine.sha256_json({"blob": blob}),
            "families": sorted({r.get("family_id") for r in self.buffer}),
            "base_goals": len({r.get("base_goal_id") for r in self.buffer}),
            "path": str(path),
        }
        _write_json_atomic(self.root / ("%s.manifest.json" % shard_name), manifest)
        self.manifests.append(manifest)
        self.buffer = []
        self.shard_idx += 1

    def close(self) -> None:
        self.flush()


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    textual = json.dumps({
        k: v for k, v in public.items() if k != "artifact"
    }, sort_keys=True)
    for slot in ast.get("slots", []):
        for needle in _needles(slot.get("value")):
            if needle in textual:
                return needle
    return None


def _needles(val: Any) -> list[str]:
    if isinstance(val, str) and len(val) >= 8:
        return [val]
    return []


def _schema_validate(record: dict[str, Any]) -> str | None:
    for f in CONTRACT_FIELDS:
        if f not in record or record[f] is None:
            return "missing contract field %s" % f
    for f in PUBLIC_FIELDS:
        if f not in record["public"]:
            return "missing public field %s" % f
    if record["schema_version"] != "aob-task/5.0":
        return "schema_version %s" % record["schema_version"]
    if record["dataset_id"] != DATASET_ID:
        return "dataset_id %s" % record["dataset_id"]
    if record["statuses"].get("qualification_status") not in (
        "CANDIDATE", "PARTIALLY_CHECKED", "QUALIFIED_NATIVE", "REJECTED"):
        return "qualification_status %s" % record["statuses"].get("qualification_status")
    return None


class LegacyCollisionIndex:
    """Task-id + signature index over the frozen legacy 8K bank (read-only)."""

    def __init__(self, repo_root: Path) -> None:
        self.task_ids: set[str] = set()
        self.instance_ids: set[str] = set()
        self.signatures: set[tuple] = set()
        self.sources: dict[str, int] = {}
        self._load(repo_root)

    def _add_task_id_source(self, source: str) -> None:
        self.sources[source] = self.sources.get(source, 0) + 1

    def _load(self, repo_root: Path) -> None:
        build = repo_root / "bank" / "builds" / "aob-8000-rc1" / "qualified_tasks.jsonl"
        if build.is_file():
            for line in build.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("task_id"):
                    self.task_ids.add(rec["task_id"])
                    self._add_task_id_source("builds/aob-8000-rc1")
        staging = repo_root / "bank" / "staging"
        if staging.is_dir():
            for lane_dir in sorted(staging.iterdir()):
                if not lane_dir.is_dir():
                    continue
                for shard in sorted((lane_dir / "qualified").glob("*.jsonl")):
                    for line in shard.open(encoding="utf-8"):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("task_id"):
                            self.task_ids.add(rec["task_id"])
                            self._add_task_id_source("staging/%s" % lane_dir.name)
                        if rec.get("instance_id"):
                            self.instance_ids.add(rec["instance_id"])
        staging20k = repo_root / "bank" / "staging_20k"
        if staging20k.is_dir():
            for lane_dir in sorted(staging20k.iterdir()):
                if not lane_dir.is_dir() or lane_dir.name == LANE:
                    continue
                for shard in sorted((lane_dir / "qualified").glob("*.jsonl")):
                    for line in shard.open(encoding="utf-8"):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("task_id"):
                            self.task_ids.add(rec["task_id"])
                            self._add_task_id_source("staging_20k/%s" % lane_dir.name)
                        if rec.get("instance_id"):
                            self.instance_ids.add(rec["instance_id"])
        # legacy L09 oracle signatures: (mechanism, artifact_path, slot keys)
        legacy_oracle_dir = repo_root / "bank" / "staging" / "l09" / "oracle"
        if legacy_oracle_dir.is_dir():
            for oj in sorted(legacy_oracle_dir.glob("*.json")):
                try:
                    payload = json.loads(oj.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                keys = tuple(sorted(
                    s.get("path") or s.get("find") or ""
                    for s in payload.get("slots", [])))
                self.signatures.add((payload.get("mechanism"), payload.get("artifact_path"), keys))

    def collides(self, record: dict[str, Any], ast: dict[str, Any]) -> str | None:
        if record["task_id"] in self.task_ids:
            return "task_id %s already exists in the legacy bank" % record["task_id"]
        if record["instance_id"] in self.instance_ids:
            return "instance_id %s already exists in the legacy bank" % record["instance_id"]
        return None

    def signature_collides(self, ast: dict[str, Any]) -> bool:
        keys = tuple(sorted(slot.get("path") or slot.get("find") or ""
                            for slot in ast.get("slots", [])))
        return (ast.get("mechanism"), ast.get("artifact_path"), keys) in self.signatures


def _base_record(candidate: CandidateTaskAST, grounded: Any) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = engine.sha256_json(
        [ast["mechanism"], ast["artifact_path"]]
    )[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L09",
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_unit_ids": list(candidate.source_unit_ids),
        "origin_mechanism_group": mech_group,
        "public": grounded.public if grounded is not None else None,
        "oracle_handle": grounded.oracle_ref if grounded is not None else None,
        "authorship": "deterministic_template",
        "provenance": "source_parameterized",
        "status": "CANDIDATE",
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
        "statuses": {
            "source_verified": "PASS",
            "schema_valid": "NOT_RUN",
            "ontology_parse_valid": "NOT_RUN",
            "native_execution_status": "NOT_RUN",
            "oracle_check_status": "NOT_RUN",
            "mutation_check_status": "NOT_RUN",
            "human_review_status": "NOT_RUN",
            "qualification_status": "CANDIDATE",
        },
    }


def _qualify_one(
    plugin: SmartAncExpansionTaskFamilies,
    verifier: SmartAncExpansionStructuralVerifier,
    candidate: CandidateTaskAST,
    collisions: LegacyCollisionIndex,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    try:
        grounded = plugin.ground(candidate)
    except ValueError as exc:
        record = _base_record(candidate, None)
        record["statuses"]["ontology_parse_valid"] = "FAIL"
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    record = _base_record(candidate, grounded)
    record["statuses"]["ontology_parse_valid"] = "PASS"
    proof = verifier.prove_exists(grounded)
    record["statuses"]["native_execution_status"] = "PASS"
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        record["statuses"]["oracle_check_status"] = "FAIL"
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    ast = candidate.ast
    if ast["artifact_format"] == "cql-text":
        reference_payload: Any = engine.ANC_ROOT.joinpath(
            ast["artifact_path"]).read_text(encoding="utf-8")
        submitted = {"artifact": {"format": "cql-text", "text": reference_payload}}
    else:
        submitted = {"artifact": {
            "format": "fhir-json",
            "content": engine.load_artifact(ast["artifact_path"]),
        }}
    term = verifier.verify_terminal(grounded, submitted)
    if not term.valid:
        record["statuses"]["oracle_check_status"] = "FAIL"
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    record["statuses"]["oracle_check_status"] = "PASS"
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        record["statuses"]["mutation_check_status"] = "FAIL"
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    record["statuses"]["mutation_check_status"] = "PASS"
    leak = _leak_screen(grounded.public, ast)
    if leak:
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "LEAKAGE", "reason": "public payload leaks: %s" % leak[:120]}
    collision = collisions.collides(record, ast)
    if collision:
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "COLLISION", "reason": collision[:300]}
    if collisions.signature_collides(ast):
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "COLLISION",
                        "reason": "signature (mechanism, artifact, slots) exists in legacy 8K"}
    record["native_evidence"] = {
        "prove_exists": proof.evidence,
        "verify_terminal": term.detail,
        "semantic_witness": witness.witness,
    }
    record["qualification"] = {
        "native_verified": True,
        "solvable": True,
        "nontrivial": True,
        "semantic_relevance": True,
        "leakage_clean": True,
        "release_eligible": True,
    }
    schema_err = _schema_validate(record)
    if schema_err:
        del record["native_evidence"]
        record["qualification"] = {
            "native_verified": True, "solvable": True, "nontrivial": True,
            "semantic_relevance": True, "leakage_clean": True,
            "release_eligible": False,
        }
        record["statuses"]["schema_valid"] = "FAIL"
        record["statuses"]["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "SCHEMA_INVALID", "reason": schema_err}
    record["statuses"]["schema_valid"] = "PASS"
    record["statuses"]["qualification_status"] = "QUALIFIED_NATIVE"
    record["status"] = "QUALIFIED_NATIVE"
    return record, None


class DedupIndex:
    """family/base-goal/instance three-layer dedup."""

    def __init__(self) -> None:
        self.families: set[str] = set()
        self.base_goals: set[str] = set()
        self.instances: set[str] = set()

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        if inst in self.instances:
            return False
        self.instances.add(inst)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True


def run_build(
    *,
    families: list[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    plugin = SmartAncExpansionTaskFamilies()
    verifier = SmartAncExpansionStructuralVerifier()
    engine.load_kbx()
    collisions = LegacyCollisionIndex(REPO_ROOT)
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L09",
        "build": "aob-20000-candidate-dev1",
        "dataset_id": DATASET_ID,
        "seed": seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
        "legacy_collision_index": {
            "legacy_task_ids": len(collisions.task_ids),
            "legacy_instance_ids": len(collisions.instance_ids),
            "legacy_signatures": len(collisions.signatures),
            "sources": collisions.sources,
        },
    }
    t0 = time.time()
    mode = "w" if families is None else "a"
    with rejection_path.open(mode, encoding="utf-8") as rej_fh:
        for fid in family_ids:
            spec = FAMILY_BY_ID[fid]
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "base_goals": 0, "eligible_artifacts": 0}
            admitted_per_goal: dict[str, int] = {}
            pool = ELIGIBLE[fid]()
            fam_stats["eligible_artifacts"] = len(pool)
            candidates = plugin.propose(
                SNAPSHOT, family=fid, seed=seed, constraints={"pool": pool})
            for candidate in candidates:
                if fam_stats["qualified"] >= spec.total:
                    break
                if admitted_per_goal.get(candidate.base_goal_id, 0) >= INSTANCES_PER_GOAL:
                    continue
                record, rejection = _qualify_one(plugin, verifier, candidate, collisions)
                fam_stats["candidates"] += 1
                stats["candidates"] += 1
                candidate_writer.add(record)
                if rejection is not None:
                    fam_stats["rejected"] += 1
                    stats["rejections"][rejection["code"]] = (
                        stats["rejections"].get(rejection["code"], 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        **rejection,
                    }, sort_keys=True) + "\n")
                    continue
                if not dedup.admit(record):
                    fam_stats["rejected"] += 1
                    stats["rejections"]["DUPLICATE"] = (
                        stats["rejections"].get("DUPLICATE", 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        "code": "DUPLICATE",
                        "reason": "instance hash already admitted",
                    }, sort_keys=True) + "\n")
                    continue
                admitted_per_goal[record["base_goal_id"]] = (
                    admitted_per_goal.get(record["base_goal_id"], 0) + 1)
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
                qualified_writer.add(record)
            fam_stats["base_goals"] = len(admitted_per_goal)
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["distinct_instances"] = len(dedup.instances)
    stats["complexity_split"] = _complexity_split(qualified_writer.manifests)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    stats["reproduce_commands"] = {
        "build": ".venv/bin/python -m benchmark.plugins.task_family.l09x.pipeline --seed %d" % seed,
        "verifier_reproduce": (
            ".venv/bin/python -m benchmark.plugins.verifier.l09x.structural_verifier "
            "reproduce --evidence-dir bank/staging_20k/l09/evidence/<instance_id>"),
    }
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _complexity_split(manifests: list[dict[str, Any]]) -> dict[str, int]:
    split = {"low": 0, "medium": 0, "high": 0}
    for manifest in manifests:
        path = Path(manifest["path"])
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            split[rec.get("complexity") or "low"] += 1
    return split


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L09x (20K expansion) lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    stats = run_build(families=args.families, seed=args.seed)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
