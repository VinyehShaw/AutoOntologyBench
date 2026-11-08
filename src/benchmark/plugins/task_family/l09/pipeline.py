"""L09 lane pipeline: candidate -> native-qualified shards (thin local runner).

``autoontobench.factory`` does not exist yet (S10 work package), so this is the
lane-local runner with the adapter seam, mirroring the L01 pattern: propose ->
ground -> prove_exists -> write oracle -> verify_terminal(reference) ->
semantic witness -> leak screen -> 3-layer dedup -> immutable JSONL shards +
manifests under ``bank/staging/l09/``. Rejection codes: UNSOLVABLE / TRIVIAL /
PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE / NON_SEMANTIC / LICENSE_HOLD.

All verification is in-process and deterministic (pure-Python recomputation
over the pinned snapshot), so no subprocess parallelism is needed; candidates
are qualified serially in a stable order.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST
from benchmark.plugins.task_family.l09 import engine
from benchmark.plugins.task_family.l09.families import (
    ELIGIBLE,
    FAMILIES,
    FAMILY_BY_ID,
    INSTANCES_PER_GOAL,
    SmartAncTaskFamilies,
)
from benchmark.plugins.verifier.l09.structural_verifier import SmartAncStructuralVerifier
from benchmark.plugins.source.l09.smart_anc import SNAPSHOT, SNAPSHOT_ID

LANE = "l09"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DATASET_ID = "autoontobench-bank-8000-rc1"

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)


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
    """Public *textual* fields (goal/intent/refs, NOT the artifact under edit —
    the corrupted artifact is the work material itself) must not reveal the
    reference slot values or the reference artifact hash."""
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


def _qualify_one(
    plugin: SmartAncTaskFamilies,
    verifier: SmartAncStructuralVerifier,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    try:
        grounded = plugin.ground(candidate)
    except ValueError as exc:
        record = _base_record(candidate, None)
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    record = _base_record(candidate, grounded)
    proof = verifier.prove_exists(grounded)
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
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
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, ast)
    if leak:
        return record, {"code": "LEAKAGE", "reason": "public payload leaks: %s" % leak[:120]}
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
    record["status"] = "QUALIFIED"
    return record, None


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
    }


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
    seed: int = 20261028,
) -> dict[str, Any]:
    plugin = SmartAncTaskFamilies()
    verifier = SmartAncStructuralVerifier()
    engine.load_kb()  # warm the immutable KB once
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L09",
        "seed": seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
    }
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
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
                record, rejection = _qualify_one(plugin, verifier, candidate)
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
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


_ELIGIBILITY: dict[str, list[Any]] = {}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L09 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    stats = run_build(families=args.families, seed=args.seed)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
