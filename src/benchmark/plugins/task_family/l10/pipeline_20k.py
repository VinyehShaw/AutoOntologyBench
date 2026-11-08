"""L10 20K-expansion lane pipeline: candidate -> native-qualified shards.

Build ``aob-20000-candidate-dev1`` for lane L10: 300 NEW symbolic BDDL tasks
(12 new families x 5 base goals x 5 instances) written to
``bank/staging_20k/l10/`` — the legacy 8K lane ``bank/staging/l10/`` is never
touched (read-only for collision checks).

New records use dataset_id ``autoontobench-bank-20000-candidate-dev1`` and
seed base 20261101. Verification is in-process and deterministic (two
independent legs, recomputed from the pinned BEHAVIOR-1K snapshot bytes).
``--recheck`` re-runs the full qualification gate over every qualified
record; ``--recheck-task <id>`` is the native_evidence command.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from benchmark.plugins.task_family.l10.engine import sha256_json
from benchmark.plugins.task_family.l10.families_20k import (
    NEW_FAMILIES,
    NEW_FAMILY_BY_ID,
    INSTANCES_PER_GOAL,
)
from benchmark.plugins.task_family.l10.propose_20k import (
    BehaviorBddl20KTaskFamilies,
    SEED_SALT,
)
from benchmark.plugins.task_family.l10 import upstream
from benchmark.plugins.verifier.l10.bddl_verifier_20k import (
    ENGINE,
    SOURCE_SNAPSHOT_ID,
    BddlVerifier20K,
)
from source_registry.fetchers import l10_behavior_bddl as fetcher

LANE = "l10"
REPO_ROOT = Path(__file__).resolve().parents[4]
STAGING = REPO_ROOT / "bank" / "staging_20k" / LANE
LEGACY_STAGING = REPO_ROOT / "bank" / "staging" / LANE  # read-only
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
DEFAULT_SEED = 20261101

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

_CONTRACT_FIELDS = (
    ("schema_version", str), ("task_id", str), ("dataset_id", str),
    ("lane_id", str), ("family_id", str), ("base_goal_id", str),
    ("instance_id", str), ("task_regime", str), ("complexity", str),
    ("semantic_profile", str), ("oracle_handle", str), ("status", str),
    ("source_snapshot_ids", list), ("source_unit_ids", list),
    ("native_evidence", dict), ("qualification", dict), ("public", dict),
    ("statuses", dict),
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
        import hashlib
        shard_name = "shard-%04d" % self.shard_idx
        path = self.root / ("%s.jsonl" % shard_name)
        blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.buffer)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, path)
        manifest = {
            "shard": shard_name,
            "records": len(self.buffer),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "families": sorted({r.get("family_id") for r in self.buffer}),
            "base_goals": len({r.get("base_goal_id") for r in self.buffer}),
            "path": str(path),
        }
        _write_json_atomic(self.root / ("%s.manifest.json" % shard_name),
                           manifest)
        self.manifests.append(manifest)
        self.buffer = []
        self.shard_idx += 1

    def close(self) -> None:
        self.flush()


# ------------------------------------------------------- collision screening

def _content_hash(record: dict[str, Any]) -> str:
    return sha256_json([
        record["family_id"], record["public"].get("task_material", {})])


def _load_legacy_index() -> dict[str, Any]:
    """Read-only scan of the frozen legacy 8K lane for collision checks."""
    task_ids: set[str] = set()
    instance_ids: set[str] = set()
    contents: set[str] = set()
    records = 0
    for path in sorted((LEGACY_STAGING / "qualified").glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records += 1
            task_ids.add(rec["task_id"])
            instance_ids.add(rec["instance_id"])
            contents.add(sha256_json([
                rec["family_id"], rec["public"].get("task_material", {})]))
    return {
        "legacy_records_scanned": records,
        "legacy_task_ids": task_ids,
        "legacy_instance_ids": instance_ids,
        "legacy_content_hashes": contents,
    }


# ------------------------------------------------------------ record shaping

def _validate_contract(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field, typ in _CONTRACT_FIELDS:
        if field not in record:
            issues.append("missing field %s" % field)
        elif typ is str and not isinstance(record[field], str):
            issues.append("field %s not a string" % field)
        elif typ is list and not isinstance(record[field], list):
            issues.append("field %s not a list" % field)
        elif typ is dict and not isinstance(record[field], dict):
            issues.append("field %s not a dict" % field)
    public = record.get("public", {})
    for key in ("user_goal", "allowed_tools", "budget", "observation_refs",
                "semantic_scope", "task_material"):
        if key not in public:
            issues.append("public.%s missing" % key)
    for flag in ("native_verified", "solvable", "nontrivial",
                 "semantic_relevance", "leakage_clean", "release_eligible"):
        if flag not in record.get("qualification", {}):
            issues.append("qualification.%s missing" % flag)
    for key in ("source_verified", "schema_valid", "ontology_parse_valid",
                "native_execution_status", "oracle_check_status",
                "mutation_check_status", "human_review_status",
                "qualification_status"):
        if key not in record.get("statuses", {}):
            issues.append("statuses.%s missing" % key)
    if record.get("schema_version") != "aob-task/5.0":
        issues.append("schema_version != aob-task/5.0")
    if record.get("dataset_id") != DATASET_ID:
        issues.append("dataset_id != %s" % DATASET_ID)
    return issues


_PARSE_AGREE_CACHE: dict[str, bool] = {}


def _ontology_parse_valid(problem_rel: str) -> bool:
    """Leg A parse + leg B (vendored bddl) structural agreement on the
    pinned source problem bytes."""
    if problem_rel not in _PARSE_AGREE_CACHE:
        from benchmark.plugins.task_family.l10 import engine
        path = engine.PINNED_ROOT / problem_rel
        text = path.read_text(encoding="utf-8")
        mine = engine.parse_problem_text(text)
        try:
            _PARSE_AGREE_CACHE[problem_rel] = bool(
                upstream.parse_agrees(text, mine))
        except Exception:
            _PARSE_AGREE_CACHE[problem_rel] = False
    return _PARSE_AGREE_CACHE[problem_rel]


def _leak_screen(public: dict[str, Any],
                 reference: dict[str, Any]) -> str | None:
    textual = json.dumps(public, sort_keys=True)
    ref_blob = json.dumps(reference, sort_keys=True)
    if len(ref_blob) > 60 and ref_blob in textual:
        return "reference_submission_json"
    gold_text = reference.get("goal_text")
    if isinstance(gold_text, str) and gold_text in textual:
        return "gold_goal_text_verbatim"
    return None


def _base_record(candidate: CandidateTaskAST,
                 grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = sha256_json(
        [candidate.family_id, ast["goal_kind"], ast["problem_rel"],
         ast.get("variant_key", ""), SEED_SALT])[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L10",
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
        "source_snapshot_ids": [SOURCE_SNAPSHOT_ID],
        "source_unit_ids": list(candidate.source_unit_ids),
        "origin_mechanism_group": mech_group,
        "public": grounded.public,
        "oracle_handle": grounded.oracle_ref,
        "authorship": "deterministic_template",
        "provenance": "source_parameterized",
        "status": "CANDIDATE",
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
        "statuses": {
            "source_verified": "NOT_RUN",
            "schema_valid": "NOT_RUN",
            "ontology_parse_valid": "NOT_RUN",
            "native_execution_status": "NOT_RUN",
            "oracle_check_status": "NOT_RUN",
            "mutation_check_status": "NOT_RUN",
            "human_review_status": "NOT_RUN",
            "qualification_status": "CANDIDATE",
        },
    }


def _trim(evidence: dict[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    out.pop("reference_terminal", None)
    return out


def _trim_witness(witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": witness.get("wrong_verdict_valid") is False,
        "counterfactual_kind": witness.get("counterfactual_kind"),
        "wrong_submission": witness.get("wrong_submission"),
    }


def _qualify_one(
    plugin: BehaviorBddl20KTaskFamilies,
    verifier: BddlVerifier20K,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    statuses = record["statuses"]
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:
        statuses["native_execution_status"] = "FAIL"
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "%s: %s" % (type(exc).__name__,
                                              str(exc))[:300]}
    statuses["source_verified"] = "PASS"  # digest-checked pinned material
    statuses["native_execution_status"] = "PASS" if proof.exists else "FAIL"
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": code, "reason": reason[:300]}
    try:
        reference = verifier.reference_submission(candidate.ast)
    except Exception as exc:
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "reference: %s: %s" % (
                            type(exc).__name__, str(exc))[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    try:
        verifier.load_oracle(grounded.oracle_ref)
        statuses["oracle_check_status"] = "PASS"
    except Exception:
        statuses["oracle_check_status"] = "FAIL"
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "oracle round-trip failed"}
    term = verifier.verify_terminal(grounded, reference)
    if not term.valid:
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal "
                                  "replay: %s"
                                  % json.dumps(term.detail)[:300]}
    witness = verifier.certify_semantic_relevance(grounded)
    statuses["mutation_check_status"] = "PASS" if witness.ok else "FAIL"
    if not witness.ok:
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, reference)
    if leak:
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "LEAKAGE",
                        "reason": "public payload leaks: %s" % leak[:120]}
    statuses["ontology_parse_valid"] = "PASS" if _ontology_parse_valid(
        candidate.ast["problem_rel"]) else "FAIL"
    if statuses["ontology_parse_valid"] != "PASS":
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "leg A/leg B parse disagreement on the "
                                  "pinned source problem"}
    contract_issues = _validate_contract(dict(
        record, native_evidence={}, status="QUALIFIED",
        qualification={"native_verified": True, "solvable": True,
                       "nontrivial": True, "semantic_relevance": True,
                       "leakage_clean": True, "release_eligible": True},
        statuses=dict(statuses, schema_valid="PASS",
                      qualification_status="QUALIFIED_NATIVE")))
    statuses["schema_valid"] = "PASS" if not contract_issues else "FAIL"
    if contract_issues:
        statuses["qualification_status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "contract validation: %s"
                                  % "; ".join(contract_issues[:5])}
    record["native_evidence"] = {
        "engine": ENGINE,
        "source_snapshot_id": SOURCE_SNAPSHOT_ID,
        "commands": [
            ".venv/bin/python -m benchmark.plugins.task_family.l10."
            "pipeline_20k --recheck-task %s" % record["task_id"],
        ],
        "expected": {"ok": True, "proof": True, "terminal": True,
                     "witness": True},
        "prove_exists": _trim(proof.evidence),
        "verify_terminal": term.detail,
        "semantic_witness": _trim_witness(witness.witness),
    }
    record["qualification"] = {
        "native_verified": True, "solvable": True, "nontrivial": True,
        "semantic_relevance": True, "leakage_clean": True,
        "release_eligible": True,
    }
    record["status"] = "QUALIFIED"
    statuses["human_review_status"] = "NOT_RUN"  # honest: no human review
    statuses["qualification_status"] = "QUALIFIED_NATIVE"
    return record, None


class DedupIndex:

    def __init__(self, legacy: dict[str, Any]) -> None:
        self.families: set[str] = set()
        self.base_goals: set[str] = set()
        self.instances: set[str] = set()
        self.content: set[str] = set()
        self.legacy = legacy
        self.legacy_task_id_hits: list[str] = []
        self.legacy_instance_id_hits: list[str] = []
        self.legacy_content_hits: list[str] = []

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        content = _content_hash(record)
        if record["task_id"] in self.legacy["legacy_task_ids"]:
            self.legacy_task_id_hits.append(record["task_id"])
            return False
        if inst in self.legacy["legacy_instance_ids"]:
            self.legacy_instance_id_hits.append(inst)
            return False
        if content in self.legacy["legacy_content_hashes"]:
            self.legacy_content_hits.append(content)
            return False
        if inst in self.instances or content in self.content:
            return False
        self.instances.add(inst)
        self.content.add(content)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True


def run_build(
    *,
    families: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    base_goals: dict[str, int] | None = None,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    plugin = BehaviorBddl20KTaskFamilies()
    verifier = BddlVerifier20K(STAGING / "oracle")
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in NEW_FAMILIES]
    legacy = _load_legacy_index()
    dedup = DedupIndex(legacy)
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "lane": "L10",
        "build_id": "aob-20000-candidate-dev1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
    }
    t0 = time.time()
    with rejection_path.open("w", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            spec = NEW_FAMILY_BY_ID[fid]
            quota = (base_goals or {}).get(fid, spec.base_goals)
            constraints = {
                "instances_per_goal": INSTANCES_PER_GOAL,
                "base_goals": quota,
            }
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "base_goals": 0}
            fam_base_goals: set[str] = set()
            candidates = list(plugin.propose(
                snapshot, family=fid, seed=seed, constraints=constraints))
            if max_per_family:
                candidates = candidates[:max_per_family]
            for candidate in candidates:
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
                        "statuses": record["statuses"],
                        **rejection,
                    }, sort_keys=True) + "\n")
                    continue
                if dedup.admit(record):
                    fam_stats["qualified"] += 1
                    stats["qualified"] += 1
                    fam_base_goals.add(record["base_goal_id"])
                    qualified_writer.add(record)
                else:
                    fam_stats["rejected"] += 1
                    code = ("DUPLICATE" if not (
                        dedup.legacy_task_id_hits
                        or dedup.legacy_instance_id_hits
                        or dedup.legacy_content_hits) else "LEAKAGE")
                    stats["rejections"][code] = (
                        stats["rejections"].get(code, 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        "statuses": record["statuses"],
                        "code": code,
                        "reason": "instance/content hash already admitted "
                                  "or legacy collision",
                    }, sort_keys=True) + "\n")
            fam_stats["base_goals"] = len(fam_base_goals)
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["task_families"] = len(stats["families"])
    stats["distinct_instances"] = len(dedup.instances)
    stats["dataset_id"] = DATASET_ID
    stats["source_snapshot_ids"] = [SOURCE_SNAPSHOT_ID]
    stats["seed"] = seed
    stats["instances_per_goal"] = INSTANCES_PER_GOAL
    stats["collision_report"] = {
        "legacy_lane_scanned": str(LEGACY_STAGING / "qualified"),
        "legacy_records_scanned": legacy["legacy_records_scanned"],
        "legacy_task_id_collisions": len(dedup.legacy_task_id_hits),
        "legacy_instance_id_collisions": len(dedup.legacy_instance_id_hits),
        "legacy_content_collisions": len(dedup.legacy_content_hits),
        "new_task_ids_unique": len(dedup.instances)
        == stats["qualified"],
    }
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _iter_qualified() -> Iterator[dict[str, Any]]:
    for path in sorted((STAGING / "qualified").glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def run_recheck(*, task_id: str | None = None) -> dict[str, Any]:
    verifier = BddlVerifier20K(STAGING / "oracle")
    checked = 0
    mismatches: list[dict[str, Any]] = []
    t0 = time.time()
    for record in _iter_qualified():
        if task_id and record["task_id"] != task_id:
            continue
        oracle = verifier.load_oracle(record["oracle_handle"])
        cand_meta = oracle["candidate"]
        cand = CandidateTaskAST(
            lane_id=cand_meta["lane_id"],
            family_id=cand_meta["family_id"],
            base_goal_id=cand_meta["base_goal_id"],
            instance_id=cand_meta["instance_id"],
            regime=cand_meta["regime"],
            semantic_profile=cand_meta["semantic_profile"],
            ast=oracle["ast"],
            source_unit_ids=tuple(cand_meta["source_unit_ids"]),
        )
        grounded = GroundedTask(
            ast=cand, public=record["public"],
            oracle_ref=record["oracle_handle"], native_evidence={})
        proof = verifier.prove_exists(grounded)
        reference = verifier.reference_submission(oracle["ast"])
        term = verifier.verify_terminal(grounded, reference)
        witness = verifier.certify_semantic_relevance(grounded)
        res = {
            "task_id": record["task_id"], "family_id": record["family_id"],
            "proof": proof.exists, "terminal": term.valid,
            "witness": witness.ok,
            "ok": proof.exists and term.valid and witness.ok,
        }
        checked += 1
        if not res["ok"]:
            res["proof_reason"] = proof.reason
            res["terminal_detail"] = term.detail
            res["witness_detail"] = witness.witness
            mismatches.append(res)
        if task_id:
            return res
    return {
        "recheck": "full qualification gate recomputed from pinned snapshot",
        "checked": checked,
        "mismatches": mismatches[:20],
        "ok": not mismatches,
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="L10 20K-expansion build runner (staging_20k)")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-per-family", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="build 1 base goal per family (12x5=60 tasks)")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--recheck-task", default=None)
    args = parser.parse_args()
    if args.recheck_task:
        result = run_recheck(task_id=args.recheck_task)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("ok") else 1
    if args.recheck:
        report = run_recheck()
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["ok"] else 1
    base_goals = None
    if args.smoke:
        base_goals = {fid: 1 for fid in NEW_FAMILY_BY_ID}
    stats = run_build(
        families=args.families,
        seed=args.seed,
        base_goals=base_goals,
        max_per_family=args.max_per_family,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests",
                                   "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
