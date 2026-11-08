"""L01 20K expansion runner: eight new families -> ``bank/staging_20k/l01``.

Reuses the legacy lane building blocks unchanged:

* ``expansion_20k`` registers the new mechanisms/families into the existing
  ``engine.MECHANISMS`` / ``families.FAMILY_BY_ID`` registries;
* ``GatekeeperTaskFamilies.propose`` / ``ground`` (legacy) generate the tasks;
* ``GatekeeperOpaVerifier`` (legacy) proves solvability, replays the reference
  terminal state and certifies semantic relevance with pinned OPA.

The only lane-local logic is the staging root, the 20K dataset id/seed, the §1
check-status envelope, and the oracle directory relocation
(``bank/staging_20k/l01/oracle``). The legacy ``bank/staging/l01`` tree is never
written.

Usage (repo-relative):

    .venv/bin/python -m benchmark.plugins.task_family.l01.pipeline_20k
    .venv/bin/python -m benchmark.plugins.task_family.l01.pipeline_20k \
        --staging bank/staging_20k/_smoke/l01 --max-per-family 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    FetchReceipt,
    GroundedTask,
)
from benchmark.plugins.task_family.l01 import engine
from benchmark.plugins.task_family.l01 import expansion_20k
from benchmark.plugins.task_family.l01.families import GatekeeperTaskFamilies
from benchmark.plugins.task_family.l01.pipeline import (
    DedupIndex,
    ShardWriter,
    _leak_screen,
    _trim_evals,
    _trim_witness,
    _write_json_atomic,
)
from benchmark.plugins.verifier.l01.opa_verifier import GatekeeperOpaVerifier
from source_registry.fetchers import l01_gatekeeper as fetcher

REPO_ROOT = Path(__file__).resolve().parents[4]
LANE = "l01"
LANE_ID = "L01"
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
BUILD_ID = "aob-20000-candidate-dev1"
SEED_BASE = 20261101
STAGING_DEFAULT = REPO_ROOT / "bank" / "staging_20k" / LANE
GENERATOR_VERSION = expansion_20k.GENERATOR_VERSION

REQUIRED_TOP = (
    "schema_version", "task_id", "dataset_id", "lane_id", "family_id",
    "base_goal_id", "instance_id", "task_regime", "complexity",
    "semantic_profile", "public", "oracle_handle", "status", "qualification",
    "native_evidence", "source_snapshot_ids", "source_unit_ids", "checks",
)
REQUIRED_PUBLIC = ("user_goal", "allowed_tools", "budget", "observation_refs")


class GatekeeperOpaVerifier20K(GatekeeperOpaVerifier):
    """Legacy verifier with the oracle directory relocated to the 20K staging."""

    def __init__(self, oracle_dir: Path) -> None:
        self.oracle_dir = Path(oracle_dir)

    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "policy_units": ast["policy_units"],
            "guards": ast.get("guards", []),
            "reference_terminal_sha256": extra.get("reference_terminal_sha256"),
            "reference_answer": extra.get("reference_answer"),
            "engine": engine.ENGINE,
            "engine_version": engine.OPA_VERSION,
            "source_commit": engine.GK_COMMIT,
        }
        path = self.oracle_dir / ("%s.json" % grounded_task.oracle_ref)
        _write_json_atomic(path, payload)
        return path

    def _oracle(self, task: GroundedTask) -> dict[str, Any] | None:
        path = self.oracle_dir / ("%s.json" % task.oracle_ref)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _check_record_contract(record: dict[str, Any]) -> list[str]:
    """Structural §1 contract check for one record (no native claims)."""
    errors = [k for k in REQUIRED_TOP if k not in record]
    public = record.get("public") or {}
    errors += ["public.%s" % k for k in REQUIRED_PUBLIC if k not in public]
    if record.get("schema_version") != "aob-task/5.0":
        errors.append("schema_version")
    if record.get("dataset_id") != DATASET_ID:
        errors.append("dataset_id")
    if not isinstance(record.get("source_snapshot_ids"), list) or not record["source_snapshot_ids"]:
        errors.append("source_snapshot_ids")
    if not isinstance(record.get("source_unit_ids"), list) or not record["source_unit_ids"]:
        errors.append("source_unit_ids")
    if not isinstance(record.get("qualification"), dict) or len(record["qualification"]) < 6:
        errors.append("qualification_flags")
    if not isinstance(record.get("checks"), dict):
        errors.append("checks")
    return errors


def _new_checks() -> dict[str, str]:
    return {
        "source_verified": "PASS",
        "schema_valid": "NOT_RUN",
        "ontology_parse_valid": "NOT_APPLICABLE",
        "native_execution_status": "NOT_RUN",
        "oracle_check_status": "NOT_RUN",
        "mutation_check_status": "NOT_RUN",
        "leakage_check_status": "NOT_RUN",
        "human_review_status": "NOT_RUN",
        "qualification_status": "CANDIDATE",
    }


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = engine.sha256_json(
        [(p["unit"], p["sample"]) for p in ast.get("policy_units", [])]
    )[:16]
    public = dict(grounded.public)
    public["observation_refs"] = {
        "source_units": list(candidate.source_unit_ids),
        "objects": sorted((ast.get("objects") or {}).keys()),
    }
    checks = _new_checks()
    record = {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": LANE_ID,
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
        "source_snapshot_ids": ["gatekeeper-library@commit-a8d4799d8daf"],
        "source_unit_ids": list(candidate.source_unit_ids),
        "origin_mechanism_group": mech_group,
        "public": public,
        "oracle_handle": grounded.oracle_ref,
        "authorship": "deterministic_template",
        "provenance": "source_parameterized",
        "status": "CANDIDATE",
        "native_evidence": {},
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
        "checks": checks,
    }
    errors = _check_record_contract(record)
    if errors:
        raise ValueError("record contract violation: %s" % sorted(errors))
    checks["schema_valid"] = "PASS"
    return record


def _leak_screen_20k(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    return _leak_screen({k: v for k, v in public.items() if k != "observation_refs"}, ast)


def _qualify_one(
    plugin: GatekeeperTaskFamilies,
    verifier: GatekeeperOpaVerifier20K,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    checks = record["checks"]
    try:
        proof = verifier.prove_exists(grounded)
    except engine.OpaRunError as exc:
        checks["native_execution_status"] = "BLOCKED"
        checks["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    checks["native_execution_status"] = "PASS"
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in ("UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE",
                        "DUPLICATE", "NON_SEMANTIC", "LICENSE_HOLD"):
            code = "UNSOLVABLE"
        checks["oracle_check_status"] = "FAIL"
        checks["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    ast = candidate.ast
    if ast["goal_kind"] == "exact_decision_set":
        submitted = {"decisions": proof.evidence["reference_answer"]}
    else:
        submitted = {"objects": ast.get("reference_terminal") or {}}
    term = verifier.verify_terminal(grounded, submitted)
    if not term.valid:
        checks["oracle_check_status"] = "FAIL"
        checks["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {
            "code": "UNSOLVABLE",
            "reason": "reference solution failed terminal replay: %s"
                      % json.dumps(term.detail)[:300],
        }
    checks["oracle_check_status"] = "PASS"
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        checks["mutation_check_status"] = "FAIL"
        checks["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    checks["mutation_check_status"] = "PASS"
    leak = _leak_screen_20k(grounded.public, ast)
    if leak:
        checks["leakage_check_status"] = "FAIL"
        checks["qualification_status"] = "REJECTED"
        record["status"] = "REJECTED"
        return record, {"code": "LEAKAGE", "reason": "public payload leaks: %s" % leak[:120]}
    checks["leakage_check_status"] = "PASS"
    record["native_evidence"] = {
        "prove_exists": _trim_evals(proof.evidence),
        "verify_terminal": term.detail,
        "semantic_witness": _trim_witness(witness.witness),
    }
    record["qualification"] = {
        "native_verified": True,
        "solvable": True,
        "nontrivial": True,
        "semantic_relevance": True,
        "leakage_clean": True,
        "release_eligible": True,
    }
    checks["qualification_status"] = "QUALIFIED_NATIVE"
    record["status"] = "QUALIFIED"
    return record, None


def _verify_source() -> dict[str, Any]:
    src = fetcher.GatekeeperLibrarySource()
    release = src.discover_releases()[0]
    receipt = FetchReceipt(
        source_id=fetcher.SOURCE_ID,
        release_id=release.release_id,
        status="FETCHED_COMPLETE",
        bytes_downloaded=0,
        file_count=0,
        cache_dir=str(fetcher.PINNED_ROOT),
        sha256_manifest="",
        fetched_at="",
    )
    report = src.verify_snapshot(receipt)
    return {
        "ok": report.ok,
        "checks": report.checks,
        "missing": list(report.missing),
        "release_id": release.release_id,
    }


def _bump(coverage: dict[str, dict[str, int]], checks: dict[str, str]) -> None:
    for key, value in checks.items():
        coverage.setdefault(key, {})
        coverage[key][value] = coverage[key].get(value, 0) + 1


def run_build(
    *,
    staging: Path | None = None,
    families: list[str] | None = None,
    quota_scale: float = 1.0,
    instances_per_goal: int = 5,
    workers: int = 12,
    seed: int = SEED_BASE,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    expansion_20k.register()
    staging = Path(staging) if staging else STAGING_DEFAULT
    source = _verify_source()
    if not source["ok"]:
        raise RuntimeError("pinned source verification failed: %s" % source)

    plugin = GatekeeperTaskFamilies()
    verifier = GatekeeperOpaVerifier20K(staging / "oracle")
    snapshot = fetcher.load_snapshot()
    family_ids = families or expansion_20k.list_families()
    qualified_writer = ShardWriter(staging / "qualified")
    candidate_writer = ShardWriter(staging / "candidates")
    rejection_path = staging / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": LANE_ID,
        "build_id": BUILD_ID,
        "dataset_id": DATASET_ID,
        "seed_base": seed,
        "generator_version": GENERATOR_VERSION,
        "staging": str(staging),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_verification": source,
        "families": {},
        "rejections": {},
        "check_coverage": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
    }
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0}
            constraints = {"quota_scale": quota_scale,
                           "instances_per_goal": instances_per_goal}
            candidates = list(plugin.propose(snapshot, family=fid, seed=seed,
                                             constraints=constraints))
            if max_per_family:
                candidates = candidates[:max_per_family]
            # Threads only wait on OPA subprocesses; the writers stay on this thread.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_qualify_one, plugin, verifier, c)
                           for c in candidates]
                for fut in futures:
                    record, rejection = fut.result()
                    fam_stats["candidates"] += 1
                    stats["candidates"] += 1
                    candidate_writer.add(record)
                    _bump(stats["check_coverage"], record["checks"])
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
                    if dedup.admit(record):
                        fam_stats["qualified"] += 1
                        stats["qualified"] += 1
                        qualified_writer.add(record)
                    else:
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
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["distinct_instances"] = len(dedup.instances)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["expected_new"] = expansion_20k.expected_total()
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json_atomic(staging / "build_report.json", stats)
    _write_shard_manifest(staging, stats)
    return stats


def _write_shard_manifest(staging: Path, stats: dict[str, Any]) -> None:
    payload = {
        "lane": LANE_ID,
        "build_id": BUILD_ID,
        "dataset_id": DATASET_ID,
        "qualified": stats["qualified_manifests"],
        "candidates": stats["candidate_manifests"],
    }
    _write_json_atomic(staging / "manifest_20k.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="L01 20K expansion runner")
    parser.add_argument("--staging", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--quota-scale", type=float, default=1.0)
    parser.add_argument("--instances-per-goal", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--max-per-family", type=int, default=None)
    args = parser.parse_args()
    stats = run_build(
        staging=Path(args.staging) if args.staging else None,
        families=args.families,
        quota_scale=args.quota_scale,
        instances_per_goal=args.instances_per_goal,
        workers=args.workers,
        seed=args.seed,
        max_per_family=args.max_per_family,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())