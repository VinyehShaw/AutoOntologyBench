"""Build the L07 20K-candidate expansion: 550 new SBML tasks.

Writes to ``bank/staging_20k/l07`` by default and never touches the legacy
``bank/staging/l07`` (8K) tree. The build is shard-incremental and resumable:
a complete shard is loaded from disk only when its qualified row count matches
the family target and its candidate/qualified/manifest files all exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import assert_no_private_keys, canonical_sha
from benchmark.plugins.source.l07.sbml_struct import LIBSBML_VERSION
from benchmark.plugins.source.l07.simulate import PROTOCOL, load_roadrunner
from benchmark.plugins.task_family.l07.build import (
    _cand_row, _grouped, _load_existing, _manifest, _rejection_row,
    _utcnow, _write_json, _write_jsonl,
)
from benchmark.plugins.task_family.l07.families_20k import (
    FAMILY_TABLE_20K, SIM_KINDS_20K,
)
from benchmark.plugins.task_family.l07.propose_20k import (
    L07TaskFamilyPlugin20K, legacy_models,
)
from benchmark.plugins.verifier.l07.verifier_20k import SbmlNativeVerifier20K
from benchmark.schema.plugin_protocols import GroundedTask
from source_registry.fetchers.l07_biomodels import INDEX_ROOT, RELEASE_ID

DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
LANE = "L07"
DEFAULT_SEED = 20261101
DEFAULT_OUT = Path("bank/staging_20k/l07")
TARGET_INSTANCES = 550

SET_KINDS = ("retention_set", "pulse_set", "reaction_degree_query")
INIT_KINDS = ("init_threshold_set", "init_window_set")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _legacy_qualified_count(root: Path = Path("bank/staging/l07")) -> int:
    total = 0
    for path in sorted((root / "qualified").glob("shard-*.jsonl")):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                     if line)
    return total


def _solver_block(ast: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "engine": "libsbml+libroadrunner",
        "libsbml_version": LIBSBML_VERSION,
        "roadrunner_version": load_roadrunner().__version__,
    }
    if ast["kind"] in SIM_KINDS_20K:
        block["protocol"] = dict(PROTOCOL)
    return block


def _wrong_submission(verifier: SbmlNativeVerifier20K, grounded: GroundedTask,
                      ref: Any) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind in SET_KINDS:
        want = set(ref["answer_ids"])
        universe = set(ast.get("universe_ids") or ast["universe_reaction_ids"])
        if len(want) > 1:
            return {"answer_ids": sorted(want)[1:]}
        return {"answer_ids": sorted(want | set(sorted(universe - want)[:1]))}
    if kind == "peak_time_ranking":
        order = list(ref["ranking"])
        if len(order) >= 2:
            order[-1], order[-2] = order[-2], order[-1]
        else:
            order = list(reversed(order))
        return {"ranking": order}
    if kind in INIT_KINDS:
        accepted = sorted(ref["accepted_values"])
        if len(accepted) > 1:
            return {"accepted_values": accepted[1:]}
        extra = sorted(set(ast["candidates"]) - set(accepted))[:1]
        return {"accepted_values": accepted + extra}
    raise ValueError(f"unknown kind {kind!r}")


def _leakage_ok(public: dict[str, Any], ast: dict[str, Any], ref: Any) -> bool:
    blob = json.dumps(public, sort_keys=True)
    kind = ast["kind"]
    if kind in SET_KINDS:
        return json.dumps(sorted(ref["answer_ids"])) not in blob
    if kind == "peak_time_ranking":
        return json.dumps(ref["ranking"]) not in blob
    if kind in INIT_KINDS:
        return json.dumps(sorted(ref["accepted_values"])) not in blob
    return False


def _qualify(plugin: L07TaskFamilyPlugin20K, verifier: SbmlNativeVerifier20K,
             grounded: GroundedTask) -> dict[str, Any]:
    ast = grounded.ast.ast
    row: dict[str, Any] = {"grounded": grounded, "qualified": False, "rejection": "",
                           "reason_code": "", "model_id": ast["model_id"],
                           "kind": ast["kind"]}
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:  # noqa: BLE001 - recorded as a status, never swallowed
        row["rejection"] = f"NATIVE_ERROR:{type(exc).__name__}:{exc}"[:300]
        row["reason_code"] = "NATIVE_ERROR"
        return row
    if not proof.exists:
        row["rejection"] = proof.reason or "PROFILE_UNSUPPORTED"
        row["reason_code"] = row["rejection"].split(":", 1)[0]
        return row
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        row["rejection"] = "NON_SEMANTIC:no_counterfactual"
        row["reason_code"] = "NON_SEMANTIC"
        return row
    try:
        ref = verifier.reference_submission(ast)
        positive = verifier.verify_terminal(grounded, ref)
        negative = verifier.verify_terminal(
            grounded, _wrong_submission(verifier, grounded, ref))
    except Exception as exc:  # noqa: BLE001
        row["rejection"] = f"NATIVE_ERROR:{type(exc).__name__}:{exc}"[:300]
        row["reason_code"] = "NATIVE_ERROR"
        return row
    if not positive.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_positive_failed"
        row["reason_code"] = "PROFILE_UNSUPPORTED"
        return row
    if negative.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_negative_failed"
        row["reason_code"] = "PROFILE_UNSUPPORTED"
        return row
    public = grounded.public
    try:
        assert_no_private_keys(public)
    except ValueError as exc:
        row["rejection"] = f"LEAKAGE:private_key:{exc}"[:300]
        row["reason_code"] = "LEAKAGE"
        return row
    try:
        leak_free = _leakage_ok(public, ast, ref)
    except Exception as exc:  # noqa: BLE001
        row["rejection"] = f"NATIVE_ERROR:leakage_check:{type(exc).__name__}"[:300]
        row["reason_code"] = "NATIVE_ERROR"
        return row
    if not leak_free:
        row["rejection"] = "LEAKAGE:answer_in_public"
        row["reason_code"] = "LEAKAGE"
        return row
    row.update(qualified=True, proof=proof, witness=witness, reference=ref,
               self_check={"positive": positive.valid,
                           "negative_must_fail": negative.valid})
    return row


def _semantic_signature(ast: dict[str, Any]) -> str:
    """Semantics-only identity (model + predicate parameters), for dedup audits."""
    return canonical_sha(ast)


def _record(row: dict[str, Any], spec: Any, public_ref: str,
            oracle_rel: str, seed: int) -> dict[str, Any]:
    grounded: GroundedTask = row["grounded"]
    cand = grounded.ast
    ast = cand.ast
    task_id = "l07-" + canonical_sha(
        {"instance": cand.instance_id, "ast": ast})[:16]
    solver = _solver_block(ast)
    reference_sha = _sha256_bytes(
        json.dumps(row["reference"], sort_keys=True).encode())
    witness = row["witness"].witness
    native_evidence = {
        "verdict": "PASS",
        "commands": [f".venv/bin/python -m "
                     f"benchmark.plugins.task_family.l07.verify_cli_20k "
                     f"--oracle {oracle_rel}"],
        "solver": solver,
        "reference_sha256": reference_sha,
        "differential_agree": bool(
            row["proof"].evidence.get("differential_agree", True)),
        "differential_kind": row["proof"].evidence.get("differential_kind"),
        "nontrivial_gate": True,
        "semantic_witness": {k: witness.get(k) for k in
                             ("counterfactual", "answer_diff", "max_rel_diff")
                             if k in witness},
        "self_check": row["self_check"],
    }
    public = dict(grounded.public)
    public["observation_refs"] = [
        public_ref,
        f"external_knowledge/pinned/biomodels/{RELEASE_ID}/models/{ast['model_id']}",
    ]
    public["solver"] = solver
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "family_id": cand.family_id,
        "base_goal_id": cand.base_goal_id,
        "instance_id": cand.instance_id,
        "task_id": task_id,
        "task_regime": spec.regime,
        "task_family": spec.regime,
        "complexity": ast["complexity"],
        "semantic_profile": cand.semantic_profile,
        "domain": "systems_biology",
        "track": "numeric",
        "public": public,
        "oracle_handle": grounded.oracle_ref,
        "status": "QUALIFIED",
        "qualification": {
            "native_verified": True,
            "solvable": True,
            "nontrivial": True,
            "semantic_relevance": True,
            "leakage_clean": True,
            "release_eligible": True,
        },
        "native_evidence": native_evidence,
        "source_snapshot_ids": [f"biomodels/{RELEASE_ID}"],
        "source_unit_ids": list(cand.source_unit_ids),
        "source_model_id": ast["model_id"],
        "semantic_signature": _semantic_signature(ast),
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "split": "dev",
        "seed": seed,
    }


def _oracle_payload(row: dict[str, Any], task_id: str,
                    verifier: SbmlNativeVerifier20K) -> dict[str, Any]:
    grounded: GroundedTask = row["grounded"]
    ast = grounded.ast.ast
    return {
        "oracle_handle": grounded.oracle_ref,
        "task_id": task_id,
        "instance_id": grounded.ast.instance_id,
        "family_id": grounded.ast.family_id,
        "ast": ast,
        "reference_submission": row["reference"],
        "proof": row["proof"].evidence,
        "witness": row["witness"].witness,
        "self_check": row["self_check"],
        "solver": _solver_block(ast),
        "model_sha256": verifier.sim.struct(ast["model_id"]).sha256,
    }


def build_shards(index_dir: Path, out_root: Path, *, seed: int = DEFAULT_SEED,
                 max_attempt_goals: int = 80, resume: bool = True
                 ) -> dict[str, Any]:
    started = _utcnow()
    t0 = time.time()
    excluded = legacy_models()
    legacy_count = _legacy_qualified_count()
    plugin = L07TaskFamilyPlugin20K(index_dir, seed=seed,
                                    exclude_models=excluded)
    verifier = SbmlNativeVerifier20K(index_dir)
    out_root = Path(out_root)
    for sub in ("candidates", "qualified", "oracle", "public", "rejections"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    target_goals = sum(spec.base_goals for spec in FAMILY_TABLE_20K)
    stats: dict[str, Any] = {
        "lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
        "index_dir": str(index_dir), "out_dir": str(out_root),
        "target": {"instances": TARGET_INSTANCES, "base_goals": target_goals,
                   "families": len(FAMILY_TABLE_20K)},
        "seed": seed,
        "candidates": 0, "qualified": 0, "base_goals": 0,
        "families_count": 0, "families_detail": {}, "rejections": {},
        "status_counts": {}, "shards": [], "gaps": [],
        "legacy_models_excluded": len(excluded),
        "legacy_models_excluded_sha256": _sha256_bytes(
            json.dumps(sorted(excluded), sort_keys=True).encode()),
        "legacy_count": legacy_count,
        "lane_total_target": legacy_count + TARGET_INSTANCES,
        "libsbml_version": LIBSBML_VERSION,
        "roadrunner_version": load_roadrunner().__version__,
        "protocol": dict(PROTOCOL), "started_at": started,
    }
    shard_no = 0
    for spec in FAMILY_TABLE_20K:
        existing = _load_existing(out_root, shard_no, spec) if resume else None
        if existing is not None:
            stats["candidates"] += existing["candidates"]
            stats["qualified"] += existing["qualified"]
            stats["base_goals"] += existing["base_goals"]
            stats["families_count"] += 1
            stats["families_detail"][spec.family_id] = existing
            for code, n in existing["rejections"].items():
                stats["rejections"][code] = stats["rejections"].get(code, 0) + n
            stats["shards"].append({"shard": f"shard-{shard_no:04d}",
                                    "family_id": spec.family_id,
                                    "qualified": existing["qualified"],
                                    "resumed": True})
            shard_no += 1
            continue
        cand_rows: list[dict[str, Any]] = []
        qual_rows: list[dict[str, Any]] = []
        fam_rej: list[dict[str, Any]] = []
        complete_goals = 0
        attempted_goals = 0
        generator = plugin.propose(
            None, family=spec.family_id, seed=seed,
            constraints={"max_base_goals": max_attempt_goals})
        for _base_goal_id, cands in _grouped(generator):
            attempted_goals += 1
            rows = [_qualify(plugin, verifier, plugin.ground(c)) for c in cands]
            complete = (len(rows) == spec.instances_per_goal
                        and all(r["qualified"] for r in rows))
            if complete:
                complete_goals += 1
                for row in rows:
                    cand = row["grounded"].ast
                    task_id = "l07-" + canonical_sha(
                        {"instance": cand.instance_id, "ast": cand.ast})[:16]
                    oracle_rel = (f"{out_root.as_posix()}/oracle/"
                                  f"{row['grounded'].oracle_ref}.json")
                    public_ref = (f"{out_root.as_posix()}/public/"
                                  f"{cand.instance_id}.json")
                    oracle_path = (out_root / "oracle"
                                   / f"{row['grounded'].oracle_ref}.json")
                    public_path = out_root / "public" / f"{cand.instance_id}.json"
                    _write_json(public_path, row["grounded"].public, mode=0o644)
                    _write_json(oracle_path,
                                _oracle_payload(row, task_id, verifier),
                                mode=0o600)
                    qual_rows.append(_record(row, spec, public_ref, oracle_rel, seed))
                    cand_rows.append(_cand_row(row))
            else:
                for row in rows:
                    if row["qualified"]:
                        row["qualified"] = False
                        row["reason_code"] = "BASE_GOAL_INCOMPLETE"
                        row["rejection"] = "BASE_GOAL_INCOMPLETE:peer_rejected"
                    cand_rows.append(_cand_row(row))
                    fam_rej.append(_rejection_row(row))
            if complete_goals >= spec.base_goals:
                break
        candidates_path = out_root / "candidates" / f"shard-{shard_no:04d}.jsonl"
        qualified_path = out_root / "qualified" / f"shard-{shard_no:04d}.jsonl"
        rejection_path = out_root / "rejections" / f"shard-{shard_no:04d}.jsonl"
        _write_jsonl(candidates_path, cand_rows)
        _write_jsonl(qualified_path, qual_rows)
        _write_jsonl(rejection_path, fam_rej)
        cand_manifest = _manifest(f"shard-{shard_no:04d}", spec.family_id,
                                  candidates_path, cand_rows)
        qual_manifest = _manifest(f"shard-{shard_no:04d}", spec.family_id,
                                  qualified_path, qual_rows)
        _write_json(out_root / "candidates"
                    / f"shard-{shard_no:04d}.manifest.json", cand_manifest)
        _write_json(out_root / "qualified"
                    / f"shard-{shard_no:04d}.manifest.json", qual_manifest)
        fam_rejections: dict[str, int] = {}
        for row in fam_rej:
            fam_rejections[row["reason_code"]] = (
                fam_rejections.get(row["reason_code"], 0) + 1)
        detail = {"candidates": len(cand_rows), "qualified": len(qual_rows),
                  "base_goals": complete_goals,
                  "attempted_base_goals": attempted_goals,
                  "rejections": fam_rejections}
        stats["families_detail"][spec.family_id] = detail
        stats["candidates"] += len(cand_rows)
        stats["qualified"] += len(qual_rows)
        stats["base_goals"] += complete_goals
        stats["families_count"] += 1
        for code, n in fam_rejections.items():
            stats["rejections"][code] = stats["rejections"].get(code, 0) + n
        stats["shards"].append({"shard": f"shard-{shard_no:04d}",
                                "family_id": spec.family_id,
                                "qualified": len(qual_rows),
                                "candidates": len(cand_rows),
                                "manifest_sha256": qual_manifest["sha256"]})
        if len(qual_rows) < spec.base_goals * spec.instances_per_goal:
            stats["gaps"].append({
                "family_id": spec.family_id,
                "expected_instances": spec.base_goals * spec.instances_per_goal,
                "actual_instances": len(qual_rows),
                "expected_goals": spec.base_goals,
                "actual_goals": complete_goals,
                "attempted_goals": attempted_goals,
            })
        _write_json(out_root / "build_report.json", stats)
        shard_no += 1
    all_rejections: list[dict[str, Any]] = []
    for path in sorted((out_root / "rejections").glob("shard-*.jsonl")):
        all_rejections.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line)
    _write_jsonl(out_root / "rejections" / "rejections.jsonl", all_rejections)
    oracle_manifest: list[dict[str, Any]] = []
    for path in sorted((out_root / "oracle").glob("*.json")):
        raw = path.read_bytes()
        oracle_manifest.append({"path": str(path), "bytes": len(raw),
                                "sha256": _sha256_bytes(raw)})
    _write_json(out_root / "oracle" / "manifest.json",
                {"count": len(oracle_manifest), "files": oracle_manifest},
                mode=0o600)
    stats["wall_seconds"] = round(time.time() - t0, 1)
    stats["finished_at"] = _utcnow()
    total_rej = sum(stats["rejections"].values())
    stats["status_counts"] = {"QUALIFIED": stats["qualified"],
                              "REJECTED": total_rej}
    stats["lane_total_qualified"] = stats["legacy_count"] + stats["qualified"]
    stats["reproduce"] = {
        "build": (f".venv/bin/python -m benchmark.plugins.task_family.l07.build_20k "
                  f"--index {index_dir} --out {out_root} --seed {seed}"),
        "verify_sample": (
            ".venv/bin/python -m benchmark.plugins.task_family.l07.verify_cli_20k "
            "--oracle $(ls bank/staging_20k/l07/oracle/orl07_*.json | shuf -n 1)"),
        "collisions": (".venv/bin/python "
                       "scripts/check_l07_20k_collisions.py"),
    }
    _write_json(out_root / "build_report.json", stats)
    _write_json(out_root / "manifest.json", {
        "lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
        "seed": seed, "candidates": stats["candidates"],
        "qualified": stats["qualified"], "base_goals": stats["base_goals"],
        "legacy_count": stats.get("legacy_count"),
        "lane_total_qualified": stats.get("lane_total_qualified"),
        "families": stats["families_count"], "shards": stats["shards"],
    })
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L07 20K-candidate shards")
    ap.add_argument("--index", default=str(INDEX_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-attempt-goals", type=int, default=80)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    stats = build_shards(Path(args.index), Path(args.out), seed=args.seed,
                         max_attempt_goals=args.max_attempt_goals,
                         resume=not args.no_resume)
    print(json.dumps({k: stats[k] for k in
                      ("candidates", "qualified", "base_goals",
                       "families_count", "rejections", "gaps",
                       "wall_seconds")}, indent=1, sort_keys=True))
    return 0 if stats["qualified"] == TARGET_INSTANCES else 1


if __name__ == "__main__":
    raise SystemExit(main())