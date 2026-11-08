from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autoontobench.contracts.models import assert_no_private_keys, canonical_sha
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit
from benchmark.plugins.source.l07.sbml_struct import LIBSBML_VERSION, parse_sbml_bytes
from benchmark.plugins.source.l07.simulate import PROTOCOL, load_roadrunner
from benchmark.plugins.task_family.l07.families import BY_ID, FAMILY_TABLE, PROFILE
from benchmark.plugins.task_family.l07.goals_sim import swap_species_in_formula
from benchmark.plugins.task_family.l07.propose import L07TaskFamilyPlugin
from benchmark.plugins.verifier.l07.verifier import SbmlNativeVerifier
from benchmark.schema.plugin_protocols import GroundedTask
from source_registry.fetchers.l07_biomodels import INDEX_ROOT, PINNED_ROOT, RELEASE_ID

DATASET_ID = "autoontobench-bank-8000-rc1"
LANE = "L07"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_text_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json(path: Path, payload: Any, *, mode: int | None = None) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=1, sort_keys=True) + "\n",
                       mode=mode)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _manifest(shard: str, family_id: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"shard": shard, "family_id": family_id, "path": str(path),
            "records": len(rows), "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _grouped(cands: Iterator[Any]) -> Iterator[tuple[str, list[Any]]]:
    current: str | None = None
    buf: list[Any] = []
    for cand in cands:
        if current is None:
            current = cand.base_goal_id
        if cand.base_goal_id != current:
            yield current, buf
            current = cand.base_goal_id
            buf = []
        buf.append(cand)
    if buf:
        yield current or "", buf


def _wrong_submission(verifier: SbmlNativeVerifier, grounded: GroundedTask,
                      ref: Any) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind in ("threshold_set", "role_query"):
        want = set(ref["answer_ids"])
        universe = set(ast.get("universe_ids") or ast["universe_reaction_ids"])
        if len(want) > 1:
            return {"answer_ids": sorted(want)[1:]}
        return {"answer_ids": sorted(want | set(sorted(universe - want)[:1]))}
    if kind == "ranking":
        order = list(ref["ranking"])
        if len(order) >= 2:
            order[-1], order[-2] = order[-2], order[-1]
        else:
            order = list(reversed(order))
        return {"ranking": order}
    if kind in ("shape_org", "species_role_org"):
        assignment = dict(ref["assignment"])
        keys = sorted(assignment)
        alt = next((k for k in keys[1:] if assignment[k] != assignment[keys[0]]), None)
        if alt is not None:
            assignment[keys[0]], assignment[alt] = assignment[alt], assignment[keys[0]]
        elif assignment:
            assignment[keys[0]] = "other" if assignment[keys[0]] != "other" else "flat"
        return {"assignment": assignment}
    if kind in ("param_threshold", "order_inversion"):
        accepted = sorted(ref["accepted_values"])
        if len(accepted) > 1:
            return {"accepted_values": accepted[1:]}
        extra = sorted(set(ast["candidates"]) - set(accepted))[:1]
        return {"accepted_values": accepted + extra}
    if kind == "param_repair":
        p0 = verifier.sim.struct(ast["model_id"]).parameter_value(ast["parameter"])
        return {"parameter_value": p0 * 3.0}
    if kind == "init_repair":
        info = verifier.sim.struct(ast["model_id"]).species_info(ast["species"])
        v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
        return {"initial_value": (v0 or 0.0) * 3.0}
    if kind == "law_repair":
        struct = verifier.sim.struct(ast["model_id"])
        formula = struct.reaction_info(ast["reaction"]).kinetic_formula or ""
        corrupted = swap_species_in_formula(
            formula, [tuple(p) for p in ast["swap_pairs"]])
        return {"repaired_formula": corrupted}
    raise ValueError(f"unknown kind {kind!r}")


def _leakage_ok(public: dict[str, Any], ast: dict[str, Any], ref: Any,
                verifier: SbmlNativeVerifier) -> bool:
    blob = json.dumps(public, sort_keys=True)
    kind = ast["kind"]
    if kind in ("threshold_set", "role_query"):
        return json.dumps(sorted(ref["answer_ids"])) not in blob
    if kind == "ranking":
        return json.dumps(ref["ranking"]) not in blob
    if kind in ("shape_org", "species_role_org"):
        return json.dumps(ref["assignment"], sort_keys=True) not in blob
    if kind in ("param_threshold", "order_inversion"):
        return json.dumps(sorted(ref["accepted_values"])) not in blob
    if kind == "param_repair":
        p0 = verifier.sim.struct(ast["model_id"]).parameter_value(ast["parameter"])
        edited = parse_sbml_bytes(public["model"]["sbml_xml"].encode("utf-8"))
        value = edited.parameter_value(ast["parameter"])
        return abs(value - p0 * ast["corrupt_factor"]) <= abs(p0) * 1e-12
    if kind == "init_repair":
        info = verifier.sim.struct(ast["model_id"]).species_info(ast["species"])
        v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
        edited = parse_sbml_bytes(public["model"]["sbml_xml"].encode("utf-8"))
        new_info = edited.species_info(ast["species"])
        value = new_info.initial_amount if new_info.initial_amount is not None else new_info.initial_concentration
        return v0 is not None and value is not None and abs(value - v0 * ast["corrupt_factor"]) <= abs(v0) * 1e-12
    if kind == "law_repair":
        formula = verifier.sim.struct(ast["model_id"]).reaction_info(ast["reaction"]).kinetic_formula or ""
        return formula not in blob
    return False


def _qualify(plugin: L07TaskFamilyPlugin, verifier: SbmlNativeVerifier,
             grounded: GroundedTask) -> dict[str, Any]:
    ast = grounded.ast.ast
    row: dict[str, Any] = {"grounded": grounded, "qualified": False, "rejection": "",
                           "reason_code": "", "model_id": ast["model_id"],
                           "kind": ast["kind"]}
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:
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
    except Exception as exc:
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
        leak_free = _leakage_ok(public, ast, ref, verifier)
    except Exception as exc:
        row["rejection"] = f"NATIVE_ERROR:leakage_check:{type(exc).__name__}"[:300]
        row["reason_code"] = "NATIVE_ERROR"
        return row
    if not leak_free:
        row["rejection"] = "LEAKAGE:answer_in_public"
        row["reason_code"] = "LEAKAGE"
        return row
    row.update(qualified=True, proof=proof, witness=witness, reference=ref,
               self_check={"positive": positive.valid, "negative_must_fail": negative.valid})
    return row


def _solver_block(ast: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "engine": "libsbml+libroadrunner",
        "libsbml_version": LIBSBML_VERSION,
        "roadrunner_version": load_roadrunner().__version__,
    }
    if ast["kind"] not in ("role_query", "species_role_org"):
        block["protocol"] = dict(PROTOCOL)
    return block


def _record(row: dict[str, Any], spec: Any, public_ref: str,
            oracle_rel: str) -> dict[str, Any]:
    grounded: GroundedTask = row["grounded"]
    cand = grounded.ast
    ast = cand.ast
    task_id = "l07-" + canonical_sha({"instance": cand.instance_id, "ast": ast})[:16]
    model_id = ast["model_id"]
    public_budget = grounded.public["budget"]
    solver = _solver_block(ast)
    reference_sha = _sha256_bytes(json.dumps(row["reference"], sort_keys=True).encode())
    witness = row["witness"].witness
    native_evidence = {
        "verdict": "PASS",
        "commands": [f".venv/bin/python -m benchmark.plugins.task_family.l07.verify_cli --oracle {oracle_rel}"],
        "solver": solver,
        "reference_sha256": reference_sha,
        "differential_agree": bool(row["proof"].evidence.get("differential_agree", True)),
        "nontrivial_gate": True,
        "semantic_witness": {k: witness.get(k) for k in
                             ("counterfactual", "answer_diff", "max_rel_diff") if k in witness},
        "self_check": row["self_check"],
    }
    public = {
        "user_goal": grounded.public["user_goal"],
        "allowed_tools": list(grounded.public["allowed_tools"]),
        "budget": {"actions": int(public_budget["actions"]),
                   "probes": int(public_budget["probes"])},
        "observation_refs": [
            public_ref,
            f"external_knowledge/pinned/biomodels/{RELEASE_ID}/models/{model_id}",
        ],
        "semantic_profile": PROFILE,
        "complexity": ast["complexity"],
        "in_silico": True,
        "solver": solver,
    }
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
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "split": "dev",
        "seed": 0,
    }


def _oracle_payload(row: dict[str, Any], task_id: str, verifier: SbmlNativeVerifier) -> dict[str, Any]:
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


def _cand_row(row: dict[str, Any]) -> dict[str, Any]:
    cand = row["grounded"].ast
    return {"instance_id": cand.instance_id, "family_id": cand.family_id,
            "base_goal_id": cand.base_goal_id, "qualified": row["qualified"],
            "rejection": row["rejection"], "reason_code": row["reason_code"],
            "model_id": row["model_id"], "kind": row["kind"]}


def _rejection_row(row: dict[str, Any]) -> dict[str, Any]:
    cand = row["grounded"].ast
    return {"instance_id": cand.instance_id, "family_id": cand.family_id,
            "base_goal_id": cand.base_goal_id, "reason_code": row["reason_code"],
            "reason": row["rejection"], "model_id": row["model_id"],
            "kind": row["kind"]}


def _load_existing(out_root: Path, shard_no: int, spec: Any) -> dict[str, Any] | None:
    qpath = out_root / "qualified" / f"shard-{shard_no:04d}.jsonl"
    cpath = out_root / "candidates" / f"shard-{shard_no:04d}.jsonl"
    qman = out_root / "qualified" / f"shard-{shard_no:04d}.manifest.json"
    if not (qpath.is_file() and cpath.is_file() and qman.is_file()):
        return None
    qrows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line]
    crows = [json.loads(line) for line in cpath.read_text(encoding="utf-8").splitlines() if line]
    if len(qrows) != spec.base_goals * spec.instances_per_goal:
        return None
    rejections: dict[str, int] = {}
    for row in crows:
        if not row["qualified"]:
            rejections[row["reason_code"]] = rejections.get(row["reason_code"], 0) + 1
    return {"candidates": len(crows), "qualified": len(qrows),
            "base_goals": len({r["base_goal_id"] for r in qrows}),
            "rejections": rejections, "resumed": True}


def build_shards(index_dir: Path, out_root: Path, *, seed: int = 20261027,
                 max_attempt_goals: int = 24, families: list[str] | None = None,
                 resume: bool = True) -> dict[str, Any]:
    started = _utcnow()
    t0 = time.time()
    plugin = L07TaskFamilyPlugin(index_dir, seed=seed)
    verifier = SbmlNativeVerifier(index_dir)
    out_root = Path(out_root)
    for sub in ("candidates", "qualified", "oracle", "public", "rejections"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    os.chmod(out_root / "oracle", 0o700)
    stats: dict[str, Any] = {
        "lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
        "index_dir": str(index_dir), "out_dir": str(out_root),
        "target": {"instances": 400, "base_goals": 80, "families": 10},
        "candidates": 0, "qualified": 0, "base_goals": 0, "families_count": 0,
        "families_detail": {}, "rejections": {}, "shards": [], "gaps": [],
        "libsbml_version": LIBSBML_VERSION,
        "roadrunner_version": load_roadrunner().__version__,
        "protocol": dict(PROTOCOL), "started_at": started,
    }
    rejection_rows: list[dict[str, Any]] = []
    shard_no = 0
    for spec in FAMILY_TABLE:
        if families and spec.family_id not in families:
            continue
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
        generator = plugin.propose(None, family=spec.family_id, seed=seed,
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
                    oracle_rel = f"bank/staging/l07/oracle/{row['grounded'].oracle_ref}.json"
                    public_ref = f"bank/staging/l07/public/{cand.instance_id}.json"
                    oracle_path = out_root / "oracle" / f"{row['grounded'].oracle_ref}.json"
                    public_path = out_root / "public" / f"{cand.instance_id}.json"
                    _write_json(public_path, row["grounded"].public, mode=0o644)
                    _write_json(oracle_path, _oracle_payload(row, task_id, verifier),
                                mode=0o600)
                    qual_rows.append(_record(row, spec, public_ref, oracle_rel))
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
        _write_json(out_root / "candidates" / f"shard-{shard_no:04d}.manifest.json",
                    cand_manifest)
        _write_json(out_root / "qualified" / f"shard-{shard_no:04d}.manifest.json",
                    qual_manifest)
        fam_rejections: dict[str, int] = {}
        for row in fam_rej:
            fam_rejections[row["reason_code"]] = fam_rejections.get(row["reason_code"], 0) + 1
        detail = {"candidates": len(cand_rows), "qualified": len(qual_rows),
                  "base_goals": complete_goals, "attempted_base_goals": attempted_goals,
                  "rejections": fam_rejections}
        stats["families_detail"][spec.family_id] = detail
        stats["candidates"] += len(cand_rows)
        stats["qualified"] += len(qual_rows)
        stats["base_goals"] += complete_goals
        stats["families_count"] += 1
        for code, n in fam_rejections.items():
            stats["rejections"][code] = stats["rejections"].get(code, 0) + n
        rejection_rows.extend(fam_rej)
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
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    _write_jsonl(out_root / "rejections" / "rejections.jsonl", all_rejections)
    oracle_manifest: list[dict[str, Any]] = []
    for path in sorted((out_root / "oracle").glob("*.json")):
        raw = path.read_bytes()
        oracle_manifest.append({"path": str(path), "bytes": len(raw),
                                "sha256": _sha256_bytes(raw)})
    _write_json(out_root / "oracle" / "manifest.json",
                {"count": len(oracle_manifest), "files": oracle_manifest}, mode=0o600)
    stats["wall_seconds"] = round(time.time() - t0, 1)
    stats["finished_at"] = _utcnow()
    stats["reproduce"] = {
        "build": (f".venv/bin/python -m benchmark.plugins.task_family.l07.build "
                  f"--index {index_dir} --out {out_root} --seed {seed}"),
        "verify_sample": ("for i in $(seq 1 3); do .venv/bin/python -m "
                          "benchmark.plugins.task_family.l07.verify_cli --oracle "
                          "$(ls bank/staging/l07/oracle/orl07_*.json | shuf -n 1); done"),
    }
    _write_json(out_root / "build_report.json", stats)
    _write_json(out_root / "manifest.json",
                {"lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
                 "candidates": stats["candidates"], "qualified": stats["qualified"],
                 "base_goals": stats["base_goals"], "families": stats["families_count"],
                 "shards": stats["shards"]})
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L07 qualified shards")
    ap.add_argument("--index", default=str(INDEX_ROOT))
    ap.add_argument("--out", default="bank/staging/l07")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--max-attempt-goals", type=int, default=24)
    ap.add_argument("--families", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    fams = args.families.split(",") if args.families else None
    stats = build_shards(Path(args.index), Path(args.out), seed=args.seed,
                         max_attempt_goals=args.max_attempt_goals, families=fams,
                         resume=not args.no_resume)
    print(json.dumps({k: stats[k] for k in
                      ("candidates", "qualified", "base_goals", "families_count",
                       "rejections", "gaps", "wall_seconds")}, indent=1, sort_keys=True))
    return 0 if stats["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())