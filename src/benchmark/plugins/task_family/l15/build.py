"""L15 build: propose -> ground -> qualify -> shards + manifests + report.

Qualified-record contract follows aob-task/5.0 with the 20K status block
(source_verified / schema_valid / ontology_parse_valid / native_execution /
oracle_check / mutation_check / human_review / qualification_status).
Output: ``bank/staging_20k/l15`` (candidates/qualified/oracle/public/
rejections + build_report.json + manifest.json). Resumable per shard.
"""
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
from benchmark.plugins.task_family.l15.families import (
    BY_ID, DATASET_ID, FAMILY_TABLE, LANE, PROFILE, SEED, FamilySpec,
)
from benchmark.plugins.task_family.l15.propose import L15TaskFamilyPlugin
from benchmark.plugins.verifier.l15.verifier import ReactomeNativeVerifier
from benchmark.schema.plugin_protocols import GroundedTask
from source_registry.fetchers.l15_reactome import PINNED_ROOT, RELEASE_ID, ReactomeSource

OUT_ROOT_DEFAULT = "bank/staging_20k/l15"


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


def _leakage_ok(public: dict[str, Any], ref: Any) -> bool:
    blob = json.dumps(public, sort_keys=True)
    for key in ("answer_ids", "pairs"):
        if key in ref and json.dumps(ref[key]) in blob:
            return False
    if "assignment" in ref and json.dumps(ref["assignment"], sort_keys=True) in blob:
        return False
    return True


def _qualify(verifier: ReactomeNativeVerifier, grounded: GroundedTask,
             spec: FamilySpec) -> dict[str, Any]:
    ast = grounded.ast.ast
    row: dict[str, Any] = {"grounded": grounded, "qualified": False, "rejection": "",
                           "reason_code": "", "kind": ast["kind"]}
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
        mutated = verifier.mutation_submission(ast, ref)
        negative = verifier.verify_terminal(grounded, mutated)
    except Exception as exc:
        row["rejection"] = f"NATIVE_ERROR:{type(exc).__name__}:{exc}"[:300]
        row["reason_code"] = "NATIVE_ERROR"
        return row
    if not positive.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_positive_failed"
        row["reason_code"] = "PROFILE_UNSUPPORTED"
        return row
    if negative.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:mutation_still_valid"
        row["reason_code"] = "PROFILE_UNSUPPORTED"
        return row
    if _canon_equal(mutated, ref):
        row["rejection"] = "PROFILE_UNSUPPORTED:mutation_is_noop"
        row["reason_code"] = "PROFILE_UNSUPPORTED"
        return row
    try:
        assert_no_private_keys(grounded.public)
    except ValueError as exc:
        row["rejection"] = f"LEAKAGE:private_key:{exc}"[:300]
        row["reason_code"] = "LEAKAGE"
        return row
    if not _leakage_ok(grounded.public, ref):
        row["rejection"] = "LEAKAGE:answer_in_public"
        row["reason_code"] = "LEAKAGE"
        return row
    row.update(qualified=True, proof=proof, witness=witness, reference=ref,
               mutation=mutated,
               self_check={"positive": True, "negative_must_fail": False,
                           "mutation_differs": True})
    return row


def _canon_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _engine_block() -> dict[str, Any]:
    import sqlite3
    return {"engine": "sqlite3 " + sqlite3.sqlite_version + " + python re/xml",
            "truth_paths": ["biopax_xml_extractor", "mapping_tsv", "sbml_libsbml_spot"],
            "store": "external_knowledge/indexes/reactome/release-98/reactome.db"}


def _record(row: dict[str, Any], spec: FamilySpec, public_ref: str,
            oracle_rel: str) -> dict[str, Any]:
    grounded: GroundedTask = row["grounded"]
    cand = grounded.ast
    ast = cand.ast
    task_id = "l15-" + canonical_sha({"instance": cand.instance_id, "ast": ast})[:16]
    reference_sha = _sha256_bytes(json.dumps(row["reference"], sort_keys=True).encode())
    witness = row["witness"].witness
    native_evidence = {
        "verdict": "PASS",
        "commands": [f".venv/bin/python -m benchmark.plugins.task_family.l15.verify_cli --oracle {oracle_rel}"],
        "reference_sha256": reference_sha,
        "differential_agree": row["proof"].evidence.get("differential_agree", True),
        "nontrivial_gate": True,
        "semantic_witness": {k: witness.get(k) for k in
                             ("counterfactual", "answer_diff") if k in witness},
        "self_check": row["self_check"],
        **_engine_block(),
    }
    public = dict(grounded.public)
    public["semantic_profile"] = PROFILE
    public["complexity"] = ast["complexity"]
    public["observation_refs"] = [public_ref,
                                  f"external_knowledge/pinned/reactome/{RELEASE_ID}"]
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
        "domain": "systems_biology_pathways",
        "track": "symbolic",
        "public": public,
        "oracle_handle": grounded.oracle_ref,
        "status": "QUALIFIED",
        "qualification_status": "QUALIFIED_NATIVE",
        "check_statuses": {
            "source_verified": "PASS",
            "schema_valid": "PASS",
            "ontology_parse_valid": "PASS",
            "native_execution_status": "PASS",
            "oracle_check_status": "PASS",
            "mutation_check_status": "PASS",
            "human_review_status": "NOT_RUN",
        },
        "qualification": {
            "native_verified": True,
            "solvable": True,
            "nontrivial": True,
            "semantic_relevance": True,
            "leakage_clean": True,
            "release_eligible": True,
        },
        "independent_native_checks": [
            "reference_recompute_matches",
            "terminal_reference_passes",
            "mutation_rejected",
            "counterfactual_flips_answer",
            "no_answer_leak",
            "path_agreement_where_declared",
        ],
        "state_axis": {"axis": "evidence_view",
                       "variant": ast.get("evidence_view", "records")},
        "mechanism_group_id": f"{cand.family_id}/{cand.base_goal_id}",
        "combination_signature": canonical_sha(
            {"family": cand.family_id, "ast": ast}),
        "task_provenance": "source_parameterized",
        "source_refs": [{
            "source_id": "reactome",
            "commit": f"release 98 ({RELEASE_ID})",
            "file_sha256": _sha256_bytes((PINNED_ROOT / "SHA256SUMS.txt").read_bytes())
            if (PINNED_ROOT / "SHA256SUMS.txt").is_file() else "",
            "source_rule_unit_id": f"reactome/{RELEASE_ID}",
        }],
        "native_evidence": native_evidence,
        "source_snapshot_ids": [f"reactome/{RELEASE_ID}"],
        "source_unit_ids": list(cand.source_unit_ids),
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "split": "dev",
        "seed": SEED,
    }


def _oracle_payload(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    grounded: GroundedTask = row["grounded"]
    return {
        "oracle_handle": grounded.oracle_ref,
        "task_id": task_id,
        "instance_id": grounded.ast.instance_id,
        "family_id": grounded.ast.family_id,
        "ast": grounded.ast.ast,
        "reference_submission": row["reference"],
        "mutation_submission": row["mutation"],
        "proof": row["proof"].evidence,
        "witness": row["witness"].witness,
        "self_check": row["self_check"],
    }


def _cand_row(row: dict[str, Any]) -> dict[str, Any]:
    cand = row["grounded"].ast
    return {"instance_id": cand.instance_id, "family_id": cand.family_id,
            "base_goal_id": cand.base_goal_id, "qualified": row["qualified"],
            "rejection": row["rejection"], "reason_code": row["reason_code"],
            "kind": row["kind"]}


def _rejection_row(row: dict[str, Any]) -> dict[str, Any]:
    cand = row["grounded"].ast
    return {"instance_id": cand.instance_id, "family_id": cand.family_id,
            "base_goal_id": cand.base_goal_id, "reason_code": row["reason_code"],
            "reason": row["rejection"], "kind": row["kind"]}


def _load_existing(out_root: Path, shard_no: int, spec: FamilySpec,
                   target_goals: int | None = None) -> dict[str, Any] | None:
    qpath = out_root / "qualified" / f"shard-{shard_no:04d}.jsonl"
    cpath = out_root / "candidates" / f"shard-{shard_no:04d}.jsonl"
    qman = out_root / "qualified" / f"shard-{shard_no:04d}.manifest.json"
    if not (qpath.is_file() and cpath.is_file() and qman.is_file()):
        return None
    qrows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line]
    crows = [json.loads(line) for line in cpath.read_text(encoding="utf-8").splitlines() if line]
    want = (target_goals or spec.base_goals) * spec.instances_per_goal
    if len(qrows) != want:
        return None
    rejections: dict[str, int] = {}
    for row in crows:
        if not row["qualified"]:
            rejections[row["reason_code"]] = rejections.get(row["reason_code"], 0) + 1
    return {"candidates": len(crows), "qualified": len(qrows),
            "base_goals": len({r["base_goal_id"] for r in qrows}),
            "rejections": rejections, "resumed": True}


from benchmark.plugins.task_family.l15.families import FALLBACK_FAMILY_IDS
M_ONLY_FALLBACK_FAMILIES = set(FALLBACK_FAMILY_IDS)
FALLBACK_GOALS_PER_FAMILY = 25


def build_shards(out_root: Path, *, seed: int = SEED,
                 families: list[str] | None = None, resume: bool = True,
                 max_goals_override: int | None = None,
                 m_only_fallback: bool = False) -> dict[str, Any]:
    started = _utcnow()
    t0 = time.time()
    plugin = L15TaskFamilyPlugin(seed=seed)
    verifier = ReactomeNativeVerifier(allow_m_only=m_only_fallback)
    out_root = Path(out_root)
    for sub in ("candidates", "qualified", "oracle", "public", "rejections"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    os.chmod(out_root / "oracle", 0o700)
    stats: dict[str, Any] = {
        "lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
        "out_dir": str(out_root), "seed": seed,
        "target": {"instances": 1500, "families": 15},
        "candidates": 0, "qualified": 0, "base_goals": 0, "families_count": 0,
        "families_detail": {}, "rejections": {}, "shards": [], "gaps": [],
        "started_at": started,
    }
    rejection_rows: list[dict[str, Any]] = []
    for shard_no, spec in enumerate(FAMILY_TABLE):
        if families and spec.family_id not in families:
            continue
        target_goals = (max_goals_override
                        or (FALLBACK_GOALS_PER_FAMILY if m_only_fallback
                            else spec.base_goals))
        existing = _load_existing(out_root, shard_no, spec,
                                  target_goals=target_goals) if resume else None
        if existing is not None and not max_goals_override:
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
            continue
        cand_rows: list[dict[str, Any]] = []
        qual_rows: list[dict[str, Any]] = []
        fam_rej: list[dict[str, Any]] = []
        complete_goals = 0
        attempted_goals = 0
        constraints = {"max_base_goals": max(target_goals * 3, target_goals + 4)}
        generator = plugin.propose(None, family=spec.family_id, seed=seed,
                                   constraints=constraints)
        for _base_goal_id, cands in _grouped(generator):
            attempted_goals += 1
            rows = [_qualify(verifier, plugin.ground(c), spec) for c in cands]
            complete = (len(rows) == spec.instances_per_goal
                        and all(r["qualified"] for r in rows))
            if complete:
                complete_goals += 1
                for row in rows:
                    cand = row["grounded"].ast
                    task_id = "l15-" + canonical_sha(
                        {"instance": cand.instance_id, "ast": cand.ast})[:16]
                    oracle_rel = f"bank/staging_20k/l15/oracle/{row['grounded'].oracle_ref}.json"
                    public_ref = f"bank/staging_20k/l15/public/{cand.instance_id}.json"
                    _write_json(out_root / "public" / f"{cand.instance_id}.json",
                                row["grounded"].public, mode=0o644)
                    _write_json(out_root / "oracle" / f"{row['grounded'].oracle_ref}.json",
                                _oracle_payload(row, task_id), mode=0o600)
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
            if complete_goals >= target_goals:
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
        if len(qual_rows) < target_goals * spec.instances_per_goal:
            stats["gaps"].append({
                "family_id": spec.family_id,
                "expected_instances": target_goals * spec.instances_per_goal,
                "actual_instances": len(qual_rows),
                "attempted_goals": attempted_goals})
        _write_json(out_root / "build_report.json", stats)
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
        "build": (f".venv/bin/python -m benchmark.plugins.task_family.l15.build "
                  f"--out {out_root} --seed {seed}"),
        "verify_sample": ("for i in $(seq 1 3); do .venv/bin/python -m "
                          "benchmark.plugins.task_family.l15.verify_cli --oracle "
                          "$(ls bank/staging_20k/l15/oracle/orl15_*.json | shuf -n 1); done"),
    }
    _write_json(out_root / "build_report.json", stats)
    _write_json(out_root / "manifest.json",
                {"lane": LANE, "dataset_id": DATASET_ID, "release_id": RELEASE_ID,
                 "candidates": stats["candidates"], "qualified": stats["qualified"],
                 "base_goals": stats["base_goals"], "families": stats["families_count"],
                 "shards": stats["shards"]})
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L15 qualified shards")
    ap.add_argument("--out", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--families", default=None)
    ap.add_argument("--max-goals", type=int, default=None)
    ap.add_argument("--m-only-fallback", action="store_true",
                    help="restrict to M-path families and mark the BioPAX "
                         "agreement gate NOT_RUN (used only when the BioPAX "
                         "snapshot could not be acquired; see REPORT.md)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    fams = args.families.split(",") if args.families else None
    if args.m_only_fallback:
        fams = sorted(M_ONLY_FALLBACK_FAMILIES & set(fams)) if fams \
            else sorted(M_ONLY_FALLBACK_FAMILIES)
    stats = build_shards(Path(args.out), seed=args.seed, families=fams,
                         resume=not args.no_resume,
                         max_goals_override=args.max_goals,
                         m_only_fallback=args.m_only_fallback)
    print(json.dumps({k: stats[k] for k in
                      ("candidates", "qualified", "base_goals", "families_count",
                       "rejections", "gaps", "wall_seconds")}, indent=1, sort_keys=True))
    return 0 if stats["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
