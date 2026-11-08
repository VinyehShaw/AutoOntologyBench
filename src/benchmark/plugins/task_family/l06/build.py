"""L06 shard builder: candidates -> native gates -> immutable qualified shards.

Gate chain per candidate (cheap -> expensive, Build-order §A):
  1. ground                    — public view (fragment + question), oracle ref
  2. prove_exists              — leg1 recompute + leg2 owlrl agreement +
                                 solvable/nontrivial gates
                                 (reject: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED)
  3. certify_semantic_relevance — counterfactual witness (reject: NON_SEMANTIC)
  4. verifier self-check        — verify_terminal(gold) MUST pass and the
                                 tampered/deranged submission MUST fail
  5. leakage scan               — serialized reference must not appear in the
                                 public bytes (reject: LEAKAGE)
  6. dedup                      — instance_id unique (reject: DUPLICATE)

Layout (L01-style flat staging):
  bank/staging/l06/candidates/shard-XXXX.jsonl + .manifest.json
  bank/staging/l06/qualified/shard-XXXX.jsonl + .manifest.json
  bank/staging/l06/oracle/<oracle_handle>.json
  bank/staging/l06/rejections/rejections.jsonl
  bank/staging/l06/build_report.json

CLI: ``python -m benchmark.plugins.task_family.l06.build --out bank/staging/l06``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.source.l06.plugin import L06SourcePlugin
from benchmark.plugins.task_family.l06.families import BY_ID, FAMILY_TABLE, FamilySpec
from benchmark.plugins.task_family.l06.propose import L06TaskFamilyPlugin, PROFILE
from benchmark.plugins.verifier.l06.fragment import fragment_edges, induce
from benchmark.plugins.verifier.l06.verifier import L06NativeVerifier, REPAIR_KINDS
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l06_go import RELEASE_ID

DATASET_ID = "autoontobench-bank-8000-rc1"
SHARD_SIZE = 250
INSTANCE_OVERDRAW = 8

# One extra deterministic variant (next ``inst_seq``) per filled base goal.
# Totals 49 = 7 twelve-goal families x 3 + 8 ten-goal families x 3
# + 2 eight-goal families x 2; base goals stay at 180.
TOP_UP: dict[str, int] = {
    "l06.isa_ancestors_exact": 3,
    "l06.isa_descendants_exact": 3,
    "l06.isa_common_ancestors": 3,
    "l06.isa_lowest_common_ancestor": 3,
    "l06.partof_ancestors_closure": 3,
    "l06.regulates_closure": 3,
    "l06.cross_isa_partof_intersection": 3,
    "l06.branch_difference": 3,
    "l06.depth_band_terms": 3,
    "l06.multi_parent_terms": 3,
    "l06.repair_isa_parent_swap": 3,
    "l06.repair_missing_isa_edge": 3,
    "l06.repair_relation_type_confusion": 3,
    "l06.repair_orphan_reconnect": 3,
    "l06.org_align_gene_aspect_annotations": 3,
    "l06.org_annotation_scope_closure": 2,
    "l06.org_shared_ancestors_multigene": 2,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference_submission(grounded: GroundedTask, verifier: L06NativeVerifier,
                          frag: dict[str, Any]) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind in ("repair_isa_parent_swap", "repair_missing_isa_edge"):
        return {"term": ast["term"],
                "parent": verifier._gold_fix(ast)["parent"]}
    if kind == "repair_orphan_reconnect":
        return {"term": ast["term"], "parents": verifier._gold_fix(ast)["parents"]}
    if kind == "repair_relation_type_confusion":
        return {"subject": ast["subject"], "object": ast["object"],
                "predicate": verifier._gold_fix(ast)["predicate"]}
    if kind == "align_gene_terms":
        return {"alignment": verifier._reference(ast, frag)}
    return {"answer_ids": sorted(verifier._reference(ast, frag))}


def _wrong_submission(grounded: GroundedTask, verifier: L06NativeVerifier,
                      frag: dict[str, Any]) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind == "repair_isa_parent_swap":
        return {"term": ast["term"], "parent": ast["wrong_parent"]}
    if kind == "repair_missing_isa_edge":
        u = sorted(set(ast["universe_ids"]) - {ast["removed_parent"], ast["term"]})
        return {"term": ast["term"], "parent": u[0]}
    if kind == "repair_orphan_reconnect":
        return {"term": ast["term"], "parents": sorted(ast["removed_parents"])[:-1] or ["GO:0008150"]}
    if kind == "repair_relation_type_confusion":
        return {"subject": ast["subject"], "object": ast["object"],
                "predicate": ast["wrong_pred"]}
    if kind == "align_gene_terms":
        correct = verifier._reference(ast, frag)
        return {"alignment": {g: correct[wrong] for g, wrong in ast["derangement"].items()}}
    want = set(verifier._reference(ast, frag))
    universe = set(ast["universe_ids"])
    wrong = sorted((universe - want))[: max(1, len(want))]
    return {"answer_ids": wrong}


def _leakage_scan(public_blob: str, reference: Any, kind: str) -> bool:
    """True when no serialized reference answer appears in the public bytes."""
    needles: list[str] = []
    if isinstance(reference, dict) and "answer_ids" in reference:
        needles.append(json.dumps(sorted(reference["answer_ids"])))
    if isinstance(reference, dict) and "alignment" in reference:
        needles += [json.dumps(sorted(v)) for v in reference["alignment"].values()]
    if kind == "repair_isa_parent_swap":
        # gold edge must not be an edge of the tampered public fragment
        needles.append(json.dumps({"o": reference["parent"], "p": "is_a",
                                   "s": reference["term"]}, sort_keys=True))
    if kind == "repair_missing_isa_edge":
        needles.append(json.dumps({"o": reference["parent"], "p": "is_a",
                                   "s": reference["term"]}, sort_keys=True))
    if kind == "repair_relation_type_confusion":
        needles.append(json.dumps({"o": reference["object"], "p": reference["predicate"],
                                   "s": reference["subject"]}, sort_keys=True))
    return not any(n and n in public_blob for n in needles)


def _record(grounded: GroundedTask, spec: FamilySpec, native_evidence: dict[str, Any],
            oracle_handle: str, evidence_cmd: str) -> dict[str, Any]:
    cand = grounded.ast
    task_id = "l06-" + cand.instance_id
    native_evidence = {
        "commands": [evidence_cmd],
        "verdict": {"prove_exists": True, "witness_ok": True,
                    "positive_valid": True, "negative_rejected": True},
        **native_evidence,
    }
    return {
        "schema_version": "aob-task/5.0",
        "task_id": task_id,
        "dataset_id": DATASET_ID,
        "lane_id": "L06",
        "family_id": cand.family_id,
        "base_goal_id": cand.base_goal_id,
        "instance_id": cand.instance_id,
        "task_regime": spec.regime,
        "complexity": spec.complexity,
        "semantic_profile": PROFILE,
        "source_snapshot_ids": [f"gene-ontology@{RELEASE_ID}"],
        "source_unit_ids": list(cand.source_unit_ids),
        "origin_mechanism_group": canonical_sha(
            {"lane": "L06", "family": cand.family_id, "release": RELEASE_ID})[:16],
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "public": grounded.public,
        "oracle_handle": oracle_handle,
        "native_evidence": native_evidence,
        "qualification": {"native_verified": True, "solvable": True,
                          "nontrivial": True, "semantic_relevance": True,
                          "leakage_clean": True, "release_eligible": True},
        "status": "QUALIFIED",
    }


def build_shards(index_dir: Path, out_root: Path, *, seed: int = 20261027,
                  shard_size: int = SHARD_SIZE,
                  max_base_goals: dict[str, int] | None = None,
                  instance_overdraw: int = INSTANCE_OVERDRAW,
                  top_up: dict[str, int] | None = None) -> dict[str, Any]:
    top_up = TOP_UP if top_up is None else top_up
    t_start = time.time()
    started_at = _utcnow()
    shared_index = L06SourcePlugin.load_index(Path(index_dir))
    plugin = L06TaskFamilyPlugin(index_dir, seed=seed, index=shared_index)
    verifier = L06NativeVerifier(index_dir, index=shared_index)
    out_root = Path(out_root)
    cand_dir = out_root / "candidates"
    qual_dir = out_root / "qualified"
    oracle_dir = out_root / "oracle"
    rej_dir = out_root / "rejections"
    for d in (cand_dir, qual_dir, oracle_dir, rej_dir):
        d.mkdir(parents=True, exist_ok=True)
    cmd_prefix = (".venv/bin/python -m benchmark.plugins.verifier.l06.reverify "
                  f"--oracle-file {out_root}/oracle/")

    stats: dict[str, Any] = {"candidates": 0, "qualified": 0, "rejections": {},
                             "families": {}, "base_goals": set(),
                             "base_goals_qualified": set()}
    seen_instances: set[str] = set()
    shard_rows: list[dict[str, Any]] = []
    shard_no = 0
    rejection_rows: list[dict[str, Any]] = []

    def flush_shard() -> None:
        nonlocal shard_no, shard_rows
        if not shard_rows:
            return
        name = f"shard-{shard_no:04d}"
        cand_path = cand_dir / f"{name}.jsonl"
        qual_path = qual_dir / f"{name}.jsonl"
        fams: set[str] = set()
        bgoals: set[str] = set()
        n_qual = 0
        with cand_path.open("w", encoding="utf-8") as cfh, \
                qual_path.open("w", encoding="utf-8") as qfh:
            for row in shard_rows:
                fams.add(row["family_id"])
                bgoals.add(row["base_goal_id"])
                cfh.write(json.dumps({
                    "instance_id": row["instance_id"], "family_id": row["family_id"],
                    "base_goal_id": row["base_goal_id"], "qualified": row["qualified"],
                    "rejection": row["rejection"]}, sort_keys=True) + "\n")
                if row["qualified"]:
                    qfh.write(json.dumps(row["record"], sort_keys=True) + "\n")
                    n_qual += 1
                    (oracle_dir / f"{row['record']['oracle_handle']}.json").write_text(
                        json.dumps(row["oracle_payload"], indent=1, sort_keys=True) + "\n")
        for path, records in ((cand_path, len(shard_rows)), (qual_path, n_qual)):
            manifest = {"shard": name, "path": str(path), "records": records,
                        "sha256": _sha256_file(path),
                        "families": sorted(fams), "base_goals": len(bgoals)}
            path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        shard_no += 1
        shard_rows = []

    for spec in FAMILY_TABLE:
        fam_stats: dict[str, Any] = {"candidates": 0, "qualified": 0, "rejected": 0,
                                     "goals_filled": 0, "goals_dropped": 0}
        max_goals = (max_base_goals or {}).get(spec.family_id, spec.base_goals)
        goal_budget = max_goals * 3
        top_up_target = top_up.get(spec.family_id, 0)
        top_up_added = 0
        filled = 0

        def record(row: dict[str, Any]) -> None:
            shard_rows.append(row)
            if row["qualified"]:
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
            else:
                fam_stats["rejected"] += 1
                code = row["rejection"].split(":", 1)[0]
                stats["rejections"][code] = stats["rejections"].get(code, 0) + 1
                rejection_rows.append({"instance_id": row["instance_id"],
                                       "family_id": row["family_id"],
                                       "base_goal_id": row["base_goal_id"],
                                       "rejection": row["rejection"]})
            if len(shard_rows) >= shard_size:
                flush_shard()

        for base_goal_id, base in plugin.iter_base_goals(
                spec.family_id, max_goals=goal_budget, seed=seed):
            if filled >= max_goals:
                break
            goal_rows: list[dict[str, Any]] = []
            goal_qual = 0
            inst = 0
            max_inst = spec.instances_per_goal * instance_overdraw
            while goal_qual < spec.instances_per_goal and inst < max_inst:
                cand = plugin.instantiate(spec, base, base_goal_id, inst, seed)
                inst += 1
                if cand is None:
                    continue
                fam_stats["candidates"] += 1
                stats["candidates"] += 1
                stats["base_goals"].add(base_goal_id)
                row = _qualify_one(plugin, verifier, cand, spec, seen_instances,
                                   cmd_prefix)
                goal_rows.append(row)
                if row["qualified"]:
                    goal_qual += 1
            if (goal_qual >= spec.instances_per_goal
                    and top_up_added < top_up_target):
                extra_budget = inst + max_inst
                while inst < extra_budget:
                    cand = plugin.instantiate(spec, base, base_goal_id, inst, seed)
                    inst += 1
                    if cand is None:
                        continue
                    fam_stats["candidates"] += 1
                    stats["candidates"] += 1
                    row = _qualify_one(plugin, verifier, cand, spec, seen_instances,
                                       cmd_prefix)
                    goal_rows.append(row)
                    if row["qualified"]:
                        top_up_added += 1
                        break
            if goal_qual >= spec.instances_per_goal:
                filled += 1
                fam_stats["goals_filled"] += 1
                stats["base_goals_qualified"].add(base_goal_id)
                for row in goal_rows:
                    record(row)
            else:
                fam_stats["goals_dropped"] += 1
                for row in goal_rows:
                    if row["qualified"]:
                        row["qualified"] = False
                        row["rejection"] = (
                            f"GOAL_UNDERFILLED:{goal_qual}of{spec.instances_per_goal}")
                    record(row)
        fam_stats["top_up_target"] = top_up_target
        fam_stats["top_up_added"] = top_up_added
        stats["families"][spec.family_id] = fam_stats
    flush_shard()

    rej_path = rej_dir / "rejections.jsonl"
    with rej_path.open("w", encoding="utf-8") as fh:
        for r in rejection_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    wall = round(time.time() - t_start, 1)
    report = {
        "lane": "L06",
        "dataset_id": DATASET_ID,
        "source_snapshot_ids": [f"gene-ontology@{RELEASE_ID}"],
        "semantic_profile": PROFILE,
        "seed": seed,
        "started_at": started_at,
        "wall_seconds": wall,
        "candidates": stats["candidates"],
        "qualified": stats["qualified"],
        "distinct_instances": len(seen_instances),
        "base_goals": len(stats["base_goals"]),
        "base_goals_qualified": len(stats["base_goals_qualified"]),
        "instance_overdraw": instance_overdraw,
        "top_up": dict(top_up),
        "top_up_added": sum(v.get("top_up_added", 0)
                            for v in stats["families"].values()),
        "families": stats["families"],
        "rejections": stats["rejections"],
        "qualified_per_hour": round(stats["qualified"] / max(wall / 3600, 1e-9), 1),
        "candidate_manifests": _shard_manifests(cand_dir),
        "qualified_manifests": _shard_manifests(qual_dir),
        "oracle_count": len(list(oracle_dir.glob("l06-oracle-*.json"))),
        "oracle_sha256": _oracle_manifest(oracle_dir, out_root),
        "rejections_file": str(rej_path),
        "rejections_sha256": _sha256_file(rej_path),
    }
    (out_root / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _shard_manifests(d: Path) -> list[dict[str, Any]]:
    out = []
    for m in sorted(d.glob("shard-*.manifest.json")):
        out.append(json.loads(m.read_text()))
    return out


def _oracle_manifest(oracle_dir: Path, out_root: Path) -> str:
    entries = {}
    for p in sorted(oracle_dir.glob("l06-oracle-*.json")):
        entries[p.name] = _sha256_file(p)
    payload = {"count": len(entries), "files": entries}
    manifest_path = oracle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return _sha256_file(manifest_path)


def _qualify_one(plugin: L06TaskFamilyPlugin, verifier: L06NativeVerifier,
                 cand: CandidateTaskAST, spec: FamilySpec,
                 seen_instances: set[str], cmd_prefix: str) -> dict[str, Any]:
    row: dict[str, Any] = {"instance_id": cand.instance_id,
                           "family_id": cand.family_id,
                           "base_goal_id": cand.base_goal_id,
                           "qualified": False, "rejection": ""}
    if cand.instance_id in seen_instances:
        row["rejection"] = "DUPLICATE:instance_id"
        return row
    seen_instances.add(cand.instance_id)
    grounded = plugin.ground(cand)
    proof = verifier.prove_exists(grounded)
    if not proof.exists:
        row["rejection"] = proof.reason
        return row
    kind = cand.ast["kind"]
    if (spec.min_answer > 1 and kind not in REPAIR_KINDS
            and kind != "align_gene_terms"):
        size = int(proof.evidence.get("answer_size", 0))
        if size < spec.min_answer:
            row["rejection"] = f"TRIVIAL:below_min_answer({size}<{spec.min_answer})"
            return row
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        row["rejection"] = "NON_SEMANTIC:no_counterfactual"
        return row
    frag = induce(cand.ast["universe_ids"], verifier.graph)
    ref_sub = _reference_submission(grounded, verifier, frag)
    pos = verifier.verify_terminal(grounded, ref_sub)
    neg = verifier.verify_terminal(grounded, _wrong_submission(grounded, verifier, frag))
    if not pos.valid or neg.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_selfcheck"
        return row
    public_blob = json.dumps(grounded.public, sort_keys=True)
    scan_ref = ref_sub
    if kind in ("repair_isa_parent_swap", "repair_missing_isa_edge"):
        scan_ref = {"term": cand.ast["term"],
                    "parent": verifier._gold_fix(cand.ast)["parent"]}
    elif kind == "repair_orphan_reconnect":
        scan_ref = None  # parents are ordinary GO ids; the edge JSON is the secret
        gold_parents = verifier._gold_fix(cand.ast)["parents"]
        t = cand.ast["term"]
        needles = [json.dumps({"o": p, "p": "is_a", "s": t}, sort_keys=True)
                   for p in gold_parents]
        if any(n in public_blob for n in needles):
            row["rejection"] = "LEAKAGE:gold_edge_in_public"
            return row
    elif kind == "repair_relation_type_confusion":
        scan_ref = {"subject": cand.ast["subject"], "object": cand.ast["object"],
                    "predicate": verifier._gold_fix(cand.ast)["predicate"]}
    elif kind == "align_gene_terms":
        # the term-sets are the public puzzle input by design; the secret is the
        # correct gene->set assignment, gated by the row-correct check below
        scan_ref = None
    if scan_ref is not None and not _leakage_scan(public_blob, scan_ref, kind):
        row["rejection"] = "LEAKAGE:answer_in_public"
        return row
    if kind == "align_gene_terms":
        correct = verifier._reference(cand.ast, frag)
        table = grounded.public["misaligned_table"]
        if any(sorted(table.get(g, [])) == sorted(correct[g]) for g in correct):
            row["rejection"] = "LEAKAGE:alignment_row_correct_in_public"
            return row
    native_evidence = {
        "prove_exists": proof.evidence,
        "semantic_witness": witness.witness,
        "verify_terminal": {"positive": pos.valid, "negative_rejected": not neg.valid,
                            "positive_detail": pos.detail,
                            "negative_detail": neg.detail},
    }
    oracle_handle = grounded.oracle_ref
    evidence_cmd = f"{cmd_prefix}{oracle_handle}.json"
    row["record"] = _record(grounded, spec, native_evidence, oracle_handle,
                            evidence_cmd)
    row["qualified"] = True
    oracle_payload = {
        "oracle_handle": oracle_handle,
        "lane_id": "L06",
        "family_id": cand.family_id,
        "base_goal_id": cand.base_goal_id,
        "instance_id": cand.instance_id,
        "regime": cand.regime,
        "semantic_profile": PROFILE,
        "source_release": RELEASE_ID,
        "source_unit_ids": list(cand.source_unit_ids),
        "ast": cand.ast,
        "reference_submission": ref_sub,
        "proof": proof.evidence,
        "witness": witness.witness,
        "self_check": {"positive": pos.valid, "negative_valid": neg.valid},
    }
    return row | {"oracle_payload": oracle_payload}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L06 qualified shards")
    ap.add_argument("--index", default=str(
        Path("external_knowledge/indexes/gene-ontology") / RELEASE_ID))
    ap.add_argument("--out", default="bank/staging/l06")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = ap.parse_args()
    report = build_shards(Path(args.index), Path(args.out), seed=args.seed,
                          shard_size=args.shard_size)
    print(json.dumps({"qualified": report["qualified"],
                      "candidates": report["candidates"],
                      "base_goals": report["base_goals"],
                      "rejections": report["rejections"],
                      "wall_seconds": report["wall_seconds"]}, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
