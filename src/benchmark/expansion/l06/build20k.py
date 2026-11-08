"""L06 20K-expansion shard builder: candidates -> native gates -> immutable
qualified shards under ``bank/staging_20k/l06/``.

Gate chain per candidate (cheap -> expensive, identical order to the legacy
8K builder):
  1. ground                    — public view (fragment + question), oracle ref
  2. prove_exists              — leg1 recompute + leg2 owlrl agreement +
                                 solvable/nontrivial gates (and, for the new
                                 repair kinds, the owlrl signature dual-leg)
                                 (reject: UNSOLVABLE / TRIVIAL /
                                 PROFILE_UNSUPPORTED)
  3. certify_semantic_relevance — counterfactual witness (NON_SEMANTIC)
  4. verifier self-check        — verify_terminal(gold) MUST pass and the
                                  tampered/deranged submission MUST fail
  5. leakage scan               — serialized reference / gold edge must not
                                  appear in the public bytes (LEAKAGE)
  6. dedup                      — instance_id unique (DUPLICATE), plus the
                                  legacy-collision check (task_id /
                                  instance_id / base_goal_id / public blob
                                  sha256 vs the frozen 8K build)

Layout (flat staging, legacy-compatible):
  bank/staging_20k/l06/candidates/shard-XXXX.jsonl + .manifest.json
  bank/staging_20k/l06/qualified/shard-XXXX.jsonl + .manifest.json
  bank/staging_20k/l06/oracle/l06x-oracle-*.json + manifest.json
  bank/staging_20k/l06/rejections/rejections.jsonl
  bank/staging_20k/l06/build_report.json

Per-task §4.1 status coverage: every qualified record carries a ``checks``
block (source_verified / schema_valid / ontology_parse_valid /
native_execution_status / oracle_check_status / mutation_check_status /
human_review_status / qualification_status) with PASS only for checks that
actually ran; human_review_status is honestly NOT_RUN. Rejected candidates
are recorded in rejections.jsonl with qualification_status REJECTED.

CLI: ``.venv/bin/python -m benchmark.expansion.l06.build20k
--out bank/staging_20k/l06`` (deterministic; seed 20261101).
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
from benchmark.expansion.l06.families20k import (
    BY_ID_20K,
    DATASET_ID_20K,
    FAMILY_TABLE_20K,
    PROFILE,
    SEED_20K,
    TARGET_NEW_20K,
    TOP_UP_20K,
    FamilySpec20k,
)
from benchmark.expansion.l06.fragment20k import (
    GENE_ANSWER_KINDS,
    REPAIR_KINDS_20K,
    gold_fix20k,
)
from benchmark.expansion.l06.propose20k import L06ExpansionProposer
from benchmark.expansion.l06.verify20k import L06ExpansionVerifier
from benchmark.plugins.verifier.l06.fragment import induce
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l06_go import RELEASE_ID

SHARD_SIZE = 250
INSTANCE_OVERDRAW = 8
LEGACY_STAGING = Path("bank/staging/l06")
LEGACY_GLOBAL_MANIFEST = Path("bank/builds/aob-8000-rc1/task_manifest.jsonl")
STATUS_PATH = Path("data/erule/review5/build8000/expansion/lanes/L06/STATUS.md")

REQUIRED_FIELDS = (
    "schema_version", "task_id", "dataset_id", "lane_id", "family_id",
    "base_goal_id", "instance_id", "task_regime", "complexity",
    "semantic_profile", "source_snapshot_ids", "source_unit_ids",
    "origin_mechanism_group", "provenance", "authorship", "public",
    "oracle_handle", "native_evidence", "qualification", "status", "checks",
)
REQUIRED_PUBLIC = ("user_goal", "fragment", "allowed_tools", "budget",
                   "semantic_profile", "complexity")
REQUIRED_CHECKS = ("source_verified", "schema_valid", "ontology_parse_valid",
                   "native_execution_status", "oracle_check_status",
                   "mutation_check_status", "human_review_status",
                   "qualification_status")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_blob(public: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(public, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------- reference/wrong submissions

def _reference_submission20k(grounded: GroundedTask,
                             verifier: L06ExpansionVerifier,
                             frag: dict[str, Any]) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind in REPAIR_KINDS_20K:
        return gold_fix20k(ast)
    ref = verifier._reference(ast, frag)
    if kind in GENE_ANSWER_KINDS:
        return {"answer_genes": sorted(ref)}
    return {"answer_ids": sorted(ref)}


def _wrong_submission20k(grounded: GroundedTask,
                         verifier: L06ExpansionVerifier,
                         frag: dict[str, Any]) -> Any:
    ast = grounded.ast.ast
    kind = ast["kind"]
    if kind == "repair_partof_edge_deletion":
        s, o = ast["subject"], ast["object"]
        from benchmark.plugins.source.l06.closure import transitive_ancestors
        forbidden = (set(transitive_ancestors(s, frag["isa_parents"]))
                     | set(verifier._frag(ast)["rel_out"].get(s, {})
                           .get("part_of", []))
                     | {o, s})
        u = sorted(set(ast["universe_ids"]) - forbidden)
        return {"subject": s, "object": u[0] if u else s}
    if kind == "repair_spurious_isa_edge":
        t, spurious = ast["term"], ast["spurious_parent"]
        true_parents = [p for p in frag["isa_parents"].get(t, [])
                        if p != spurious]
        if true_parents:
            return {"term": t, "remove_parent": sorted(true_parents)[0]}
        u = sorted(set(ast["universe_ids"]) - {t, spurious})
        return {"term": t, "remove_parent": u[0] if u else t}
    if kind == "repair_regulation_polarity_swap":
        return {"subject": ast["subject"], "object": ast["object"],
                "predicate": ast["wrong_pred"]}
    if kind == "repair_spurious_partof_edge":
        s, spurious = ast["subject"], ast["spurious_target"]
        true_targets = [o for o in frag["rel_out"].get(s, {}).get("part_of", [])
                        if o != spurious]
        if true_targets:
            return {"subject": s, "remove_object": sorted(true_targets)[0]}
        u = sorted(set(ast["universe_ids"]) - {s, spurious})
        return {"subject": s, "remove_object": u[0] if u else s}
    ref = verifier._reference(ast, frag)
    if kind in GENE_ANSWER_KINDS:
        want = set(ref)
        panel = set(ast["candidate_genes"])
        wrong = sorted(panel - want)[: max(1, len(want))]
        return {"answer_genes": wrong}
    want = set(ref)
    universe = set(ast["universe_ids"])
    wrong = sorted(universe - want)[: max(1, len(want))]
    return {"answer_ids": wrong}


def _leakage_scan20k(public_blob: str, reference: Any,
                     kind: str) -> bool:
    """True when no serialized reference answer / gold edge appears in the
    public bytes. For the two spurious-edge repair kinds and the reverse
    gene-panel kinds the puzzle input is public BY DESIGN (the corrupt edge
    / the candidate panel); the gated secret is the answer identity, which
    is exactly what is scanned."""
    needles: list[str] = []
    if isinstance(reference, dict) and "answer_ids" in reference:
        needles.append(json.dumps(sorted(reference["answer_ids"])))
    if isinstance(reference, dict) and "answer_genes" in reference:
        needles.append(json.dumps(sorted(reference["answer_genes"])))
    if kind == "repair_partof_edge_deletion":
        needles.append(json.dumps({"o": reference["object"], "p": "part_of",
                                   "s": reference["subject"]}, sort_keys=True))
    if kind == "repair_regulation_polarity_swap":
        needles.append(json.dumps({"o": reference["object"],
                                   "p": reference["predicate"],
                                   "s": reference["subject"]}, sort_keys=True))
    if kind in ("repair_spurious_isa_edge", "repair_spurious_partof_edge"):
        needles.append(json.dumps(reference, sort_keys=True))
    return not any(n and n in public_blob for n in needles)


# ------------------------------------------------------------------- records

def _validate_record(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in record:
            issues.append(f"missing field {f}")
    for f in REQUIRED_PUBLIC:
        if f not in record.get("public", {}):
            issues.append(f"missing public.{f}")
    for f in REQUIRED_CHECKS:
        if f not in record.get("checks", {}):
            issues.append(f"missing checks.{f}")
    if record.get("schema_version") != "aob-task/5.0":
        issues.append("schema_version != aob-task/5.0")
    if record.get("dataset_id") != DATASET_ID_20K:
        issues.append("dataset_id mismatch")
    if record.get("lane_id") != "L06":
        issues.append("lane_id mismatch")
    if not record.get("native_evidence", {}).get("commands"):
        issues.append("native_evidence.commands empty")
    return issues


def _record(grounded: GroundedTask, spec: FamilySpec20k,
            native_evidence: dict[str, Any], oracle_handle: str,
            evidence_cmd: str) -> dict[str, Any]:
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
        "dataset_id": DATASET_ID_20K,
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
            {"lane": "L06", "family": cand.family_id, "release": RELEASE_ID,
             "expansion": "20k-dev1"})[:16],
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "public": grounded.public,
        "oracle_handle": oracle_handle,
        "native_evidence": native_evidence,
        "qualification": {"native_verified": True, "solvable": True,
                          "nontrivial": True, "semantic_relevance": True,
                          "leakage_clean": True, "release_eligible": True},
        "status": "QUALIFIED_NATIVE",
        "checks": {
            "source_verified": "PASS",
            "schema_valid": "PASS",
            "ontology_parse_valid": "PASS",
            "native_execution_status": "PASS",
            "oracle_check_status": "PASS",
            "mutation_check_status": "PASS",
            "human_review_status": "NOT_RUN",
            "qualification_status": "QUALIFIED_NATIVE",
        },
    }


# --------------------------------------------------------- legacy collisions

def _load_legacy_ids() -> dict[str, Any]:
    """Legacy 8K collision surface: the global task_id manifest (all lanes)
    plus the frozen L06 staging records (instance ids, base goal ids,
    public blob sha256). Read-only; the legacy build is never touched."""
    out: dict[str, Any] = {"task_ids": set(), "instance_ids": set(),
                           "base_goal_ids": set(), "public_blobs": set(),
                           "l06_task_ids": set()}
    if LEGACY_GLOBAL_MANIFEST.is_file():
        with LEGACY_GLOBAL_MANIFEST.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                tid = row.get("task_id") or row.get("id")
                if tid:
                    out["task_ids"].add(tid)
                    if tid.startswith("l06-"):
                        out["l06_task_ids"].add(tid)
    for path in sorted(LEGACY_STAGING.glob("qualified/shard-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out["instance_ids"].add(row["instance_id"])
                out["base_goal_ids"].add(row["base_goal_id"])
                out["public_blobs"].add(_public_blob(row["public"]))
    return out


def _verify_pinned_source() -> dict[str, Any]:
    """source_verified check: sha256 + sanity of the pinned snapshot via the
    legacy fetcher's verify path (read-only)."""
    from source_registry.fetchers.l06_go import GeneOntologySource
    from benchmark.schema.plugin_protocols import FetchReceipt
    src = GeneOntologySource()
    receipt = FetchReceipt(
        source_id="gene-ontology", release_id=RELEASE_ID,
        status="FETCHED_COMPLETE", bytes_downloaded=0, file_count=0,
        cache_dir=str(src.release_dir), sha256_manifest="",
        fetched_at="")
    report = src.verify_snapshot(receipt)
    return {"ok": bool(report.ok),
            "checks": {k: bool(v) for k, v in
                       getattr(report, "checks", {}).items()}
            if isinstance(getattr(report, "checks", None), dict)
            else repr(getattr(report, "checks", None))[:500]}


# -------------------------------------------------------------------- build

def build_shards20k(index_dir: Path, out_root: Path, *,
                    seed: int = SEED_20K,
                    shard_size: int = SHARD_SIZE,
                    max_base_goals: dict[str, int] | None = None,
                    instance_overdraw: int = INSTANCE_OVERDRAW,
                    top_up: dict[str, int] | None = None,
                    status_path: Path | None = None,
                    legacy: dict[str, Any] | None = None,
                    source_check: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    top_up = TOP_UP_20K if top_up is None else top_up
    status_path = status_path or STATUS_PATH
    t_start = time.time()
    started_at = _utcnow()
    if legacy is None:
        legacy = _load_legacy_ids()
    if source_check is None:
        source_check = _verify_pinned_source()
    if not source_check.get("ok"):
        raise RuntimeError(f"pinned source verification failed: "
                           f"{source_check}")
    shared_index = L06ExpansionProposer(index_dir, seed=seed).index
    proposer = L06ExpansionProposer(index_dir, seed=seed, index=shared_index)
    verifier = L06ExpansionVerifier(index_dir, index=shared_index,
                                     evidence=proposer.evidence)
    out_root = Path(out_root)
    cand_dir = out_root / "candidates"
    qual_dir = out_root / "qualified"
    oracle_dir = out_root / "oracle"
    rej_dir = out_root / "rejections"
    for d in (cand_dir, qual_dir, oracle_dir, rej_dir):
        d.mkdir(parents=True, exist_ok=True)
    # a re-run must be byte-clean: remove stale shards/oracles/rejections from
    # any previous (deterministic) run so the output reflects exactly this run
    for d in (cand_dir, qual_dir, rej_dir):
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
    for p in oracle_dir.glob("l06x-oracle-*.json"):
        p.unlink()
    (oracle_dir / "manifest.json").unlink(missing_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_prefix = (".venv/bin/python -m benchmark.expansion.l06.reverify20k "
                  f"--oracle-file {out_root}/oracle/")

    stats: dict[str, Any] = {"candidates": 0, "qualified": 0, "rejections": {},
                             "families": {}, "base_goals": set(),
                             "base_goals_qualified": set(),
                             "gate_fail": {}, "gate_not_run": {}}
    seen_instances: set[str] = set()
    seen_task_ids: set[str] = set()
    seen_public_blobs: set[str] = set()
    legacy_collisions: dict[str, int] = {"task_id": 0, "instance_id": 0,
                                         "base_goal_id": 0, "public_blob": 0}
    shard_rows: list[dict[str, Any]] = []
    shard_no = 0
    rejection_rows: list[dict[str, Any]] = []
    families_done: list[str] = []

    def _write_status() -> None:
        lines = [
            "# L06 20K-expansion STATUS (auto-written by build20k)",
            "",
            f"- updated: {_utcnow()}",
            f"- dataset_id: {DATASET_ID_20K} (seed {seed})",
            f"- target new tasks: {TARGET_NEW_20K}",
            f"- candidates gated so far: {stats['candidates']}",
            f"- qualified so far: {stats['qualified']}",
            f"- rejections so far: {json.dumps(stats['rejections'], sort_keys=True)}",
            f"- families completed: {len(families_done)} / {len(FAMILY_TABLE_20K)}",
            f"- legacy collisions: {json.dumps(legacy_collisions, sort_keys=True)}",
            f"- source_verified: {source_check.get('ok')}",
            "",
            "Shards are flushed incrementally every "
            f"{shard_size} gated candidates (candidates/qualified/oracle "
            "written atomically per shard, manifests regenerated).",
            "",
        ]
        status_path.write_text("\n".join(lines), encoding="utf-8")

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
                    "instance_id": row["instance_id"],
                    "family_id": row["family_id"],
                    "base_goal_id": row["base_goal_id"],
                    "qualified": row["qualified"],
                    "rejection": row["rejection"]}, sort_keys=True) + "\n")
                if row["qualified"]:
                    qfh.write(json.dumps(row["record"], sort_keys=True) + "\n")
                    n_qual += 1
                    (oracle_dir / f"{row['record']['oracle_handle']}.json").write_text(
                        json.dumps(row["oracle_payload"], indent=1,
                                   sort_keys=True) + "\n")
        for path, records in ((cand_path, len(shard_rows)),
                              (qual_path, n_qual)):
            manifest = {"shard": name, "path": str(path),
                        "records": records, "sha256": _sha256_file(path),
                        "families": sorted(fams),
                        "base_goals": len(bgoals)}
            path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        shard_no += 1
        shard_rows = []

    for spec in FAMILY_TABLE_20K:
        fam_stats: dict[str, Any] = {"candidates": 0, "qualified": 0,
                                     "rejected": 0, "goals_filled": 0,
                                     "goals_dropped": 0}
        max_goals = (max_base_goals or {}).get(spec.family_id,
                                               spec.base_goals)
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
                rejection_rows.append({
                    "instance_id": row["instance_id"],
                    "family_id": row["family_id"],
                    "base_goal_id": row["base_goal_id"],
                    "rejection": row["rejection"],
                    "qualification_status": "REJECTED"})
            if len(shard_rows) >= shard_size:
                flush_shard()
                _write_status()

        for base_goal_id, base in proposer.iter_base_goals(
                spec.family_id, max_goals=goal_budget, seed=seed):
            if filled >= max_goals:
                break
            goal_rows: list[dict[str, Any]] = []
            goal_qual = 0
            inst = 0
            max_inst = spec.instances_per_goal * instance_overdraw
            while goal_qual < spec.instances_per_goal and inst < max_inst:
                cand = proposer.instantiate(spec, base, base_goal_id, inst,
                                            seed)
                inst += 1
                if cand is None:
                    continue
                fam_stats["candidates"] += 1
                stats["candidates"] += 1
                stats["base_goals"].add(base_goal_id)
                row = _qualify_one20k(proposer, verifier, cand, spec,
                                      seen_instances, seen_task_ids,
                                      seen_public_blobs, legacy,
                                      legacy_collisions, cmd_prefix)
                goal_rows.append(row)
                if row["qualified"]:
                    goal_qual += 1
            if (goal_qual >= spec.instances_per_goal
                    and top_up_added < top_up_target):
                extra_budget = inst + max_inst
                while inst < extra_budget:
                    cand = proposer.instantiate(spec, base, base_goal_id, inst,
                                                seed)
                    inst += 1
                    if cand is None:
                        continue
                    fam_stats["candidates"] += 1
                    stats["candidates"] += 1
                    row = _qualify_one20k(proposer, verifier, cand, spec,
                                          seen_instances, seen_task_ids,
                                          seen_public_blobs, legacy,
                                          legacy_collisions, cmd_prefix)
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
                            f"GOAL_UNDERFILLED:{goal_qual}of"
                            f"{spec.instances_per_goal}")
                    record(row)
        fam_stats["top_up_target"] = top_up_target
        fam_stats["top_up_added"] = top_up_added
        stats["families"][spec.family_id] = fam_stats
        families_done.append(spec.family_id)
        _write_status()
    flush_shard()
    _write_status()

    rej_path = rej_dir / "rejections.jsonl"
    with rej_path.open("w", encoding="utf-8") as fh:
        for r in rejection_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    wall = round(time.time() - t_start, 1)
    n_qual = stats["qualified"]
    report = {
        "lane": "L06",
        "build_kind": "20k-expansion",
        "dataset_id": DATASET_ID_20K,
        "source_snapshot_ids": [f"gene-ontology@{RELEASE_ID}"],
        "semantic_profile": PROFILE,
        "seed": seed,
        "started_at": started_at,
        "wall_seconds": wall,
        "target_new": TARGET_NEW_20K,
        "candidates": stats["candidates"],
        "qualified": n_qual,
        "distinct_instances": len(seen_instances),
        "distinct_task_ids": len(seen_task_ids),
        "distinct_public_blobs": len(seen_public_blobs),
        "base_goals": len(stats["base_goals"]),
        "base_goals_qualified": len(stats["base_goals_qualified"]),
        "instance_overdraw": instance_overdraw,
        "top_up": dict(top_up),
        "top_up_added": sum(v.get("top_up_added", 0)
                            for v in stats["families"].values()),
        "families": stats["families"],
        "rejections": stats["rejections"],
        "qualified_per_hour": round(n_qual / max(wall / 3600, 1e-9), 1),
        # §4.2 statistics (denominators explicit)
        "stats_4_2": {
            "N_candidate": stats["candidates"],
            "N_source_verified": n_qual,
            "N_schema_valid": n_qual,
            "N_native_checked": n_qual,
            "N_native_pass": n_qual,
            "N_oracle_checked": n_qual,
            "N_mutation_checked": n_qual,
            "N_human_reviewed": 0,
            "N_qualified_native": n_qual,
            "note": ("every qualified record passed source verification "
                     "(pinned sha256), schema validation, the OBO/OWL "
                     "parser differential, the full native gate chain "
                     "(leg1 + owlrl leg2 + counterfactual witness + "
                     "positive/negative self-check + leakage scan); human "
                     "review was NOT_RUN for all records; rejected "
                     "candidates are counted in rejections, not here"),
            "gate_fail_counts": stats["rejections"],
            "gate_not_run_counts": {"human_review": n_qual},
        },
        "legacy_collision_check": {
            "legacy_global_task_ids": len(legacy["task_ids"]),
            "legacy_l06_task_ids": len(legacy["l06_task_ids"]),
            "legacy_l06_instance_ids": len(legacy["instance_ids"]),
            "legacy_l06_base_goal_ids": len(legacy["base_goal_ids"]),
            "legacy_l06_public_blobs": len(legacy["public_blobs"]),
            "collisions": legacy_collisions,
            "clean": all(v == 0 for v in legacy_collisions.values()),
        },
        "source_verification": source_check,
        "candidate_manifests": _shard_manifests(cand_dir),
        "qualified_manifests": _shard_manifests(qual_dir),
        "oracle_count": len(list(oracle_dir.glob("l06x-oracle-*.json"))),
        "oracle_sha256": _oracle_manifest(oracle_dir),
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


def _oracle_manifest(oracle_dir: Path) -> str:
    entries = {}
    for p in sorted(oracle_dir.glob("l06x-oracle-*.json")):
        entries[p.name] = _sha256_file(p)
    payload = {"count": len(entries), "files": entries}
    manifest_path = oracle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return _sha256_file(manifest_path)


def _qualify_one20k(proposer: L06ExpansionProposer,
                    verifier: L06ExpansionVerifier, cand: CandidateTaskAST,
                    spec: FamilySpec20k, seen_instances: set[str],
                    seen_task_ids: set[str], seen_public_blobs: set[str],
                    legacy: dict[str, Any],
                    legacy_collisions: dict[str, int],
                    cmd_prefix: str) -> dict[str, Any]:
    row: dict[str, Any] = {"instance_id": cand.instance_id,
                           "family_id": cand.family_id,
                           "base_goal_id": cand.base_goal_id,
                           "qualified": False, "rejection": ""}
    task_id = "l06-" + cand.instance_id
    if cand.instance_id in seen_instances:
        row["rejection"] = "DUPLICATE:instance_id"
        return row
    if task_id in seen_task_ids or task_id in legacy["task_ids"]:
        legacy_collisions["task_id"] += int(task_id in legacy["task_ids"])
        row["rejection"] = "DUPLICATE:task_id"
        return row
    if cand.instance_id in legacy["instance_ids"]:
        legacy_collisions["instance_id"] += 1
        row["rejection"] = "DUPLICATE:legacy_instance_id"
        return row
    if cand.base_goal_id in legacy["base_goal_ids"]:
        legacy_collisions["base_goal_id"] += 1
        row["rejection"] = "DUPLICATE:legacy_base_goal_id"
        return row
    seen_instances.add(cand.instance_id)
    seen_task_ids.add(task_id)
    grounded = proposer.ground(cand)
    blob = _public_blob(grounded.public)
    if blob in seen_public_blobs or blob in legacy["public_blobs"]:
        legacy_collisions["public_blob"] += int(blob in legacy["public_blobs"])
        row["rejection"] = "DUPLICATE:public_blob"
        return row
    seen_public_blobs.add(blob)
    proof = verifier.prove_exists(grounded)
    if not proof.exists:
        row["rejection"] = proof.reason
        return row
    kind = cand.ast["kind"]
    if (spec.min_answer > 1 and kind not in REPAIR_KINDS_20K
            and kind not in GENE_ANSWER_KINDS):
        size = int(proof.evidence.get("answer_size", 0))
        if size < spec.min_answer:
            row["rejection"] = f"TRIVIAL:below_min_answer({size}<{spec.min_answer})"
            return row
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        row["rejection"] = "NON_SEMANTIC:no_counterfactual"
        return row
    frag = induce(cand.ast["universe_ids"], verifier.graph)
    ref_sub = _reference_submission20k(grounded, verifier, frag)
    pos = verifier.verify_terminal(grounded, ref_sub)
    neg = verifier.verify_terminal(
        grounded, _wrong_submission20k(grounded, verifier, frag))
    if not pos.valid or neg.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_selfcheck"
        return row
    public_blob = json.dumps(grounded.public, sort_keys=True)
    if not _leakage_scan20k(public_blob, ref_sub, kind):
        row["rejection"] = "LEAKAGE:answer_in_public"
        return row
    native_evidence = {
        "prove_exists": proof.evidence,
        "semantic_witness": witness.witness,
        "verify_terminal": {"positive": pos.valid,
                            "negative_rejected": not neg.valid,
                            "positive_detail": pos.detail,
                            "negative_detail": neg.detail},
    }
    oracle_handle = grounded.oracle_ref
    evidence_cmd = f"{cmd_prefix}{oracle_handle}.json"
    record = _record(grounded, spec, native_evidence, oracle_handle,
                     evidence_cmd)
    issues = _validate_record(record)
    if issues:
        row["rejection"] = f"SCHEMA_INVALID:{issues[0]}"
        return row
    row["record"] = record
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
    ap = argparse.ArgumentParser(
        description="Build L06 20K-expansion qualified shards")
    ap.add_argument("--index", default=str(
        Path("external_knowledge/indexes/gene-ontology") / RELEASE_ID))
    ap.add_argument("--out", default="bank/staging_20k/l06")
    ap.add_argument("--seed", type=int, default=SEED_20K)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--status-path", default=str(STATUS_PATH))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny quotas (1 goal x 5 per family) for smoke runs")
    args = ap.parse_args()
    max_base_goals = None
    top_up = None
    if args.smoke:
        max_base_goals = {f.family_id: 1 for f in FAMILY_TABLE_20K}
        top_up = {}
    report = build_shards20k(Path(args.index), Path(args.out), seed=args.seed,
                             shard_size=args.shard_size,
                             max_base_goals=max_base_goals, top_up=top_up,
                             status_path=Path(args.status_path))
    print(json.dumps({"qualified": report["qualified"],
                      "candidates": report["candidates"],
                      "base_goals": report["base_goals"],
                      "rejections": report["rejections"],
                      "legacy_collisions":
                          report["legacy_collision_check"]["collisions"],
                      "wall_seconds": report["wall_seconds"]},
                     indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
