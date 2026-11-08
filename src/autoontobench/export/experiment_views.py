"""experiment_views: E3/E4/E5/E6 Codex-alignment views (Review-5 alignment doc §4).

- ``build_task_transitions``: §4-E3 paired Gold/Corrupt/Repaired outcome
  transitions. Rows are emitted ONLY for real, joinable, claimable paired runs
  (same base goal, model, seeds, experiment scope) — never simulated. When no
  complete triple exists the JSONL file is empty and the summary lands in
  ``causal_interventions.transitions_summary`` / ``claim_provenance``.
- ``build_mapping_ablation``: §4-E4 defect x workload cells; measured only from
  qualified Mapping tasks with real defect runs (none yet -> BLOCKED/PENDING
  with instrument evidence cited, no synthetic aliases).
- ``build_active_probe_diagnostics``: §4-E5 Discovery-only diagnostics; the
  current discovery stratum is INSTRUMENT_ONLY (9/10 pairs min_probe_cost==1).
- ``build_generalization``: §4-E6 reuse/OOD schema with the Shared-contract
  reuse points {1,2,4,8,16} (override of the old Review5B {1,2,5,10,20}).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from autoontobench.export.aggregate import LoadReport
from autoontobench.export.aux_inputs import AuxData
from autoontobench.export.interfaces import SCHEMA, STATUS_MEASURED, STATUS_PENDING

STATUS_BLOCKED = "BLOCKED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_INSTRUMENT_ONLY = "INSTRUMENT_ONLY"
ALIGNMENT_EXT = "w17-codex-alignment/1.0"

GOLD_ARMS = frozenset({"oracle_assisted", "gold"})
REUSE_POINTS: Tuple[int, ...] = (1, 2, 4, 8, 16)
PROBE_POLICIES: Tuple[str, ...] = (
    "NO_PROBE", "RANDOM", "BOUNDARY", "MATCHED_PASSIVE", "AGENT_ACTIVE",
    "FINITE_CATALOGUE_REFERENCE",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_task_transitions(report: LoadReport) -> Tuple[str, Dict[str, Any]]:
    """Return (jsonl_text, summary). Rows only for real gold/corrupt/repaired triples."""
    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in report.claimable_rows:
        if not r.row.launched:
            continue
        key = (r.row.base_goal_id, r.row.model_exact_id, r.row.source_seed,
               r.row.repeat_seed, r.experiment_scope)
        groups.setdefault(key, {})[(r.row.arm or "").lower()] = r

    lines: List[str] = []
    partial: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        arms = groups[key]
        gold = next((arms[a] for a in arms if a in GOLD_ARMS), None)
        corrupt = arms.get("corrupt")
        repaired = arms.get("repaired")
        if gold is None or corrupt is None or repaired is None:
            missing = [n for n, a in (("gold", gold), ("corrupt", corrupt), ("repaired", repaired)) if a is None]
            partial.append({
                "base_goal_id": key[0], "model": key[1], "source_seed": key[2],
                "repeat_seed": key[3], "experiment_scope": key[4],
                "arms_present": sorted(arms), "missing_arms": missing,
            })
            continue

        def _s(cr: Any) -> Optional[bool]:
            return cr.row.task_success if cr.row.status == "SCORED" else None

        outcomes = {"gold": _s(gold), "corrupt": _s(corrupt), "repaired": _s(repaired)}
        transition = "->".join(
            "success" if v else ("fail" if v is not None else "unscored")
            for v in (outcomes["gold"], outcomes["corrupt"], outcomes["repaired"])
        )
        row = {
            "task_id": gold.row.task_id,
            "base_goal_id": gold.row.base_goal_id,
            "source_group": gold.row.independent_source_group,
            "model": gold.row.model_exact_id,
            "seed": gold.row.source_seed,
            "fault_type": None,  # requires per-task fault-assignment join (gold_o certification)
            "fault_type_status": STATUS_PENDING,
            "governed_preassigned": sorted(gold.row.source_rule_units),
            "applicable": True,
            "gold_run_id": gold.run_ref,
            "corrupt_run_id": corrupt.run_ref,
            "repaired_run_id": repaired.run_ref,
            "gold_S": outcomes["gold"],
            "corrupt_S": outcomes["corrupt"],
            "repaired_S": outcomes["repaired"],
            "equivalent_S": None,   # equivalence-control arm not run as an agent arm
            "irrelevant_S": None,   # irrelevant-edit control arm not run as an agent arm
            "controls_status": STATUS_PENDING,
            "replay_changed": None,  # consumer-replay evidence lives in causal_interventions.affected_vs_unaffected
            "final_task_transition": transition,
            "evidence_sha": _sha(json.dumps(
                {"gold": gold.run_ref, "corrupt": corrupt.run_ref, "repaired": repaired.run_ref},
                sort_keys=True)),
            "status": STATUS_MEASURED,
        }
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))

    if lines:
        summary: Dict[str, Any] = {
            "status": STATUS_MEASURED, "rows_emitted": len(lines), "partial_pairs": partial,
        }
    else:
        summary = {
            "status": STATUS_PENDING,
            "rows_emitted": 0,
            "reason": "no claimable (gold, corrupt, repaired) triple exists: preflight_p1 ran "
                      "gold/direct/ontology only; corrupt/repaired exist solely as W13 mock "
                      "rehearsals (non-claimable, data/erule/review5/agent_eval/rehearsal_w13)",
            "partial_pairs": partial,
            "missing_arms": ["corrupt", "repaired"],
        }
    return ("\n".join(lines) + "\n" if lines else ""), summary


def build_mapping_ablation(report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    rhea_supplies = [s for s in auxd.supplies if any(
        r["task_family"] == "organization" for r in s.rows)]
    for supply in rhea_supplies:
        t57 = supply.manifest_checks_summary.get("controlled_o_content_sensitivity") or {}
        t57_pass = t57.get("PASS", 0)
        goals = sorted({r["base_goal_id"] for r in supply.rows if r["task_family"] == "organization"})
        mech = sorted({r["mechanism_group_id"] for r in supply.rows if r["mechanism_group_id"]})
        for defect in ("mapping_missing", "mapping_swapped"):
            cells.append({
                "source_group": supply.rows[0]["source_group_id"] if supply.rows else None,
                "defect_type": defect,
                "task_type": "cross_source_join",
                "governed_tested": t57_pass,
                "gold_S": None, "corrupt_S": None, "repaired_S": None,
                "paired_delta_tsr": None,
                "mapping_precision": None, "mapping_recall": None, "mapping_f1": None,
                "witness_count": t57_pass,
                "n_goals": len(goals), "n_groups": len(mech),
                "status": STATUS_BLOCKED,
                "reason": "no agent runs under mapping-defect arms on Organization tasks; "
                          "run-level cells cannot be computed. Instrument-level Phi causality "
                          "IS verified (T5.7 cohort: Phi_correct==truth, Phi_empty/Phi_swapped "
                          "change the governed answer, non-effect control unchanged).",
            })
    instrument_evidence: List[Dict[str, Any]] = []
    if auxd.phi_demonstration is not None:
        phi = auxd.phi_demonstration
        instrument_evidence.append({
            "kind": "phi_causality_demonstration",
            "governed_query": phi.get("governed_query"),
            "control_query": phi.get("control_query"),
            "governed_changed": phi.get("governed_changed"),
            "control_unchanged": phi.get("control_unchanged"),
            "phi_correct_matches_truth": phi.get("phi_correct_matches_truth"),
            "truth_governed_count": phi.get("truth_governed_count"),
        })
    if auxd.gold_o_validation is not None:
        instrument_evidence.append({
            "kind": "gatekeeper_constraint_defects_NOT_cross_source_mapping",
            "ref": "data/erule/review5/agent_eval/gold_o/validation.json",
            "note": "W13 certified per-unit corrupt mutations (repo prefix / label key / "
                    "inverted range / swapped service type); admission-control constraint "
                    "defects, deliberately NOT counted as cross-source Mapping (doc §4-E4)",
        })
    return {
        "schema": SCHEMA,
        "alignment_extension": ALIGNMENT_EXT,
        "paper_experiment_id": ["E4"],
        "implementation_stage": ["stage-E2"],
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "status": STATUS_BLOCKED,
        "blocked_reason": "no qualified Mapping paper-runs (agent defect-cohort runs) exist; "
                          "only instrument-level evidence is available",
        "cells": cells,
        "instrument_evidence": instrument_evidence,
        "no_synthetic_aliases": True,
    }


def build_active_probe_diagnostics(report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    cert = auxd.discovery_cert_rows
    non_control = [r for r in cert if not r.get("control") and r.get("min_probe_cost") is not None]
    one_probe = sum(1 for r in non_control if r.get("min_probe_cost") == 1)
    observed_probe_usage = sorted(
        (
            {
                "run_ref": r.run_ref,
                "arm": r.row.arm,
                "base_goal_id": r.row.base_goal_id,
                "probe_cost": r.row.probe_cost,
                "tool_calls": r.row.tool_calls,
            }
            for r in report.claimable_rows if r.row.launched
        ),
        key=lambda d: d["run_ref"],
    )
    return {
        "schema": SCHEMA,
        "alignment_extension": ALIGNMENT_EXT,
        "paper_experiment_id": ["E5"],
        "implementation_stage": ["stage-E3"],
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "status": STATUS_INSTRUMENT_ONLY,
        "scope": "Discovery tasks only; NOT_APPLICABLE for non-discovery regimes",
        "strata": {
            "easy": {
                "status": STATUS_INSTRUMENT_ONLY,
                "evidence": "data/erule/identifiability_certificate.jsonl",
                "certificate_rows": len(cert),
                "non_control_pairs": len(non_control),
                "one_probe_pairs": one_probe,
                "note": "old two-candidate pairs are usable only as Easy / instrument sanity "
                        "(alignment doc §4-E5 acceptance)",
            },
            "medium": {"status": STATUS_BLOCKED,
                       "reason": "needs >=4 compatible hypotheses or a real stateful "
                                 "prerequisite with a valid distinction-cost certificate; none exist"},
            "hard": {"status": STATUS_BLOCKED,
                     "reason": "same as medium; no hard discovery task has been qualified"},
        },
        "diagnostics": [],
        "diagnostics_status": STATUS_PENDING,
        "diagnostics_reason": "no Discovery agent runs; hypothesis-catalogue fields "
                              "(initial_hypotheses, remaining_hypotheses, identifiable_by_B, "
                              "oracle_min_cost, distinction_found) apply only to verified "
                              "finite-catalogue Discovery tasks and stay null/NOT_APPLICABLE "
                              "otherwise (never fabricated)",
        "probe_policies": [{"policy": p, "status": STATUS_PENDING,
                            "reason": "no budget-sweep runs (B in {0,1,2,4,8,16})"} for p in PROBE_POLICIES],
        "observed_probe_usage": observed_probe_usage,
        "observed_probe_usage_note": "real probe charges on non-discovery (gatekeeper) tasks; "
                                     "observational only — NOT the E5 budget sweep",
    }


def build_generalization(report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "alignment_extension": ALIGNMENT_EXT,
        "paper_experiment_id": ["E6"],
        "implementation_stage": ["stage-E4"],
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "status": STATUS_PENDING,
        "pending_reason": "no reuse/OOD runs exist; only the dev split is qualified "
                          "(hidden_source/hidden_mechanism/hidden_goal await preregistration lock)",
        "reuse_points": list(REUSE_POINTS),
        "reuse_points_note": "n in {1,2,4,8,16} per alignment doc §3.1 item 6; this OVERRIDES the "
                             "old Review5B §6 Stage-E4 points {1,2,5,10,20}. If fewer than 16 "
                             "independent base goals exist, only computable points are reported.",
        "shift_types": [
            {"shift_type": "entity_combination_zero_edit", "zero_edit": True, "status": STATUS_PENDING},
            {"shift_type": "mechanism_holdout", "zero_edit": False, "status": STATUS_PENDING},
            {"shift_type": "source_holdout", "zero_edit": False, "status": STATUS_PENDING},
            {"shift_type": "new_mechanism_research", "zero_edit": False, "status": STATUS_PENDING,
             "note": "fresh bounded research task; NOT zero-shot reuse"},
        ],
        "cells": [],
        "cell_schema": ["shift_type", "zero_edit", "source_group", "task_ids", "arm", "model",
                        "n_goals", "n_launched", "n_success", "tsr", "valid_tsr",
                        "new_build_cost", "patch_cost", "run_cost", "status"],
        "cost_divisor_rule": "cost_per_* is null when the divisor is 0 — never 0.0",
    }
