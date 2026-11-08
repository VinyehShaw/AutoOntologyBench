"""interfaces: JSON results-interface builders for the P8 exporter (Contract §F).

Metric cells without claimable measured runs are ``null`` + ``status=PENDING``
(never 0, never fabricated); observed denominators may be true 0. Deterministic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from autoontobench.export.aggregate import ClassifiedRow, LoadReport

SCHEMA = "aob-export/1.0"
STATUS_PENDING = "PENDING"
STATUS_MEASURED = "MEASURED"

# Canonical paired-arm set (Shared Contract §E; preregistration arms).
CANONICAL_ARMS: Sequence[str] = (
    "direct", "rag", "code_dsl", "ontology", "oracle_assisted", "corrupt", "repaired",
)
INTERVENTION_ARMS: Sequence[str] = ("oracle_assisted", "corrupt", "repaired")

# Reliability categories (Implementation §11.5 / preregistration.yaml).
RELIABILITY_CATEGORIES: Sequence[str] = (
    "MODEL_FAILURE", "TASK_UNSOLVABLE", "NATIVE_ORACLE_ERROR", "ONTOLOGY_COMPILE_ERROR",
    "RUNNER_TIMEOUT", "PROVIDER_RATE_LIMIT", "SECURITY_ABORT", "BUDGET_EXCEEDED", "UNKNOWN",
)

# Planned source families (Implementation §2) for the provenance-yield skeleton.
PLANNED_SOURCES: Sequence[str] = (
    "gatekeeper", "rhea+chebi", "rdkit", "camunda-dmn", "kubernetes-cel",
    "go", "biomodels", "cql/who-smart-anc", "bddl",
)

PROBE_BUDGETS: Sequence[int] = (0, 1, 2, 4, 8, 16)

CLAIM_PROVENANCE_POLICY = (
    "claimable := provider not in {mock} AND model_exact_id is not scripted/mock "
    "AND domain is not a dev fixture. Mock/scripted dev rows (e.g. G0-respecting "
    "MockModel pilot runs) are engineering evidence only and must never appear as "
    "paper results. PENDING = not measured from claimable runs; never 0, never SIM."
)


def _is_launched_failure(row: ClassifiedRow) -> bool:
    r = row.row
    return bool(r.launched) and r.status not in (None, "SCORED", "PENDING")


def arm_cell(arm: str, rows: List[ClassifiedRow]) -> Dict[str, Any]:
    """One main-TSR cell. Launched denominator includes launched-but-failed runs."""
    arm_rows = [r for r in rows if (r.row.arm or "").lower() == arm]
    launched = [r for r in arm_rows if r.row.launched]
    scored = [r for r in launched if r.row.status == "SCORED"]
    filters = {"arm": arm, "claimable": True}
    if not launched:
        return {
            "arm": arm, "status": STATUS_PENDING, "launched": 0, "launched_failures": 0,
            "scored": 0, "task_successes": None, "tsr": None,
            "valid_task_successes": None, "v_tsr": None, "run_ids": [], "filters": filters,
        }
    successes = sum(1 for r in scored if r.row.task_success is True)
    valid = sum(1 for r in scored if r.row.valid_task_success is True)
    n = len(launched)
    return {
        "arm": arm, "status": STATUS_MEASURED, "launched": n,
        "launched_failures": sum(1 for r in launched if _is_launched_failure(r)),
        "scored": len(scored), "task_successes": successes, "tsr": successes / n,
        "valid_task_successes": valid, "v_tsr": valid / n,
        "run_ids": sorted(r.row.run_id for r in launched), "filters": filters,
    }


def build_main_tsr(report: LoadReport) -> Dict[str, Any]:
    claimable = report.claimable_rows
    observed_arms = sorted({(r.row.arm or "") for r in claimable if r.row.arm})
    cells = [arm_cell(arm, claimable) for arm in CANONICAL_ARMS]
    for extra in observed_arms:
        if extra not in CANONICAL_ARMS:
            cells.append(arm_cell(extra, claimable))
    overall = STATUS_MEASURED if any(c["status"] == STATUS_MEASURED for c in cells) else STATUS_PENDING
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "status": overall,
        "primary_outcome": "valid_task_success_rate (V-TSR); fallback task_success_rate (TSR)",
        "denominator_policy": "launched runs; launched-but-failed count as task_success=false",
        "cells": cells,
        "non_claimable_dev_rows": len(report.non_claimable_rows),
    }


def build_evidence_registry(report: LoadReport) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "files": report.files,
        "totals": {
            "rows": len(report.rows),
            "claimable_rows": len(report.claimable_rows),
            "non_claimable_rows": len(report.non_claimable_rows),
            "launched_rows": sum(1 for r in report.rows if r.row.launched),
            "scored_rows": sum(1 for r in report.rows if r.row.status == "SCORED"),
        },
        "warnings": sorted(report.warnings),
    }


def build_task_manifest(report: LoadReport) -> str:
    by_task: Dict[str, List[ClassifiedRow]] = {}
    for r in report.rows:
        by_task.setdefault(r.row.task_id, []).append(r)
    lines: List[str] = []
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        lines.append(
            json.dumps(
                {
                    "task_id": task_id,
                    "base_goal_id": rows[0].row.base_goal_id,
                    "domain": rows[0].row.domain,
                    "split": rows[0].row.split,
                    "arms": sorted({r.row.arm for r in rows if r.row.arm}),
                    "runs": len(rows),
                    "claimable_runs": sum(1 for r in rows if r.claimable),
                    "statuses": sorted({r.row.status for r in rows if r.row.status}),
                    "run_ids": sorted(r.row.run_id for r in rows),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_source_funnel(report: LoadReport) -> Dict[str, Any]:
    groups: Dict[str, List[ClassifiedRow]] = {}
    for r in report.rows:
        groups.setdefault(r.row.independent_source_group or "UNGROUPED", []).append(r)
    entries = []
    for key in sorted(groups):
        rows = groups[key]
        claimable = [r for r in rows if r.claimable]
        entries.append(
            {
                "independent_source_group": key if key != "UNGROUPED" else None,
                "group_label": key,
                "domains": sorted({r.row.domain for r in rows if r.row.domain}),
                "rows": len(rows),
                "claimable_rows": len(claimable),
                "launched": sum(1 for r in claimable if r.row.launched),
                "scored": sum(1 for r in claimable if r.row.status == "SCORED"),
                "status": STATUS_MEASURED if claimable else STATUS_PENDING,
            }
        )
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "status": STATUS_MEASURED if any(e["status"] == STATUS_MEASURED for e in entries) else STATUS_PENDING,
        "planned_sources": list(PLANNED_SOURCES),
        "groups": entries,
    }


def pending_interface(name: str, report: LoadReport, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "interface": name,
        "status": STATUS_PENDING,
        "pending_reason": "no claimable runs observed for this interface",
    }
    if extra:
        payload.update(extra)
    return payload


def build_causal_interventions(report: LoadReport) -> Dict[str, Any]:
    claimable = [r for r in report.claimable_rows if (r.row.arm or "").lower() in INTERVENTION_ARMS]
    if not claimable:
        return pending_interface(
            "causal_interventions", report, {"required_arms": list(INTERVENTION_ARMS), "pairs": []}
        )
    # Paired replay deltas require frozen action sequences and are a later
    # statistical wave; P8 scaffolding reports per-arm cells only.
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "interface": "causal_interventions",
        "status": STATUS_MEASURED,
        "required_arms": list(INTERVENTION_ARMS),
        "cells": [arm_cell(arm, claimable) for arm in INTERVENTION_ARMS],
        "pairs": [],
        "note": "paired replay deltas require frozen action sequences; not aggregated in P8 scaffolding",
    }


def build_failure_taxonomy(report: LoadReport) -> Dict[str, Any]:
    def taxonomy(rows: List[ClassifiedRow]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in rows:
            if _is_launched_failure(r):
                cat = (r.row.fail_reason or "UNKNOWN").strip().upper()
                if cat not in RELIABILITY_CATEGORIES:
                    cat = "UNKNOWN"
                counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items()))

    claimable = report.claimable_rows
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "status": STATUS_MEASURED if claimable else STATUS_PENDING,
        "categories_known": list(RELIABILITY_CATEGORIES),
        "launched_claimable": sum(1 for r in claimable if r.row.launched),
        "categories": taxonomy(claimable),
        "non_claimable_dev": taxonomy(report.non_claimable_rows),
        "note": "only observed failure categories are listed; absent category = zero observed failures",
    }


def build_claim_provenance(
    report: LoadReport, main_tsr: Dict[str, Any], causal: Dict[str, Any]
) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = [
        {
            "cell": f"main_tsr.{cell['arm']}",
            "interface": "results/main_tsr.json",
            "status": cell["status"],
            "claimable": cell["status"] == STATUS_MEASURED,
            "run_ids": cell["run_ids"],
            "filters": cell["filters"],
        }
        for cell in main_tsr["cells"]
    ]
    claims.append(
        {
            "cell": "causal_interventions",
            "interface": "results/causal_interventions.json",
            "status": causal["status"],
            "claimable": causal["status"] == STATUS_MEASURED,
            "run_ids": sorted(
                r.row.run_id for r in report.claimable_rows if (r.row.arm or "").lower() in INTERVENTION_ARMS
            ),
            "filters": {"arm": list(INTERVENTION_ARMS), "claimable": True},
        }
    )
    claims.extend(
        {
            "cell": name,
            "interface": f"results/{name}.json",
            "status": STATUS_PENDING,
            "claimable": False,
            "run_ids": [],
            "filters": {"claimable": True},
        }
        for name in ("semantics_vs_utility", "active_budget", "cost_reuse", "failure_taxonomy")
    )
    return {
        "schema": SCHEMA,
        "input_digest": report.input_digest,
        "policy": CLAIM_PROVENANCE_POLICY,
        "claims": claims,
        "non_claimable_rows": [
            {
                "run_id": r.row.run_id,
                "task_id": r.row.task_id,
                "arm": r.row.arm,
                "model_exact_id": r.row.model_exact_id,
                "reason": r.non_claimable_reason,
                "source_path": r.source_path,
            }
            for r in sorted(report.non_claimable_rows, key=lambda r: r.row.run_id)
        ],
    }
