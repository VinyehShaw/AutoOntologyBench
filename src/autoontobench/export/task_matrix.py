"""task_matrix: E0/E2 Codex-alignment views (Review-5 alignment doc §4-E0/E2).

- ``build_task_matrix``: ``source x regime x output_profile x eligibility x
  qualified_goal_count`` from the qualified-task supplies + the source
  execution matrix (Review5B §6-required interface, added in W17).
- ``build_task_regime_metrics``: rows at
  ``source_group x domain x regime x difficulty x arm x model x split`` with
  the §4-E2 column set; unmeasured cells are PENDING/BLOCKED/NOT_APPLICABLE
  with explicit reasons — never 0, never simulated.

``task_regime`` is undeclared (null) in the current ``aob-task/5.0`` task
population, so regime keys are honest nulls and the declared ``task_family`` /
``track`` / ``ontology_contract`` (output profile) are carried alongside.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from autoontobench.export.aggregate import LoadReport
from autoontobench.export.aux_inputs import AuxData
from autoontobench.export.interfaces import (
    CANONICAL_ARMS,
    SCHEMA,
    STATUS_MEASURED,
    STATUS_PENDING,
)

STATUS_BLOCKED = "BLOCKED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_INSTRUMENT_ONLY = "INSTRUMENT_ONLY"
ALIGNMENT_EXT = "w17-codex-alignment/1.0"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _supply_eligibility_rows(auxd: AuxData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for supply in auxd.supplies:
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in supply.rows:
            key = (r["source_group_id"], r["domain"], r["track"], r["task_family"],
                   r["task_regime"], r["ontology_contract"], r["split"])
            groups.setdefault(key, []).append(r)
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            rs = groups[key]
            goals = sorted({r["base_goal_id"] for r in rs if r["base_goal_id"]})
            rows.append({
                "supply": supply.label,
                "source_group": key[0],
                "domain": key[1],
                "track": key[2],
                "task_family": key[3],
                "task_regime": key[4],
                "output_profile": key[5],
                "split": key[6],
                "eligibility": "QUALIFIED",
                "qualified_goal_count": len(goals),
                "qualified_row_count": len(rs),
                "mechanism_group_count": len({r["mechanism_group_id"] for r in rs if r["mechanism_group_id"]}),
                "task_provenance": sorted({r["task_provenance"] for r in rs if r["task_provenance"]}),
                "status": STATUS_MEASURED,
                "evidence": {"tasks_sha256": supply.tasks_sha256, "manifest_sha256": supply.manifest_sha256},
                "manifest_crosscheck": {
                    "unique_goals": supply.manifest_counts.get("unique_goals"),
                    "matches_computed": supply.manifest_counts.get("unique_goals") == len(goals),
                },
            })
    return rows


def _source_eligibility_rows(auxd: AuxData, covered_groups: set) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sm in auxd.source_matrix_rows:
        sid = sm.get("source_id", "")
        status = (sm.get("status") or "").strip()
        if status == "RUNNABLE" and sid in ("gatekeeper", "rhea", "chebi"):
            continue  # covered by qualified supplies above
        if status == "BLOCKED":
            eligibility, reason = STATUS_BLOCKED, (sm.get("blocked_reason") or "source blocked").strip()
        elif status == "RUNNABLE-SMOKE":
            eligibility, reason = STATUS_PENDING, "source runnable at smoke level only; no qualified tasks yet"
        else:
            eligibility, reason = STATUS_PENDING, "no qualified task supply for this source"
        rows.append({
            "supply": None, "source_group": sid, "domain": None, "track": None,
            "task_family": None, "task_regime": None, "output_profile": None, "split": None,
            "eligibility": eligibility, "qualified_goal_count": 0, "qualified_row_count": 0,
            "mechanism_group_count": 0, "task_provenance": [], "status": eligibility,
            "reason": reason, "source_status": status,
            "evidence": {"log_path": sm.get("log_path"), "native_smoke_command": sm.get("native_smoke_command")},
        })
    return rows


def _regime_availability(auxd: AuxData) -> Dict[str, Any]:
    org_goals = sorted({
        r["base_goal_id"] for s in auxd.supplies for r in s.rows
        if r["task_family"] == "organization" and r["decision"] == "QUALIFIED"
    })
    return {
        "repair": {
            "status": STATUS_PENDING, "eligible_goals": 0,
            "reason": "no task with a declared repair regime is qualified; repair evidence is "
                      "instrument-level (24 semantic faults, 8/8) plus W13 corrupt/repaired arm "
                      "configs; task_regime is undeclared (null) in the aob-task/5.0 population",
        },
        "organization": {
            "status": STATUS_MEASURED if org_goals else STATUS_BLOCKED,
            "eligible_goals": len(org_goals), "goal_ids": org_goals,
            "reason": None if org_goals else "no qualified multi-source Organization tasks",
        },
        "construction": {
            "status": STATUS_BLOCKED, "eligible_goals": 0,
            "reason": "discovery stratum is INSTRUMENT_ONLY (9/10 identifiable pairs "
                      "min_probe_cost==1, data/erule/identifiability_certificate.jsonl); no "
                      ">=4-hypothesis construction task has been qualified",
        },
        "note": "paper regimes are Repair/Organization/Construction (alignment doc §4-E2); "
                "gatekeeper supplies declare task_family=application (compliance-remediation "
                "user goals), not a paper regime — no relabeling was performed",
    }


def build_task_matrix(report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    rows = _supply_eligibility_rows(auxd)
    rows += _source_eligibility_rows(auxd, set())
    status = STATUS_MEASURED if any(r["eligibility"] == "QUALIFIED" for r in rows) else STATUS_PENDING
    if not auxd.supplies and not auxd.source_matrix_rows:
        status = STATUS_PENDING
        reason = "aux inputs missing: " + ", ".join(auxd.missing or ["none provided"])
    else:
        reason = None
    return {
        "schema": SCHEMA,
        "alignment_extension": ALIGNMENT_EXT,
        "paper_experiment_id": ["E0", "E2"],
        "implementation_stage": ["stage-E0", "task-factory"],
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "status": status,
        "pending_reason": reason,
        "grain": "supply x source x regime x output_profile x eligibility",
        "rows": rows,
        "regime_availability": _regime_availability(auxd),
        "funnel_rows": len(auxd.funnel_rows),
        "funnel_ref": "audit/review5/TASK_FACTORY_FUNNEL.csv",
    }


def _difficulty_of(goal: Optional[str], auxd: AuxData) -> Dict[str, Any]:
    if goal is None or goal not in auxd.blueprints:
        return {"min_action_count": None, "interacting_units": None, "goal_mode": None}
    bp = auxd.blueprints[goal]
    return {"min_action_count": bp["min_action_count"],
            "interacting_units": bp["interacting_units"], "goal_mode": bp.get("goal_mode")}


def _cohorts(auxd: AuxData) -> Dict[tuple, Dict[str, Any]]:
    cohorts: Dict[tuple, Dict[str, Any]] = {}
    for supply in auxd.supplies:
        by_goal: Dict[str, List[Dict[str, Any]]] = {}
        for r in supply.rows:
            by_goal.setdefault(r["base_goal_id"], []).append(r)
        for goal, rs in sorted(by_goal.items()):
            diff = _difficulty_of(goal, auxd)
            key = (rs[0]["source_group_id"], rs[0]["domain"], rs[0]["task_regime"],
                   json.dumps(diff, sort_keys=True), rs[0]["split"])
            c = cohorts.setdefault(key, {
                "source_group": key[0], "domain": key[1], "regime": key[2],
                "difficulty": diff, "split": key[4],
                "task_family": rs[0]["task_family"], "track": rs[0]["track"],
                "ontology_contract": rs[0]["ontology_contract"],
                "eligible_task_ids": [], "base_goal_ids": [],
            })
            c["eligible_task_ids"].extend(r["task_id"] for r in rs if r["task_id"])
            c["base_goal_ids"].append(goal)
    for c in cohorts.values():
        c["eligible_task_ids"] = sorted(set(c["eligible_task_ids"]))
        c["base_goal_ids"] = sorted(set(c["base_goal_ids"]))
    return cohorts


def build_task_regime_metrics(report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    claimable = report.claimable_rows
    models = sorted({r.row.model_exact_id for r in claimable if r.row.model_exact_id})
    arms = sorted(set(CANONICAL_ARMS) | {(r.row.arm or "") for r in claimable if r.row.arm})
    cohorts = _cohorts(auxd)
    ordered = [cohorts[k] for k in sorted(cohorts, key=lambda k: (str(k[0]), str(k[1]), k[4], str(k[3])))]
    rows: List[Dict[str, Any]] = []
    for c in ordered:
        goals = set(c["base_goal_ids"])
        for arm in arms:
            for model in models or [None]:
                matched = [r for r in claimable
                           if (r.row.arm or "") == arm and r.row.model_exact_id == model
                           and r.row.base_goal_id in goals and r.row.split == c["split"]]
                launched = [r for r in matched if r.row.launched]
                scored = [r for r in launched if r.row.status == "SCORED"]
                n_success = sum(1 for r in scored if r.row.task_success is True)
                status = STATUS_MEASURED if launched else STATUS_PENDING
                rows.append({
                    "source_group": c["source_group"], "domain": c["domain"],
                    "regime": c["regime"], "regime_declared": c["regime"] is not None,
                    "task_family": c["task_family"], "difficulty": c["difficulty"],
                    "arm": arm, "model": model, "split": c["split"],
                    "eligible_task_ids": c["eligible_task_ids"],
                    "n_base_goals": len(goals),
                    "n_launched": len(launched), "n_scored": len(scored),
                    "n_success": n_success if launched else None,
                    "tsr": (n_success / len(launched)) if launched else None,
                    "semantic_fidelity_status": (
                        STATUS_NOT_APPLICABLE if c["ontology_contract"] == "native-only/0"
                        else STATUS_PENDING),
                    "n_independent_groups": len({r.row.independent_source_group for r in launched if r.row.independent_source_group}),
                    "experiment_scopes": sorted({r.experiment_scope for r in launched}),
                    "run_refs": sorted(r.run_ref for r in launched),
                    "evidence_sha": _sha(json.dumps(
                        {"goals": sorted(goals), "runs": sorted(r.run_ref for r in launched)},
                        sort_keys=True)),
                    "status": status,
                    "pending_reason": None if launched else "no claimable runs for this cohort/arm/model",
                })
    eligibility_matrix = [
        {"source_group": c["source_group"], "domain": c["domain"], "regime": c["regime"],
         "task_family": c["task_family"], "output_profile": c["ontology_contract"],
         "split": c["split"], "eligibility": "QUALIFIED", "n_base_goals": len(c["base_goal_ids"])}
        for c in ordered
    ]
    org = _regime_availability(auxd)["organization"]
    metrics_by_regime = {
        "repair": {"status": STATUS_PENDING,
                   "reason": "no declared repair-regime tasks; repair arms (corrupt/repaired) "
                             "certified at instrument level (W13), no agent runs"},
        "organization": {"status": STATUS_PENDING if org["eligible_goals"] else STATUS_BLOCKED,
                         "qualified_goals": org["eligible_goals"],
                         "agent_runs": sum(1 for r in claimable if r.row.domain == "biochemical_linked_data"),
                         "instrument_evidence": ["phi_causality T5.7 cohort (manifest_rhea.json)",
                                                 "data/erule/review5/scientific/logs/phi_demonstration.json"],
                         "reason": "12 qualified Organization goals exist; zero agent runs on them"},
        "construction": {"status": STATUS_INSTRUMENT_ONLY,
                         "reason": "old two-candidate discovery pairs only (9/10 min_probe_cost==1); "
                                   "no sealed-O construction task qualified"},
    }
    return {
        "schema": SCHEMA,
        "alignment_extension": ALIGNMENT_EXT,
        "paper_experiment_id": ["E2"],
        "implementation_stage": ["stage-E1", "stage-E3"],
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "status": STATUS_MEASURED if any(r["status"] == STATUS_MEASURED for r in rows) else STATUS_PENDING,
        "grain": "source_group x domain x regime x difficulty x arm x model x split",
        "rows": rows,
        "eligibility_matrix": eligibility_matrix,
        "metrics_by_regime": metrics_by_regime,
        "blueprint_conflicts": auxd.blueprint_conflicts,
    }
