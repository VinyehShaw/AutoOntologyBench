"""alignment: W17 Codex-alignment extensions of the legacy results interfaces.

Paper ``E0``–``E7`` (research questions) and development ``Stage E0``–``E5``
are DIFFERENT numbering spaces (alignment doc §2): every extended payload
carries both ``paper_experiment_id`` and ``implementation_stage``; no legacy
field is renamed or reinterpreted (append-only).

Extends: ``main_tsr`` (§4-E1 cells), ``causal_interventions`` (E3
transitions/affected-vs-unaffected), ``failure_taxonomy`` (E7 derived layer,
see failure_analysis.py). Registry/provenance extensions live in
``registry_ext.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from autoontobench.export.aggregate import ClassifiedRow, LoadReport
from autoontobench.export.aux_inputs import AuxData
from autoontobench.export.failure_analysis import build_derived_analysis
from autoontobench.export.interfaces import STATUS_MEASURED, STATUS_PENDING

STATUS_BLOCKED = "BLOCKED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_INSTRUMENT_ONLY = "INSTRUMENT_ONLY"
ALIGNMENT_EXT = "w17-codex-alignment/1.0"

# Preregistration uncertainty policy: paired cluster bootstrap ONLY when >= 30
# independent mechanism clusters; otherwise uncertainty is explicitly
# not estimable (no fabricated 95% CI from 1-4 units).
CI_MIN_INDEPENDENT_GROUPS = 30

PAIRED_PARTNERS: Sequence[str] = ("code", "direct", "rag")


def _sums(rows: List[ClassifiedRow]) -> Dict[str, Any]:
    def total(field: str) -> Optional[int]:
        vals = [getattr(r.row, field) for r in rows if getattr(r.row, field) is not None]
        return sum(vals) if vals else None
    return {
        "tokens_input_total": total("tokens_input"),
        "tokens_output_total": total("tokens_output"),
        "api_calls_total": total("api_calls"),
        "wall_seconds_total": total("wall_seconds"),
    }


def _paired_delta(all_rows: List[ClassifiedRow], arm: str, partner: str,
                  scope_filter: Optional[str] = None) -> Dict[str, Any]:
    """Paired per-goal TSR delta of `arm` vs `partner`, per experiment scope."""
    by_scope: Dict[str, List[ClassifiedRow]] = {}
    for r in all_rows:
        if scope_filter is None or r.experiment_scope == scope_filter:
            by_scope.setdefault(r.experiment_scope, []).append(r)
    deltas: List[Dict[str, Any]] = []
    for scope in sorted(by_scope):
        scope_rows = by_scope[scope]
        arms_in_scope = {r.row.arm for r in scope_rows}
        if arm not in arms_in_scope:
            continue  # only scopes where the reference arm actually ran

        def _per_goal(arm: str) -> Dict[str, List[bool]]:
            per_goal: Dict[str, List[bool]] = {}
            for r in scope_rows:
                if r.row.status == "SCORED" and r.row.arm == arm and r.row.base_goal_id:
                    per_goal.setdefault(r.row.base_goal_id, []).append(r.row.task_success is True)
            return per_goal

        goals_arm = _per_goal(arm)
        goals_partner = _per_goal(partner)
        paired = sorted(set(goals_arm) & set(goals_partner))
        if partner not in arms_in_scope:
            deltas.append({"experiment_scope": scope, "status": STATUS_PENDING,
                           "reason": f"no '{partner}' arm runs in this scope"})
            continue
        if not paired:
            deltas.append({"experiment_scope": scope, "status": STATUS_PENDING,
                           "reason": "no base goal scored under both arms in this scope"})
            continue

        def _mean(bits: List[bool]) -> float:
            return sum(bits) / len(bits)

        delta = sum(_mean(goals_arm[g]) - _mean(goals_partner[g]) for g in paired) / len(paired)
        deltas.append({
            "experiment_scope": scope, "status": STATUS_MEASURED,
            "n_paired_goals": len(paired),
            "n_discordant": sum(1 for g in paired
                                if any(goals_arm[g]) != any(goals_partner[g])),
            "delta_tsr": delta,
            "rule": "per-goal mean task_success rate, averaged over paired goals",
        })
    status = STATUS_MEASURED if any(d["status"] == STATUS_MEASURED for d in deltas) else STATUS_PENDING
    return {"status": status, "deltas": deltas,
            "policy": "paired within (experiment_scope, base_goal_id, model, split); "
                      "GOLD is oracle-assisted and excluded from fairness contrasts"}


def _cell_extension(cell_rows: List[ClassifiedRow], pairing_pool: List[ClassifiedRow],
                    arm: str, scope_filter: Optional[str] = None) -> Dict[str, Any]:
    launched = [r for r in cell_rows if (r.row.arm or "").lower() == arm and r.row.launched]
    goals = sorted({r.row.base_goal_id for r in launched if r.row.base_goal_id})
    groups = sorted({r.row.independent_source_group for r in launched if r.row.independent_source_group})
    scopes = sorted({r.experiment_scope for r in launched})
    paired: Dict[str, Any] = {}
    if arm == "ontology" and launched:
        for partner in PAIRED_PARTNERS:
            paired[f"paired_vs_{partner}"] = _paired_delta(pairing_pool, arm, partner, scope_filter)
    elif arm in ("oracle_assisted", "gold"):
        for partner in PAIRED_PARTNERS:
            paired[f"paired_vs_{partner}"] = {
                "status": STATUS_NOT_APPLICABLE, "deltas": [],
                "reason": "GOLD/oracle-assisted is an upper reference with EXTRA information, "
                          "excluded from paired causal fairness claims (preregistration parity_rules)"}
    else:
        for partner in PAIRED_PARTNERS:
            paired[f"paired_vs_{partner}"] = {
                "status": STATUS_NOT_APPLICABLE, "deltas": [],
                "reason": "preregistered paired contrasts are Self-O vs {code,direct,rag}"}
    return {
        "n_goals": len(goals), "n_groups": len(groups),
        "source_group_ids": groups,
        "model_exact_ids": sorted({r.row.model_exact_id for r in launched if r.row.model_exact_id}),
        "experiment_scopes": scopes,
        "mixed_generations": len(scopes) > 1,
        "run_refs": [r.run_ref for r in sorted(launched, key=lambda r: r.run_ref)],
        "ci_status": "ESTIMABLE" if len(groups) >= CI_MIN_INDEPENDENT_GROUPS else "NOT_ESTIMABLE",
        "cost_total": None,
        "cost_status": STATUS_PENDING,
        "cost_reason": "USD rate card pending PI decision; exact token/call totals are reported "
                       "in tokens_*_total / api_calls_total",
        "cost_per_valid_success": None,
        "cost_per_valid_success_reason": "null when cost or valid-success denominator is "
                                         "unavailable/zero — never 0.0",
        **paired,
        **_sums(launched),
    }


def extend_main_tsr(main_tsr: Dict[str, Any], report: LoadReport) -> Dict[str, Any]:
    """Append the §4-E1 field list to each legacy cell + add cohort cells."""
    claimable = report.claimable_rows
    for cell in main_tsr["cells"]:
        cell.update(_cell_extension(claimable, claimable, cell["arm"]))

    cohorts: Dict[tuple, List[ClassifiedRow]] = {}
    for r in claimable:
        key = (r.experiment_scope, r.row.domain, r.row.arm, r.row.model_exact_id, r.row.split)
        cohorts.setdefault(key, []).append(r)
    cells_e1: List[Dict[str, Any]] = []
    for key in sorted(cohorts, key=lambda k: tuple(str(x) for x in k)):
        rows = cohorts[key]
        launched = [r for r in rows if r.row.launched]
        scored = [r for r in launched if r.row.status == "SCORED"]
        n_success = sum(1 for r in scored if r.row.task_success is True)
        n_valid = sum(1 for r in scored if r.row.valid_task_success is True)
        groups = sorted({r.row.independent_source_group for r in launched if r.row.independent_source_group})
        ext = _cell_extension(rows, claimable, (key[2] or "").lower(), scope_filter=key[0])
        cells_e1.append({
            "domain": key[1],
            "task_regime": None,  # undeclared in aob-task/5.0 population (see regime_declared)
            "regime_declared": False,
            "semantic_profile": None,  # not recorded in RunRowV5; not derivable — never guessed
            "arm": key[2],
            "model_exact_id": key[3],
            "split": key[4],
            "experiment_scope": key[0],
            "source_group_ids": groups,
            "n_goals": len({r.row.base_goal_id for r in launched if r.row.base_goal_id}),
            "n_groups": len(groups),
            "n_launched": len(launched),
            "n_scored": len(scored),
            "n_success": n_success,
            "n_valid_success": n_valid,
            "tsr": (n_success / len(launched)) if launched else None,
            "v_tsr": (n_valid / len(launched)) if launched else None,
            "paired_vs_code": ext["paired_vs_code"],
            "paired_vs_direct": ext["paired_vs_direct"],
            "paired_vs_rag": ext["paired_vs_rag"],
            "cost_total": ext["cost_total"],
            "cost_status": ext["cost_status"],
            "cost_reason": ext["cost_reason"],
            "cost_per_valid_success": ext["cost_per_valid_success"],
            "ci_status": ext["ci_status"],
            "run_refs": ext["run_refs"],
            **{k: v for k, v in ext.items() if k.endswith("_total")},
            "status": STATUS_MEASURED if launched else STATUS_PENDING,
        })
    main_tsr["paper_experiment_id"] = ["E1"]
    main_tsr["implementation_stage"] = ["stage-E1", "stage-E3"]
    main_tsr["alignment_extension"] = ALIGNMENT_EXT
    main_tsr["ci_min_independent_groups"] = CI_MIN_INDEPENDENT_GROUPS
    main_tsr["ci_rule"] = ("ci_status=NOT_ESTIMABLE when independent source groups < "
                           f"{CI_MIN_INDEPENDENT_GROUPS} (preregistration uncertainty policy: "
                           "cluster bootstrap only at >= 30 clusters)")
    main_tsr["cells_v2_e1"] = cells_e1
    return main_tsr


def extend_causal_interventions(causal: Dict[str, Any], auxd: AuxData,
                                transitions_summary: Dict[str, Any]) -> Dict[str, Any]:
    replay: Dict[str, Any] = {"status": STATUS_PENDING, "cases": []}
    for entry in auxd.interventions:
        data = entry["data"]
        case: Dict[str, Any] = {
            "blueprint": data.get("blueprint"),
            "evidence_path": entry["path"],
            "evidence_sha256": entry["sha256"],
            "artifact_shas": data.get("artifact_shas"),
            "milestone": data.get("milestone"),
        }
        comparisons = data.get("comparisons") or {}
        if comparisons:
            case["comparisons"] = {
                name: {k: comp.get(k) for k in
                       ("variant", "governed_changes", "governed_total",
                        "unaffected_changes", "unaffected_total", "changed_steps")}
                for name, comp in sorted(comparisons.items())
            }
            if data.get("outcome"):
                case["outcome"] = data["outcome"]
        else:
            checks = data.get("checks") or []
            governed = [c for c in checks if c.get("governed")]
            unaffected = [c for c in checks if not c.get("governed")]

            def _changed(c: Dict[str, Any]) -> bool:
                d = c.get("decisions") or {}
                return (d.get("gold") or {}).get("prediction") != (d.get("corrupt") or {}).get("prediction")

            case.update({
                "governed_checks": len(governed),
                "governed_changed": sum(1 for c in governed if _changed(c)),
                "unaffected_checks": len(unaffected),
                "unaffected_changed": sum(1 for c in unaffected if _changed(c)),
            })
        replay["cases"].append(case)
    if replay["cases"]:
        replay["status"] = STATUS_MEASURED
        replay["level"] = "consumer_replay_only"
        replay["note"] = ("fixed-input O-swap replay decisions; NOT whole-task replanned "
                          "causality (alignment doc §4-E3 two-layer rule)")
    causal["paper_experiment_id"] = ["E3"]
    causal["implementation_stage"] = ["stage-E2"]
    causal["alignment_extension"] = ALIGNMENT_EXT
    causal["transitions_summary"] = transitions_summary
    causal["affected_vs_unaffected"] = {
        "replay_level": replay,
        "run_level": {"status": STATUS_PENDING,
                      "reason": transitions_summary.get("reason", "no paired intervention runs")},
    }
    return causal


def extend_failure_taxonomy(taxonomy: Dict[str, Any], report: LoadReport, auxd: AuxData) -> Dict[str, Any]:
    taxonomy["paper_experiment_id"] = ["E7"]
    taxonomy["implementation_stage"] = ["stage-E0", "stage-E5-export"]
    taxonomy["alignment_extension"] = ALIGNMENT_EXT
    taxonomy["derived_analysis"] = build_derived_analysis(report, auxd)
    return taxonomy

