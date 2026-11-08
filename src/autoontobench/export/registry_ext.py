"""registry_ext: W17 extensions of evidence_registry + claim_provenance.

Adds the E0-E7 paper-experiment index to ``evidence_registry.json`` (no
parallel experiment-registry file is created — alignment doc §0 allows the
lightweight index inside the existing registry) and the new-interface claims
to ``claim_provenance.json``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from autoontobench.export.aggregate import ClassifiedRow, LoadReport
from autoontobench.export.alignment import ALIGNMENT_EXT, STATUS_MEASURED, STATUS_PENDING
from autoontobench.export.aux_inputs import AuxData

PAPER_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "E0": {"research_goal": "source authenticity, rule-to-task funnel, difficulty",
           "implementation_stage": ["stage-E0", "task-factory"],
           "interfaces": ["results/source_funnel.json", "results/task_manifest.jsonl",
                          "results/task_matrix.json"]},
    "E1": {"research_goal": "real task value of each representation arm",
           "implementation_stage": ["stage-E1", "stage-E3"],
           "interfaces": ["results/main_tsr.json"]},
    "E2": {"research_goal": "Repair / Organization / Construction three-regime competence",
           "implementation_stage": ["stage-E1", "stage-E3"],
           "interfaces": ["results/task_regime_metrics.json", "results/task_matrix.json"]},
    "E3": {"research_goal": "semantic corruption, repair, negative controls, task attribution",
           "implementation_stage": ["stage-E2"],
           "interfaces": ["results/causal_interventions.json", "results/task_transitions.jsonl"]},
    "E4": {"research_goal": "mapping / relation / constraint defect impact",
           "implementation_stage": ["stage-E2"],
           "interfaces": ["results/mapping_ablation.json", "results/semantics_vs_utility.json"]},
    "E5": {"research_goal": "active evidence / probe identification efficiency and budget",
           "implementation_stage": ["stage-E3"],
           "interfaces": ["results/active_budget.json", "results/active_probe_diagnostics.json"]},
    "E6": {"research_goal": "generalization, mechanism holdout, knowledge reuse and cost",
           "implementation_stage": ["stage-E4"],
           "interfaces": ["results/cost_reuse.json", "results/generalization.json"]},
    "E7": {"research_goal": "failure attribution, semantic-utility mismatch, instrument validity",
           "implementation_stage": ["stage-E0", "stage-E5-export"],
           "interfaces": ["results/failure_taxonomy.json", "results/semantics_vs_utility.json"]},
}


def _panel(rows: List[ClassifiedRow], filters: Dict[str, Any]) -> Dict[str, Any]:
    scored = [r for r in rows if r.row.status == "SCORED"]
    return {
        "run_ids": sorted(r.run_ref for r in rows),
        "task_ids": sorted({r.row.task_id for r in rows if r.row.task_id}),
        "base_goal_ids": sorted({r.row.base_goal_id for r in rows if r.row.base_goal_id}),
        "source_units": sorted({u for r in rows for u in (r.row.source_rule_units or [])}),
        "source_groups": sorted({r.row.independent_source_group for r in rows if r.row.independent_source_group}),
        "model_exact_ids": sorted({r.row.model_exact_id for r in rows if r.row.model_exact_id}),
        "oracle_shas": sorted({r.row.native_terminal_verifier_sha for r in scored if r.row.native_terminal_verifier_sha}),
        "code_sha": None,
        "code_sha_note": "repository intentionally uncommitted (docs/DECISIONS.md D-12); "
                         "code identity is the input_digest plus per-file sha256 entries",
        "prompt_version": None,
        "prompt_version_note": "no prompt-version field exists in RunRowV5; per-call prompt_sha "
                               "lives in runs/<scope>/<run>/model_log.jsonl",
        "filters": filters,
    }


def extend_evidence_registry(registry: Dict[str, Any], report: LoadReport, auxd: AuxData,
                             interface_status: Dict[str, str]) -> Dict[str, Any]:
    """Add the E0-E7 experiment index. No parallel registry file is created."""
    claimable = report.claimable_rows
    failed = [r for r in claimable
              if r.row.launched and r.row.status == "SCORED" and r.row.task_success is False]
    panels: Dict[str, List[Dict[str, Any]]] = {
        "E0": [{"panel": "qualified_task_supplies", "run_ids": [],
                "task_ids": sorted({r["task_id"] for s in auxd.supplies for r in s.rows if r["task_id"]}),
                "source_groups": sorted({r["source_group_id"] for s in auxd.supplies for r in s.rows if r["source_group_id"]}),
                "evidence_files": [f for f in auxd.files if f["role"].startswith("qualified")],
                "filters": {"qualification": "QUALIFIED"}}],
        "E1": [_panel(claimable, {"claimable": True})],
        "E2": [_panel(claimable, {"claimable": True, "regime": None}),
               {"panel": "organization_eligibility",
                "run_ids": [], "task_ids": sorted(
                    {r["task_id"] for s in auxd.supplies for r in s.rows
                     if r["task_family"] == "organization" and r["task_id"]}),
                "filters": {"task_family": "organization"}}],
        "E3": [_panel([r for r in claimable if (r.row.arm or "").lower() in
                       ("oracle_assisted", "gold", "corrupt", "repaired")],
                      {"arm": ["gold/oracle_assisted", "corrupt", "repaired"]}),
               {"panel": "replay_instrument",
                "evidence_files": [f for f in auxd.files if f["role"] == "intervention_evidence"],
                "run_ids": []}],
        "E4": [{"panel": "mapping_instrument",
                "evidence_files": [f for f in auxd.files
                                   if f["role"] in ("phi_demonstration", "gold_o_validation")],
                "run_ids": []}],
        "E5": [{"panel": "discovery_instrument",
                "evidence_files": [f for f in auxd.files if f["role"] == "discovery_certificate"],
                "run_ids": []}],
        "E6": [_panel([], {"reuse_points": [1, 2, 4, 8, 16]})],
        "E7": [_panel(failed, {"task_success": False, "claimable": True})],
    }
    experiments: Dict[str, Any] = {}
    for eid, meta in PAPER_EXPERIMENTS.items():
        experiments[eid] = {
            **meta,
            "status": interface_status.get(eid, STATUS_PENDING),
            "panels": panels.get(eid, []),
        }
    registry["alignment_extension"] = ALIGNMENT_EXT
    registry["experiments"] = experiments
    registry["numbering_note"] = ("paper_experiment_id (E0-E7, research questions) and "
                                  "implementation_stage (stage-E0..E5, engineering phases) are "
                                  "disjoint numbering spaces (alignment doc §2)")
    registry["aux_files"] = auxd.files
    registry["aux_digest"] = auxd.digest
    registry["skipped_files"] = report.skipped_files
    return registry


def extend_claim_provenance(provenance: Dict[str, Any], new_interfaces: Dict[str, Dict[str, Any]],
                            transitions_summary: Dict[str, Any]) -> Dict[str, Any]:
    claims = provenance["claims"]
    for name in sorted(new_interfaces):
        payload = new_interfaces[name]
        claims.append({
            "cell": name,
            "interface": f"results/{name}.json",
            "status": payload.get("status", STATUS_PENDING),
            "claimable": payload.get("status") == STATUS_MEASURED,
            "run_ids": [],
            "filters": {"claimable": True},
            "note": payload.get("pending_reason") or payload.get("blocked_reason"),
        })
    claims.append({
        "cell": "task_transitions",
        "interface": "results/task_transitions.jsonl",
        "status": transitions_summary.get("status", STATUS_PENDING),
        "claimable": transitions_summary.get("status") == STATUS_MEASURED,
        "run_ids": [],
        "filters": {"arms": ["gold", "corrupt", "repaired"], "paired": True},
        "note": transitions_summary.get("reason"),
        "rows_emitted": transitions_summary.get("rows_emitted", 0),
        "partial_pairs": transitions_summary.get("partial_pairs", []),
    })
    provenance["alignment_extension"] = ALIGNMENT_EXT
    return provenance
