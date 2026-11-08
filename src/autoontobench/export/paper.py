"""paper: P8 export driver — emit the results interfaces from run rows.

Contract: Shared Contract §F (fixed spelling of the 13 legacy interfaces) + §G
(PENDING cells, not guessed numbers), Implementation §6 E5 + §7 P8 + §11.7, and
the W17 Codex-alignment increment (alignment doc §4: six derived views —
task_matrix, task_regime_metrics, task_transitions, mapping_ablation,
active_probe_diagnostics, generalization — plus append-only extensions of
main_tsr / causal_interventions / failure_taxonomy / evidence_registry /
claim_provenance).

Interface builders live in ``interfaces.py`` (legacy), ``task_matrix.py`` /
``experiment_views.py`` / ``alignment.py`` / ``failure_analysis.py`` (W17),
LaTeX rendering in ``latex.py``, placeholder figures in ``figures.py``, input
validation in ``aggregate.py``, auxiliary evidence loading in ``aux_inputs.py``.

Output is deterministic: sorted keys/order everywhere, no timestamps, so
regeneration over the same input is byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from autoontobench.export import (
    alignment,
    aux_inputs,
    experiment_views,
    figures,
    interfaces,
    latex,
    registry_ext,
    task_matrix,
)
from autoontobench.export.aggregate import StrictExportError, load_runs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def export_all(
    input_dir: Path,
    output_dir: Path,
    strict: bool = True,
    aux: Optional[aux_inputs.ExportAux] = None,
) -> Dict[str, Any]:
    """Run the full export; returns a summary dict. Raises StrictExportError.

    ``aux=None`` keeps the call hermetic (unit tests): aux-derived views emit
    PENDING with a named reason. The CLI passes ``default_aux(REPO_ROOT)``.
    """
    report = load_runs(input_dir, strict=strict)
    auxd = aux_inputs.load_aux(aux) if aux is not None else aux_inputs.AuxData()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_tsr = interfaces.build_main_tsr(report)
    causal = interfaces.build_causal_interventions(report)
    failure_taxonomy = interfaces.build_failure_taxonomy(report)
    evidence_registry = interfaces.build_evidence_registry(report)
    claim_provenance = interfaces.build_claim_provenance(report, main_tsr, causal)

    transitions_text, transitions_summary = experiment_views.build_task_transitions(report)
    new_views: Dict[str, Dict[str, Any]] = {
        "task_matrix": task_matrix.build_task_matrix(report, auxd),
        "task_regime_metrics": task_matrix.build_task_regime_metrics(report, auxd),
        "mapping_ablation": experiment_views.build_mapping_ablation(report, auxd),
        "active_probe_diagnostics": experiment_views.build_active_probe_diagnostics(report, auxd),
        "generalization": experiment_views.build_generalization(report, auxd),
    }

    alignment.extend_main_tsr(main_tsr, report)
    alignment.extend_causal_interventions(causal, auxd, transitions_summary)
    alignment.extend_failure_taxonomy(failure_taxonomy, report, auxd)
    interface_status = {
        "E0": new_views["task_matrix"]["status"],
        "E1": main_tsr["status"],
        "E2": _e2_status(new_views["task_regime_metrics"]),
        "E3": transitions_summary["status"] if transitions_summary["rows_emitted"] else _e3_status(auxd),
        "E4": new_views["mapping_ablation"]["status"],
        "E5": new_views["active_probe_diagnostics"]["status"],
        "E6": new_views["generalization"]["status"],
        "E7": failure_taxonomy.get("derived_analysis", {}).get("status", "PENDING"),
    }
    registry_ext.extend_evidence_registry(evidence_registry, report, auxd, interface_status)
    registry_ext.extend_claim_provenance(claim_provenance, new_views, transitions_summary)

    _write_json(output_dir / "evidence_registry.json", evidence_registry)
    (output_dir / "task_manifest.jsonl").write_text(
        interfaces.build_task_manifest(report), encoding="utf-8"
    )
    _write_json(output_dir / "source_funnel.json", interfaces.build_source_funnel(report))
    _write_json(output_dir / "main_tsr.json", main_tsr)
    _write_json(output_dir / "causal_interventions.json", causal)
    _write_json(
        output_dir / "semantics_vs_utility.json",
        interfaces.pending_interface("semantics_vs_utility", report, {"points": []}),
    )
    _write_json(
        output_dir / "active_budget.json",
        interfaces.pending_interface(
            "active_budget", report, {"budgets": list(interfaces.PROBE_BUDGETS), "series": []}
        ),
    )
    _write_json(
        output_dir / "cost_reuse.json",
        interfaces.pending_interface("cost_reuse", report, {"reuse_counts": []}),
    )
    _write_json(output_dir / "failure_taxonomy.json", failure_taxonomy)
    _write_json(output_dir / "claim_provenance.json", claim_provenance)
    for name, payload in new_views.items():
        _write_json(output_dir / f"{name}.json", payload)
    (output_dir / "task_transitions.jsonl").write_text(transitions_text, encoding="utf-8")

    (output_dir / "paper_macros.tex").write_text(
        latex.render_macros(main_tsr, len(report.claimable_rows), len(report.non_claimable_rows)),
        encoding="utf-8",
    )
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "main_results.tex").write_text(
        latex.render_main_results_table(main_tsr), encoding="utf-8"
    )
    (tables_dir / "source_provenance_yield.tex").write_text(
        latex.render_source_yield_table(), encoding="utf-8"
    )
    figure_paths = figures.write_figures(
        output_dir, note=f"input_digest {report.input_digest[:16]}; no claimable measured data"
    )

    return {
        "input_digest": report.input_digest,
        "aux_digest": auxd.digest,
        "rows": len(report.rows),
        "claimable_rows": len(report.claimable_rows),
        "non_claimable_rows": len(report.non_claimable_rows),
        "status": main_tsr["status"],
        "interface_status": interface_status,
        "new_views": {k: v["status"] for k, v in new_views.items()},
        "task_transitions": transitions_summary["status"],
        "skipped_files": len(report.skipped_files),
        "figures": figure_paths,
        "warnings": sorted(report.warnings),
    }


def _e2_status(regime_metrics: Dict[str, Any]) -> str:
    """E2 requires agent runs in a declared paper regime (Repair/Org/Construction)."""
    declared = [r for r in regime_metrics.get("rows", []) if r["regime_declared"]]
    if any(r["status"] == "MEASURED" for r in declared):
        return "MEASURED"
    return "PENDING"


def _e3_status(auxd: aux_inputs.AuxData) -> str:
    """E3 without run-level transitions: replay-level instrument evidence exists."""
    return "INSTRUMENT_ONLY" if auxd.interventions else "PENDING"


def run_cli(
    input_dir: Path,
    output_dir: Path,
    strict: bool = True,
    aux: Optional[aux_inputs.ExportAux] = None,
) -> int:
    """Shared CLI entry (scripts/export_paper_results.py + python -m autoontobench)."""
    if aux is None:
        aux = aux_inputs.default_aux(REPO_ROOT)
    try:
        summary = export_all(input_dir, output_dir, strict=strict, aux=aux)
    except StrictExportError as exc:
        sys.stderr.write(f"[export-paper] STRICT FAILURE: {exc}\n")
        return 1
    print(
        "[export-paper] status={status} rows={rows} claimable={claimable_rows} "
        "non_claimable={non_claimable_rows} digest={input_digest}".format(**summary)
    )
    return 0
