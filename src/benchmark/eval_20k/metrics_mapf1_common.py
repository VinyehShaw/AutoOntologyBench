"""Common-family MapF1 aggregation over the existing MapF1 report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

COMMON_FAMILIES = (
    "L05/l05.org_align_group_molecule_sets",
    "L06/l06.org_align_gene_aspect_annotations",
    "L07/l07-f03-trajectory-shape-organization",
    "L07/l07-f09-species-role-organization",
    "L12/l12-f04-model-transition-relabel",
    "L12/l12-f16-dual-transition-relabel",
    "L16/l16.con_ec_rhea_catalyst_map",
)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_common7(per_family: Mapping[str, Mapping[str, Any]],
                    *, allow_missing: bool = False) -> dict[str, Any]:
    missing = [family for family in COMMON_FAMILIES
               if family not in per_family]
    if missing:
        if not allow_missing:
            raise KeyError("common family missing from MapF1 report: %s"
                           % missing[0])
        return {
            "status": "NOT_APPLICABLE",
            "family_count": len(COMMON_FAMILIES),
            "missing_families": missing,
            "families": {},
            "tasks": None,
            "scored_tasks": None,
            "coverage": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "micro_f1": None,
            "macro_f1": None,
        }
    tp = fp = fn = scored = tasks = 0
    family_f1: dict[str, float | None] = {}
    family_rows: dict[str, dict[str, Any]] = {}
    for family in COMMON_FAMILIES:
        row = per_family[family]
        row_tp = int(row["tp"])
        row_fp = int(row["fp"])
        row_fn = int(row["fn"])
        row_tasks = int(row["family_tasks"])
        row_scored = int(row["scored_tasks"])
        row_denom = 2 * row_tp + row_fp + row_fn
        row_f1 = _ratio(2 * row_tp, row_denom)
        tp += row_tp
        fp += row_fp
        fn += row_fn
        scored += row_scored
        tasks += row_tasks
        family_f1[family] = row_f1
        family_rows[family] = {
            "family_tasks": row_tasks,
            "scored_tasks": row_scored,
            "coverage": _ratio(row_scored, row_tasks),
            "tp": row_tp,
            "fp": row_fp,
            "fn": row_fn,
            "micro_f1": row_f1,
        }
    micro_denom = 2 * tp + fp + fn
    known = [value for value in family_f1.values() if value is not None]
    return {
        "status": "OK",
        "families": family_rows,
        "family_count": len(COMMON_FAMILIES),
        "tasks": tasks,
        "scored_tasks": scored,
        "coverage": _ratio(scored, tasks),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": _ratio(2 * tp, micro_denom),
        "macro_f1": sum(known) / len(known) if known else None,
    }


def build_report(mapf1: Mapping[str, Any], *,
                 allow_missing: bool = False) -> dict[str, Any]:
    models = {}
    for model, block in mapf1["models"].items():
        common = compute_common7(block["per_family"],
                                 allow_missing=allow_missing)
        common["original_9_family_macro_f1"] = block["overall"]["macro_f1"]
        common["original_9_family_count"] = block["overall"]["scored_families"]
        models[model] = common
    return {
        "schema": "aob-mapf1-common7/1.0",
        "source": "reports/extra_metrics/mapf1.json",
        "common_families": list(COMMON_FAMILIES),
        "definitions": {
            "micro_f1": "2*TP / (2*TP + FP + FN) over the same 7 families",
            "macro_f1": "unweighted mean of the 7 per-family micro-F1 values",
            "coverage": "scored_tasks / family_tasks over the same 7 families",
            "excluded": "unparseable or non-mapping predictions are excluded from TP/FP/FN and reflected in coverage",
        },
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute common-7 MapF1")
    parser.add_argument("--mapf1", default="reports/extra_metrics/mapf1.json")
    parser.add_argument("--out", default="reports/extra_metrics/mapf1_common7.json")
    parser.add_argument("--allow-missing", action="store_true",
                        help="emit NOT_APPLICABLE for a run missing common families")
    args = parser.parse_args(argv)
    source = Path(args.mapf1)
    report = build_report(json.loads(source.read_text()),
                          allow_missing=args.allow_missing)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())