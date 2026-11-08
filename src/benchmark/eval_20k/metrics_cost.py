"""Offline CPS (cost per valid success) metrics for completed 20K v3 runs.

The module only reads ``report.json`` artifacts written by
:mod:`benchmark.eval_20k.run`.  It never calls a model, reruns inference, or
mutates a report.

Primary metric::

    cps_tokens = (tokens_input.total + tokens_output.total) / verified_success

Secondary diagnostics: ``api_calls_per_success`` and
``wall_seconds_per_success``.  A ratio is ``null`` (with an explicit reason)
when its denominator is zero, its denominator is missing, or a required cost
field is missing.  It is never silently reported as zero.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_GLOB = "eval_20k_full2-*/report.json"
PRIMARY_COST_KEYS = ("tokens_input", "tokens_output")
DIAGNOSTIC_COST_KEYS = ("api_calls", "wall_seconds")
COUNT_FIELDS = ("verified_success", "tasks", "scored", "success")
GROUP_RE = re.compile(r"^eval_20k_full2-(g\d+)-")

CPS_SCHEMA = "aob-eval-20k/cps/1"


@dataclass(frozen=True)
class RunRecord:
    """Normalized view of one completed report."""

    run_id: str
    group: str
    model: str
    report_path: str
    counts: Mapping[str, int | None]
    costs: Mapping[str, Mapping[str, float | int | None]]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    return None if number is None else int(number)


def _cost_block(raw: Any) -> dict[str, float | int | None]:
    if not isinstance(raw, Mapping):
        return {"total": None, "rows": None}
    return {"total": _as_number(raw.get("total")),
            "rows": _as_int(raw.get("rows"))}


def _ordered_cost_keys(raw: Mapping[str, Any]) -> list[str]:
    preferred = PRIMARY_COST_KEYS + DIAGNOSTIC_COST_KEYS
    keys = list(preferred)
    keys.extend(sorted(key for key in raw if key not in keys))
    return keys


def summarize_run(report: Mapping[str, Any], run_id: str,
                  report_path: str = "") -> RunRecord:
    """Normalize one report's overall counts and cost blocks."""
    predictions = report.get("predictions")
    model = predictions.get("model") if isinstance(predictions, Mapping) else None
    metrics = report.get("metrics")
    overall = metrics.get("overall") if isinstance(metrics, Mapping) else None
    overall = overall if isinstance(overall, Mapping) else {}
    raw_costs = overall.get("costs")
    raw_costs = raw_costs if isinstance(raw_costs, Mapping) else {}
    costs = {key: _cost_block(raw_costs.get(key))
             for key in _ordered_cost_keys(raw_costs)}
    match = GROUP_RE.match(run_id)
    return RunRecord(
        run_id=run_id,
        group=match.group(1) if match else "unknown",
        model=str(model) if model is not None else "unknown",
        report_path=report_path,
        counts={field: _as_int(overall.get(field)) for field in COUNT_FIELDS},
        costs=costs,
    )


def _ratio(numerator: float | None, verified_success: int | None,
           numerator_name: str) -> tuple[float | None, str | None]:
    if numerator is None:
        return None, f"missing cost total: {numerator_name}"
    if verified_success is None:
        return None, "missing verified_success"
    if verified_success == 0:
        return None, "verified_success is zero"
    return numerator / verified_success, None


def compute_cps(costs: Mapping[str, Mapping[str, Any]],
                verified_success: int | None) -> dict[str, Any]:
    """Return rounds of CPS plus denominator-preserving diagnostics."""
    def total(key: str) -> float | None:
        block = costs.get(key)
        return block.get("total") if isinstance(block, Mapping) else None

    reasons: dict[str, str] = {}
    token_in, token_out = total("tokens_input"), total("tokens_output")
    missing = [key for key, value in
               (("tokens_input", token_in), ("tokens_output", token_out))
               if value is None]
    if missing:
        cps_tokens = None
        reasons["cps_tokens"] = f"missing cost total(s): {', '.join(missing)}"
    else:
        cps_tokens, reason = _ratio(
            token_in + token_out, verified_success, "tokens_input+tokens_output")
        if reason:
            reasons["cps_tokens"] = reason

    api_calls, reason = _ratio(
        total("api_calls"), verified_success, "api_calls")
    if reason:
        reasons["api_calls_per_success"] = reason
    wall_seconds, reason = _ratio(
        total("wall_seconds"), verified_success, "wall_seconds")
    if reason:
        reasons["wall_seconds_per_success"] = reason

    return {
        "cps_tokens": cps_tokens,
        "cost_per_valid_success": cps_tokens,
        "cps_tokens_denominator": verified_success,
        "api_calls_per_success": api_calls,
        "api_calls_per_success_denominator": verified_success,
        "wall_seconds_per_success": wall_seconds,
        "wall_seconds_per_success_denominator": verified_success,
        "null_reasons": reasons,
    }


def _sum_count(runs: Sequence[RunRecord], field: str
               ) -> tuple[int | None, str | None]:
    values = [run.counts.get(field) for run in runs]
    missing = sum(value is None for value in values)
    if missing:
        return None, f"missing {field} in {missing} of {len(values)} run(s)"
    return sum(value for value in values if value is not None), None


def _sum_cost(runs: Sequence[RunRecord], key: str, part: str
              ) -> tuple[float | int | None, str | None]:
    values: list[float | int | None] = []
    for run in runs:
        block = run.costs.get(key)
        values.append(block.get(part) if isinstance(block, Mapping) else None)
    missing = sum(value is None for value in values)
    if missing:
        return (None,
                f"missing costs.{key}.{part} in {missing} of {len(values)} run(s)")
    return sum(value for value in values if value is not None), None


def _cost_keys(runs: Sequence[RunRecord]) -> list[str]:
    present = {key for run in runs for key in run.costs}
    preferred = PRIMARY_COST_KEYS + DIAGNOSTIC_COST_KEYS
    keys = list(preferred)
    keys.extend(sorted(present.difference(keys)))
    return keys


def aggregate_runs(runs: Sequence[RunRecord]) -> dict[str, Any]:
    """Aggregate one or more runs, preserving all denominators."""
    reasons: dict[str, str] = {}
    counts: dict[str, int | None] = {}
    for field in COUNT_FIELDS:
        counts[field], reason = _sum_count(runs, field)
        if reason:
            reasons[field] = reason

    costs: dict[str, dict[str, Any]] = {}
    for key in _cost_keys(runs):
        total, total_reason = _sum_cost(runs, key, "total")
        rows, rows_reason = _sum_cost(runs, key, "rows")
        costs[key] = {"total": total, "rows": rows}
        if total_reason:
            reasons[f"costs.{key}.total"] = total_reason
        if rows_reason:
            reasons[f"costs.{key}.rows"] = rows_reason

    ratios = compute_cps(costs, counts["verified_success"])
    reasons.update(ratios.pop("null_reasons"))
    bucket: dict[str, Any] = {
        "runs": len(runs),
        "run_ids": [run.run_id for run in runs],
        "groups": sorted({run.group for run in runs}),
        "models": sorted({run.model for run in runs}),
        **counts,
        "costs": costs,
        **ratios,
        "null_reasons": reasons,
    }
    return bucket


def build_report(root: str | Path) -> dict[str, Any]:
    """Build the CPS report from a run directory or legacy run root."""
    root = Path(root)
    direct = root / "report.json"
    paths = [direct] if direct.is_file() else sorted(root.glob(RUN_GLOB))
    records: list[RunRecord] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        records.append(summarize_run(report, path.parent.name, str(path)))

    cells = []
    for record in records:
        cell = aggregate_runs([record])
        cells.append({
            "run_id": record.run_id,
            "group": record.group,
            "model": record.model,
            "report_path": record.report_path,
            **cell,
        })

    groups: dict[str, Any] = {}
    for group in sorted({record.group for record in records}):
        group_runs = [record for record in records if record.group == group]
        by_model = {}
        for model in sorted({record.model for record in group_runs}):
            by_model[model] = aggregate_runs(
                [record for record in group_runs if record.model == model])
        groups[group] = {"models": by_model,
                         "total": aggregate_runs(group_runs)}

    models = {
        model: aggregate_runs([record for record in records
                               if record.model == model])
        for model in sorted({record.model for record in records})
    }

    return {
        "schema": CPS_SCHEMA,
        "source_root": str(root),
        "run_count": len(records),
        "definitions": {
            "verified_success": "metrics.overall.verified_success",
            "cps_tokens": ("(costs.tokens_input.total + "
                           "costs.tokens_output.total) / verified_success"),
            "cost_per_valid_success": "alias of cps_tokens",
            "api_calls_per_success": "costs.api_calls.total / verified_success",
            "wall_seconds_per_success": ("costs.wall_seconds.total / "
                                         "verified_success"),
            "null_policy": ("null with a null_reasons entry when the "
                            "denominator is zero/missing or a cost field is "
                            "missing; never zero"),
        },
        "cells": cells,
        "groups": groups,
        "models": models,
        "overall": aggregate_runs(records),
    }


