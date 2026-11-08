"""Bucket arithmetic and report-overall reconciliation for regime metrics."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics_regime_core import (
    COUNT_FIELDS,
    COST_KEYS,
    STATUS_ERROR,
    STATUS_MISSING,
    STATUS_NOT_SUPPORTED,
    STATUS_ORACLE_MISSING,
    STATUS_SCORED,
    RegimeJoinError,
    RegimeValidationError,
    RunInputs,
)


def _new_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {field: 0 for field in COUNT_FIELDS}
    bucket["costs"] = {
        key: {"total": 0.0, "rows": 0} for key in COST_KEYS
    }
    return bucket


def _merge_bucket(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for field in COUNT_FIELDS:
        target[field] += int(source[field])
    for key in COST_KEYS:
        block = source["costs"][key]
        target["costs"][key]["total"] += float(block["total"])
        target["costs"][key]["rows"] += int(block["rows"])


def _number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegimeJoinError("%s must be a number, got %r" % (where, value))
    return float(value)


def _integer(value: Any, *, where: str) -> int:
    number = _number(value, where=where)
    if number != int(number):
        raise RegimeJoinError("%s must be an integer, got %r" % (where, value))
    return int(number)


def _record_task(bucket: dict[str, Any], row: Mapping[str, Any],
                 prediction: Mapping[str, Any] | None) -> None:
    status = row.get("status")
    if not isinstance(status, str):
        raise RegimeJoinError(
            "scored row %r has no string status" % (row.get("task_id"),))

    bucket["tasks"] += 1
    if status == STATUS_SCORED:
        bucket["scored"] += 1
        if bool(row.get("task_success")):
            bucket["success"] += 1
        if bool(row.get("valid_task_success")):
            bucket["verified_success"] += 1
    elif status == STATUS_MISSING:
        bucket["missing"] += 1
    elif status == STATUS_NOT_SUPPORTED:
        bucket["not_supported"] += 1
    elif status == STATUS_ORACLE_MISSING:
        bucket["oracle_missing"] += 1
    elif status == STATUS_ERROR:
        bucket["error"] += 1
    else:
        raise RegimeJoinError(
            "unexpected scored-row status %r for task %r"
            % (status, row.get("task_id")))

    if prediction is None:
        return
    for key in COST_KEYS:
        if key not in prediction:
            continue
        bucket["costs"][key]["total"] += _number(
            prediction[key], where="prediction[%r][%r]" % (
                prediction.get("task_id"), key))
        bucket["costs"][key]["rows"] += 1


def _finalize_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    scored = int(bucket["scored"])
    tasks = int(bucket["tasks"])
    success = int(bucket["success"])
    return {
        "tasks": tasks,
        "scored": scored,
        "success": success,
        "verified_success": int(bucket["verified_success"]),
        "error": int(bucket["error"]),
        "missing": int(bucket["missing"]),
        "not_supported": int(bucket["not_supported"]),
        "oracle_missing": int(bucket["oracle_missing"]),
        "tsr": (success / scored) if scored else None,
        "tsr_denominator": scored,
        "coverage": (scored / tasks) if tasks else None,
        "coverage_denominator": tasks,
        "costs": {
            key: {
                "total": float(bucket["costs"][key]["total"]),
                "rows": int(bucket["costs"][key]["rows"]),
            }
            for key in COST_KEYS
        },
    }


def _expected_bucket(runs: Sequence[RunInputs]) -> dict[str, Any]:
    expected = _new_bucket()
    for run in runs:
        overall = run.report_overall
        for field in COUNT_FIELDS:
            if field not in overall:
                raise RegimeValidationError(
                    "%s: report.metrics.overall is missing %r"
                    % (run.run_id, field))
            expected[field] += _integer(
                overall[field],
                where="%s: report.metrics.overall.%s" % (run.run_id, field))
        costs = overall.get("costs")
        if not isinstance(costs, Mapping):
            raise RegimeValidationError(
                "%s: report.metrics.overall.costs is not an object"
                % (run.run_id,))
        for key in COST_KEYS:
            block = costs.get(key)
            if not isinstance(block, Mapping):
                raise RegimeValidationError(
                    "%s: report.metrics.overall.costs.%s is not an object"
                    % (run.run_id, key))
            expected["costs"][key]["total"] += _number(
                block.get("total"),
                where="%s: report.metrics.overall.costs.%s.total"
                      % (run.run_id, key))
            expected["costs"][key]["rows"] += _integer(
                block.get("rows"),
                where="%s: report.metrics.overall.costs.%s.rows"
                      % (run.run_id, key))
    return expected


def _validate_report_totals(model: str, actual: Mapping[str, Any],
                            expected: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for field in COUNT_FIELDS:
        if int(actual[field]) != int(expected[field]):
            failures.append("%s: actual=%d expected=%d" % (
                field, int(actual[field]), int(expected[field])))
    for key in COST_KEYS:
        a_block, e_block = actual["costs"][key], expected["costs"][key]
        if not math.isclose(float(a_block["total"]), float(e_block["total"]),
                            rel_tol=1e-12, abs_tol=1e-6):
            failures.append("costs.%s.total: actual=%.12f expected=%.12f"
                            % (key, a_block["total"], e_block["total"]))
        if int(a_block["rows"]) != int(e_block["rows"]):
            failures.append("costs.%s.rows: actual=%d expected=%d"
                            % (key, a_block["rows"], e_block["rows"]))
    if failures:
        raise RegimeValidationError(
            "model %r does not reconcile with report.metrics.overall: %s"
            % (model, "; ".join(failures)))
    return {"actual": _finalize_bucket(actual),
            "expected": _finalize_bucket(expected),
            "matches_report_overall": True}