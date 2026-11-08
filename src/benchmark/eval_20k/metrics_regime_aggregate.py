"""Pure per-model/per-``task_regime`` aggregation for the 20K reporter."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics_regime_buckets import (
    _expected_bucket,
    _finalize_bucket,
    _merge_bucket,
    _new_bucket,
    _record_task,
    _validate_report_totals,
)
from .metrics_regime_core import (
    COUNT_FIELDS,
    EXPECTED_GROUPS,
    REGIMES,
    SCHEMA,
    RegimeError,
    RegimeJoinError,
    RegimeValidationError,
    RunInputs,
)


def _join_run(run: RunInputs) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return ``(run_overall, per_regime)`` for one run."""
    task_regimes: dict[str, str] = {}
    for task in run.task_rows:
        if not task.task_id:
            raise RegimeJoinError(
                "%s: task row has an empty task_id (%s)"
                % (run.run_id, run.tasks_path))
        if task.task_id in task_regimes:
            raise RegimeJoinError(
                "%s: duplicate task_id %r in %s"
                % (run.run_id, task.task_id, run.tasks_path))
        if task.task_regime not in REGIMES:
            raise RegimeJoinError(
                "%s: task %r has unsupported task_regime %r; expected one of %s"
                % (run.run_id, task.task_id, task.task_regime,
                   ", ".join(REGIMES)))
        task_regimes[task.task_id] = task.task_regime

    predictions: dict[str, Mapping[str, Any]] = {}
    for prediction in run.prediction_rows:
        task_id = prediction.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RegimeJoinError(
                "%s: prediction row without a string task_id (%s)"
                % (run.run_id, run.predictions_path))
        if task_id not in task_regimes:
            raise RegimeJoinError(
                "%s: prediction task_id %r cannot be joined to a task_regime "
                "in %s" % (run.run_id, task_id, run.tasks_path))
        if task_id in predictions:
            raise RegimeJoinError(
                "%s: duplicate prediction task_id %r in %s"
                % (run.run_id, task_id, run.predictions_path))
        predictions[task_id] = prediction

    scored: dict[str, Mapping[str, Any]] = {}
    for row in run.scored_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RegimeJoinError(
                "%s: scored row without a string task_id (%s)"
                % (run.run_id, run.rows_path))
        if task_id not in task_regimes:
            raise RegimeJoinError(
                "%s: scored task_id %r cannot be joined to a task_regime "
                "in %s" % (run.run_id, task_id, run.tasks_path))
        if task_id in scored:
            raise RegimeJoinError(
                "%s: duplicate scored task_id %r in %s"
                % (run.run_id, task_id, run.rows_path))
        scored[task_id] = row

    missing_rows = sorted(set(task_regimes) - set(scored))
    if missing_rows:
        preview = ", ".join(repr(task_id) for task_id in missing_rows[:5])
        if len(missing_rows) > 5:
            preview += ", ..."
        raise RegimeJoinError(
            "%s: %d task(s) have no scored row in %s (%s)"
            % (run.run_id, len(missing_rows), run.rows_path, preview))

    run_overall = _new_bucket()
    per_regime = {regime: _new_bucket() for regime in REGIMES}
    for task in run.task_rows:
        row = scored[task.task_id]
        prediction = predictions.get(task.task_id)
        _record_task(run_overall, row, prediction)
        _record_task(per_regime[task.task_regime], row, prediction)
    return run_overall, per_regime


def aggregate_regimes(runs: Sequence[RunInputs]) -> dict[str, Any]:
    """Aggregate loaded runs by model and ``task_regime``.

    The function is pure: it performs no file I/O and never mutates its
    inputs.  Every model gets all four regime cells, even when a cell has zero
    tasks, so the key set is stable.  Counts are preserved exactly and the
    per-model overall block is reconciled against the sum of the report
    ``metrics.overall`` blocks; a mismatch raises :class:`RegimeValidationError`.
    """
    run_list = sorted(runs, key=lambda run: run.run_id)
    if not run_list:
        raise RegimeError("no runs to aggregate")

    model_runs: dict[str, list[RunInputs]] = {}
    model_overall: dict[str, dict[str, Any]] = {}
    model_regimes: dict[str, dict[str, dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []

    for run in run_list:
        model_runs.setdefault(run.model, []).append(run)
        overall = model_overall.setdefault(run.model, _new_bucket())
        regimes = model_regimes.setdefault(
            run.model, {regime: _new_bucket() for regime in REGIMES})
        run_overall, run_regimes = _join_run(run)
        _merge_bucket(overall, run_overall)
        for regime in REGIMES:
            _merge_bucket(regimes[regime], run_regimes[regime])
        sources.append({
            "run_id": run.run_id,
            "group": run.group,
            "model": run.model,
            "report_path": run.report_path,
            "report_sha256": run.report_sha256,
            "tasks_path": run.tasks_path,
            "tasks_sha256": run.tasks_sha256,
            "task_rows": len(run.task_rows),
            "predictions_path": run.predictions_path,
            "predictions_sha256": run.predictions_sha256,
            "prediction_rows": len(run.prediction_rows),
            "rows_path": run.rows_path,
            "rows_sha256": run.rows_sha256,
            "scored_rows": len(run.scored_rows),
        })

    models: dict[str, Any] = {}
    reconciliation_models: dict[str, Any] = {}
    for model in sorted(model_runs):
        runs = model_runs[model]
        actual = model_overall[model]
        expected = _expected_bucket(runs)

        regimes_sum = _new_bucket()
        for regime in REGIMES:
            _merge_bucket(regimes_sum, model_regimes[model][regime])
        for field in COUNT_FIELDS:
            if int(regimes_sum[field]) != int(actual[field]):
                raise RegimeValidationError(
                    "model %r: regime %s sum to %d but overall has %d"
                    % (model, field, int(regimes_sum[field]),
                       int(actual[field])))

        _validate_report_totals(model, actual, expected)
        groups = sorted({run.group for run in runs})
        observed = [regime for regime in REGIMES
                    if int(model_regimes[model][regime]["tasks"]) > 0]
        models[model] = {
            "run_ids": [run.run_id for run in runs],
            "groups": groups,
            "observed_regimes": observed,
            "missing_regimes": [regime for regime in REGIMES
                                if regime not in observed],
            "overall": _finalize_bucket(actual),
            "regimes": {regime: _finalize_bucket(model_regimes[model][regime])
                        for regime in REGIMES},
            "report_expected": _finalize_bucket(expected),
            "matches_report_overall": True,
        }
        reconciliation_models[model] = {
            "report_count": len(runs),
            "groups": groups,
            "tasks": int(actual["tasks"]),
            "scored": int(actual["scored"]),
            "success": int(actual["success"]),
            "verified_success": int(actual["verified_success"]),
            "expected_tasks": int(expected["tasks"]),
            "expected_scored": int(expected["scored"]),
            "expected_success": int(expected["success"]),
            "expected_verified_success": int(expected["verified_success"]),
            "matches_report_overall": True,
        }

    all_four = all(
        set(entry["groups"]) >= set(EXPECTED_GROUPS)
        for entry in reconciliation_models.values())
    all_regimes = all(
        not entry["missing_regimes"] for entry in models.values())
    total_tasks = sum(int(entry["tasks"])
                      for entry in reconciliation_models.values())
    total_success = sum(int(entry["success"])
                        for entry in reconciliation_models.values())
    total_verified = sum(int(entry["verified_success"])
                         for entry in reconciliation_models.values())

    return {
        "schema": SCHEMA,
        "definitions": {
            "task_regime": (
                "task_regime from the run's tasks JSONL; exactly "
                + ", ".join(REGIMES)),
            "tsr": (
                "success / scored; tsr_denominator is scored. The harness "
                "report's overall tsr uses scored + missing; they are equal "
                "when missing = 0, as in these runs."),
            "coverage": "scored / tasks; coverage_denominator is tasks.",
            "success": "rows whose task_success is true",
            "verified_success": "rows whose valid_task_success is true",
            "costs": (
                "sum over joined prediction rows that carry the field; "
                "costs.<key>.rows is the number of such rows"),
            "reconciliation": (
                "per-model counts, cost totals and cost row counts must equal "
                "the sum of report.metrics.overall across that model's runs"),
        },
        "regimes": list(REGIMES),
        "run_count": len(run_list),
        "models": models,
        "reconciliation": {
            "ok": True,
            "report_count": len(run_list),
            "models": sorted(models),
            "all_models_have_four_groups": all_four,
            "all_models_have_four_regimes": all_regimes,
            "counts_match_report_overall": True,
            "costs_match_report_overall": True,
            "tasks_total_across_models": total_tasks,
            "success_total_across_models": total_success,
            "verified_success_total_across_models": total_verified,
            "per_model": reconciliation_models,
        },
        "sources": sources,
    }