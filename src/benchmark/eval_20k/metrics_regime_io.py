"""File loading, hashing, and path resolution for the regime reporter."""
from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .metrics_regime_aggregate import aggregate_regimes
from .metrics_regime_core import (
    DEFAULT_RUNS_GLOB,
    DEFAULT_TASKS_DIR,
    REPO_ROOT,
    RUN_DIR_RE,
    RegimeJoinError,
    RunInputs,
    TaskRow,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _load_tasks(path: Path) -> tuple[tuple[TaskRow, ...], str]:
    digest = hashlib.sha256()
    tasks: list[TaskRow] = []
    with path.open("rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            digest.update(raw)
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RegimeJoinError(
                    "%s line %d: task row is not an object" % (path, line_no))
            task_id = row.get("task_id")
            regime = row.get("task_regime")
            if not isinstance(task_id, str) or not task_id:
                raise RegimeJoinError(
                    "%s line %d: task_id must be a non-empty string"
                    % (path, line_no))
            if not isinstance(regime, str) or not regime:
                raise RegimeJoinError(
                    "%s line %d: task %r has no task_regime"
                    % (path, line_no, task_id))
            tasks.append(TaskRow(task_id=task_id, task_regime=regime))
    return tuple(tasks), digest.hexdigest()


def _load_predictions(path: Path) -> tuple[tuple[Mapping[str, Any], ...], str]:
    digest = hashlib.sha256()
    rows: list[Mapping[str, Any]] = []
    first_record = True
    with path.open("rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            digest.update(raw)
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RegimeJoinError(
                    "%s line %d: prediction row is not an object"
                    % (path, line_no))
            if first_record:
                first_record = False
                if "schema" in row and "task_id" not in row:
                    schema = row.get("schema")
                    if schema != "aob-submission-20k/1":
                        raise RegimeJoinError(
                            "%s line %d: unsupported prediction schema %r"
                            % (path, line_no, schema))
                    continue
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise RegimeJoinError(
                    "%s line %d: prediction task_id must be a non-empty "
                    "string" % (path, line_no))
            rows.append(row)
    return tuple(rows), digest.hexdigest()


def _load_scored_rows(path: Path) -> tuple[tuple[Mapping[str, Any], ...], str]:
    digest = hashlib.sha256()
    rows: list[Mapping[str, Any]] = []
    with path.open("rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            digest.update(raw)
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RegimeJoinError(
                    "%s line %d: scored row is not an object" % (path, line_no))
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise RegimeJoinError(
                    "%s line %d: scored task_id must be a non-empty string"
                    % (path, line_no))
            rows.append(row)
    return tuple(rows), digest.hexdigest()


def _resolve_tasks_path(raw: str, tasks_dir: Path, repo_root: Path) -> Path:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        repo_root / path,
        Path.cwd() / path,
        tasks_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "tasks file not found for %r (tried %s)"
        % (raw, ", ".join(str(candidate) for candidate in candidates)))


def _resolve_predictions_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        repo_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "predictions file not found for %r (tried %s)"
        % (raw, ", ".join(str(candidate) for candidate in candidates)))


def load_run_inputs(report_path: str | Path, *,
                    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
                    repo_root: str | Path = REPO_ROOT,
                    task_cache: dict[str, tuple[tuple[TaskRow, ...], str]]
                    | None = None) -> RunInputs:
    """Load one completed run into a :class:`RunInputs` value.

    ``rows.jsonl`` lives beside ``predictions.jsonl`` and is required because
    it is the only per-task source of ``task_success`` / ``valid_task_success``.
    """
    root = Path(repo_root)
    report_file = Path(report_path)
    if not report_file.is_file():
        raise FileNotFoundError("report not found: %s" % report_file)
    report_file = report_file.resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise RegimeJoinError("%s: report is not an object" % report_file)

    run_id = report_file.parent.name
    match = RUN_DIR_RE.match(run_id)
    if match:
        group = match.group("group")
    else:
        declared_group = report.get("group")
        group = (declared_group if isinstance(declared_group, str)
                 and declared_group else "local")

    predictions_block = report.get("predictions")
    tasks_block = report.get("tasks")
    metrics = report.get("metrics")
    if not isinstance(predictions_block, Mapping):
        raise RegimeJoinError("%s: missing predictions block" % report_file)
    if not isinstance(tasks_block, Mapping):
        raise RegimeJoinError("%s: missing tasks block" % report_file)
    if not isinstance(metrics, Mapping) or not isinstance(
            metrics.get("overall"), Mapping):
        raise RegimeJoinError(
            "%s: missing metrics.overall block" % report_file)
    model = predictions_block.get("model")
    if not isinstance(model, str) or not model:
        raise RegimeJoinError("%s: predictions.model is missing" % report_file)

    tasks_path = _resolve_tasks_path(
        str(tasks_block.get("path")), Path(tasks_dir), root)
    cache = task_cache if task_cache is not None else {}
    cache_key = str(tasks_path)
    if cache_key in cache:
        task_rows, tasks_sha = cache[cache_key]
    else:
        task_rows, tasks_sha = _load_tasks(tasks_path)
        cache[cache_key] = (task_rows, tasks_sha)
    declared_tasks_sha = tasks_block.get("sha256")
    if declared_tasks_sha and declared_tasks_sha != tasks_sha:
        raise RegimeJoinError(
            "%s: tasks sha256 mismatch: report=%s actual=%s"
            % (run_id, declared_tasks_sha, tasks_sha))

    predictions_path = _resolve_predictions_path(
        str(predictions_block.get("path")), root)
    prediction_rows, predictions_sha = _load_predictions(predictions_path)
    declared_predictions_sha = predictions_block.get("sha256")
    if declared_predictions_sha and declared_predictions_sha != predictions_sha:
        raise RegimeJoinError(
            "%s: predictions sha256 mismatch: report=%s actual=%s"
            % (run_id, declared_predictions_sha, predictions_sha))

    rows_path = predictions_path.with_name("rows.jsonl")
    if not rows_path.is_file():
        raise FileNotFoundError(
            "%s: scored rows file required for per-regime success: %s"
            % (run_id, rows_path))
    scored_rows, rows_sha = _load_scored_rows(rows_path)

    return RunInputs(
        run_id=run_id,
        group=group,
        model=model,
        report_path=_display_path(report_file, root),
        report_sha256=_sha256_file(report_file),
        report_overall=metrics["overall"],
        tasks_path=_display_path(tasks_path, root),
        tasks_sha256=tasks_sha,
        task_rows=task_rows,
        predictions_path=_display_path(predictions_path, root),
        predictions_sha256=predictions_sha,
        prediction_rows=prediction_rows,
        rows_path=_display_path(rows_path, root),
        rows_sha256=rows_sha,
        scored_rows=scored_rows,
    )


def build_report(runs_glob: str | Path = DEFAULT_RUNS_GLOB, *,
                 tasks_dir: str | Path = DEFAULT_TASKS_DIR,
                 repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Load every matching report and return the regime breakdown."""
    pattern = str(runs_glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no report.json matched %r" % pattern)
    cache: dict[str, tuple[tuple[TaskRow, ...], str]] = {}
    runs = [
        load_run_inputs(path, tasks_dir=tasks_dir, repo_root=repo_root,
                        task_cache=cache)
        for path in paths
    ]
    return aggregate_regimes(runs)