"""Offline per-regime metrics for the completed 20K v3 evaluation.

The module aggregates existing ``runs/eval_20k_full2-*/report.json`` artifacts
by the harness ``task_regime`` label.  It never calls a model, reruns
inference, invokes a verifier, or mutates a report.

Inputs per run
--------------

* ``report.json``: source paths/hashes and ``metrics.overall`` totals.  The
  overall block is the reconciliation oracle, never the per-regime source.
* ``tasks`` JSONL (``report.tasks.path``): ``task_id`` + ``task_regime``.
* ``predictions.jsonl`` (``report.predictions.path``): header row followed by
  prediction rows; per-task cost fields come from these rows.
* ``rows.jsonl`` (beside ``predictions.jsonl``): the harness' own per-task
  scoring rows.  They are the only artifact carrying ``status``,
  ``task_success``, and ``valid_task_success``; the report does not persist
  per-task outcomes anywhere else.  A missing ``rows.jsonl`` is a hard error,
  never a silent zero.

The four v3 regimes are exactly ``application``, ``construction``,
``organization``, and ``repair``.  Every task in every group must carry one of
those labels; a run row whose ``task_id`` cannot be joined to a task regime is
a hard error, never an ``unknown`` bucket.

``tsr`` here is the requested ``success / scored`` with ``tsr_denominator``
kept alongside it.  The harness report's overall ``tsr`` uses
``scored + missing`` as its denominator; the two are identical for these runs
because every task was scored.  ``coverage`` is ``scored / tasks``.

Usage::

    python -m benchmark.eval_20k.metrics_regime \
        --runs-glob 'runs/eval_20k_full2-*/report.json' \
        --tasks-dir runs/full_v3_models/tasks \
        --out reports/extra_metrics/regime_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .metrics_regime_aggregate import aggregate_regimes
from .metrics_regime_core import (
    COST_KEYS,
    COUNT_FIELDS,
    DEFAULT_OUT,
    DEFAULT_RUNS_GLOB,
    DEFAULT_TASKS_DIR,
    EXPECTED_GROUPS,
    REGIMES,
    SCHEMA,
    RegimeError,
    RegimeJoinError,
    RegimeValidationError,
    RunInputs,
    TaskRow,
)
from .metrics_regime_io import build_report, load_run_inputs

__all__ = [
    "COST_KEYS",
    "COUNT_FIELDS",
    "DEFAULT_OUT",
    "DEFAULT_RUNS_GLOB",
    "DEFAULT_TASKS_DIR",
    "EXPECTED_GROUPS",
    "REGIMES",
    "SCHEMA",
    "RegimeError",
    "RegimeJoinError",
    "RegimeValidationError",
    "RunInputs",
    "TaskRow",
    "aggregate_regimes",
    "build_report",
    "load_run_inputs",
    "main",
]


def _write_new_or_identical(path: Path, text: str) -> bool:
    """Write ``text`` unless ``path`` already holds different content.

    Returns ``True`` when a new file was written, ``False`` when the existing
    file is byte-identical.  A conflicting existing file is a hard error so a
    prior report is never silently overwritten.
    """
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RegimeValidationError(
                "refusing to overwrite existing output with different "
                "content: %s" % path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-glob", default=DEFAULT_RUNS_GLOB,
        help="glob for completed eval_20k_full2-*/report.json files")
    parser.add_argument(
        "--tasks-dir", default=DEFAULT_TASKS_DIR,
        help="fallback directory for tasks JSONL files")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="output JSON file (created; never overwritten with different data)")
    args = parser.parse_args(argv)

    try:
        report = build_report(args.runs_glob, tasks_dir=args.tasks_dir)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        written = _write_new_or_identical(Path(args.out), text)
    except (RegimeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print(("wrote %s" if written else "unchanged %s") % args.out)
    reconciliation = report["reconciliation"]
    print("reports=%d models=%d" % (
        reconciliation["report_count"], len(reconciliation["models"])))
    for model in reconciliation["models"]:
        entry = reconciliation["per_model"][model]
        print("%s: reports=%d tasks=%d scored=%d success=%d "
              "verified_success=%d groups=%s" % (
                  model, entry["report_count"], entry["tasks"],
                  entry["scored"], entry["success"],
                  entry["verified_success"], ",".join(entry["groups"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())