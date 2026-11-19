#!/usr/bin/env python3
"""Score Core-track predictions against a mounted private oracle.

Usage (repository or release package root)::

    python -m benchmark.eval_core.run \
        --tasks ../aob-data/core_worlds/data/core/tasks/tasks.jsonl \
        --predictions runs/eval_core_x/predictions.jsonl \
        --oracle ../aob-private/oracle/core_oracle.jsonl \
        --out runs/eval_core_x

Exit codes: 0 success (including a low score), 2 CLI/format error,
3 task/oracle/prediction file missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import REPORT_SCHEMA
from .oracle import OracleError, load_oracle, load_predictions
from .scoring import aggregate, format_summary
from .tasks import (
    EXPECTED_TASK_COUNT,
    TaskValidationError,
    default_tasks_path,
    load_tasks,
    parse_csv,
    select_tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = default_tasks_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.eval_core.run",
        description="Score AutoOntologyBench Core-track answer predictions.",
    )
    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASKS),
        help="public Core tasks JSONL (default: %(default)s)",
    )
    parser.add_argument("--predictions", required=True, help="submission JSONL")
    parser.add_argument(
        "--oracle",
        required=True,
        help="private Core oracle JSONL or a directory containing core_oracle.jsonl",
    )
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument(
        "--no-count-check",
        action="store_true",
        help="allow a sampled/smoke task file instead of exactly 19,950 rows",
    )
    parser.add_argument("--worlds", default="", help="comma-separated world filter")
    parser.add_argument("--lanes", default="", help="comma-separated lane filter")
    parser.add_argument("--limit", type=int, default=None, help="score first N filtered tasks")
    return parser


def _out_dir(path: str | None) -> Path:
    if path:
        return Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / f"eval_core_{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = load_tasks(
            args.tasks,
            expected_count=None if args.no_count_check else EXPECTED_TASK_COUNT,
        )
    except TaskValidationError as exc:
        print(f"error: tasks: {exc}", file=sys.stderr)
        return 3
    try:
        oracle = load_oracle(args.oracle)
    except OracleError as exc:
        print(f"error: oracle: {exc}", file=sys.stderr)
        return 3
    try:
        predictions = load_predictions(args.predictions)
    except OracleError as exc:
        print(f"error: predictions: {exc}", file=sys.stderr)
        return 3

    selected = select_tasks(
        tasks.records,
        worlds=parse_csv(args.worlds),
        lanes=parse_csv(args.lanes),
        limit=args.limit,
    )
    report_metrics, rows = aggregate(selected, oracle, predictions)
    out = _out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks_file": str(Path(args.tasks)),
        "tasks_sha256": tasks.sha256,
        "predictions_file": str(Path(args.predictions)),
        "oracle_file": str(Path(args.oracle)),
        **report_metrics,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(format_summary(report))
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())