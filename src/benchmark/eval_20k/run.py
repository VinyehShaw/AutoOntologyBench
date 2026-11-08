"""CLI entry point for the 20K load-and-evaluate harness.

Usage (repo root)::

    python -m benchmark.eval_20k.run \
        --tasks release_20k/tasks/qualified_tasks_test.jsonl \
        --predictions submissions/team.jsonl \
        --oracle /mount/aob-20000-private-oracle \
        --out runs/eval_001

Fallback when the public file is not staged yet (derives the public test file
from a private build with the release packager's own projection; the derived
file is written to --write-derived or <out>/derived/, never into
release_20k/tasks)::

    python -m benchmark.eval_20k.run \
        --derive-from-build bank/builds/aob-20000-candidate-dev1 \
        --predictions submissions/team.jsonl \
        --out runs/eval_001

Lane support probe (no scoring)::

    python -m benchmark.eval_20k.run --probe-lanes

Exit codes: 0 success; 2 CLI/validation error; 3 required runtime missing;
1 any other failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import REPORT_SCHEMA, __version__
from .lanes import LaneAdapter, Verdict, get_adapter, probe_lanes
from .oracle import OracleStore, default_oracle_roots
from .scoring import (
    STATUS_NOT_SUPPORTED,
    aggregate,
    missing_result,
    result_from_verdict,
    unknown_task_result,
    write_rows,
)
from .submission import (
    Prediction,
    SubmissionSet,
    SubmissionValidationError,
    load_submission,
)
from .tasks import (
    EXPECTED_TASK_COUNT,
    TaskSet,
    TaskValidationError,
    build_fallback_tasks,
    load_tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_REL = Path("release_20k") / "tasks" / "qualified_tasks_test.jsonl"


def _parse_lane_kw(pairs: list[str]) -> dict[str, dict[str, str]]:
    lane_kw: dict[str, dict[str, str]] = {}
    for pair in pairs or []:
        if "=" not in pair or "." not in pair.split("=", 1)[0]:
            raise argparse.ArgumentTypeError(
                "--lane-kw expects LANE.key=value, got %r" % pair)
        lhs, value = pair.split("=", 1)
        lane, key = lhs.split(".", 1)
        lane = lane.upper()
        lane_kw.setdefault(lane, {})[key] = value
    return lane_kw


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m benchmark.eval_20k.run",
        description="Load-and-evaluate harness for the AutoOntologyBench 20K "
                    "test release. Scores a submission JSONL against the "
                    "mounted private oracle via the per-lane native verifiers.",
    )
    ap.add_argument("--tasks", default=None,
                    help="public tasks JSONL (default: %s when present)"
                         % DEFAULT_TASKS_REL)
    ap.add_argument("--predictions", default=None,
                    help="submission JSONL (one row per task_id)")
    ap.add_argument("--oracle", action="append", default=[],
                    help="mounted private oracle root (repeatable; env "
                         "AOB_20K_ORACLE also accepted, os.pathsep-separated)")
    ap.add_argument("--out", default=None,
                    help="output directory for report.json + rows.jsonl "
                         "(default: stdout only)")
    ap.add_argument("--expect-count", type=int, default=EXPECTED_TASK_COUNT,
                    help="expected task count (default %(default)s)")
    ap.add_argument("--no-count-check", action="store_true",
                    help="disable the exact task-count check (fixtures/smoke)")
    ap.add_argument("--derive-from-build", default=None, metavar="BUILD_DIR",
                    help="fallback: derive the public test file from a private "
                         "build dir (must contain qualified_tasks.jsonl)")
    ap.add_argument("--write-derived", default=None, metavar="PATH",
                    help="where to write the derived tasks file "
                         "(default: <out>/derived/qualified_tasks_test.jsonl)")
    ap.add_argument("--lane", action="append", default=[], dest="lanes",
                    help="restrict scoring to this lane (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N tasks (a deliberate, reported "
                         "subset)")
    ap.add_argument("--lane-kw", action="append", default=[],
                    help="per-lane resource override, e.g. "
                         "L05.index_dir=external_knowledge/indexes/rdkit-molecules/...")
    ap.add_argument("--no-repo-staging", action="store_true",
                    help="do not search bank/staging_20k and bank/staging for "
                         "oracle payloads (use only --oracle / env roots)")
    ap.add_argument("--probe-lanes", action="store_true",
                    help="report per-lane verifier support and exit")
    ap.add_argument("--version", action="version",
                    version="%s (%s)" % (__version__, REPORT_SCHEMA))
    return ap


def _resolve_tasks(args: argparse.Namespace) -> TaskSet:
    expected = None if args.no_count_check else args.expect_count
    candidates: list[Path] = []
    if args.tasks:
        candidates.append(Path(args.tasks))
    else:
        candidates.append(REPO_ROOT / DEFAULT_TASKS_REL)
    for candidate in candidates:
        if candidate.is_file():
            return load_tasks(candidate, expected_count=expected)
    if args.derive_from_build:
        if args.write_derived:
            out_path = Path(args.write_derived)
        elif args.out:
            out_path = Path(args.out) / "derived" / "qualified_tasks_test.jsonl"
        else:
            raise TaskValidationError(
                "--derive-from-build needs --write-derived or --out to place "
                "the derived tasks file (release_20k/tasks is never written)")
        return build_fallback_tasks(args.derive_from_build, out_path,
                                    expected_count=expected)
    tried = ", ".join(str(c) for c in candidates)
    raise TaskValidationError(
        "public tasks file not found (tried: %s). Stage the release or pass "
        "--derive-from-build <build_dir> for the fallback." % tried)


def _lane_kw_or_exit(raw: list[str]) -> dict[str, dict[str, str]]:
    try:
        return _parse_lane_kw(raw)
    except argparse.ArgumentTypeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)


def _score(tasks: TaskSet, predictions: SubmissionSet, oracle_store: OracleStore,
           lane_kw: dict[str, dict[str, str]],
           lanes: list[str], limit: int | None) -> list:
    from .scoring import TaskResult

    selected_lanes = {l.upper() for l in lanes} if lanes else None
    adapters: dict[str, LaneAdapter] = {}

    def adapter_for(lane_id: str) -> LaneAdapter:
        if lane_id not in adapters:
            adapters[lane_id] = get_adapter(lane_id, lane_kw.get(lane_id))
        return adapters[lane_id]

    results: list[TaskResult] = []
    by_task_id = predictions.by_task_id
    consumed: set[str] = set()
    task_rows = tasks.records[:limit] if limit else tasks.records
    for task in task_rows:
        if selected_lanes is not None and task.lane_id not in selected_lanes:
            continue
        prediction = by_task_id.get(task.task_id)
        if prediction is None:
            adapter = adapter_for(task.lane_id)
            supported, _, reason = adapter.support()
            if supported:
                results.append(missing_result(task))
            else:
                results.append(result_from_verdict(
                    task, _empty_prediction(task.task_id),
                    Verdict(status=STATUS_NOT_SUPPORTED, valid=None,
                            reason=reason)))
            continue
        consumed.add(task.task_id)
        adapter = adapter_for(task.lane_id)
        verdict = adapter.evaluate(task, prediction, oracle_store)
        results.append(result_from_verdict(task, prediction, verdict,
                                           oracle_path=verdict.oracle_path))
    for prediction in predictions.predictions:
        if prediction.task_id not in consumed and prediction.task_id not in tasks.by_id:
            results.append(unknown_task_result(prediction))
    return results


def _empty_prediction(task_id: str) -> Prediction:
    return Prediction(task_id=task_id, answer=None, actions=None, costs={},
                      provenance={}, line_no=0)


def _print_summary(report: dict[str, Any]) -> None:
    overall = report["metrics"]["overall"]
    print("== 20K evaluation summary ==")
    print("tasks: %d  scored: %d  missing: %d  not_supported: %d  "
          "oracle_missing: %d  error: %d"
          % (overall["tasks"], overall["scored"], overall["missing"],
             overall["not_supported"], overall["oracle_missing"],
             overall["error"]))
    tsr = overall["tsr"]
    vtsr = overall["v_tsr"]
    print("TSR: %s   V-TSR: %s"
          % (("%.4f (n=%d)" % (tsr, overall["tsr_denominator"]))
             if tsr is not None else "n/a",
             ("%.4f (n=%d)" % (vtsr, overall["v_tsr_denominator"]))
             if vtsr is not None else "n/a"))
    print("unsafe: %d rows / %d actions (rate over checked: %s); "
          "unchecked-action rows: %d"
          % (overall["unsafe_rows"], overall["unsafe_action_count"],
             ("%.4f" % overall["unsafe_rate"])
             if overall["unsafe_rate"] is not None else "n/a",
             overall["unsafe_unchecked"]))
    costs = {k: v["total"] for k, v in overall["costs"].items() if v["rows"]}
    if costs:
        print("costs: " + ", ".join("%s=%.1f" % (k, v)
                                    for k, v in sorted(costs.items())))
    print("-- per lane --")
    for lane, block in report["metrics"]["per_lane"].items():
        lane_status = report["lanes"].get(lane, {})
        note = "" if lane_status.get("status") == "SUPPORTED" else \
            "  [%s: %s]" % (lane_status.get("status", "?"),
                            lane_status.get("reason", ""))
        tsr_lane = block["tsr"]
        print("%-4s scored=%-6d success=%-6d tsr=%s%s"
              % (lane, block["scored"], block["success"],
                 ("%.4f" % tsr_lane) if tsr_lane is not None else "n/a",
                 note[:160]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lane_kw = _lane_kw_or_exit(args.lane_kw)

    if args.probe_lanes:
        report = probe_lanes(lane_kw)
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0

    if not args.predictions:
        print("error: --predictions is required", file=sys.stderr)
        return 2

    try:
        tasks = _resolve_tasks(args)
    except TaskValidationError as exc:
        print("error: tasks: %s" % exc, file=sys.stderr)
        return 2

    try:
        submission = load_submission(args.predictions)
    except SubmissionValidationError as exc:
        print("error: predictions: %s" % exc, file=sys.stderr)
        return 2

    oracle_roots = default_oracle_roots(
        include_repo_staging=not args.no_repo_staging, extra_roots=args.oracle)
    if not oracle_roots:
        print("error: no oracle roots available; mount the private oracle with "
              "--oracle PATH (or AOB_20K_ORACLE)", file=sys.stderr)
        return 2
    oracle_store = OracleStore(oracle_roots)

    results = _score(tasks, submission, oracle_store, lane_kw,
                     args.lanes, args.limit)

    lane_report = probe_lanes(lane_kw)  # lane support block for the report
    metrics = aggregate(results)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "harness_version": __version__,
        "tasks": {
            "path": tasks.path, "sha256": tasks.sha256,
            "count": len(tasks.records), "derived_from": tasks.derived_from,
            "limit": args.limit, "lanes_filter": args.lanes or None,
        },
        "predictions": {
            "path": submission.path, "sha256": submission.sha256,
            "rows": len(submission.predictions), "model": submission.model,
        },
        "oracle": {"roots": [str(r) for r in oracle_roots]},
        "lanes": lane_report,
        "metrics": metrics,
    }
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        write_rows(results, out_dir / "rows.jsonl")
        print("wrote %s and %s" % (out_dir / "report.json",
                                   out_dir / "rows.jsonl"))
    else:
        print(json.dumps(report, indent=1, sort_keys=True))
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
