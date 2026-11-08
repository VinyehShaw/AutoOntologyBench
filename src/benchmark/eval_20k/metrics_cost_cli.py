"""Markdown/JSON writer and CLI for :mod:`benchmark.eval_20k.metrics_cost`."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics_cost import build_report


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a short human-readable companion to ``cps.json``."""
    lines = [
        "# CPS (cost per valid success) - 20K v3",
        "",
        f"Offline aggregation of {report['run_count']} completed "
        "`runs/eval_20k_full2-*/report.json` artifacts. No model was called "
        "and inference was not rerun.",
        "",
        "## Definitions",
        "",
        "- `verified_success` = `metrics.overall.verified_success`",
        "- `cps_tokens` = `(tokens_input.total + tokens_output.total) / "
        "verified_success`",
        "- `cost_per_valid_success` = alias of `cps_tokens`",
        "- `api_calls_per_success` = `api_calls.total / verified_success`",
        "- `wall_seconds_per_success` = `wall_seconds.total / verified_success`",
        "- A zero or missing denominator, or a missing cost field, yields "
        "`null` plus a `null_reasons` entry - never `0`.",
        "",
        "## Per group / model",
        "",
        "| Group | Model | Runs | Verified success | Tasks | Tokens in | "
        "Tokens out | CPS tokens | API calls/success | Wall s/success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        costs = cell["costs"]
        lines.append(
            f"| {cell['group']} | {cell['model']} | {cell['runs']} | "
            f"{_fmt(cell['verified_success'])} | {_fmt(cell['tasks'])} | "
            f"{_fmt(costs.get('tokens_input', {}).get('total'))} | "
            f"{_fmt(costs.get('tokens_output', {}).get('total'))} | "
            f"{_fmt(cell['cps_tokens'])} | "
            f"{_fmt(cell['api_calls_per_success'])} | "
            f"{_fmt(cell['wall_seconds_per_success'])} |")

    lines += [
        "",
        "## Model totals",
        "",
        "| Model | Runs | Verified success | Tasks | CPS tokens | "
        "API calls/success | Wall s/success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, total in report["models"].items():
        lines.append(
            f"| {model} | {total['runs']} | "
            f"{_fmt(total['verified_success'])} | {_fmt(total['tasks'])} | "
            f"{_fmt(total['cps_tokens'])} | "
            f"{_fmt(total['api_calls_per_success'])} | "
            f"{_fmt(total['wall_seconds_per_success'])} |")

    lines += [
        "",
        "## Group totals",
        "",
        "| Group | Runs | Verified success | Tasks | CPS tokens | "
        "API calls/success | Wall s/success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, total in report["groups"].items():
        aggregate = total["total"]
        lines.append(
            f"| {group} | {aggregate['runs']} | "
            f"{_fmt(aggregate['verified_success'])} | "
            f"{_fmt(aggregate['tasks'])} | "
            f"{_fmt(aggregate['cps_tokens'])} | "
            f"{_fmt(aggregate['api_calls_per_success'])} | "
            f"{_fmt(aggregate['wall_seconds_per_success'])} |")

    lines += ["", "## Sources", ""]
    lines.extend(f"- `{cell['report_path']}`" for cell in report["cells"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], out_dir: str | Path
                  ) -> tuple[Path, Path]:
    """Write ``cps.json`` and ``CPS.md`` without touching any run report."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "cps.json"
    markdown_path = directory / "CPS.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs",
                        help="directory containing eval_20k_full2-* runs")
    parser.add_argument("--out", default="reports/extra_metrics",
                        help="output directory for cps.json and CPS.md")
    args = parser.parse_args(argv)
    report = build_report(args.root)
    json_path, markdown_path = write_outputs(report, args.out)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(f"runs={report['run_count']} models={len(report['models'])} "
          f"groups={len(report['groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())