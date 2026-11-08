"""CLI for offline balanced F_rule on the completed 20K v3 runs.

No model is called and inference is not rerun; see
:mod:`benchmark.eval_20k.frule_report` for the report builder and
:mod:`benchmark.eval_20k.frule` for the metric definition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .frule_report import DEFAULT_ORACLE_ROOT, build_report, discover_runs

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--oracle-root", action="append", default=[],
                        help="oracle root to treat as in-scope (repeatable)")
    parser.add_argument("--out-json", default=str(
        REPO_ROOT / "reports" / "extra_metrics" / "frule.json"))
    parser.add_argument("--out-md", default=str(
        REPO_ROOT / "reports" / "extra_metrics" / "FRULE.md"))
    args = parser.parse_args(argv)
    from .frule_markdown import render_markdown

    run_root = Path(args.run_root)
    runs = discover_runs(run_root)
    roots = tuple(args.oracle_root or [DEFAULT_ORACLE_ROOT])
    report = build_report(runs, run_root=str(run_root), oracle_roots=roots)
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print("wrote %s (%d runs, %d models)" % (
        out_json, len(runs), len(report["models"])))
    print("wrote %s" % out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())