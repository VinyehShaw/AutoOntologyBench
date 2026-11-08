from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_RE = re.compile(r"^eval_20k_full2-(?P<group>g\d+)-")
SCHEMA = "aob-eval-20k/batch-tsr/1"


def _paths(run_dirs: list[str], runs_glob: str | None) -> list[Path]:
    paths: list[Path] = []
    for raw in run_dirs:
        path = Path(raw) / "report.json"
        if not path.is_file():
            raise FileNotFoundError("report.json not found under %s" % raw)
        paths.append(path)
    if runs_glob:
        paths.extend(sorted(Path().glob(runs_glob)))
    unique = {str(path.resolve()): path for path in paths}
    return [unique[key] for key in sorted(unique)]


def _batch_name(path: Path, label: str | None) -> str:
    if label:
        return label
    match = GROUP_RE.match(path.parent.name)
    return match.group("group") if match else path.parent.name


def _overall(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    overall = metrics.get("overall") or {}
    return {
        key: overall.get(key)
        for key in (
            "tasks", "scored", "success", "verified_success", "missing",
            "not_supported", "oracle_missing", "error",
            "tsr", "tsr_denominator", "v_tsr", "v_tsr_denominator",
        )
    }


def build(paths: list[Path], label: str | None = None) -> dict[str, Any]:
    batches: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "tasks": 0, "scored": 0, "success": 0,
                 "verified_success": 0, "missing": 0, "models": {}})
    rows = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        batch = _batch_name(path, label)
        model = str((report.get("predictions") or {}).get("model") or "unknown")
        values = _overall(report)
        entry = {"batch": batch, "model": model, "run": path.parent.name,
                 "path": str(path), **values}
        rows.append(entry)
        b = batches[batch]
        for key in ("tasks", "scored", "success", "verified_success", "missing"):
            value = values.get(key)
            if isinstance(value, int):
                b[key] += value
        b["runs"] += 1
        b["models"][model] = entry
    for b in batches.values():
        denominator = (b["scored"] or 0) + (b["missing"] or 0)
        b["tsr"] = (b["success"] / denominator) if denominator else None
        b["v_tsr"] = ((b["verified_success"] / denominator)
                      if denominator else None)
    return {
        "schema": SCHEMA,
        "batches": {name: batches[name] for name in sorted(batches)},
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-batch TSR from completed 20K reports")
    parser.add_argument("--run-dir", action="append", default=[],
                        help="run directory containing report.json (repeatable)")
    parser.add_argument("--runs-glob", default=None,
                        help="optional glob for report.json files")
    parser.add_argument("--batch-label", default=None,
                        help="override batch label for a single run")
    parser.add_argument("--out", default=None, help="output JSON")
    args = parser.parse_args(argv)
    try:
        paths = _paths(args.run_dir, args.runs_glob)
        if not paths:
            raise FileNotFoundError("no report.json inputs")
        report = build(paths, args.batch_label)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: %s" % exc)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print("wrote %s" % out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())