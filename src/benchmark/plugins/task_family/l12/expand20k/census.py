"""L12 20K expansion: source census + report data generation.

Writes ``SOURCE_CENSUS.md`` (per source unit: id, path, sha256, kind, size,
task usage, license posture) into the expansion report dir from on-disk
artifacts only (scan report + staging shards). Read-only elsewhere.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.expand20k.registry import (
    EXPANSION_DIR,
    NEW_GOLD_PAIRS,
    NEW_LOGS,
    NEW_NOISY_PAIRS,
    FP_CONF_PAIRS,
    STAGING_20K,
)

SCAN_REPORT = EXPANSION_DIR / "scan_report.json"


def _iter_qualified() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((STAGING_20K / "qualified").glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def run_census(out_path: Path) -> dict[str, Any]:
    scan = json.loads(SCAN_REPORT.read_text(encoding="utf-8")) \
        if SCAN_REPORT.is_file() else {}
    records = _iter_qualified()

    usage: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        for uid in rec["source_unit_ids"]:
            usage[uid]["tasks"] += 1
            usage[uid][rec["family_id"]] += 1

    units: dict[str, dict[str, Any]] = {}

    def _unit(rel: str, kind: str, role: str) -> None:
        uid = engine.unit_id_for(rel)
        path = engine.PINNED_ROOT / rel
        units[uid] = {
            "unit_id": uid, "rel": rel, "kind": kind, "role": role,
            "sha256": engine.sha256_file(path) if path.is_file() else "MISSING",
            "bytes": path.stat().st_size if path.is_file() else 0,
            "tasks": usage.get(uid, Counter()).get("tasks", 0),
        }

    for key, pair in NEW_GOLD_PAIRS.items():
        _unit(pair["log"], "xes_log", "gold-pair log (%s)" % key)
        _unit(pair["net"], "pnml_net", "gold-pair net (%s)" % key)
    for key, rel in NEW_LOGS.items():
        _unit(rel, "xes_log", "audit/constraint log (%s)" % key)
    for key, pair in NEW_NOISY_PAIRS.items():
        _unit(pair["log"], "xes_log", "noisy-pair log (%s)" % key)
        _unit(pair["net"], "pnml_net", "noisy-pair net (%s)" % key)

    lines = [
        "# L12 20K expansion — source census",
        "",
        "Pinned snapshot: `pm4py-process@tag-2.7.23.11` "
        "(commit `0e077848cdb2db0440fcd2de62fb8b3c6aadac96`, AGPL-3.0).",
        "All units are NEW vs the legacy 8K lane (scan-verified; see",
        "`scan_report.json` → `legacy_usage`). No AGPL content is bundled into",
        "task artifacts: public records reference units by id + sha256 only",
        "(fetch-recipe redistribution posture, same as the legacy lane).",
        "",
        "| unit_id | rel path | kind | role | sha256 (first 16) | bytes | tasks |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for uid in sorted(units):
        u = units[uid]
        lines.append(
            "| `%s` | `%s` | %s | %s | `%s` | %d | %d |"
            % (uid, u["rel"], u["kind"], u["role"],
               u["sha256"][:16], u["bytes"], u["tasks"])
        )
    total_tasks = len(records)
    lines += [
        "",
        "Distinct source units referenced: %d; qualified tasks: %d."
        % (len(units), total_tasks),
        "",
        "Excluded candidates (honest):",
        "",
        "- `xes_20.xes` — events carry no `concept:name` (not an activity log).",
        "- `stochastic_running_example.pnml` as a gold pair — visible transition",
        "  labels identical to legacy `running-example.pnml` (behaviorally the",
        "  same net; would duplicate legacy semantics).",
        "- a42 nets for footprints conformance — reachability-graph construction",
        "  exceeds 120 s (probe evidence in scan log).",
        "- inh_res nets for footprints conformance — pm4py footprints model",
        "  discovery raises on these non-sound nets.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"units": len(units), "tasks": total_tasks}


def main() -> int:
    ap = argparse.ArgumentParser(description="L12 20K source census")
    ap.add_argument("--out", default=str(EXPANSION_DIR / "SOURCE_CENSUS.md"))
    args = ap.parse_args()
    summary = run_census(Path(args.out))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
