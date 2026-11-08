"""Read-only evidence audits for the 20K FRR/MRR/OOD feasibility report."""
from __future__ import annotations

import glob
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .extra_metrics_data import (
    BUILD_MANIFEST,
    PAPER_SPLITS,
    REPO_ROOT,
    RUNS,
    TASKS,
    manifest_rows,
    read_jsonl,
)

_DECISION_LIKE = re.compile(
    r"legal|illegal|consumer|ground.?truth|decision|accept|reject|expected_ok",
    re.IGNORECASE,
)


def task_schema_audit() -> dict[str, object]:
    """Scan public task records for decision/label field names."""
    rows = 0
    top_keys: Counter[str] = Counter()
    public_keys: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    for row in read_jsonl(TASKS):
        rows += 1
        top_keys.update(row.keys())
        for key in row:
            if _DECISION_LIKE.search(str(key)):
                matches["top:" + str(key)] += 1
        public = row.get("public")
        if not isinstance(public, Mapping):
            continue
        public_keys.update(public.keys())
        for key, value in public.items():
            if _DECISION_LIKE.search(str(key)):
                matches["public:" + str(key)] += 1
            if isinstance(value, Mapping):
                for nested in value:
                    if _DECISION_LIKE.search(str(nested)):
                        matches["public.%s.%s" % (key, nested)] += 1
    return {
        "tasks_scanned": rows,
        "top_level_keys": sorted(top_keys),
        "public_top_level_keys": sorted(public_keys),
        "decision_like_key_paths": dict(sorted(matches.items())),
        "has_top_level_legal_illegal_consumer_ground_truth": any(
            key in top_keys
            for key in ("legal", "illegal", "consumer", "ground_truth")),
    }


def prediction_audit() -> dict[str, object]:
    files = sorted(RUNS.glob("*/predictions.jsonl"))
    rows = 0
    keys: Counter[str] = Counter()
    actions = row_decisions = split_rows = answer_decisions = 0
    for path in files:
        for row in read_jsonl(path):
            if "task_id" not in row:
                continue
            rows += 1
            keys.update(row.keys())
            actions += int(bool(row.get("actions")))
            row_decisions += int("decisions" in row or "trace" in row)
            split_rows += int("split" in row)
            answer = row.get("answer")
            answer_decisions += int(isinstance(answer, Mapping)
                                    and "decisions" in answer)
    return {
        "files": len(files), "data_rows": rows, "row_keys": dict(keys),
        "rows_with_actions": actions, "rows_with_row_level_decisions": row_decisions,
        "rows_with_split_field": split_rows,
        "rows_with_answer_decisions": answer_decisions,
    }


def rows_audit() -> dict[str, object]:
    files = sorted(RUNS.glob("*/rows.jsonl"))
    statuses: Counter[str] = Counter()
    verdict_keys: Counter[str] = Counter()
    rows = unsafe = 0
    for path in files:
        for row in read_jsonl(path):
            rows += 1
            statuses[str(row.get("status"))] += 1
            verdict_keys.update((row.get("verdict_detail") or {}).keys())
            unsafe += int(bool(row.get("unsafe_unchecked")))
    return {
        "files": len(files), "rows": rows, "statuses": dict(statuses),
        "rows_with_unsafe_unchecked": unsafe,
        "verdict_detail_keys": sorted(verdict_keys),
    }


def oracle_audit() -> dict[str, object]:
    files = sorted(
        glob.glob(str(REPO_ROOT / "bank" / "staging_20k" / "**" / "oracle" / "*.json"),
                  recursive=True)
        + glob.glob(str(REPO_ROOT / "bank" / "staging" / "**" / "oracle" / "*.json"),
                    recursive=True))
    per_lane: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    scanned = errors = 0
    for path in files:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors += 1
            continue
        scanned += 1
        parts = Path(path).parts
        root = "staging_20k" if "staging_20k" in parts else "staging"
        index = parts.index(root)
        per_lane[parts[index + 1]] += 1
        if isinstance(payload, Mapping):
            keys = {str(key) for key in payload}
            flags["files_with_reference_answer"] += int("reference_answer" in keys)
            flags["files_with_reference_terminal"] += int("reference_terminal" in keys)
            flags["files_with_suite"] += int("suite" in keys)
            flags["files_with_step_modes"] += int("step_modes" in keys)
            flags["files_with_decision_id"] += int("decision_id" in keys)
            flags["files_with_legal_or_illegal_key"] += int(
                bool({"legal", "illegal", "consumer", "ground_truth"} & keys))
            ref = payload.get("reference_answer")
            flags["files_with_reference_answer_decisions"] += int(
                isinstance(ref, Mapping) and "decisions" in ref)
    return {
        "files_scanned": scanned, "errors": errors,
        "per_root_and_lane": dict(sorted(per_lane.items())),
        "flags": dict(flags),
    }


def split_audit(tasks: dict[str, dict]) -> dict[str, object]:
    public = Counter(row.get("split") for row in tasks.values())
    build = Counter(row.get("split") for row in
                    manifest_rows(BUILD_MANIFEST).values())
    paper: dict[str, object] = {}
    task_ids = set(tasks)
    for name in ("dev_labels", "hidden_iid", "hidden_comp", "hidden_transfer"):
        path = PAPER_SPLITS / (name + ".jsonl")
        rows = list(read_jsonl(path))
        ids = {row.get("task_id") for row in rows}
        paper[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": len(rows), "task_ids": len(ids),
            "overlap_with_20k_task_ids": len(ids & task_ids),
            "sample_task_ids": sorted(ids)[:3],
        }
    reports = sorted(RUNS.glob("*/report.json"))
    return {
        "public_20k_split_counts": dict(public),
        "internal_build_split_counts": dict(build),
        "paper_world_split_files": paper,
        "report_files_scanned": len(reports),
        "report_files_with_split_key": sum(
            1 for path in reports
            if '"split"' in path.read_text(encoding="utf-8")),
    }