"""Shared read-only paths and loaders for the 20K extra-metric audit."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = REPO_ROOT / "release_20k" / "tasks" / "qualified_tasks_test.jsonl"
TASK_MANIFEST = REPO_ROOT / "release_20k" / "tasks" / "task_manifest_test.jsonl"
BUILD_MANIFEST = (REPO_ROOT / "bank" / "builds" / "aob-20000-candidate-dev1"
                  / "task_manifest.jsonl")
PAPER_SPLITS = REPO_ROOT / "data" / "archive" / "world_0001_v4" / "private"
RUNS = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "reports" / "extra_metrics"
OUT_JSON = OUT_DIR / "decision_ood.json"
OUT_MD = OUT_DIR / "decision_ood.md"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_rows(path: Path) -> dict[str, dict]:
    return {row["task_id"]: row for row in read_jsonl(path)}


def load_predictions(path: Path) -> tuple[str | None, dict[str, object]]:
    model: str | None = None
    answers: dict[str, object] = {}
    for row in read_jsonl(path):
        if "task_id" not in row:
            continue
        model = model or row.get("model")
        answers[row["task_id"]] = row.get("answer")
    return model, answers


def prediction_files(group: str) -> dict[str, tuple[Path, dict[str, object]]]:
    found: dict[str, tuple[Path, dict[str, object]]] = {}
    for report_path in sorted(RUNS.glob("eval_20k_full2-%s-*/report.json" % group)):
        pred_path = report_path.parent / "predictions.jsonl"
        if not pred_path.is_file():
            continue
        model, answers = load_predictions(pred_path)
        if model:
            found[model] = (pred_path, answers)
    return found


def cell_keys(value: object) -> set[tuple[str, str]]:
    if not isinstance(value, Mapping):
        return set()
    return {
        (object_key, policy_key)
        for object_key, policies in value.items()
        if isinstance(object_key, str) and isinstance(policies, Mapping)
        for policy_key in policies
        if isinstance(policy_key, str)
    }