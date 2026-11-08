"""Report builder for offline balanced F_rule (CLI in metrics_frule).

Reads already-written ``predictions.jsonl`` / ``rows.jsonl`` artifacts and the
mounted 20K oracle payloads.  It never calls a model or reruns inference; the
only native code it invokes is the lane's own build-time negative-submission
helper (see :mod:`benchmark.eval_20k.frule`).

The metric is deliberately coverage-aware: a lane or family without an oracle
reference plus an independent verifier-provided rejected answer is reported as
BLOCKED/N/A with an exact reason, never as zero.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .frule import (
    ELIGIBLE_LANES,
    L06NegativeBuilder,
    SCHEMA,
    add_task,
    classify,
    finalize,
    lane_status,
    new_bucket,
    negative_witness,
    positive_witness,
    self_check_ok,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR_RE = re.compile(
    r"^eval_20k_full2-(?P<group>g[0-9]+)-(?P<model>.+?)_(?P<suffix>[A-Za-z0-9]+)$")
DEFAULT_ORACLE_ROOT = "bank/staging_20k"

DEFINITIONS = {
    "task": "one (lane, family, task_id) oracle instance in the 20K run",
    "rule_unit": "(lane, family) pair",
    "positive_witness": "the oracle reference submission",
    "negative_witness": (
        "an independent verifier-provided rejected answer: stored "
        "mutation_submission (L15) or the lane's own build-time negative "
        "helper under the oracle's self_check (L05/L06/L07/L08)"),
    "outcome": (
        "positive -> TP on positive witness + TN on negative witness; "
        "negative -> FN + FP; any other answer -> UNKNOWN"),
    "balanced_frule": "(TPR + TNR) / 2",
    "macro": "unweighted mean of per-family balanced F_rule (unit: family)",
    "unknown": (
        "UNKNOWN is excluded from TP/FN/TN/FP and from TPR/TNR; it is "
        "reported as unknown and reflected in coverage, never counted as "
        "positive or negative"),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _oracle_json(row: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    path = Path(row.get("oracle_path") or "")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return _load_json(path), path


def discover_runs(root: Path) -> list[dict[str, str]]:
    """Pick completed runs from a run directory or a legacy run root."""
    direct_report = root / "report.json"
    if direct_report.is_file() and (root / "rows.jsonl").is_file() and (
            root / "predictions.jsonl").is_file():
        report = json.loads(direct_report.read_text(encoding="utf-8"))
        model = str((report.get("predictions") or {}).get("model") or "unknown")
        return [{"group": "local", "model": model, "path": str(root)}]
    chosen: dict[tuple[str, str], tuple[int, Path]] = {}
    for path in sorted(root.glob("eval_20k_full2-*")):
        match = RUN_DIR_RE.match(path.name)
        if not match or not (path / "rows.jsonl").is_file():
            continue
        if not (path / "predictions.jsonl").is_file():
            continue
        key = (match.group("group"), match.group("model"))
        size = sum(1 for _ in (path / "rows.jsonl").open(encoding="utf-8"))
        if key not in chosen or size > chosen[key][0]:
            chosen[key] = (size, path)
    return [{"group": key[0], "model": key[1], "path": str(path)}
            for key, (_, path) in sorted(chosen.items())]


def _in_scope(oracle_path: Path | None, roots: Iterable[str]) -> bool:
    if oracle_path is None:
        return False
    text = str(oracle_path).replace("\\", "/")
    return any(str(root).strip("/") in text for root in roots)


def _evidence_for(lane: str, oracle_path: Path | None) -> dict[str, Any] | None:
    if lane != "L08" or oracle_path is None:
        return None
    text = str(oracle_path).replace("\\", "/")
    if "/oracle/" not in text:
        return None
    return _load_json(Path(text.replace("/oracle/", "/evidence/")))


class _Witnesses:
    """Per-oracle witness cache shared across model runs."""

    def __init__(self) -> None:
        self.cache: dict[str, tuple[Any, Any, str]] = {}
        self.l05_adapter: Any = None
        self.l06_builder = L06NegativeBuilder()

    def get(self, lane: str, payload: dict[str, Any],
            evidence: dict[str, Any] | None, key: str) -> tuple[Any, Any, str]:
        if key in self.cache:
            return self.cache[key]
        result = self._build(lane, payload, evidence)
        self.cache[key] = result
        return result

    def _build(self, lane: str, payload: dict[str, Any],
               evidence: dict[str, Any] | None) -> tuple[Any, Any, str]:
        ok, reason = self_check_ok(lane, payload)
        if not ok:
            return None, None, reason
        positive, reason = positive_witness(lane, payload)
        if positive is None:
            return None, None, reason
        if lane == "L05" and self.l05_adapter is None:
            from benchmark.eval_20k.lanes import get_adapter
            self.l05_adapter = get_adapter("L05")
        negative, reason = negative_witness(
            lane, payload, evidence=evidence, l05_adapter=self.l05_adapter,
            l06_builder=self.l06_builder)
        if negative is None:
            return None, None, reason
        if json.dumps(positive, sort_keys=True) == json.dumps(
                negative, sort_keys=True):
            return None, None, "negative_equals_positive"
        return positive, negative, ""


def _buckets(block: dict[str, Any], lane: str, family: str
             ) -> tuple[dict[str, Any], ...]:
    family_key = "%s/%s" % (lane, family)
    lane_block = block["per_lane"].setdefault(lane, {
        "overall": new_bucket(), "families": {}})
    return (
        block["overall"],
        lane_block["overall"],
        block["per_family"].setdefault(family_key, new_bucket()),
        lane_block["families"].setdefault(family_key, new_bucket()),
    )


def _score_run(run: dict[str, str], per_model: dict[str, Any],
               witnesses: _Witnesses, oracle_roots: tuple[str, ...],
               lane_scope_ids: dict[str, set[str]]) -> None:
    block = per_model.setdefault(run["model"], {
        "runs": [], "overall": new_bucket(), "per_lane": {},
        "per_family": {}})
    block["runs"].append({"run": Path(run["path"]).name,
                          "group": run["group"], "path": run["path"]})
    preds = {p["task_id"]: p for p in _load_jsonl(
        Path(run["path"]) / "predictions.jsonl") if "task_id" in p}
    for row in _load_jsonl(Path(run["path"]) / "rows.jsonl"):
        lane = str(row.get("lane_id") or "")
        family = str(row.get("family_id") or "")
        if not lane or not family:
            continue
        payload, oracle_path = _oracle_json(row)
        if not _in_scope(oracle_path, oracle_roots):
            continue
        task_id = str(row.get("task_id") or "")
        lane_scope_ids[lane].add(task_id)
        status, reason = lane_status(lane)
        if status != "ELIGIBLE":
            for bucket in _buckets(block, lane, family):
                add_task(bucket, "blocked", reason)
            continue
        if payload is None:
            for bucket in _buckets(block, lane, family):
                add_task(bucket, "blocked", "oracle_payload_missing")
            continue
        key = str(oracle_path)
        positive, negative, witness_reason = witnesses.get(
            lane, payload, _evidence_for(lane, oracle_path), key)
        if positive is None or negative is None:
            for bucket in _buckets(block, lane, family):
                add_task(bucket, "blocked", witness_reason)
            continue
        prediction = preds.get(task_id)
        answer = prediction.get("answer") if prediction is not None else None
        outcome, outcome_reason = classify(answer, positive, negative)
        for bucket in _buckets(block, lane, family):
            add_task(bucket, outcome, outcome_reason)


def _family_macro(families: dict[str, dict[str, Any]]) -> float | None:
    values = [row["balanced_accuracy"] for row in families.values()
              if row.get("balanced_accuracy") is not None]
    return sum(values) / len(values) if values else None


def _finalize_model(block: dict[str, Any]) -> dict[str, Any]:
    families = {key: finalize(bucket)
                for key, bucket in sorted(block["per_family"].items())}
    lanes = {}
    for lane_id, lane in sorted(block["per_lane"].items()):
        lane_families = {key: finalize(bucket)
                         for key, bucket in sorted(lane["families"].items())}
        lane_overall = finalize(lane["overall"])
        lane_overall["macro_balanced_frule"] = _family_macro(lane_families)
        lane_overall["macro_unit"] = "family"
        lane_overall["scored_families"] = sum(
            1 for row in lane_families.values()
            if row.get("balanced_accuracy") is not None)
        lanes[lane_id] = {"overall": lane_overall,
                          "per_family": lane_families}
    overall = finalize(block["overall"])
    overall["macro_balanced_frule"] = _family_macro(families)
    overall["macro_unit"] = "family"
    overall["scored_families"] = sum(
        1 for row in families.values()
        if row.get("balanced_accuracy") is not None)
    return {"runs": sorted(block["runs"], key=lambda r: r["run"]),
            "overall": overall, "per_lane": lanes, "per_family": families}


def build_report(run_dirs: Iterable[dict[str, str]], *,
                 run_root: str = "runs",
                 oracle_roots: tuple[str, ...] = (DEFAULT_ORACLE_ROOT,)
                 ) -> dict[str, Any]:
    runs = list(run_dirs)
    per_model: dict[str, Any] = {}
    witnesses = _Witnesses()
    lane_scope_ids: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        _score_run(run, per_model, witnesses, oracle_roots, lane_scope_ids)
    lanes = {}
    for lane_id in sorted({*ELIGIBLE_LANES, *lane_scope_ids}):
        status, reason = lane_status(lane_id)
        lanes[lane_id] = {
            "status": status, "reason": reason,
            "tasks_unique": len(lane_scope_ids.get(lane_id, set())),
        }
    return {
        "schema": SCHEMA,
        "generated_from": {
            "run_root": run_root, "run_count": len(runs), "runs": runs,
            "oracle_roots": list(oracle_roots), "model_calls": 0,
            "inference_reruns": 0,
        },
        "definitions": DEFINITIONS,
        "eligible_lanes": list(ELIGIBLE_LANES),
        "lanes": lanes,
        "models": {model: _finalize_model(block)
                   for model, block in sorted(per_model.items())},
    }


