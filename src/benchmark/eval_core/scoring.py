"""Deterministic answer scoring for the Core-track evaluator.

Answer contract (also documented in the package README):

* one-step query: ``answer`` is the exact JSON payload (list/dict/string/int);
* one-step validation: ``answer`` is ``"ok"``, ``"reject:<reason>"``, or a
  boolean;
* multi-step task: ``answer`` is a list with one entry per action, or a
  ``{"outcomes": [...]}`` object. A string equal to the private
  semicolon-joined ``expected`` value is also accepted for compatibility.
  Each entry is either the action payload or
  ``{"ok": bool, "reason": "...", "payload": ...}``.

Scoring is exact: no model judge is used. ``TSR`` is the fraction of tasks
whose every action is correct. ``step_accuracy`` is the fraction of correct
actions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .oracle import OracleRecord
from .tasks import TaskRecord


def _expected_verdict(outcome: dict[str, Any]) -> str:
    if bool(outcome.get("ok")):
        return "ok"
    reason = str(outcome.get("reason") or "")
    return f"reject:{reason}" if reason else "reject"


def _unwrap_payload(value: Any) -> Any:
    if isinstance(value, dict) and "payload" in value:
        allowed = {"ok", "reason", "payload"}
        if set(value) <= allowed:
            return value["payload"]
    return value


def _step_correct(answer: Any, outcome: dict[str, Any]) -> tuple[bool, str, str]:
    payload = outcome.get("payload")
    if payload is not None:
        expected = payload
        predicted = _unwrap_payload(answer)
        return predicted == expected, str(predicted), str(expected)

    expected = _expected_verdict(outcome)
    if isinstance(answer, bool):
        return answer == bool(outcome.get("ok")), str(answer), expected
    if isinstance(answer, str):
        predicted = answer.strip()
        accepted = {expected}
        if not bool(outcome.get("ok")):
            reason = str(outcome.get("reason") or "")
            if reason:
                accepted.add(reason)
        return predicted in accepted, predicted, expected
    if isinstance(answer, dict) and "ok" in answer:
        predicted_ok = bool(answer.get("ok"))
        if predicted_ok != bool(outcome.get("ok")):
            return False, str(answer), expected
        if not predicted_ok and "reason" in answer:
            reason = str(answer.get("reason") or "")
            expected_reason = str(outcome.get("reason") or "")
            return reason == expected_reason, str(answer), expected
        return True, str(answer), expected
    return False, str(answer), expected


def _step_values(answer: Any, count: int) -> list[Any] | None:
    if isinstance(answer, dict) and isinstance(answer.get("outcomes"), list):
        values = answer["outcomes"]
        return list(values) if len(values) == count else None
    if count == 1:
        return [answer]
    if isinstance(answer, list) and len(answer) == count:
        return list(answer)
    return None


def score_task(task: TaskRecord, oracle: OracleRecord, answer: Any) -> dict[str, Any]:
    """Score one answer; return per-step evidence and task success."""
    count = len(oracle.outcomes)
    values = _step_values(answer, count)
    details: list[dict[str, Any]] = []

    if values is None:
        # Compatibility path for multi-step verdict tasks: the private
        # `expected` is the canonical semicolon-joined verdict sequence.
        whole_correct = isinstance(answer, str) and answer.strip() == oracle.expected
        for index, outcome in enumerate(oracle.outcomes):
            details.append(
                {
                    "step": index,
                    "correct": whole_correct,
                    "predicted": answer,
                    "expected": _expected_verdict(outcome),
                }
            )
        correct = count if whole_correct else 0
    else:
        for index, (value, outcome) in enumerate(zip(values, oracle.outcomes)):
            ok, predicted, expected = _step_correct(value, outcome)
            details.append(
                {
                    "step": index,
                    "correct": ok,
                    "predicted": predicted,
                    "expected": expected,
                }
            )
        correct = sum(1 for detail in details if detail["correct"])

    return {
        "task_id": task.release_task_id,
        "world_id": task.world_id,
        "lane_id": task.lane_id,
        "split": task.split,
        "task_id_local": task.task_id,
        "success": correct == count,
        "steps_correct": correct,
        "steps_total": count,
        "details": details,
    }


def aggregate(
    tasks: Iterable[TaskRecord],
    oracle: dict[str, OracleRecord],
    predictions: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score all tasks and aggregate Core metrics with per-group breakdowns."""
    task_list = list(tasks)
    rows: list[dict[str, Any]] = []
    missing = 0
    unknown = 0
    for task in task_list:
        record = oracle.get(task.release_task_id)
        if record is None:
            missing += 1
            rows.append(
                {
                    "task_id": task.release_task_id,
                    "world_id": task.world_id,
                    "lane_id": task.lane_id,
                    "split": task.split,
                    "success": False,
                    "steps_correct": 0,
                    "steps_total": 0,
                    "error": "oracle_missing",
                }
            )
            continue
        answer = predictions.get(task.release_task_id)
        if answer is None:
            missing += 1
            rows.append(
                {
                    "task_id": task.release_task_id,
                    "world_id": task.world_id,
                    "lane_id": task.lane_id,
                    "split": task.split,
                    "success": False,
                    "steps_correct": 0,
                    "steps_total": len(record.outcomes),
                    "error": "prediction_missing",
                }
            )
            continue
        rows.append(score_task(task, record, answer))

    known = {task.release_task_id for task in task_list}
    unknown = sum(1 for task_id in predictions if task_id not in known)
    scored = len(rows) - missing
    success = sum(1 for row in rows if row.get("success"))
    steps = sum(int(row.get("steps_total") or 0) for row in rows)
    steps_correct = sum(int(row.get("steps_correct") or 0) for row in rows)

    def group(key: str) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"tasks": 0, "scored": 0, "success": 0, "steps": 0, "steps_correct": 0}
        )
        for row in rows:
            bucket = buckets[str(row[key])]
            bucket["tasks"] += 1
            if row.get("error") is None:
                bucket["scored"] += 1
                bucket["success"] += int(bool(row.get("success")))
            bucket["steps"] += int(row.get("steps_total") or 0)
            bucket["steps_correct"] += int(row.get("steps_correct") or 0)
        for bucket in buckets.values():
            bucket["tsr"] = (
                bucket["success"] / bucket["scored"] if bucket["scored"] else None
            )
            bucket["step_accuracy"] = (
                bucket["steps_correct"] / bucket["steps"] if bucket["steps"] else None
            )
        return dict(sorted(buckets.items()))

    metrics = {
        "tasks": len(task_list),
        "scored": scored,
        "missing": missing,
        "unknown_predictions": unknown,
        "success": success,
        "tsr": success / scored if scored else None,
        "steps": steps,
        "steps_correct": steps_correct,
        "step_accuracy": steps_correct / steps if steps else None,
    }
    report = {
        "metrics": metrics,
        "per_world": group("world_id"),
        "per_lane": group("lane_id"),
        "per_split": group("split"),
    }
    return report, rows


def format_summary(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.4f}"
    return "\n".join(
        [
            "== Core-track evaluation summary ==",
            f"tasks: {metrics['tasks']}  scored: {metrics['scored']}  "
            f"missing: {metrics['missing']}  unknown: {metrics['unknown_predictions']}",
            f"success: {metrics['success']}",
            f"TSR: {fmt(metrics['tsr'])}",
            f"step accuracy: {fmt(metrics['step_accuracy'])} "
            f"({metrics['steps_correct']}/{metrics['steps']})",
        ]
    )