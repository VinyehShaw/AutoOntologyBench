"""Submission format for the 20K evaluation harness.

A submission is a JSONL file with one record per attempted task::

    {"task_id": "L05-l05i-0045c8bdeaa0c2e7",
     "answer": { ... lane-specific payload ... },          # required
     "actions": [ ... ],                                    # optional action program
     "tokens_input": 1234, "tokens_output": 56,             # optional cost fields
     "api_calls": 7, "tool_calls": 12, "wall_seconds": 3.2, # optional cost fields
     "model": "my-model-id", "run_id": "run-001"}           # optional provenance

The ``answer`` payload is passed verbatim to the lane's
``NativeVerifierPlugin.verify_terminal`` — e.g. for L05 ``query`` tasks
``{"answer_ids": [...]}``, for L05 ``repair`` tasks
``{"repaired_smarts": "..."}``, for L05 ``align`` tasks
``{"alignment": {...}}``. Task statements (``public`` payloads) document the
expected answer kind per family; the harness never inspects answer semantics
itself.

Rules enforced here (format only — semantics belong to the lane verifiers):

* ``task_id`` is required and must be unique across the file;
* cost fields, when present, must be non-negative numbers;
* a row referencing a ``task_id`` that is not in the task set is rejected at
  score time (counted as ``unknown_task``), never silently dropped;
* tasks with no prediction row are scored as failures (``missing``) — an
  unsubmitted task is not a success, and the missing count is reported.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Recognised cost/provenance keys (optional, numeric where noted).
COST_KEYS: tuple[str, ...] = (
    "tokens_input", "tokens_output", "api_calls", "tool_calls", "wall_seconds",
)
PROVENANCE_KEYS: tuple[str, ...] = ("model", "run_id", "agent", "notes")
KNOWN_KEYS: frozenset[str] = frozenset(
    {"task_id", "answer", "actions", "schema"} | set(COST_KEYS) | set(PROVENANCE_KEYS)
)


class SubmissionValidationError(ValueError):
    """Raised when a submission file is malformed (hard CLI error)."""


@dataclass(frozen=True)
class Prediction:
    """One validated submission row."""

    task_id: str
    answer: Any
    actions: tuple[Any, ...] | None
    costs: dict[str, float]
    provenance: dict[str, str]
    line_no: int


@dataclass(frozen=True)
class SubmissionSet:
    predictions: tuple[Prediction, ...]
    path: str
    sha256: str
    by_task_id: dict[str, Prediction] = field(compare=False)

    @property
    def model(self) -> str | None:
        models = {p.provenance.get("model") for p in self.predictions}
        models.discard(None)
        return sorted(models)[0] if len(models) == 1 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction(row: Any, *, line_no: int) -> Prediction:
    where = "line %d" % line_no
    if not isinstance(row, dict):
        raise SubmissionValidationError("%s: record is not a JSON object" % where)
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise SubmissionValidationError(
            "%s: 'task_id' must be a non-empty string" % where)
    if "answer" not in row:
        raise SubmissionValidationError(
            "%s: missing required key 'answer' (task_id=%s)" % (where, task_id))
    unknown = sorted(set(row) - KNOWN_KEYS)
    if unknown:
        raise SubmissionValidationError(
            "%s: unknown keys %s (task_id=%s); known keys: %s"
            % (where, ", ".join(unknown), task_id, ", ".join(sorted(KNOWN_KEYS))))
    actions = row.get("actions")
    if actions is not None and not isinstance(actions, list):
        raise SubmissionValidationError(
            "%s: 'actions' must be a list when present (task_id=%s)"
            % (where, task_id))
    costs: dict[str, float] = {}
    for key in COST_KEYS:
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise SubmissionValidationError(
                "%s: %s must be a non-negative number (task_id=%s)"
                % (where, key, task_id))
        costs[key] = float(value)
    provenance = {k: str(row[k]) for k in PROVENANCE_KEYS if row.get(k) is not None}
    return Prediction(
        task_id=task_id,
        answer=row["answer"],
        actions=tuple(actions) if actions is not None else None,
        costs=costs,
        provenance=provenance,
        line_no=line_no,
    )


def load_submission(path: str | Path) -> SubmissionSet:
    """Load and validate a submission JSONL file (hard-fails on any error)."""
    path = Path(path)
    if not path.is_file():
        raise SubmissionValidationError("predictions file not found: %s" % path)
    predictions: list[Prediction] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SubmissionValidationError(
                    "line %d: invalid JSON: %s" % (line_no, exc)) from exc
            if line_no == 1 and isinstance(row, dict) and "schema" in row and "task_id" not in row:
                # optional header record: {"schema": "aob-submission-20k/1", ...}
                if row["schema"] != "aob-submission-20k/1":
                    raise SubmissionValidationError(
                        "line 1: unsupported submission schema %r" % (row["schema"],))
                continue
            pred = validate_prediction(row, line_no=line_no)
            if pred.task_id in seen:
                raise SubmissionValidationError(
                    "line %d: duplicate task_id %r" % (line_no, pred.task_id))
            seen.add(pred.task_id)
            predictions.append(pred)
    by_task_id = {p.task_id: p for p in predictions}
    return SubmissionSet(
        predictions=tuple(predictions),
        path=str(path),
        sha256=_sha256_file(path),
        by_task_id=by_task_id,
    )
