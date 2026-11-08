"""Submission format validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.eval_20k.submission import (
    SubmissionValidationError,
    load_submission,
)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_valid_submission(tmp_path: Path):
    path = _write(tmp_path, [
        {"schema": "aob-submission-20k/1"},
        {"task_id": "L05-a", "answer": {"answer_ids": ["CHEBI:1"]},
         "tokens_input": 10, "tokens_output": 2, "model": "m"},
        {"task_id": "L05-b", "answer": {"repaired_smarts": "C(=O)Cl"},
         "actions": [{"kind": "probe"}], "wall_seconds": 1.5},
    ])
    submission = load_submission(path)
    assert len(submission.predictions) == 2
    assert submission.model == "m"
    first = submission.by_task_id["L05-a"]
    assert first.costs["tokens_input"] == 10
    second = submission.by_task_id["L05-b"]
    assert second.actions == ({"kind": "probe"},)


def test_bad_schema_header(tmp_path: Path):
    path = _write(tmp_path, [{"schema": "other/9"},
                             {"task_id": "L05-a", "answer": {}}])
    with pytest.raises(SubmissionValidationError, match="schema"):
        load_submission(path)


def test_missing_answer_rejected(tmp_path: Path):
    path = _write(tmp_path, [{"task_id": "L05-a"}])
    with pytest.raises(SubmissionValidationError, match="answer"):
        load_submission(path)


def test_duplicate_task_id_rejected(tmp_path: Path):
    path = _write(tmp_path, [{"task_id": "L05-a", "answer": {}},
                             {"task_id": "L05-a", "answer": {}}])
    with pytest.raises(SubmissionValidationError, match="duplicate"):
        load_submission(path)


def test_unknown_key_rejected(tmp_path: Path):
    path = _write(tmp_path, [{"task_id": "L05-a", "answer": {}, "gold": "x"}])
    with pytest.raises(SubmissionValidationError, match="unknown keys"):
        load_submission(path)


def test_negative_cost_rejected(tmp_path: Path):
    path = _write(tmp_path, [{"task_id": "L05-a", "answer": {},
                              "tokens_input": -1}])
    with pytest.raises(SubmissionValidationError, match="non-negative"):
        load_submission(path)


def test_missing_file():
    with pytest.raises(SubmissionValidationError, match="not found"):
        load_submission("/nonexistent/predictions.jsonl")
