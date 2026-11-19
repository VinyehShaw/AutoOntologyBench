"""Private Core-track oracle loading.

The oracle is intentionally separate from the public task release. It contains
only the machine-replayed labels needed by the answer scorer; it never contains
the worker's prompt or the public materials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .tasks import read_jsonl

ORACLE_FILENAME = "core_oracle.jsonl"


class OracleError(ValueError):
    """Raised when a private Core oracle is missing or malformed."""


@dataclass(frozen=True)
class OracleRecord:
    """Labels for one task, joined by ``release_task_id``."""

    release_task_id: str
    expected: str
    expected_ok: bool
    expected_reason: str
    outcomes: tuple[dict[str, Any], ...]

    @property
    def terminal_payload(self) -> Any:
        if not self.outcomes:
            return None
        return self.outcomes[-1].get("payload")


def _oracle_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / ORACLE_FILENAME
        if candidate.is_file():
            return candidate
        # Accept a directory containing exactly one *.jsonl oracle.
        files = sorted(path.glob("*.jsonl"))
        if len(files) == 1:
            return files[0]
        raise OracleError(
            f"oracle directory {path} has no {ORACLE_FILENAME} and not exactly one *.jsonl"
        )
    return path


def load_oracle(path: str | Path) -> dict[str, OracleRecord]:
    """Load ``release_task_id -> OracleRecord`` from a private JSONL file."""
    path = _oracle_path(path)
    if not path.is_file():
        raise OracleError(f"oracle file not found: {path}")

    oracle: dict[str, OracleRecord] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        release_task_id = row.get("release_task_id")
        if not isinstance(release_task_id, str) or not release_task_id:
            raise OracleError(
                f"{path}:{line_no}: missing non-empty release_task_id"
            )
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise OracleError(f"{path}:{line_no}: outcomes must be a non-empty list")
        if not all(isinstance(outcome, dict) for outcome in outcomes):
            raise OracleError(f"{path}:{line_no}: every outcome must be an object")
        expected_ok = row.get("expected_ok")
        if not isinstance(expected_ok, bool):
            raise OracleError(f"{path}:{line_no}: expected_ok must be a boolean")
        expected = str(row.get("expected") or "")
        expected_reason = str(row.get("expected_reason") or "")
        if not expected:
            raise OracleError(f"{path}:{line_no}: expected must be non-empty")
        if release_task_id in oracle:
            raise OracleError(f"{path}:{line_no}: duplicate oracle task {release_task_id!r}")
        oracle[release_task_id] = OracleRecord(
            release_task_id=release_task_id,
            expected=expected,
            expected_ok=expected_ok,
            expected_reason=expected_reason,
            outcomes=tuple(outcomes),
        )
    if not oracle:
        raise OracleError(f"oracle is empty: {path}")
    return oracle


def load_predictions(path: str | Path) -> dict[str, Any]:
    """Load a submission JSONL, rejecting duplicate or unknown-shaped rows."""
    path = Path(path)
    if not path.is_file():
        raise OracleError(f"predictions file not found: {path}")
    predictions: dict[str, Any] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        if line_no == 1 and "task_id" not in row and "schema" in row:
            # Optional header record, matching the 20K convention.
            if row["schema"] != "aob-submission-core/1":
                raise OracleError(
                    f"{path}:{line_no}: unsupported submission schema {row['schema']!r}"
                )
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise OracleError(f"{path}:{line_no}: missing non-empty task_id")
        if "answer" not in row:
            raise OracleError(f"{path}:{line_no}: missing answer (task_id={task_id})")
        if task_id in predictions:
            raise OracleError(f"{path}:{line_no}: duplicate task_id {task_id!r}")
        predictions[task_id] = row["answer"]
    return predictions