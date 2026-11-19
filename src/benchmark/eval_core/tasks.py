"""Public Core-track task loading and deterministic sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EXPECTED_TASK_COUNT = 19950
SAMPLE_SEED = 20261119


def default_tasks_path() -> Path:
    """Locate the Core task file in the repository or a sibling data download."""
    repo_root = Path(__file__).resolve().parents[3]
    candidates = (
        repo_root / "data/core/tasks/tasks.jsonl",
        repo_root.parent / "aob-data/core_worlds/data/core/tasks/tasks.jsonl",
        repo_root.parent / "aob-data/data/core/tasks/tasks.jsonl",
        repo_root / "release_core/tasks/tasks.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]

REQUIRED_TASK_FIELDS: tuple[str, ...] = (
    "release_task_id",
    "world_id",
    "lane_id",
    "split",
    "task_id",
    "spec_version",
    "template",
    "template_family",
    "actions",
    "prompt",
    "features",
    "instance_ids",
    "paraphrase_variant",
)


class TaskValidationError(ValueError):
    """Raised when a Core public task file is malformed."""


@dataclass(frozen=True)
class TaskRecord:
    """One public Core task row."""

    release_task_id: str
    task_id: str
    world_id: str
    lane_id: str
    split: str
    prompt: str
    actions: tuple[dict[str, Any], ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class TaskSet:
    """A validated task file plus its hash provenance."""

    records: tuple[TaskRecord, ...]
    path: str
    sha256: str

    @property
    def by_id(self) -> dict[str, TaskRecord]:
        return {record.release_task_id: record for record in self.records}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Read JSONL rows with line-numbered validation errors."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskValidationError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TaskValidationError(
                    f"{path}:{line_no}: record is not a JSON object"
                )
            yield row


def validate_task_record(row: dict[str, Any], *, line_no: int) -> TaskRecord:
    missing = [key for key in REQUIRED_TASK_FIELDS if key not in row]
    if missing:
        raise TaskValidationError(
            f"line {line_no}: missing required keys: {', '.join(sorted(missing))}"
        )
    if row["split"] not in {"dev", "hidden_iid", "hidden_comp", "hidden_transfer"}:
        raise TaskValidationError(
            f"line {line_no}: invalid split {row['split']!r}"
        )
    if not isinstance(row["actions"], list) or not row["actions"]:
        raise TaskValidationError(f"line {line_no}: actions must be a non-empty list")
    if not all(isinstance(action, dict) for action in row["actions"]):
        raise TaskValidationError(f"line {line_no}: every action must be an object")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise TaskValidationError(f"line {line_no}: prompt must be non-empty")
    return TaskRecord(
        release_task_id=str(row["release_task_id"]),
        task_id=str(row["task_id"]),
        world_id=str(row["world_id"]),
        lane_id=str(row["lane_id"]),
        split=str(row["split"]),
        prompt=str(row["prompt"]),
        actions=tuple(row["actions"]),
        raw=row,
    )


def load_tasks(
    path: str | Path,
    *,
    expected_count: int | None = EXPECTED_TASK_COUNT,
) -> TaskSet:
    """Load and validate a Core public task file."""
    path = Path(path)
    if not path.is_file():
        raise TaskValidationError(f"task file not found: {path}")
    records: list[TaskRecord] = []
    seen: set[str] = set()
    for line_no, row in enumerate(read_jsonl(path), start=1):
        record = validate_task_record(row, line_no=line_no)
        if record.release_task_id in seen:
            raise TaskValidationError(
                f"line {line_no}: duplicate release_task_id {record.release_task_id!r}"
            )
        seen.add(record.release_task_id)
        records.append(record)
    if expected_count is not None and len(records) != expected_count:
        raise TaskValidationError(
            f"expected {expected_count} tasks, found {len(records)} in {path}"
        )
    return TaskSet(records=tuple(records), path=str(path), sha256=_sha256_file(path))


def parse_csv(value: str | None) -> set[str]:
    """Parse a comma-separated world/lane filter."""
    if not value:
        return set()
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def select_tasks(
    records: Iterable[TaskRecord],
    *,
    worlds: set[str] | None = None,
    lanes: set[str] | None = None,
    sample_per_world: int = 0,
    limit: int | None = None,
    seed: int = SAMPLE_SEED,
) -> list[TaskRecord]:
    """Select tasks, deterministically sampling per world.

    ``sample_per_world == 0`` means all matching tasks. The sample is stable
    for a given task file, seed, world and cap. The returned order is the
    original file order.
    """
    worlds = {w.upper() for w in (worlds or set())}
    lanes = {l.upper() for l in (lanes or set())}
    indexed = list(enumerate(records))
    by_world: dict[str, list[tuple[int, TaskRecord]]] = {}
    for index, record in indexed:
        if worlds and record.world_id.upper() not in worlds:
            continue
        if lanes and record.lane_id.upper() not in lanes:
            continue
        by_world.setdefault(record.world_id, []).append((index, record))

    selected: list[tuple[int, TaskRecord]] = []
    for world_id in sorted(by_world):
        candidates = list(by_world[world_id])
        random.Random(f"{seed}:{world_id}").shuffle(candidates)
        selected.extend(candidates[:sample_per_world] if sample_per_world else candidates)

    selected.sort(key=lambda pair: pair[0])
    result = [record for _, record in selected]
    if limit is not None:
        result = result[: max(0, limit)]
    return result