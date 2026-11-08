"""Public 20K task loading, schema validation and the build fallback.

The official public file is ``release_20k/tasks/qualified_tasks_test.jsonl``:
20,000 records, every record ``split == "test"``, and no private fields
(anything the release packager strips: ``private_*`` keys anywhere, plus the
oracle-only top-level keys such as ``native_evidence`` or ``ast``).

When the official file is absent, :func:`build_fallback_tasks` derives an
equivalent public test file from a private build directory (e.g.
``bank/builds/aob-20000-candidate-dev1``) by reusing the release packager's
own projection (:func:`autoontobench.factory.release.package._project_row`)
and stamping ``split = "test"``. The derived file is written to a
caller-chosen path -- never into ``release_20k/tasks`` (that tree is owned by
the release lane).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from autoontobench.contracts.models import assert_no_private_keys

#: Expected task count of the official 20K test file.
EXPECTED_TASK_COUNT = 20000

#: Required top-level keys of a public task record (superset of the release
#: packager's PUBLIC_TASK_KEYS, plus the harness-mandated ``split``).
REQUIRED_PUBLIC_KEYS: tuple[str, ...] = (
    "schema_version", "dataset_id", "lane_id", "family_id", "base_goal_id",
    "instance_id", "task_id", "task_regime", "complexity", "semantic_profile",
    "public", "oracle_handle", "status", "qualification",
    "source_snapshot_ids", "source_unit_ids", "difficulty", "split",
)

#: Top-level keys that belong to the private side of the bank and must never
#: appear in a public task record (defence in depth on top of the
#: ``private_*`` recursive guard from the canonical contracts).
ORACLE_ONLY_KEYS: tuple[str, ...] = (
    "native_evidence", "ast", "reference_submission", "reference_answer",
    "expected_answer", "proof", "witness", "self_check", "oracle",
)

DEFAULT_EXPECTED_COUNT = EXPECTED_TASK_COUNT


class TaskValidationError(ValueError):
    """Raised when the public task file fails schema/leak validation."""


@dataclass(frozen=True)
class TaskRecord:
    """A validated public task row."""

    task_id: str
    instance_id: str
    lane_id: str
    family_id: str
    base_goal_id: str
    task_regime: str
    complexity: str
    semantic_profile: str
    oracle_handle: str
    public: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class TaskSet:
    """Loaded task file plus provenance."""

    records: tuple[TaskRecord, ...]
    path: str
    sha256: str
    derived_from: str | None = None

    @property
    def by_id(self) -> dict[str, TaskRecord]:
        return {r.task_id: r for r in self.records}


def validate_task_record(row: Any, *, line_no: int) -> TaskRecord:
    """Validate one public task record; raises :class:`TaskValidationError`."""
    where = "line %d" % line_no
    if not isinstance(row, dict):
        raise TaskValidationError("%s: record is not a JSON object" % where)
    missing = [k for k in REQUIRED_PUBLIC_KEYS if k not in row]
    if missing:
        raise TaskValidationError(
            "%s: missing required keys: %s" % (where, ", ".join(sorted(missing))))
    leaked = [k for k in ORACLE_ONLY_KEYS if k in row]
    if leaked:
        raise TaskValidationError(
            "%s: private/oracle-only keys present in public record: %s"
            % (where, ", ".join(sorted(leaked))))
    try:
        assert_no_private_keys(row)
    except ValueError as exc:
        raise TaskValidationError("%s: %s" % (where, exc)) from exc
    if row["split"] != "test":
        raise TaskValidationError(
            "%s: split must be \"test\", got %r" % (where, row["split"]))
    if not isinstance(row["public"], dict):
        raise TaskValidationError("%s: 'public' must be an object" % where)
    if not isinstance(row["lane_id"], str) or not row["lane_id"]:
        raise TaskValidationError("%s: lane_id must be a non-empty string" % where)
    for key in ("task_id", "instance_id", "family_id", "oracle_handle"):
        if not isinstance(row[key], str) or not row[key]:
            raise TaskValidationError(
                "%s: %s must be a non-empty string" % (where, key))
    return TaskRecord(
        task_id=row["task_id"],
        instance_id=row["instance_id"],
        lane_id=row["lane_id"],
        family_id=row["family_id"],
        base_goal_id=row["base_goal_id"],
        task_regime=row["task_regime"],
        complexity=row["complexity"],
        semantic_profile=row["semantic_profile"],
        oracle_handle=row["oracle_handle"],
        public=row["public"],
        raw=row,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path: str | Path, *, expected_count: int | None = EXPECTED_TASK_COUNT,
               derived_from: str | None = None) -> TaskSet:
    """Load and validate a public tasks JSONL file.

    ``expected_count`` enforces the exact record count (20,000 for the
    official file); pass ``None`` to disable the count check (fixtures).
    Raises :class:`TaskValidationError` on any schema, leak, duplicate or
    count violation -- a partially valid task set is never returned.
    """
    path = Path(path)
    if not path.is_file():
        raise TaskValidationError("tasks file not found: %s" % path)
    records: list[TaskRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskValidationError(
                    "line %d: invalid JSON: %s" % (line_no, exc)) from exc
            record = validate_task_record(row, line_no=line_no)
            if record.task_id in seen:
                raise TaskValidationError(
                    "line %d: duplicate task_id %r" % (line_no, record.task_id))
            seen.add(record.task_id)
            records.append(record)
    if expected_count is not None and len(records) != expected_count:
        raise TaskValidationError(
            "expected %d tasks, found %d in %s"
            % (expected_count, len(records), path))
    return TaskSet(
        records=tuple(records),
        path=str(path),
        sha256=_sha256_file(path),
        derived_from=derived_from,
    )


# --------------------------------------------------------------------------
# Fallback builder
# --------------------------------------------------------------------------

def _project_with_packager(row: dict[str, Any]) -> dict[str, Any]:
    """Project a private build row to its public form with the release
    packager's own logic (imported lazily so this module stays light)."""
    from autoontobench.factory.release.package import _project_row

    return _project_row(row)


def _project_row_fallback(row: dict[str, Any]) -> dict[str, Any]:
    """Self-contained mirror of package._project_row (whitelist + strip).

    Only used when the release packager module cannot be imported; the keys
    whitelist is identical to the packager's PUBLIC_TASK_KEYS.
    """
    from autoontobench.contracts.models import _strip_private

    public_keys = (
        "schema_version", "dataset_id", "lane_id", "family_id", "base_goal_id",
        "instance_id", "task_id", "task_regime", "complexity", "semantic_profile",
        "public", "oracle_handle", "status", "qualification",
        "source_snapshot_ids", "source_unit_ids", "difficulty",
    )
    clean = _strip_private(row)
    assert_no_private_keys(clean)
    return {k: clean[k] for k in public_keys if k in clean}


def project_public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Public projection of a private build row (packager logic preferred)."""
    try:
        return _project_with_packager(row)
    except ImportError:  # pragma: no cover - defensive
        return _project_row_fallback(row)


def iter_build_rows(build_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield raw rows from a private build directory's qualified_tasks.jsonl."""
    qualified = Path(build_dir) / "qualified_tasks.jsonl"
    if not qualified.is_file():
        raise TaskValidationError(
            "build fallback: %s not found (expected a private build dir "
            "containing qualified_tasks.jsonl)" % qualified)
    with qualified.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskValidationError(
                    "build fallback: %s line %d: invalid JSON: %s"
                    % (qualified, line_no, exc)) from exc


def build_fallback_tasks(build_dir: str | Path, out_path: str | Path,
                         *, expected_count: int | None = EXPECTED_TASK_COUNT
                         ) -> TaskSet:
    """Derive the public ``qualified_tasks_test.jsonl`` from a private build.

    Applies the release packager's public projection, stamps ``split="test"``
    on every record (the 20K release is test-only by construction), validates
    each projected record, then writes the file to ``out_path`` and reloads it
    through :func:`load_tasks` so the returned TaskSet is exactly what the
    harness scores. ``out_path`` must not live inside ``release_20k/tasks``.
    """
    build_dir = Path(build_dir)
    out_path = Path(out_path)
    resolved = out_path.resolve()
    release_tasks = (Path(__file__).resolve().parents[2]
                     / "release_20k" / "tasks")
    if release_tasks in resolved.parents or resolved.parent == release_tasks:
        raise TaskValidationError(
            "refusing to write fallback tasks inside release_20k/tasks "
            "(owned by the release lane): %s" % out_path)
    rows: list[dict[str, Any]] = []
    for row in iter_build_rows(build_dir):
        projected = project_public_row(row)
        projected["split"] = "test"
        rows.append(projected)
    if expected_count is not None and len(rows) != expected_count:
        raise TaskValidationError(
            "build fallback: expected %d qualified rows in %s, found %d"
            % (expected_count, build_dir, len(rows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return load_tasks(out_path, expected_count=expected_count,
                      derived_from=str(build_dir))
