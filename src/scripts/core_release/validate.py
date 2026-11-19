"""Pack-contract validation for Core-track world packs (release contract v1).

``collect_violations`` never raises for a bad pack: it returns one ``Violation``
per machine-checkable defect, so the CLI can print every problem at once and
callers can assert on stable codes. A world is released only when the list is
empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contract import (
    DIFFICULTY_FILE,
    MANIFEST_FILES,
    PROVENANCE_FILE,
    SPLIT_NAMES,
    canonical_json,
    read_json,
    release_projection,
    release_task_row,
    sha256_bytes,
)
from .errors import PackContractError, Violation
from .labels import check_labels
from .layout import check_hashes, check_layout, check_world_identity, load_spec, object_field
from .leakage import check_leakage
from .semantics import check_semantic_leakage
from .tasks import check_families_and_tags, check_splits, parse_tasks


@dataclass(frozen=True, slots=True)
class ValidatedWorld:
    pack_dir: Path
    world_id: str
    lane_id: str
    spec_version: str
    provenance: dict[str, object]
    difficulty: dict[str, object]
    spec: dict[str, object]
    tasks: tuple[dict[str, object], ...]  # release rows, no labels
    dev_examples: tuple[dict[str, object], ...]  # sanitized dev projection, file order
    per_split: dict[str, int]
    task_set_sha256: str


def _task_set_sha256(rows: list[dict[str, object]]) -> str:
    ordered = sorted(rows, key=lambda row: str(row["task_id"]))
    body = "".join(f"{canonical_json(row)}\n" for row in ordered)
    return sha256_bytes(body.encode("utf-8"))


def collect_violations(
    pack_dir: Path, repo_root: Path
) -> tuple[list[Violation], ValidatedWorld | None]:
    """Validate one pack; returns every violation and a world only when clean."""
    pack = Path(pack_dir)
    violations: list[Violation] = []
    if not pack.is_dir():
        return [Violation("missing_pack", str(pack), "pack directory not found")], None
    if not check_layout(pack, violations):
        return violations, None
    try:
        provenance = object_field(read_json(pack / PROVENANCE_FILE), PROVENANCE_FILE)
        difficulty = object_field(read_json(pack / DIFFICULTY_FILE), DIFFICULTY_FILE)
        read_json(pack / "public/source_schema.json")
    except (ValueError, OSError) as exc:
        return [Violation("json_parse_error", MANIFEST_FILES[0], str(exc))], None
    check_hashes(pack, provenance, violations)
    spec = load_spec(repo_root, provenance, violations)
    if spec is None:
        return violations, None
    identity = check_world_identity(provenance, difficulty, spec, violations)
    parsed = parse_tasks(pack, violations)
    if identity is None or parsed is None:
        return violations, None
    world_id, lane_id = identity
    by_split, public_records = parsed
    check_splits(by_split, public_records, difficulty, violations)
    check_labels(by_split, provenance, spec, violations)
    check_families_and_tags(by_split, difficulty, spec, violations)
    check_leakage(pack, by_split, provenance, violations)
    rows = [
        release_task_row(record, world_id=world_id, lane_id=lane_id, split=split)
        for split in SPLIT_NAMES
        for record in by_split[split]
    ]
    check_semantic_leakage(rows, by_split, violations)
    if violations:
        return violations, None
    world = ValidatedWorld(
        pack_dir=pack,
        world_id=world_id,
        lane_id=lane_id,
        spec_version=str(difficulty["spec_version"]),
        provenance=provenance,
        difficulty=difficulty,
        spec=spec,
        tasks=tuple(sorted(rows, key=lambda row: str(row["task_id"]))),
        dev_examples=tuple(
            release_projection(record, world_id=world_id) for record in by_split["dev"]
        ),
        per_split={split: len(by_split[split]) for split in SPLIT_NAMES},
        task_set_sha256=_task_set_sha256(rows),
    )
    return [], world


def validate_pack(pack_dir: Path, repo_root: Path) -> ValidatedWorld:
    """Validate or raise ``PackContractError`` with all violations attached."""
    violations, world = collect_violations(pack_dir, repo_root)
    if violations or world is None:
        raise PackContractError(violations)
    return world


__all__ = ["ValidatedWorld", "collect_violations", "validate_pack"]