"""Pack structure checks: 17-file layout, artifact hashes, spec, world identity."""

from __future__ import annotations

import json
from pathlib import Path

from .contract import (
    DIFFICULTY_FILE,
    PACK_FILES,
    PROVENANCE_FILE,
    WORLD_ID_PATTERN,
    read_json,
    sha256_file,
)
from .errors import Violation


def ignored(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix == ".pyc"


def object_field(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def check_layout(pack: Path, violations: list[Violation]) -> bool:
    present = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and not ignored(path.relative_to(pack))
    }
    complete = True
    for relative in PACK_FILES:
        if relative not in present:
            complete = False
            violations.append(Violation("missing_file", relative, "required pack file absent"))
    for relative in sorted(present - set(PACK_FILES)):
        violations.append(Violation("unexpected_file", relative, "not part of contract v1"))
    return complete


def check_hashes(
    pack: Path, provenance: dict[str, object], violations: list[Violation]
) -> None:
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        violations.append(
            Violation("artifact_hash_missing", PROVENANCE_FILE, "artifacts block absent")
        )
        return
    expected = {relative for relative in PACK_FILES if relative != PROVENANCE_FILE}
    declared_paths = {str(path) for path in artifacts}
    for relative in sorted(expected - declared_paths):
        violations.append(
            Violation("artifact_hash_missing", relative, "not listed in provenance.artifacts")
        )
    for relative in sorted(declared_paths - expected):
        violations.append(
            Violation("artifact_hash_unexpected", relative, "hashed path is not a pack file")
        )
    for relative in sorted(expected & declared_paths):
        declared = artifacts[relative]
        actual = sha256_file(pack / relative)
        if declared != actual:
            violations.append(
                Violation(
                    "artifact_hash_mismatch",
                    relative,
                    f"declared {declared} != on-disk {actual}",
                )
            )


def load_spec(
    repo_root: Path, provenance: dict[str, object], violations: list[Violation]
) -> dict[str, object] | None:
    spec_block = provenance.get("spec")
    if not isinstance(spec_block, dict):
        violations.append(Violation("spec_unresolvable", PROVENANCE_FILE, "spec block absent"))
        return None
    raw_path = spec_block.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        violations.append(Violation("spec_unresolvable", PROVENANCE_FILE, "spec.path absent"))
        return None
    spec_path = Path(raw_path)
    if not spec_path.is_absolute():
        spec_path = repo_root / spec_path
    if not spec_path.is_file():
        violations.append(Violation("spec_unresolvable", raw_path, "spec file not found"))
        return None
    if sha256_file(spec_path) != spec_block.get("sha256"):
        violations.append(Violation("spec_hash_mismatch", raw_path, "spec bytes drifted"))
    if spec_block.get("status") != "frozen":
        violations.append(Violation("spec_not_frozen", raw_path, "spec.status must be 'frozen'"))
    try:
        spec = read_json(spec_path)
    except (ValueError, json.JSONDecodeError) as exc:
        violations.append(Violation("spec_parse_error", raw_path, str(exc)))
        return None
    if spec.get("spec_version") != spec_block.get("spec_version"):
        violations.append(
            Violation("spec_version_mismatch", raw_path, "spec_version != provenance.spec_version")
        )
    taxonomy = spec.get("failure_taxonomy")
    if not isinstance(taxonomy, dict) or not taxonomy:
        violations.append(
            Violation("taxonomy_missing", raw_path, "failure_taxonomy must be a non-empty object")
        )
    return spec


def check_world_identity(
    provenance: dict[str, object],
    difficulty: dict[str, object],
    spec: dict[str, object],
    violations: list[Violation],
) -> tuple[str, str] | None:
    world_id = difficulty.get("world_id")
    if not isinstance(world_id, str) or not WORLD_ID_PATTERN.match(world_id):
        violations.append(
            Violation("world_id_invalid", DIFFICULTY_FILE, f"world_id {world_id!r} is not world_LNN")
        )
        return None
    if provenance.get("world_id") != world_id:
        violations.append(
            Violation(
                "world_id_mismatch",
                PROVENANCE_FILE,
                f"provenance.world_id {provenance.get('world_id')!r} != difficulty {world_id!r}",
            )
        )
    spec_block = provenance.get("spec")
    declared_version = spec_block.get("spec_version") if isinstance(spec_block, dict) else None
    if difficulty.get("spec_version") != declared_version:
        violations.append(
            Violation("spec_version_mismatch", DIFFICULTY_FILE, "difficulty spec_version drifted")
        )
    match = WORLD_ID_PATTERN.match(world_id)
    lane_id = f"L{match.group('lane') if match else ''}"
    spec_provenance = spec.get("provenance")
    if isinstance(spec_provenance, dict) and spec_provenance.get("lane_id") not in (None, lane_id):
        violations.append(Violation("lane_id_mismatch", "spec.provenance", f"lane_id != {lane_id}"))
    return world_id, lane_id