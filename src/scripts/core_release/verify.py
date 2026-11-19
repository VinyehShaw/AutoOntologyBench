"""Verify an assembled ``release_core`` tree against its own manifest."""

from __future__ import annotations

from pathlib import Path

from .contract import (
    GOLD_ARTIFACTS,
    RELEASE_TASK_FIELDS,
    SPLIT_NAMES,
    canonical_json,
    jsonl_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from .errors import ReleaseError

_FORBIDDEN_NAMES = frozenset({Path(path).name for path in GOLD_ARTIFACTS}) | {
    "world_rules.py",
    "corruption_manifest.json",
    "dev_labels.jsonl",
    "hidden_iid.jsonl",
    "hidden_comp.jsonl",
    "hidden_transfer.jsonl",
}
_MANIFEST_NAMES = frozenset({"MANIFEST.json", "MANIFEST.sha256"})


def _tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            if relative not in _MANIFEST_NAMES:
                hashes[relative] = sha256_file(path)
    return hashes


def _forbidden_paths(root: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"symlink:{relative}")
        if "private" in path.relative_to(root).parts:
            problems.append(f"private_path:{relative}")
        if path.name in _FORBIDDEN_NAMES:
            problems.append(f"forbidden_file:{relative}")
    return problems


def _task_set_sha256(rows: list[dict[str, object]]) -> str:
    ordered = sorted(rows, key=lambda row: str(row["task_id"]))
    body = "".join(f"{canonical_json(row)}\n" for row in ordered)
    return sha256_bytes(body.encode("utf-8"))


def verify_release(out_dir: Path) -> list[str]:
    """Return every release-integrity problem (``[]`` means verified)."""
    root = Path(out_dir)
    if not (root / "MANIFEST.json").is_file():
        raise ReleaseError("missing_manifest", f"{root}/MANIFEST.json not found")
    try:
        manifest = read_json(root / "MANIFEST.json")
    except (ValueError, OSError) as exc:
        return [f"manifest_unreadable:{exc}"]
    problems: list[str] = []
    if manifest.get("schema") is None:
        problems.append("manifest_schema_missing")
    problems.extend(_forbidden_paths(root))
    sha_path = root / "MANIFEST.sha256"
    if not sha_path.is_file():
        return problems + ["missing:MANIFEST.sha256"]
    body = sha_path.read_text(encoding="utf-8")
    if manifest.get("root_sha256") != sha256_bytes(body.encode("utf-8")):
        problems.append("root_digest_mismatch")
    declared: dict[str, str] = {}
    for line in body.splitlines():
        digest, _, relative = line.partition("  ")
        if not relative:
            problems.append(f"malformed_manifest_line:{line!r}")
            continue
        declared[relative] = digest
    disk = _tree_hashes(root)
    for relative, digest in sorted(declared.items()):
        if relative not in disk:
            problems.append(f"missing:{relative}")
        elif disk[relative] != digest:
            problems.append(f"hash_mismatch:{relative}")
    for relative in sorted(set(disk) - set(declared)):
        problems.append(f"unlisted_file:{relative}")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != len(declared):
        problems.append("manifest_file_table_mismatch")
    elif isinstance(manifest_files, list):
        table = {
            str(entry.get("path")): entry.get("sha256")
            for entry in manifest_files
            if isinstance(entry, dict)
        }
        if table != declared:
            problems.append("manifest_file_table_mismatch")
    if manifest.get("file_count") != len(declared):
        problems.append("file_count_mismatch")
    tasks_path = root / "tasks/tasks.jsonl"
    worlds_path = root / "tasks/worlds.jsonl"
    if not tasks_path.is_file() or not worlds_path.is_file():
        return problems + ["missing:tasks"]
    tasks = read_jsonl(tasks_path)
    worlds = read_jsonl(worlds_path)
    problems.extend(_verify_tasks(tasks, manifest))
    problems.extend(_verify_worlds(root, tasks, worlds, manifest))
    return problems


def _verify_tasks(tasks: list[dict[str, object]], manifest: dict[str, object]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    per_split = {split: 0 for split in SPLIT_NAMES}
    for index, task in enumerate(tasks, 1):
        if set(task) != set(RELEASE_TASK_FIELDS):
            problems.append(f"task_fields:{index}:{sorted(set(task) ^ set(RELEASE_TASK_FIELDS))}")
        release_id = str(task.get("release_task_id"))
        if release_id in seen:
            problems.append(f"duplicate_release_task_id:{release_id}")
        seen.add(release_id)
        split = task.get("split")
        if split in per_split:
            per_split[str(split)] += 1
        else:
            problems.append(f"task_split_invalid:{release_id}:{split!r}")
        expected_id = f"{task.get('world_id')}:{task.get('task_id')}"
        if release_id != expected_id:
            problems.append(f"release_task_id_mismatch:{release_id}")
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        return problems + ["manifest_totals_missing"]
    if totals.get("tasks") != len(tasks):
        problems.append(f"task_count_mismatch:{totals.get('tasks')}!={len(tasks)}")
    if totals.get("per_split") != per_split:
        problems.append(f"split_totals_mismatch:{totals.get('per_split')}!={per_split}")
    return problems


def _verify_worlds(
    root: Path,
    tasks: list[dict[str, object]],
    worlds: list[dict[str, object]],
    manifest: dict[str, object],
) -> list[str]:
    problems: list[str] = []
    expected_worlds = manifest.get("worlds")
    if expected_worlds != worlds:
        problems.append("manifest_worlds_mismatch")
    by_world: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        by_world.setdefault(str(task.get("world_id")), []).append(task)
    for row in worlds:
        world_id = str(row.get("world_id"))
        rows = by_world.pop(world_id, [])
        if row.get("task_count") != len(rows):
            problems.append(f"world_task_count_mismatch:{world_id}")
        metadata_path = root / str(row.get("metadata_path"))
        if not metadata_path.is_file():
            problems.append(f"missing_metadata:{world_id}")
            continue
        try:
            metadata = read_json(metadata_path)
        except (ValueError, OSError) as exc:
            problems.append(f"metadata_unreadable:{world_id}:{exc}")
            continue
        if metadata.get("task_set_sha256") != _task_set_sha256(rows):
            problems.append(f"task_set_digest_mismatch:{world_id}")
        counts = metadata.get("task_counts")
        declared_splits = counts.get("per_split") if isinstance(counts, dict) else None
        if declared_splits != row.get("per_split"):
            problems.append(f"world_split_mismatch:{world_id}")
        public_dir = root / "worlds" / world_id / "public"
        if not public_dir.is_dir():
            problems.append(f"missing_public_dir:{world_id}")
        artifacts = metadata.get("artifacts")
        if isinstance(artifacts, dict):
            for relative, digest in artifacts.items():
                path = root / "worlds" / world_id / str(relative)
                if not path.is_file() or sha256_file(path) != digest:
                    problems.append(f"public_artifact_mismatch:{world_id}:{relative}")
        sanitized = metadata.get("sanitized_files")
        if isinstance(sanitized, dict):
            for relative, digest in sanitized.items():
                path = root / "worlds" / world_id / str(relative)
                if not path.is_file() or sha256_file(path) != digest:
                    problems.append(f"sanitized_file_mismatch:{world_id}:{relative}")
    for world_id in sorted(by_world):
        problems.append(f"task_world_not_indexed:{world_id}")
    totals = manifest.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    if totals.get("worlds") != len(worlds):
        problems.append("world_count_mismatch")
    if jsonl_bytes(tasks) != (root / "tasks/tasks.jsonl").read_bytes():
        problems.append("tasks_not_canonical")
    return problems


__all__ = ["verify_release"]