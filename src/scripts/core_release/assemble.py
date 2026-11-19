"""Assemble validated Core-track world packs into ``release_core/``.

The release is benchmark-only: label-stripped task records, each pack's public
materials, per-world metadata, and hash manifests. Private oracle files are
never read into the output tree.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contract import (
    DIFFICULTY_FILE,
    PACK_CONTRACT_VERSION,
    PROVENANCE_FILE,
    PUBLIC_FILES,
    SCHEMA_RELEASE,
    SCHEMA_WORLD_INDEX,
    SCHEMA_WORLD_METADATA,
    SPLIT_NAMES,
    json_bytes,
    jsonl_bytes,
    opaque_release_id,
    repo_relative,
    sha256_bytes,
    sha256_file,
    tree_entries,
)
from . import licenses
from .docs_templates import TARGET_TASK_COUNT, render_contract, render_readme
from .errors import PackContractError, ReleaseError
from .validate import ValidatedWorld, collect_violations
from .verify import verify_release

_MANIFEST_EXCLUDES = frozenset({"MANIFEST.json", "MANIFEST.sha256"})


def _split_floors(world: ValidatedWorld) -> tuple[dict[str, int], dict[str, bool], str]:
    counts = world.difficulty.get("task_counts")
    declared = counts.get("minimums_required") if isinstance(counts, dict) else None
    if not isinstance(declared, dict):
        return {}, {}, "NOT_DECLARED"
    required = {split: int(declared.get(split, 0)) for split in SPLIT_NAMES}
    met = {split: world.per_split[split] >= required[split] for split in SPLIT_NAMES}
    return required, met, "MET" if all(met.values()) else "PARTIAL"


def _world_metadata(
    world: ValidatedWorld, pack_path: str, sanitized_files: dict[str, str]
) -> dict[str, object]:
    provenance = world.provenance
    difficulty = world.difficulty
    required, met, floors_status = _split_floors(world)
    artifacts = provenance.get("artifacts")
    artifact_hashes = artifacts if isinstance(artifacts, dict) else {}
    counts = difficulty.get("task_counts")
    task_counts = counts if isinstance(counts, dict) else {}
    families = difficulty.get("template_families")
    family_block = families if isinstance(families, dict) else {}
    family_names = family_block.get("families")
    projected_families = (
        [
            opaque_release_id(
                "family",
                str(name),
                spec_version=world.spec_version,
                world_id=world.world_id,
            )
            for name in family_names
        ]
        if isinstance(family_names, list)
        else []
    )
    structural_transfer = family_block.get("structural_transfer")
    projected_structural_transfer = (
        opaque_release_id(
            "family",
            structural_transfer,
            spec_version=world.spec_version,
            world_id=world.world_id,
        )
        if isinstance(structural_transfer, str)
        else structural_transfer
    )
    return {
        "schema": SCHEMA_WORLD_METADATA,
        "world_id": world.world_id,
        "lane_id": world.lane_id,
        "spec_version": world.spec_version,
        "pack": {
            "path": pack_path,
            "pack_revision": provenance.get("pack_revision"),
            "provenance_sha256": sha256_file(world.pack_dir / PROVENANCE_FILE),
            "difficulty_sha256": sha256_file(world.pack_dir / DIFFICULTY_FILE),
            "spec": provenance.get("spec"),
        },
        "source_snapshot": provenance.get("source_snapshot"),
        "source_data": provenance.get("source_data"),
        "license": provenance.get("license"),
        "license_attribution": licenses.world_license_attribution(world.world_id),
        "human_review": provenance.get("human_review"),
        "task_counts": {
            "total": task_counts.get("total"),
            "per_split": dict(world.per_split),
            "scenarios": task_counts.get("scenarios"),
            "paraphrase_variants_per_scenario": task_counts.get(
                "paraphrase_variants_per_scenario"
            ),
        },
        "template_families": {
            "families": projected_families,
            "count": family_block.get("count"),
            "structural_transfer": projected_structural_transfer,
            "note": (
                "Release-projected opaque family IDs (fam_<16 hex>). The original "
                "family names and the name-to-ID mapping remain in the private pack."
            ),
        },
        "split_floors": {
            "required": required,
            "met": met,
            "status": floors_status,
        },
        "artifacts": {
            relative: artifact_hashes[relative]
            for relative in PUBLIC_FILES
            if relative in artifact_hashes and relative not in sanitized_files
        },
        "sanitized_files": dict(sanitized_files),
        "release_projection": {
            "sanitized": True,
            "note": (
                "Task rows and public/examples_dev.jsonl are the release "
                "projection of the pack records: verdict-namespace feature "
                "tokens (polarity:*, canonical/counterfactual variant naming) "
                "are removed, polarity-encoding prompt phrasings are "
                "neutralized, and the label-semantic identity fields "
                "(template, template_family, family: features, actions[].variant) "
                "are replaced with deterministic opaque release IDs "
                "(tpl_/fam_/var_<16 hex>). The original names and their "
                "mapping remain in the private pack; labels are not shipped. "
                "All other public files are byte-identical pack copies (see "
                "artifacts)."
            ),
        },
        "task_set_sha256": world.task_set_sha256,
        "oracle": {
            "included": False,
            "note": (
                "Private oracle material (labels, gold ontology/shapes, corruption "
                "manifest, world_rules.py) is not part of this release."
            ),
        },
    }


def _index_row(
    world: ValidatedWorld, pack_path: str, metadata: dict[str, object]
) -> dict[str, object]:
    provenance = world.provenance
    snapshot = provenance.get("source_snapshot")
    families = metadata.get("template_families")
    family_count = families.get("count") if isinstance(families, dict) else None
    floors = metadata.get("split_floors")
    floors_status = floors.get("status") if isinstance(floors, dict) else "UNKNOWN"
    return {
        "schema": SCHEMA_WORLD_INDEX,
        "world_id": world.world_id,
        "lane_id": world.lane_id,
        "spec_version": world.spec_version,
        "pack_path": pack_path,
        "task_count": len(world.tasks),
        "per_split": dict(world.per_split),
        "template_families": family_count,
        "split_floors_status": floors_status,
        "source_snapshot_id": snapshot.get("id") if isinstance(snapshot, dict) else None,
        "license": provenance.get("license"),
        "task_set_sha256": world.task_set_sha256,
        "metadata_path": f"worlds/{world.world_id}/metadata.json",
    }


def _check_unique(worlds: list[ValidatedWorld]) -> None:
    seen_worlds: set[str] = set()
    seen_tasks: set[str] = set()
    world_duplicates: list[str] = []
    task_duplicates: list[str] = []
    for world in worlds:
        if world.world_id in seen_worlds:
            world_duplicates.append(f"duplicate_world_id:{world.world_id}")
        seen_worlds.add(world.world_id)
        for task in world.tasks:
            release_id = str(task["release_task_id"])
            if release_id in seen_tasks:
                task_duplicates.append(f"duplicate_release_task_id:{release_id}")
            seen_tasks.add(release_id)
    if world_duplicates or task_duplicates:
        code = "duplicate_world_id" if world_duplicates else "duplicate_release_task_id"
        raise ReleaseError(code, "; ".join(world_duplicates + task_duplicates))


def _guard_output(out: Path, worlds: list[ValidatedWorld], repo_root: Path, force: bool) -> None:
    resolved = out.resolve()
    if resolved == repo_root.resolve() or resolved.parent == resolved:
        raise ReleaseError("release_path_invalid", f"refusing to assemble into {resolved}")
    for world in worlds:
        if resolved == world.pack_dir.resolve():
            raise ReleaseError("release_path_invalid", "output directory is an input pack")
    if resolved.exists() and any(resolved.iterdir()) and not force:
        raise ReleaseError("release_exists", f"{resolved} is non-empty; pass --force")


def _write_release(
    staging: Path, worlds: list[ValidatedWorld], repo_root: Path
) -> dict[str, object]:
    staging.mkdir(parents=True, exist_ok=True)
    ordered = sorted(worlds, key=lambda world: world.world_id)
    tasks = [task for world in ordered for task in world.tasks]
    index_rows: list[dict[str, object]] = []
    for world in ordered:
        world_dir = staging / "worlds" / world.world_id
        (world_dir / "public").mkdir(parents=True)
        for relative in PUBLIC_FILES:
            if relative == "public/examples_dev.jsonl":
                continue  # re-projected (sanitized) below, not byte-copied
            shutil.copyfile(world.pack_dir / relative, world_dir / relative)
        examples_bytes = jsonl_bytes(list(world.dev_examples))
        (world_dir / "public/examples_dev.jsonl").write_bytes(examples_bytes)
        pack_path = repo_relative(world.pack_dir, repo_root)
        sanitized_files = {"public/examples_dev.jsonl": sha256_bytes(examples_bytes)}
        metadata = _world_metadata(world, pack_path, sanitized_files)
        (world_dir / "metadata.json").write_bytes(json_bytes(metadata))
        index_rows.append(_index_row(world, pack_path, metadata))
    (staging / "tasks").mkdir()
    (staging / "tasks/tasks.jsonl").write_bytes(jsonl_bytes(tasks))
    (staging / "tasks/worlds.jsonl").write_bytes(jsonl_bytes(index_rows))
    per_split = {
        split: sum(world.per_split[split] for world in ordered) for split in SPLIT_NAMES
    }
    (staging / "README.md").write_text(
        render_readme(index_rows, per_split, len(tasks)), encoding="utf-8"
    )
    (staging / "CORE_RELEASE_CONTRACT.md").write_text(render_contract(), encoding="utf-8")
    licenses.write_license_layer(staging, ordered, repo_root)
    entries = tree_entries(staging, _MANIFEST_EXCLUDES)
    body = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
    (staging / "MANIFEST.sha256").write_text(body, encoding="utf-8")
    manifest = {
        "schema": SCHEMA_RELEASE,
        "release": "core-track-benchmark",
        "release_status": "PACKAGING-FRAMEWORK",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": {
            "claim": "benchmark-only packaging of validated Core-track world packs",
            "is_full_core_track_release": False,
            "target_task_count": TARGET_TASK_COUNT,
            "assembled_world_count": len(ordered),
            "assembled_task_count": len(tasks),
            "note": (
                "Not the full 19,947-task Core-track release; these are the real "
                "assembled counts and every record is label-stripped."
            ),
        },
        "worlds": index_rows,
        "totals": {"worlds": len(ordered), "tasks": len(tasks), "per_split": per_split},
        "license": licenses.manifest_license_block(),
        "validation": {
            "pack_contract_version": PACK_CONTRACT_VERSION,
            "status": "PASS",
            "violations": [],
        },
        "file_count": len(entries),
        "root_sha256": sha256_bytes(body.encode("utf-8")),
        "files": entries,
    }
    (staging / "MANIFEST.json").write_bytes(json_bytes(manifest))
    return manifest


def assemble_release(
    packs: list[Path],
    out_dir: Path,
    repo_root: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Validate every pack, then write and verify a complete release tree."""
    if not packs:
        raise ReleaseError("missing_pack", "at least one --pack is required")
    repo_root = Path(repo_root)
    worlds: list[ValidatedWorld] = []
    for pack in packs:
        violations, world = collect_violations(Path(pack), repo_root)
        if violations or world is None:
            raise PackContractError(violations)
        worlds.append(world)
    _check_unique(worlds)
    for world in worlds:
        licenses.resolve_world(world.world_id)
    out = Path(out_dir)
    _guard_output(out, worlds, repo_root, force)
    resolved = out.resolve()
    staging = resolved.parent / f".{resolved.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        manifest = _write_release(staging, worlds, repo_root)
        problems = verify_release(staging)
        if problems:
            raise ReleaseError("manifest_drift", "; ".join(problems))
        if resolved.exists():
            shutil.rmtree(resolved)
        staging.replace(resolved)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = ["assemble_release"]