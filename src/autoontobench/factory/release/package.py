"""Release packaging: public + private packages with manifests (owner: S11).

FINAL-BUILD-ORDER-v2 §6.3/§7: public packages contain agent-visible task
records, attribution, license matrix and a clean-room rebuild command; private
packages (oracle/evidence) stay physically disjoint and owner-only (0700).

Safety order of operations (public):
1. stage in a sibling temp dir
2. strip private_* keys from every task row (contracts.models stripper)
3. assert_no_private_keys + leak_audit on the staged tree
4. write MANIFEST.sha256 / MANIFEST.json
5. atomic rename into place — a failed gate never leaves a partial release
"""
from __future__ import annotations

import json
import re
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import _strip_private, assert_no_private_keys
from autoontobench.factory.gates.license_matrix import LicenseMatrix
from autoontobench.factory.gates.pipeline import gate_release_eligible, load_build
from autoontobench.factory.release.manifest import Manifest, build_manifest, write_manifest
from autoontobench.safety.leak_audit import scan_public_root
from autoontobench.safety.private_store import seal_permissions

REBUILD_COMMAND = (
    "python -m autoontobench source fetch --all --pinned --resume --max-sustainable && "
    "python -m autoontobench bank build --config configs/bank_8000.yaml "
    "--all-lanes --resume --max-sustainable --no-agent-evaluation && "
    "python -m autoontobench bank audit --build-id {build_id} --strict && "
    "python -m autoontobench bank package --build-id {build_id} --release {kind}"
)


class PackagingError(RuntimeError):
    """Raised when a safety gate blocks packaging; no partial release remains."""


@dataclass(frozen=True)
class PackageResult:
    build_id: str
    kind: str
    root: Path
    file_count: int
    root_sha256: str


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


PUBLIC_TASK_KEYS = (
    "schema_version", "dataset_id", "lane_id", "family_id", "base_goal_id",
    "instance_id", "task_id", "task_regime", "complexity", "semantic_profile",
    "public", "oracle_handle", "status", "qualification",
    "source_snapshot_ids", "source_unit_ids", "difficulty",
)

_HOST_PREFIXES = ("/mnt/", "/home/", "/root/", "/tmp/")
_SHA1_RE = re.compile(r"\b[0-9a-f]{40}\b")


def _sanitize_text(text: str) -> str:
    for prefix in _HOST_PREFIXES:
        text = text.replace(prefix, "<host>/")
    return _SHA1_RE.sub("<sha1>", text)


def _sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize_text(obj)
    if isinstance(obj, list):
        return [_sanitize_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_obj(v) for k, v in obj.items()}
    return obj


def _load_task_rows(build_dir: Path) -> list[dict[str, Any]]:
    artifacts = load_build(build_dir)
    return list(artifacts.qualified or artifacts.task_manifest or artifacts.candidates)


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    clean = _strip_private(row)
    assert_no_private_keys(clean)
    projection = {k: clean[k] for k in PUBLIC_TASK_KEYS if k in clean}
    return _sanitize_obj(projection)


def _project_task_rows(build_dir: Path) -> list[dict[str, Any]]:
    return [_project_row(row) for row in _load_task_rows(build_dir)]


def _release_meta(build_id: str, kind: str, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "schema": "aob-release/1",
        "build_id": build_id,
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "rebuild_command": REBUILD_COMMAND.format(build_id=build_id, kind=kind),
        "notes": (
            "Public package: agent-visible task projections only; gold/oracle "
            "values live exclusively in the private release (physically disjoint)."
            if kind == "public"
            else "Private package: oracle/evidence material. Owner-only modes (0700)."
        ),
    }


def _stage_and_seal(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(stage, target)


@dataclass(frozen=True)
class LicensePreview:
    total: int
    eligible: int
    excluded: int
    held_sources: dict[str, int]


def _license_filter(rows: list[dict[str, Any]],
                    matrix: LicenseMatrix) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    from collections import Counter

    kept: list[dict[str, Any]] = []
    held: Counter[str] = Counter()
    for row in rows:
        verdict = gate_release_eligible(row, matrix)
        if verdict.passed:
            kept.append(row)
            continue
        sources = (verdict.detail or {}).get("held_sources") or ["<no-source>"]
        for source_id in sources:
            held[source_id] += 1
    return kept, len(rows) - len(kept), dict(sorted(held.items()))


def preview_public(build_dir: Path, *, license_matrix: LicenseMatrix) -> LicensePreview:
    """Count public-release eligibility without writing a release."""
    rows = _load_task_rows(build_dir)
    kept, excluded, held = _license_filter(rows, license_matrix)
    return LicensePreview(len(rows), len(kept), excluded, held)


def package_public(build_dir: Path, releases_root: Path, *, build_id: str,
                   private_root: Path | None = None,
                   license_matrix: LicenseMatrix | None = None,
                   attribution_dir: Path | None = None,
                   matrix_csv: Path | None = None) -> PackageResult:
    """Stage + gate + seal the public package for `build_id`.

    When `license_matrix` is given, only release-eligible tasks are staged and
    a matrix with no eligible rows raises PackagingError (never an empty or
    unrestricted public release).
    """
    raw_rows = _load_task_rows(build_dir)
    license_excluded = 0
    if license_matrix is not None:
        kept, license_excluded, _held = _license_filter(raw_rows, license_matrix)
        if not kept:
            raise PackagingError(
                "license gate blocked public release: 0 release-eligible tasks "
                "(provide PI-approved source_redistribution_matrix.csv)")
        rows = [_project_row(row) for row in kept]
    else:
        rows = [_project_row(row) for row in raw_rows]
    target = releases_root / "public" / build_id
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage-public-", dir=target.parent))
    try:
        count = _write_jsonl(staging / "tasks" / "qualified_tasks.jsonl", rows)
        if attribution_dir is not None and attribution_dir.is_dir():
            for path in sorted(attribution_dir.rglob("*")):
                if not path.is_file():
                    continue
                dest = staging / "release_obligations" / path.relative_to(attribution_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() in (".md", ".json", ".csv", ".txt"):
                    dest.write_text(_sanitize_text(path.read_text(encoding="utf-8")),
                                    encoding="utf-8")
                else:
                    shutil.copyfile(path, dest)
        if matrix_csv is not None and matrix_csv.is_file():
            dest = staging / "licenses" / "source_redistribution_matrix.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_sanitize_text(matrix_csv.read_text(encoding="utf-8")),
                            encoding="utf-8")
        counts = {"qualified_tasks": count}
        if license_matrix is not None:
            counts["license_excluded_tasks"] = license_excluded
        (staging / "RELEASE.json").write_text(
            json.dumps(_release_meta(build_id, "public", counts),
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # safety gates on the staged tree BEFORE sealing
        leaked = [f for f in scan_public_root(staging, private_root) if f.verdict == "leaked"]
        if leaked:
            raise PackagingError(
                "leak gate blocked public release: "
                + "; ".join(f"{f.path}:{f.item}" for f in leaked[:5])
            )
        manifest = build_manifest(staging)
        public_manifest = Manifest(root=".", entries=manifest.entries,
                                   root_sha256=manifest.root_sha256)
        write_manifest(staging, public_manifest)
        _stage_and_seal(staging, target)
        return PackageResult(build_id, "public", target, len(public_manifest.entries),
                             public_manifest.root_sha256)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def package_private(build_dir: Path, releases_root: Path, *, build_id: str) -> PackageResult:
    """Private oracle package; tree sealed to owner-only after staging."""
    artifacts = load_build(build_dir)
    target = releases_root / "private" / build_id
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage-private-", dir=target.parent))
    try:
        counts = {
            "native_evidence": _write_jsonl(
                staging / "oracle" / "native_evidence_index.jsonl",
                list(artifacts.evidence_index.values()),
            ),
            "rejections": _write_jsonl(
                staging / "oracle" / "rejections.jsonl", list(artifacts.rejections)
            ),
        }
        provenance_path = build_dir / "source_provenance.jsonl"
        if provenance_path.exists():
            dest = staging / "oracle" / "source_provenance.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(provenance_path, dest)
            counts["source_provenance_rows"] = sum(
                1 for _ in dest.open("r", encoding="utf-8")
            )
        (staging / "RELEASE.json").write_text(
            json.dumps(_release_meta(build_id, "private", counts), indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest: Manifest = build_manifest(staging)
        write_manifest(staging, manifest)
        seal_permissions(staging)
        _stage_and_seal(staging, target)
        seal_permissions(target)  # os.replace can preserve staging umask effects
        return PackageResult(build_id, "private", target, len(manifest.entries),
                             manifest.root_sha256)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def package_release(build_dir: Path, releases_root: Path, *, build_id: str,
                    kind: str, private_root: Path | None = None,
                    license_matrix: LicenseMatrix | None = None,
                    attribution_dir: Path | None = None,
                    matrix_csv: Path | None = None) -> PackageResult:
    if kind == "public":
        return package_public(build_dir, releases_root, build_id=build_id,
                              private_root=private_root,
                              license_matrix=license_matrix,
                              attribution_dir=attribution_dir,
                              matrix_csv=matrix_csv)
    if kind == "private":
        return package_private(build_dir, releases_root, build_id=build_id)
    raise ValueError(f"unknown release kind: {kind}")
