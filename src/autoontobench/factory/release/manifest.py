"""Release manifest: per-file sha256 + a single root digest (owner: S11).

Format (deterministic, sorted, UTF-8):
    MANIFEST.sha256     lines: "<sha256>  <relative/path>"
    MANIFEST.json       metadata + root digest + file table
The root digest is sha256 over the MANIFEST.sha256 byte content, so any
file mutation anywhere in the tree changes it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileEntry:
    path: str  # posix, relative to the release root
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Manifest:
    root: str
    entries: tuple[FileEntry, ...]
    root_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "root_sha256": self.root_sha256,
            "file_count": len(self.entries),
            "files": [{"path": e.path, "sha256": e.sha256, "bytes": e.bytes}
                      for e in self.entries],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


MANIFEST_FILENAMES: frozenset[str] = frozenset({"MANIFEST.sha256", "MANIFEST.json"})


def build_manifest(root: Path) -> Manifest:
    """Hash every regular file under root (excluding manifest files themselves)."""
    entries: list[FileEntry] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink() or name in MANIFEST_FILENAMES:
                continue
            rel = path.relative_to(root).as_posix()
            entries.append(FileEntry(rel, _sha256_file(path), path.stat().st_size))
    entries.sort(key=lambda e: e.path)
    body = "".join(f"{e.sha256}  {e.path}\n" for e in entries)
    root_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return Manifest(root=str(root), entries=tuple(entries), root_sha256=root_sha)


def write_manifest(root: Path, manifest: Manifest) -> tuple[Path, Path]:
    sha_path = root / "MANIFEST.sha256"
    json_path = root / "MANIFEST.json"
    body = "".join(f"{e.sha256}  {e.path}\n" for e in manifest.entries)
    sha_path.write_text(body, encoding="utf-8")
    json_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha_path, json_path


def verify_manifest(root: Path, manifest: Manifest) -> list[str]:
    """Recompute all hashes; returns a list of drift descriptions ([] = clean)."""
    problems: list[str] = []
    for entry in manifest.entries:
        path = root / entry.path
        if not path.exists():
            problems.append(f"missing:{entry.path}")
            continue
        actual = _sha256_file(path)
        if actual != entry.sha256:
            problems.append(f"hash_drift:{entry.path}")
    on_disk = build_manifest(root)
    if on_disk.root_sha256 != manifest.root_sha256:
        missing = {e.path for e in on_disk.entries} - {e.path for e in manifest.entries}
        for extra in sorted(missing):
            problems.append(f"unlisted_file:{extra}")
    return problems
