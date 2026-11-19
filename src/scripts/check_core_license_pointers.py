#!/usr/bin/env python3
"""License-pointer check for the Core-track benchmark release.

Proves that every world license reference resolves inside the release or is
explicitly marked not shipped:

1. The top-level license files (LICENSE, LICENSE-DATA.md, NOTICE,
   LICENSES/README.md, LICENSES/ATTRIBUTION.md, the CC BY 4.0 legal code) exist
   on disk and are listed in MANIFEST.json's file table.
2. Every ``worlds/<world_id>/metadata.json`` carries a
   ``license_attribution`` block; each reference is either ``shipped`` (all
   listed paths exist on disk and in the manifest) or ``not_shipped`` (with a
   non-empty reason).
3. Every ``sources/registry.d/*.yaml`` token in a world's ``license`` string
   (brace expansion included) and every
   ``metadata.source_snapshot.registry_receipts`` entry appears among the
   declared reference pointers, and a ``source_snapshot`` pointer is covered.
4. ``LICENSES/ATTRIBUTION.md`` mentions every assembled world.

Exit 0 = PASS; exit 1 prints every problem.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_release.contract import read_json  # noqa: E402

_REGISTRY_TOKEN = re.compile(r"sources/registry\.d/(?P<name>\{[^}]+\}|[\w-]+)\.yaml")

_REQUIRED_FILES: tuple[str, ...] = (
    "LICENSE",
    "LICENSE-DATA.md",
    "NOTICE",
    "LICENSES/README.md",
    "LICENSES/ATTRIBUTION.md",
    "LICENSES/legalcode/CC-BY-4.0.txt",
)


def _expand_braces(token: str) -> list[str]:
    match = _REGISTRY_TOKEN.search(token)
    if match is None:
        return []
    name = match.group("name")
    names = name[1:-1].split(",") if name.startswith("{") else [name]
    return [f"sources/registry.d/{part}.yaml" for part in names]


def _check_world(
    root: Path, metadata_path: Path, manifested: set[str], problems: list[str]
) -> None:
    world_id = metadata_path.parent.name
    try:
        metadata = read_json(metadata_path)
    except (ValueError, OSError) as exc:
        problems.append(f"{world_id}: metadata unreadable: {exc}")
        return
    block = metadata.get("license_attribution")
    if not isinstance(block, dict):
        problems.append(f"{world_id}: license_attribution missing")
        return
    references = block.get("references")
    if not isinstance(references, list):
        problems.append(f"{world_id}: license_attribution.references missing")
        return
    for path in block.get("bundle_paths") or []:
        target = root / str(path)
        if not target.is_file():
            problems.append(f"{world_id}: bundle path missing on disk: {path}")
        elif str(path) not in manifested:
            problems.append(f"{world_id}: bundle path not in MANIFEST: {path}")
    pointers: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict):
            problems.append(f"{world_id}: malformed reference {reference!r}")
            continue
        pointer = str(reference.get("pointer"))
        status = reference.get("status")
        pointers.add(pointer)
        if status == "shipped":
            paths = reference.get("paths")
            if not paths:
                problems.append(f"{world_id}: shipped reference has no paths: {pointer}")
                continue
            for path in paths:
                if not (root / str(path)).is_file():
                    problems.append(f"{world_id}: shipped path missing on disk: {path}")
                elif str(path) not in manifested:
                    problems.append(f"{world_id}: shipped path not in MANIFEST: {path}")
        elif status == "not_shipped":
            if not str(reference.get("reason") or "").strip():
                problems.append(f"{world_id}: not_shipped reference lacks a reason: {pointer}")
        else:
            problems.append(f"{world_id}: reference {pointer} has invalid status {status!r}")
    license_string = str(metadata.get("license") or "")
    for token in _expand_braces(license_string):
        if token not in pointers:
            problems.append(f"{world_id}: license-string reference unresolved: {token}")
    snapshot = metadata.get("source_snapshot")
    receipts = snapshot.get("registry_receipts") if isinstance(snapshot, dict) else None
    for receipt in receipts or []:
        if str(receipt) not in pointers:
            problems.append(f"{world_id}: registry receipt unresolved: {receipt}")
    if "source_snapshot" in license_string and not any(
        pointer.startswith("source_snapshot") for pointer in pointers
    ):
        problems.append(f"{world_id}: source_snapshot pointer uncovered")


def check_license_pointers(root: Path) -> list[str]:
    """Return every license-layer problem in the release (``[]`` means PASS)."""
    problems: list[str] = []
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        return ["MANIFEST.json missing"]
    try:
        manifest = read_json(manifest_path)
    except (ValueError, OSError) as exc:
        return [f"MANIFEST.json unreadable: {exc}"]
    files = manifest.get("files")
    manifested = {
        str(entry.get("path")) for entry in files or [] if isinstance(entry, dict)
    }
    for relative in _REQUIRED_FILES:
        if not (root / relative).is_file():
            problems.append(f"missing on disk: {relative}")
        elif relative not in manifested:
            problems.append(f"not in MANIFEST: {relative}")
    license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")[:400]
    if "Apache License" not in license_text:
        problems.append("LICENSE does not start with the Apache-2.0 text")
    worlds_dir = root / "worlds"
    metadata_paths = sorted(worlds_dir.glob("world_*/metadata.json"))
    if not metadata_paths:
        problems.append("no world metadata found")
    for metadata_path in metadata_paths:
        _check_world(root, metadata_path, manifested, problems)
    attribution_path = root / "LICENSES/ATTRIBUTION.md"
    if attribution_path.is_file():
        attribution = attribution_path.read_text(encoding="utf-8")
        for metadata_path in metadata_paths:
            world_id = metadata_path.parent.name
            if world_id not in attribution:
                problems.append(f"ATTRIBUTION.md does not mention {world_id}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="release_core", metavar="DIR", help="release root")
    args = parser.parse_args(argv)
    problems = check_license_pointers(Path(args.out))
    if problems:
        print(f"FAIL {args.out}: {len(problems)} license-pointer problem(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"PASS {args.out}: every world license reference resolves or is marked not shipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
