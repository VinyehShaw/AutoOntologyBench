"""L15 source plugin: verify the pinned Reactome release-98 snapshot and build
the local truth store (SQLite) from it.

``pin_and_index`` verifies per-file SHA256 against the pinlock manifest and
(re)builds the store when missing. The store is the ONLY truth source for
task generation and native verification; the live ContentService was used
solely at acquisition time to pin the release version.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import IndexManifest, SourceSnapshot, SourceUnit
from source_registry.fetchers.l15_reactome import (
    INDEX_ROOT, PINNED_ROOT, PINNED_FILES, RELEASE_ID, SOURCE_ID,
    ReactomeSource, read_sha256sums,
)

STORE_DB = INDEX_ROOT / "reactome.db"
STORE_MANIFEST = INDEX_ROOT / "manifest.json"


class L15SourcePlugin:
    source_id = SOURCE_ID

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else INDEX_ROOT

    # -- §3.2 SourcePlugin facade -------------------------------------------

    def pin_and_index(self, source_config: dict[str, Any] | None = None) -> SourceSnapshot:
        cfg = source_config or {}
        src = ReactomeSource()
        snapshot = SourceSnapshot(
            source_id=SOURCE_ID, release_id=RELEASE_ID, root=PINNED_ROOT,
            descriptor=src.describe())
        missing = [n for n in PINNED_FILES if not (PINNED_ROOT / n).is_file()]
        if missing:
            raise FileNotFoundError(f"pinned files missing: {missing}")
        sums = read_sha256sums()
        bad = [n for n in PINNED_FILES if n not in sums]
        if bad:
            raise FileNotFoundError(f"SHA256SUMS incomplete for: {bad}")
        if cfg.get("rebuild_index") or not STORE_DB.is_file():
            from benchmark.plugins.source.l15.store import build_store
            build_store(PINNED_ROOT, self.index_dir)
        return snapshot

    def enumerate_units(self, snapshot: SourceSnapshot | None = None) -> Iterator[SourceUnit]:
        src = ReactomeSource()
        snap = snapshot or SourceSnapshot(
            source_id=SOURCE_ID, release_id=RELEASE_ID, root=PINNED_ROOT,
            descriptor=src.describe())
        yield from src.inventory(snap)

    # -- accessors ------------------------------------------------------------

    def load_manifest(self) -> dict[str, Any]:
        return json.loads(STORE_MANIFEST.read_text(encoding="utf-8"))


PLUGIN = L15SourcePlugin()
__all__ = ["L15SourcePlugin", "PLUGIN", "STORE_DB", "STORE_MANIFEST"]
