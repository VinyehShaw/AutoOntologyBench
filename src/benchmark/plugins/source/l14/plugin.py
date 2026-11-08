"""L14 SourcePlugin adapter: pin + index + unit enumeration over pinned ChEMBL.

Thin adapter: fetching lives in ``source_registry/fetchers/l14_chembl.py``;
streaming census/index lives in ``benchmark/plugins/source/l14/inventory.py``.
This module wires both into the ``SourcePlugin`` interface semantics of
FINAL-BUILD-ORDER-v2 §3.2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import SourceSnapshot
from benchmark.plugins.source.l14.inventory import (
    INDEX_SUBDIR, build_index, iter_source_units,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "chembl"
                    / "chembl-37")


class L14SourcePlugin:
    source_id = "chembl"

    def pin_and_index(self, source_config: dict[str, Any] | None = None) -> SourceSnapshot:
        """Locate (or fetch) the pinned snapshot and build the census index."""
        from source_registry.fetchers.l14_chembl import (
            DEFAULT_CACHE, L14ChemblSource, snapshot_from_receipt,
        )
        src = L14ChemblSource()
        release = src.discover_releases()[0]
        cache_dir = Path((source_config or {}).get("cache_dir", DEFAULT_CACHE))
        manifest = cache_dir / "fetch_manifest.json"
        if not manifest.is_file():
            receipt = src.fetch(release, resume=True, cache_dir=cache_dir)
            if receipt.status != "FETCHED_COMPLETE":
                raise RuntimeError(f"L14 fetch incomplete: {receipt.reason}")
        else:
            from benchmark.schema.plugin_protocols import FetchReceipt
            m = json.loads(manifest.read_text())
            receipt = FetchReceipt(
                source_id=src.source_id, release_id=release.release_id,
                status=m["status"], bytes_downloaded=m["bytes_downloaded"],
                file_count=m["file_count"], cache_dir=str(cache_dir),
                sha256_manifest=m.get("sha256_manifest", ""),
                fetched_at=m["fetched_at"])
        report = src.verify_snapshot(receipt)
        if not report.ok:
            raise RuntimeError(f"L14 snapshot verification failed: {report.missing}")
        snapshot = snapshot_from_receipt(receipt, src.describe())
        index_dir = cache_dir / INDEX_SUBDIR
        if not (index_dir / "index_manifest.json").is_file():
            build_index(snapshot, out_dir=index_dir)
        return snapshot

    def enumerate_units(self, snapshot: SourceSnapshot) -> Iterator:
        return iter_source_units(snapshot)
