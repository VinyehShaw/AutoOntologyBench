"""L16 SourcePlugin adapter: pin + index + unit enumeration over the snapshot.

Thin adapter: fetching lives in ``source_registry/fetchers/l16_uniprot.py``;
inventory/index live in ``inventory.py``. Mirrors the FINAL-BUILD-ORDER-v2
§3.2 source-plugin surface used by the other lanes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import SourceSnapshot
from benchmark.plugins.source.l16.inventory import build_index, iter_source_units

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "uniprot"
                    / "uniprot-2026_03-sprot")


class L16SourcePlugin:
    source_id = "uniprot"

    def pin_and_index(self, source_config: dict[str, Any] | None = None) -> SourceSnapshot:
        from benchmark.schema.plugin_protocols import FetchReceipt
        from source_registry.fetchers.l16_uniprot import (
            DEFAULT_CACHE, L16UniProtSource, snapshot_from_receipt,
        )
        src = L16UniProtSource()
        release = src.discover_releases()[0]
        cache_dir = Path((source_config or {}).get("cache_dir", DEFAULT_CACHE))
        manifest = cache_dir / "fetch_manifest.json"
        if not manifest.is_file():
            receipt = src.fetch(release, resume=True, cache_dir=cache_dir)
            if receipt.status != "FETCHED_COMPLETE":
                raise RuntimeError(f"L16 fetch incomplete: {receipt.reason}")
        else:
            m = json.loads(manifest.read_text())
            receipt = FetchReceipt(
                source_id=src.source_id, release_id=release.release_id,
                status=m["status"], bytes_downloaded=m["bytes_downloaded"],
                file_count=m["file_count"], cache_dir=str(cache_dir),
                sha256_manifest="", fetched_at=m["fetched_at"])
        report = src.verify_snapshot(receipt)
        if not report.ok:
            raise RuntimeError(f"L16 pin verification failed: {report.missing}")
        return snapshot_from_receipt(receipt, src.describe())

    def inventory(self, snapshot: SourceSnapshot) -> Iterator:
        return iter_source_units(snapshot)

    def index(self, snapshot: SourceSnapshot, *, out_dir: Path):
        return build_index(snapshot, out_dir=Path(out_dir))


PLUGIN = L16SourcePlugin()
