"""L01 SourcePlugin adapter: pinned gatekeeper-library snapshot -> typed units.

Thin adapter over ``source_registry.fetchers.l01_gatekeeper`` implementing the
build-order §3.2 SourcePlugin shape (``pin_and_index`` / ``enumerate_units``)
without mutating the locked ``ExternalKnowledgeSource`` protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    FetchReceipt,
    SourceSnapshot,
    SourceUnit,
)
from source_registry.fetchers import l01_gatekeeper as fetcher


class GatekeeperSourcePlugin:
    """SourcePlugin for lane L01 (OPA Gatekeeper Library)."""

    plugin_id = "l01.gatekeeper"

    def pin_and_index(self, source_config: dict[str, Any]) -> SourceSnapshot:
        """Verify (or fetch) the pinned snapshot and (re)build its index."""
        src = fetcher.GatekeeperLibrarySource()
        release = src.discover_releases()[0]
        if source_config.get("fetch_if_missing") and not fetcher.PINNED_ROOT.is_dir():
            src.fetch(release, resume=True, cache_dir=fetcher.PINNED_ROOT.parent / "_cache")
        if not fetcher.PINNED_ROOT.is_dir():
            raise FileNotFoundError(
                "pinned snapshot missing: %s (run source fetch first)" % fetcher.PINNED_ROOT
            )
        receipt = src.verify_snapshot(
            FetchReceipt(
                source_id=fetcher.SOURCE_ID,
                release_id=release.release_id,
                status="FETCHED_COMPLETE",
                bytes_downloaded=0,
                file_count=0,
                cache_dir=str(fetcher.PINNED_ROOT),
                sha256_manifest="",
                fetched_at="",
            )
        )
        if not receipt.ok:
            raise RuntimeError("snapshot verification failed: %s" % (receipt.checks,))
        snapshot = fetcher.load_snapshot()
        src.index(snapshot, out_dir=fetcher.INDEX_ROOT)
        return snapshot

    def enumerate_units(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
        return fetcher.GatekeeperLibrarySource().inventory(snapshot)


__all__ = ["GatekeeperSourcePlugin"]
