"""L02 SourcePlugin adapter: pinned kubernetes/website VAP/CEL docs shard.

Thin adapter over ``source_registry.fetchers.l02_kubernetes_cel`` implementing
the build-order §3.2 SourcePlugin shape without mutating locked protocols.
"""
from __future__ import annotations

from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    FetchReceipt,
    SourceSnapshot,
    SourceUnit,
)
from source_registry.fetchers import l02_kubernetes_cel as fetcher


class KubernetesVapCelSourcePlugin:
    """SourcePlugin for lane L02 (Kubernetes VAP/CEL docs)."""

    plugin_id = "l02.kubernetes-vap-cel"

    def pin_and_index(self, source_config: dict[str, Any]) -> SourceSnapshot:
        src = fetcher.KubernetesVapCelSource()
        release = src.discover_releases()[0]
        if source_config.get("fetch_if_missing") and not fetcher.PINNED_ROOT.is_dir():
            src.fetch(release, resume=True, cache_dir=fetcher.PINNED_ROOT.parent / "_cache")
        if not fetcher.PINNED_ROOT.is_dir():
            raise FileNotFoundError(
                "pinned snapshot missing: %s (run source fetch first)" % fetcher.PINNED_ROOT
            )
        report = src.verify_snapshot(
            FetchReceipt(
                source_id=fetcher.SOURCE_ID,
                release_id=release.release_id,
                status="FETCHED_PARTIAL",
                bytes_downloaded=0,
                file_count=0,
                cache_dir=str(fetcher.PINNED_ROOT),
                sha256_manifest="",
                fetched_at="",
            )
        )
        if not report.ok:
            raise RuntimeError("snapshot verification failed: %s" % (report.checks,))
        snapshot = fetcher.load_snapshot()
        src.index(snapshot, out_dir=fetcher.INDEX_ROOT)
        return snapshot

    def enumerate_units(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
        return fetcher.KubernetesVapCelSource().inventory(snapshot)


__all__ = ["KubernetesVapCelSourcePlugin"]
