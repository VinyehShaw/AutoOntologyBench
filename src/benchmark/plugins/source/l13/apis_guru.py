"""L13 SourcePlugin adapter: pinned openapi-directory snapshot -> typed units.

Thin adapter over ``source_registry.fetchers.l13_apis_guru`` implementing the
build-order §3.2 SourcePlugin shape (``pin_and_index`` / ``enumerate_units``),
plus cached spec loading for task-family and verifier use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    FetchReceipt,
    SourceSnapshot,
    SourceUnit,
)
from benchmark.plugins.source.l13 import oas
from source_registry.fetchers import blob_fetch, l13_apis_guru as fetcher


class ApisGuruSourcePlugin:
    """SourcePlugin for lane L13 (APIs.guru OpenAPI Directory)."""

    plugin_id = "l13.apis_guru"

    def __init__(self) -> None:
        self._spec_cache: dict[str, dict[str, Any]] = {}

    def pin_and_index(self, source_config: dict[str, Any]) -> SourceSnapshot:
        """Verify (or fetch) the pinned snapshot and (re)build its index."""
        src = fetcher.ApisGuruSource()
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
        return fetcher.ApisGuruSource().inventory(snapshot)

    # -- spec access ------------------------------------------------------------
    def spec_path(self, payload_ref: str) -> Path:
        return fetcher.PINNED_ROOT / payload_ref

    def load_spec(self, unit: dict[str, Any]) -> dict[str, Any]:
        """Parse a spec unit once; digest-checked against the unit inventory."""
        uid = unit["unit_id"]
        if uid not in self._spec_cache:
            path = self.spec_path(unit["payload_ref"])
            spec = oas.load_spec(path)
            actual = blob_fetch.sha256_file(path)
            if actual != unit["digest"]:
                raise oas.OasError(
                    "digest mismatch for %s: index=%s file=%s" % (uid, unit["digest"], actual)
                )
            self._spec_cache[uid] = spec
        return self._spec_cache[uid]

    def task_units(self, snapshot: SourceSnapshot) -> list[dict[str, Any]]:
        """OpenAPI 3.0.x spec units under the task size cap, sorted by unit_id.

        Deterministic eligibility pre-filter: dialect openapi, spec_version
        3.0.x, file size <= 300 KiB (keeps excerpts and build time bounded).
        """
        out: list[dict[str, Any]] = []
        for unit in self.enumerate_units(snapshot):
            if unit.kind != "openapi_spec":
                continue
            if not str(unit.metadata.get("spec_version", "")).startswith("3.0"):
                continue
            path = self.spec_path(unit.payload_ref)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 300 * 1024:
                continue
            out.append({
                "unit_id": unit.unit_id,
                "payload_ref": unit.payload_ref,
                "digest": unit.digest,
                "metadata": dict(unit.metadata),
                "size": size,
            })
        out.sort(key=lambda u: u["unit_id"])
        return out


def load_index_units(index_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read the persisted unit inventory (used by audits/reports)."""
    index_dir = index_dir or fetcher.INDEX_ROOT
    path = Path(index_dir) / "units.jsonl"
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


__all__ = ["ApisGuruSourcePlugin", "load_index_units"]
