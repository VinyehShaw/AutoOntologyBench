"""Source registry loader (T0). Merges `sources/registry.d/*.yaml` shards.

Shards are written independently by W2/W3/W4 (one file per source) to avoid
concurrent-write conflicts; this module is the single merge point. The directory
may be missing or partially filled while sibling workstreams are mid-flight —
loading stays tolerant and deterministic (filename-sorted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    shard: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BrokenShard:
    shard: str
    error: str


@dataclass(frozen=True)
class SourceRegistry:
    entries: dict[str, SourceEntry] = field(default_factory=dict)
    registry_dir: str = ""
    shards_loaded: tuple[str, ...] = ()
    broken_shards: tuple[BrokenShard, ...] = ()

    def get(self, source_id: str) -> SourceEntry | None:
        return self.entries.get(source_id)

    def source_ids(self) -> list[str]:
        return sorted(self.entries)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self.entries


def _entries_from_shard(shard_path: Path, data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise RegistryError(f"{shard_path}: top level must be a mapping")
    if "sources" in data:
        sources = data["sources"]
        if not isinstance(sources, list):
            raise RegistryError(f"{shard_path}: 'sources' must be a list")
        return sources
    return [data]


def load_registry(registry_dir: Path) -> SourceRegistry:
    """Merge all shards. Missing dir -> empty registry (tolerated at M1)."""
    if not registry_dir.is_dir():
        return SourceRegistry(registry_dir=str(registry_dir))
    merged: dict[str, SourceEntry] = {}
    shards: list[str] = []
    broken: list[BrokenShard] = []
    for shard_path in sorted(registry_dir.glob("*.yaml")):
        shards.append(shard_path.name)
        try:
            data = yaml.safe_load(shard_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            broken.append(BrokenShard(shard=shard_path.name, error=str(exc).split("\n")[0][:200]))
            continue
        if data is None:
            continue
        for raw in _entries_from_shard(shard_path, data):
            source_id = raw.get("source_id")
            if not source_id or not isinstance(source_id, str):
                raise RegistryError(f"{shard_path}: entry missing 'source_id'")
            prior = merged.get(source_id)
            entry = SourceEntry(source_id=source_id, shard=shard_path.name, payload=raw)
            if prior is not None and prior.payload != raw:
                raise RegistryError(
                    f"conflicting definitions for source_id={source_id!r} "
                    f"in {prior.shard} and {shard_path.name}")
            merged[source_id] = entry
    return SourceRegistry(entries=merged, registry_dir=str(registry_dir),
                          shards_loaded=tuple(shards), broken_shards=tuple(broken))
