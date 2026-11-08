"""Source Hub registry: ``source_registry/sources.yaml`` loader and
``source_registry/source_lock.json`` reader/writer/validator.

The registry is the version-controlled list of official entry points (read-only
for S10). The lock file pins every source to an immutable commit/release plus
sha256 so a third party can re-fetch the *same* snapshot. BYOK overlays live
under ``bank/byok/sources.d/`` and merge on top of the official registry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from autoontobench.factory.cli_support import (
    ConfigError,
    canonical_json_sha,
    utc_now_iso,
    write_json,
)

# Re-export: discovery machinery lives in plugin_discovery.py (back-compat).
from autoontobench.factory.plugin_discovery import (  # noqa: F401
    PLUGIN_SURFACES,
    PluginAudit,
    PluginIssue,
    PluginRef,
    audit_plugins,
    discover_plugins,
    load_plugin_file,
    plugin_covers_lane,
)

SOURCES_SCHEMA = "aob-sources/1"
LOCK_SCHEMA = "aob-source-lock/1"

_SOURCE_KEYS = {
    "source_id", "lane", "official_url", "download_url_or_api", "source_type",
    "native_engine", "fetcher", "license", "redistribution_policy",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    lane: str
    official_url: str
    download_url_or_api: str
    source_type: str
    native_engine: str
    fetcher: str
    license: str
    redistribution_policy: str
    origin: str = "official"  # official | byok-overlay


@dataclass(frozen=True)
class LockedSource:
    source_id: str
    release_id: str
    commit_or_release: str
    urls: tuple[str, ...]
    sha256: str
    upstream_version: str
    locked_at: str
    bytes_downloaded: int = 0


@dataclass(frozen=True)
class SourceLock:
    schema: str
    bank_id: str
    generated_at: str
    lock_hash: str
    entries: dict[str, LockedSource] = field(default_factory=dict)


def _parse_source(raw: Any, origin: str, where: str) -> SourceSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: source entry must be a mapping")
    unknown = set(raw) - _SOURCE_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown source keys: {sorted(unknown)}")
    for key in ("source_id", "lane", "official_url", "source_type", "native_engine"):
        if not raw.get(key):
            raise ConfigError(f"{where}: missing required key {key!r}")
    return SourceSpec(
        source_id=str(raw["source_id"]),
        lane=str(raw["lane"]),
        official_url=str(raw["official_url"]),
        download_url_or_api=str(raw.get("download_url_or_api") or raw["official_url"]),
        source_type=str(raw["source_type"]),
        native_engine=str(raw["native_engine"]),
        fetcher=str(raw.get("fetcher") or ""),
        license=str(raw.get("license") or "unspecified"),
        redistribution_policy=str(raw.get("redistribution_policy") or "fetch_recipe_only"),
        origin=origin,
    )


def load_sources(path: Path) -> dict[str, SourceSpec]:
    """Load official sources.yaml (schema aob-sources/1), strictly."""
    if not path.is_file():
        raise ConfigError(f"sources registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SOURCES_SCHEMA:
        raise ConfigError(f"{path}: expected 'schema: {SOURCES_SCHEMA}'")
    extra = set(data) - {"schema", "generated_by", "note", "sources"}
    if extra:
        raise ConfigError(f"{path}: unknown top-level keys: {sorted(extra)}")
    out: dict[str, SourceSpec] = {}
    for raw in data.get("sources") or []:
        spec = _parse_source(raw, "official", f"{path}:{raw.get('source_id', '?')}")
        if spec.source_id in out:
            raise ConfigError(f"{path}: duplicate source_id {spec.source_id!r}")
        out[spec.source_id] = spec
    return out


def load_overlay(overlay_dir: Path) -> dict[str, SourceSpec]:
    """BYOK-added sources (bank/byok/sources.d/*.yaml); overlay official registry."""
    out: dict[str, SourceSpec] = {}
    if not overlay_dir.is_dir():
        return out
    for shard in sorted(overlay_dir.glob("*.yaml")):
        data = yaml.safe_load(shard.read_text(encoding="utf-8"))
        spec = _parse_source(data, "byok-overlay", str(shard))
        out[spec.source_id] = spec
    return out


def _lock_entry(raw: Any, where: str) -> LockedSource:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: lock entry must be a mapping")
    known = {"source_id", "release_id", "commit_or_release", "urls", "sha256",
             "upstream_version", "locked_at", "bytes_downloaded"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{where}: unknown lock keys: {sorted(unknown)}")
    sha = str(raw.get("sha256") or "")
    if not _HEX64.match(sha):
        raise ConfigError(f"{where}: sha256 must be 64 lowercase hex chars")
    if not raw.get("commit_or_release"):
        raise ConfigError(f"{where}: commit_or_release must be pinned (no floating refs)")
    return LockedSource(
        source_id=str(raw["source_id"]),
        release_id=str(raw["release_id"]),
        commit_or_release=str(raw["commit_or_release"]),
        urls=tuple(str(u) for u in raw.get("urls") or ()),
        sha256=sha,
        upstream_version=str(raw.get("upstream_version") or ""),
        locked_at=str(raw.get("locked_at") or ""),
        bytes_downloaded=int(raw.get("bytes_downloaded") or 0),
    )


def load_lock(path: Path) -> SourceLock:
    """Validate source_lock.json; missing file -> empty lock (pre-fetch state)."""
    if not path.is_file():
        return SourceLock(schema=LOCK_SCHEMA, bank_id="", generated_at="", lock_hash="")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != LOCK_SCHEMA:
        raise ConfigError(f"{path}: expected 'schema: {LOCK_SCHEMA}'")
    extra = set(data) - {"schema", "bank_id", "generated_at", "sources"}
    if extra:
        raise ConfigError(f"{path}: unknown top-level keys: {sorted(extra)}")
    entries = {
        str(s["source_id"]): _lock_entry(s, f"{path}:{s.get('source_id', '?')}")
        for s in data.get("sources") or []
    }
    body = {"schema": data["schema"], "bank_id": data.get("bank_id", ""),
            "sources": data.get("sources") or []}
    return SourceLock(schema=LOCK_SCHEMA, bank_id=str(data.get("bank_id") or ""),
                      generated_at=str(data.get("generated_at") or ""),
                      lock_hash=canonical_json_sha(body), entries=entries)


def write_lock(path: Path, bank_id: str, entries: dict[str, LockedSource]) -> SourceLock:
    """Single-writer lock commit: sorted, validated, atomic replace."""
    rows = []
    for sid in sorted(entries):
        e = entries[sid]
        rows.append({"source_id": e.source_id, "release_id": e.release_id,
                     "commit_or_release": e.commit_or_release, "urls": list(e.urls),
                     "sha256": e.sha256, "upstream_version": e.upstream_version,
                     "locked_at": e.locked_at or utc_now_iso(),
                     "bytes_downloaded": e.bytes_downloaded})
    write_json(path, {"schema": LOCK_SCHEMA, "bank_id": bank_id,
                      "generated_at": utc_now_iso(), "sources": rows})
    return load_lock(path)

