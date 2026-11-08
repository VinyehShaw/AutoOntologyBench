"""``python -m autoontobench source`` — Source Hub commands (S10).

list / fetch / audit / add. Lane-owned fetchers live in
``source_registry/fetchers/lNN_*.py`` and expose ``get_source()`` returning an
``ExternalKnowledgeSource`` (benchmark/schema/plugin_protocols.py). When a lane
fetcher has not landed yet, ``fetch`` falls back to a generic pinned GitHub
tarball fetcher so the source lock still records a real, re-fetchable pin.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

from autoontobench.factory import cli_support as cs
from autoontobench.factory.cache import ContentAddressedCache, HostThrottle
from autoontobench.factory.registry import (
    LockedSource, load_lock, load_overlay, load_sources, write_lock)

REPO = cs.REPO_ROOT
EK = REPO / "external_knowledge"
OFFICIAL_LOCK = REPO / "source_registry" / "source_lock.json"
OVERLAY_DIR = REPO / "bank" / "byok" / "sources.d"
OVERLAY_LOCK = REPO / "bank" / "byok" / "source_lock.json"

FETCH_STATUSES = ("FETCHED_COMPLETE", "FETCHED_PARTIAL", "INDEXED_REMOTE_ONLY",
                  "ACCESS_RESTRICTED", "BLOCKED_RUNTIME", "LICENSE_HOLD", "FAILED_FETCH")


def _registry() -> dict[str, object]:
    sources = load_sources(REPO / "source_registry" / "sources.yaml")
    sources.update(load_overlay(OVERLAY_DIR))
    return sources


def cmd_list(args: Namespace) -> int:
    lock = load_lock(OVERLAY_LOCK if getattr(args, "overlay", False) else OFFICIAL_LOCK)
    print(f"{'source_id':<24} {'lane':<5} {'type':<14} {'engine':<10} lock")
    for sid, spec in sorted(_registry().items()):
        pinned = lock.entries.get(sid)
        ref = (pinned.commit_or_release[:12] if pinned else "-")
        print(f"{sid:<24} {spec.lane:<5} {spec.source_type:<12} {spec.native_engine:<10} {ref}")
    return 0


def _load_lane_fetcher(spec: object, cache: ContentAddressedCache) -> object | None:
    path = REPO / str(getattr(spec, "fetcher", ""))
    if not path.is_file():
        return None
    mspec = importlib.util.spec_from_file_location(f"aob_fetcher_{path.stem}", path)
    module = importlib.util.module_from_spec(mspec)  # type: ignore[arg-type]
    mspec.loader.exec_module(module)  # type: ignore[union-attr]
    getter = getattr(module, "get_source", None)
    return getter(cache=cache) if callable(getter) else getattr(module, "SOURCE", None)


def _github_repo(url: str) -> tuple[str, str] | None:
    parts = url.rstrip("/").split("/")
    if "github.com" in url and len(parts) >= 5:
        return parts[-2], parts[-1].removesuffix(".git")
    return None


def _generic_git_fetch(spec: object, cache: ContentAddressedCache, lock_entry: LockedSource | None,
                       *, resume: bool, metadata_only: bool, dry_run: bool,
                       throttle: HostThrottle) -> dict[str, object]:
    import httpx

    sid = getattr(spec, "source_id")
    repo = _github_repo(getattr(spec, "official_url")) or _github_repo(
        getattr(spec, "download_url_or_api"))
    if repo is None:
        return {"source_id": sid, "status": "FAILED_FETCH",
                "reason": "no fetcher module and not a GitHub source"}
    owner, name = repo
    if lock_entry is not None:
        commit = lock_entry.commit_or_release
    else:
        if dry_run:
            commit = "<resolve-at-runtime>"
        else:
            resp = httpx.get(f"https://api.github.com/repos/{owner}/{name}/commits",
                             params={"per_page": 1}, timeout=30.0)
            resp.raise_for_status()
            commit = str(resp.json()[0]["sha"])
    tarball = f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}"
    if metadata_only:
        return {"source_id": sid, "status": "INDEXED_REMOTE_ONLY", "commit": commit,
                "note": "metadata-only: tarball not downloaded"}
    if dry_run:
        return {"source_id": sid, "status": "DRY_RUN", "commit": commit, "url": tarball}
    try:
        result = cache.download(tarball, resume=resume, throttle=throttle,
                                expected_sha256=lock_entry.sha256 if lock_entry else None)
    except cs.FetchError:
        mirror = f"https://ghfast.top/https://github.com/{owner}/{name}/archive/{commit}.tar.gz"
        result = cache.download(mirror, resume=resume, throttle=throttle,
                                expected_sha256=lock_entry.sha256 if lock_entry else None)
    release_id = commit[:12]
    dest = cache.pinned / sid / release_id
    n_files = _extract_tarball(cache.object_path(result.digest), dest)
    manifest = json.dumps(sorted(p.name for p in dest.rglob("*") if p.is_file()))
    receipt = {"source_id": sid, "release_id": release_id, "status": "FETCHED_COMPLETE",
               "bytes_downloaded": result.bytes_written, "file_count": n_files,
               "cache_dir": str(dest), "sha256_manifest": cs.sha256_hex(manifest.encode()),
               "tarball_sha256": result.digest, "fetched_at": cs.utc_now_iso(),
               "resumed": result.resumed}
    cache.write_receipt(receipt)
    receipt["commit"] = commit
    return receipt


def _extract_tarball(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise cs.FetchError(f"unsafe tar member: {name}")
            if member.isfile():
                tf.extract(member, dest, filter="data")
                count += 1
    return count


def cmd_fetch(args: Namespace) -> int:
    cache = ContentAddressedCache(EK)
    throttle = HostThrottle(initial=4, ramp=(8, 16))
    lock = load_lock(OVERLAY_LOCK if getattr(args, "overlay", False) else OFFICIAL_LOCK)
    selected = [args.source] if getattr(args, "source", None) else sorted(_registry())
    results: list[dict[str, object]] = []
    new_entries = dict(lock.entries)
    for sid in selected:
        spec = _registry().get(sid)
        if spec is None:
            cs.eprint(f"[source fetch] unknown source {sid!r}")
            return 2
        if getattr(args, "dry_run", False):
            print(f"DRY-RUN fetch {sid} (lane {spec.lane}) pinned={sid in lock.entries}")
            continue
        fetcher = _load_lane_fetcher(spec, cache)
        if fetcher is not None:
            receipt = _run_lane_fetcher(fetcher, cache, lock.entries.get(sid),
                                        resume=bool(args.resume),
                                        metadata_only=bool(args.metadata_only))
        else:
            receipt = _generic_git_fetch(spec, cache, lock.entries.get(sid),
                                         resume=bool(args.resume),
                                         metadata_only=bool(args.metadata_only),
                                         dry_run=False, throttle=throttle)
        results.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        if receipt.get("status") == "FETCHED_COMPLETE" and sid not in lock.entries:
            new_entries[sid] = LockedSource(
                source_id=sid, release_id=str(receipt["release_id"]),
                commit_or_release=str(receipt.get("commit", receipt["release_id"])),
                urls=(str(getattr(spec, "official_url")),),
                sha256=str(receipt.get("tarball_sha256", receipt.get("sha256_manifest"))),
                upstream_version=str(receipt.get("commit", ""))[:40],
                locked_at=cs.utc_now_iso(),
                bytes_downloaded=int(receipt.get("bytes_downloaded", 0)))
    if new_entries != lock.entries and not getattr(args, "dry_run", False):
        path = OVERLAY_LOCK if getattr(args, "overlay", False) else OFFICIAL_LOCK
        write_lock(path, "aob-8000-rc1", new_entries)
        print(f"[source fetch] lock updated: {path}")
    return 0 if all(r.get("status") in ("FETCHED_COMPLETE", "INDEXED_REMOTE_ONLY")
                    for r in results) else (0 if not results else 1)


def _run_lane_fetcher(fetcher: object, cache: ContentAddressedCache,
                      lock_entry: LockedSource | None, *, resume: bool,
                      metadata_only: bool) -> dict[str, object]:
    release = None
    if lock_entry is not None:
        from benchmark.schema.plugin_protocols import SourceRelease

        release = SourceRelease(release_id=lock_entry.release_id,
                                version=lock_entry.upstream_version,
                                urls=lock_entry.urls)
    else:
        releases = fetcher.discover_releases()  # type: ignore[attr-defined]
        release = releases[0] if releases else None
    if release is None:
        return {"source_id": fetcher.source_id, "status": "FAILED_FETCH",  # type: ignore[attr-defined]
                "reason": "no releases discovered"}
    if metadata_only:
        return {"source_id": fetcher.source_id, "release_id": release.release_id,  # type: ignore[attr-defined]
                "status": "INDEXED_REMOTE_ONLY"}
    receipt = fetcher.fetch(release, resume=resume, cache_dir=cache.pinned)  # type: ignore[attr-defined]
    return dict(receipt.__dict__) if hasattr(receipt, "__dict__") else dict(receipt)  # type: ignore[arg-type]


def cmd_audit(args: Namespace) -> int:
    cache = ContentAddressedCache(EK)
    lock = load_lock(OFFICIAL_LOCK)
    overlay = load_lock(OVERLAY_LOCK)
    rows: list[dict[str, object]] = []
    for sid, spec in sorted(_registry().items()):
        entry = lock.entries.get(sid) or overlay.entries.get(sid)
        receipts = sorted((cache.receipts / sid).glob("*.json")) if (cache.receipts / sid).is_dir() else []
        last = json.loads(receipts[-1].read_text()) if receipts else {}
        rows.append({"lane": spec.lane, "source_id": sid,
                     "pinned": entry.commit_or_release[:12] if entry else "-",
                     "status": last.get("status", "NOT_FETCHED"),
                     "bytes": last.get("bytes_downloaded", 0),
                     "files": last.get("file_count", 0),
                     "license": spec.license})
    print(f"{'lane':<5} {'source_id':<24} {'status':<20} {'bytes':>12} {'files':>7} pinned")
    for r in rows:
        print(f"{r['lane']:<5} {r['source_id']:<24} {r['status']:<20} "
              f"{r['bytes']:>12} {r['files']:>7} {r['pinned']}")
    if args.out:
        cs.write_json(Path(args.out) / "source_hub_audit.json", {"sources": rows})
    return 0


def cmd_add(args: Namespace) -> int:
    cfg = cs.load_yaml_strict(Path(args.config), cs.BYOK_SOURCE_CONFIG_SPEC)
    data = cfg.data
    sid = data["source_id"]
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    cs.write_json(OVERLAY_DIR / f"{sid}.yaml", data)
    pinned = data["pinned"]
    lock = load_lock(OVERLAY_LOCK)
    entries = dict(lock.entries)
    entries[sid] = LockedSource(
        source_id=sid, release_id=pinned["commit_or_release"][:12],
        commit_or_release=pinned["commit_or_release"], urls=tuple(pinned["urls"]),
        sha256=pinned["sha256"], upstream_version=pinned["upstream_version"],
        locked_at=cs.utc_now_iso())
    write_lock(OVERLAY_LOCK, data.get("lane", "byok"), entries)
    print(f"[source add] registered {sid} -> {OVERLAY_DIR / (sid + '.yaml')}")
    return 0


def run(args: Namespace) -> int:
    handler = {"list": cmd_list, "fetch": cmd_fetch,
               "audit": cmd_audit, "add": cmd_add}.get(args.source_command)
    if handler is None:
        cs.eprint("[source] subcommand required: list|fetch|audit|add")
        return 2
    return handler(args)
