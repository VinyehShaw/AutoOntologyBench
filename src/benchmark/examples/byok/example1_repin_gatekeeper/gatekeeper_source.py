"""BYOK source plugin: gatekeeper-library re-pinned at commit 1946bc28 (example 1).

Downloads the pinned tarball into the content-addressed cache, extracts it
under pinned/<source_id>/<release_id>/, writes a fetch receipt, and inventories
policy units (library/general/<policy>/template.yaml with an embedded Rego
target). License: Apache-2.0 (upstream repo); see README.md.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Iterator

import yaml

from autoontobench.factory import cli_support as cs
from autoontobench.factory.cache import ContentAddressedCache, HostThrottle
from benchmark.schema.plugin_protocols import (
    FetchReceipt, IndexManifest, SourceDescriptor, SourceRelease, SourceSnapshot,
    SourceUnit, VerificationReport)

SOURCE_ID = "gatekeeper-library-repin-1946bc28"
RELEASE_ID = "1946bc28e133"
COMMIT = "1946bc28e1337b90b7e675e3f0fa94aa2e92fd6e"
TARBALL_SHA256 = "5d809aada71b307f7451da191e8c518d3ab573cb88d9a5799c6095d00a8435a7"
TARBALL_URL = f"https://codeload.github.com/open-policy-agent/gatekeeper-library/tar.gz/{COMMIT}"
MIRROR_URL = f"https://ghfast.top/https://github.com/open-policy-agent/gatekeeper-library/archive/{COMMIT}.tar.gz"


def _cache(cache_dir: Path) -> ContentAddressedCache:
    return ContentAddressedCache(Path(cache_dir).parent)


class GatekeeperRepinSource:
    source_id = SOURCE_ID

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_id=SOURCE_ID,
            official_url="https://github.com/open-policy-agent/gatekeeper-library",
            download_url_or_api=TARBALL_URL,
            source_type="git",
            upstream_version=f"commit {COMMIT}",
            commit_or_release=COMMIT,
            retrieved_at="2026-10-27T00:00:00+00:00",
            sha256=TARBALL_SHA256,
            native_engine="opa",
            license_url="https://github.com/open-policy-agent/gatekeeper-library/blob/master/LICENSE",
            redistribution_policy="fetch_recipe_only",
            estimated_bytes=556722,
            is_complete_snapshot=True,
            attribution="Open Policy Agent Gatekeeper Library authors",
            usage_constraints="Apache-2.0; per-asset verification advised")

    def discover_releases(self) -> list[SourceRelease]:
        return [SourceRelease(release_id=RELEASE_ID, version=f"commit {COMMIT}",
                              urls=(TARBALL_URL, MIRROR_URL),
                              published_at="2026-09-28T22:23:50Z")]

    def fetch(self, release: SourceRelease, *, resume: bool, cache_dir: Path) -> FetchReceipt:
        cache = _cache(cache_dir)
        dest = cache.pinned / SOURCE_ID / release.release_id
        marker = dest / "MANIFEST.json"
        if resume and marker.is_file():
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            return FetchReceipt(source_id=SOURCE_ID, release_id=release.release_id,
                                status="FETCHED_COMPLETE",
                                bytes_downloaded=int(manifest["bytes_downloaded"]),
                                file_count=int(manifest["file_count"]), cache_dir=str(dest),
                                sha256_manifest=str(manifest["sha256_manifest"]),
                                fetched_at=str(manifest["fetched_at"]), resumed=True)
        try:
            result = cache.download(TARBALL_URL, expected_sha256=TARBALL_SHA256,
                                    resume=resume, throttle=HostThrottle())
        except cs.FetchError:
            result = cache.download(MIRROR_URL, expected_sha256=TARBALL_SHA256,
                                    resume=resume, throttle=HostThrottle())
        n_files = _extract(cache.object_path(result.digest), dest)
        sha_manifest = cs.sha256_hex(json.dumps(
            sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())).encode())
        manifest = {"source_id": SOURCE_ID, "release_id": release.release_id,
                    "commit": COMMIT, "tarball_sha256": result.digest,
                    "bytes_downloaded": result.bytes_written, "file_count": n_files,
                    "sha256_manifest": sha_manifest, "fetched_at": cs.utc_now_iso()}
        cs.write_json(marker, manifest)
        receipt = FetchReceipt(source_id=SOURCE_ID, release_id=release.release_id,
                               status="FETCHED_COMPLETE",
                               bytes_downloaded=result.bytes_written, file_count=n_files,
                               cache_dir=str(dest), sha256_manifest=sha_manifest,
                               fetched_at=str(manifest["fetched_at"]), resumed=result.resumed)
        cache.write_receipt({**receipt.__dict__, "commit": COMMIT,
                             "tarball_sha256": result.digest})
        return receipt

    def verify_snapshot(self, receipt: FetchReceipt) -> VerificationReport:
        root = Path(receipt.cache_dir)
        manifest_path = root / "MANIFEST.json"
        if not manifest_path.is_file():
            return VerificationReport(ok=False, missing=("MANIFEST.json",))
        units = list(self.inventory(SourceSnapshot(SOURCE_ID, receipt.release_id, root,
                                                   self.describe())))
        ok = bool(units) and receipt.status == "FETCHED_COMPLETE"
        return VerificationReport(ok=ok, checks={"policy_units": len(units)})

    def inventory(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
        root = Path(snapshot.root)
        for inner in root.iterdir():
            lib = inner / "library" / "general"
            if not lib.is_dir():
                continue
            for policy in sorted(p for p in lib.iterdir() if p.is_dir()):
                tpl = policy / "template.yaml"
                if not tpl.is_file():
                    continue
                digest = cs.sha256_hex(tpl.read_bytes())
                yield SourceUnit(unit_id=f"{SOURCE_ID}/{policy.name}", kind="rego-policy",
                                 payload_ref=str(tpl), digest=digest,
                                 metadata={"policy": policy.name})

    def index(self, snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest:
        units = list(self.inventory(snapshot))
        out_dir.mkdir(parents=True, exist_ok=True)
        cs.write_json(out_dir / "index.json",
                      {"units": [u.unit_id for u in units], "count": len(units)})
        return IndexManifest(source_id=SOURCE_ID, release_id=snapshot.release_id,
                             out_dir=str(out_dir), unit_count=len(units),
                             kind_counts={"rego-policy": len(units)})


def _extract(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise cs.FetchError(f"unsafe tar member: {member.name}")
            if member.isfile():
                tf.extract(member, dest, filter="data")
                count += 1
    return count


PLUGIN = GatekeeperRepinSource()
