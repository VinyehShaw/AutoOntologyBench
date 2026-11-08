"""BYOK source plugin (example 2): DCMI Metadata Terms (dublin_core_terms.ttl).

Small, real, legally redistributable external knowledge: DCMI publishes its
specifications under CC-BY 4.0 (see README.md). The plugin fetches the pinned
Turtle document through the factory content-addressed cache (sha256-verified,
resume-safe), materializes it under pinned/<source_id>/<release_id>/, and
inventories rdfs:Class / rdf:Property units with rdflib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from autoontobench.factory import cli_support as cs
from autoontobench.factory.cache import ContentAddressedCache, HostThrottle
from benchmark.schema.plugin_protocols import (
    FetchReceipt, IndexManifest, SourceDescriptor, SourceRelease, SourceSnapshot,
    SourceUnit, VerificationReport)

SOURCE_ID = "dcmi-terms"
RELEASE_ID = "dcmi-20120614"
TERMS_URL = "https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_terms.ttl"
TERMS_SHA256 = "13df401072dd7015bf9d75162f3e41c8138075304b7b9cc1aa1e9c16db976797"
FILE_NAME = "dublin_core_terms.ttl"


class DcmiTermsSource:
    source_id = SOURCE_ID

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_id=SOURCE_ID,
            official_url="https://www.dublincore.org/specifications/dublin-core/dcmi-terms/",
            download_url_or_api=TERMS_URL,
            source_type="single-file",
            upstream_version="DCMI Metadata Terms (dcterms:modified 2012-06-14)",
            commit_or_release=RELEASE_ID,
            retrieved_at="2026-10-27T00:00:00+00:00",
            sha256=TERMS_SHA256,
            native_engine="rdflib",
            license_url="https://www.dublincore.org/specifications/dublin-core/dcmi-terms/",
            redistribution_policy="redistributable_with_attribution",
            estimated_bytes=47834,
            is_complete_snapshot=True,
            attribution="Dublin Core Metadata Initiative (DCMI), CC-BY 4.0",
            usage_constraints="CC-BY 4.0: attribution required")

    def discover_releases(self) -> list[SourceRelease]:
        return [SourceRelease(release_id=RELEASE_ID,
                              version="dcterms:modified 2012-06-14",
                              urls=(TERMS_URL,), published_at="2012-06-14")]

    def fetch(self, release: SourceRelease, *, resume: bool, cache_dir: Path) -> FetchReceipt:
        cache = ContentAddressedCache(Path(cache_dir).parent)
        dest = cache.pinned / SOURCE_ID / release.release_id
        target = dest / FILE_NAME
        if resume and target.is_file() and cs.sha256_hex(target.read_bytes()) == TERMS_SHA256:
            return FetchReceipt(source_id=SOURCE_ID, release_id=release.release_id,
                                status="FETCHED_COMPLETE", bytes_downloaded=target.stat().st_size,
                                file_count=1, cache_dir=str(dest), sha256_manifest=TERMS_SHA256,
                                fetched_at=cs.utc_now_iso(), resumed=True)
        result = cache.download(TERMS_URL, expected_sha256=TERMS_SHA256, resume=resume,
                                throttle=HostThrottle())
        cache.materialize(SOURCE_ID, release.release_id, {FILE_NAME: result.digest})
        receipt = FetchReceipt(source_id=SOURCE_ID, release_id=release.release_id,
                               status="FETCHED_COMPLETE", bytes_downloaded=result.bytes_written,
                               file_count=1, cache_dir=str(dest), sha256_manifest=result.digest,
                               fetched_at=cs.utc_now_iso(), resumed=result.resumed)
        cache.write_receipt({**receipt.__dict__, "sha256": result.digest})
        return receipt

    def verify_snapshot(self, receipt: FetchReceipt) -> VerificationReport:
        target = Path(receipt.cache_dir) / FILE_NAME
        if not target.is_file():
            return VerificationReport(ok=False, missing=(FILE_NAME,))
        digest = cs.sha256_hex(target.read_bytes())
        ok = digest == TERMS_SHA256
        return VerificationReport(ok=ok, checks={"sha256": digest, "expected": TERMS_SHA256})

    def inventory(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
        graph = _load_graph(snapshot.root)
        from rdflib import RDF, RDFS

        for kind, rdf_type in (("rdfs-class", RDFS.Class), ("rdf-property", RDF.Property)):
            for subj in sorted(set(graph.subjects(RDF.type, rdf_type))):
                iri = str(subj)
                yield SourceUnit(unit_id=f"{SOURCE_ID}/{iri.rsplit('/', 1)[-1]}",
                                 kind=kind, payload_ref=iri,
                                 digest=cs.sha256_hex(iri.encode()),
                                 metadata={"iri": iri})

    def index(self, snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest:
        units = list(self.inventory(snapshot))
        kinds: dict[str, int] = {}
        for u in units:
            kinds[u.kind] = kinds.get(u.kind, 0) + 1
        out_dir.mkdir(parents=True, exist_ok=True)
        cs.write_json(out_dir / "index.json",
                      {"units": [u.unit_id for u in units], "kind_counts": kinds})
        return IndexManifest(source_id=SOURCE_ID, release_id=snapshot.release_id,
                             out_dir=str(out_dir), unit_count=len(units), kind_counts=kinds)


def _load_graph(root: Path):
    import rdflib

    graph = rdflib.Graph()
    graph.parse(str(Path(root) / FILE_NAME), format="turtle")
    return graph


PLUGIN = DcmiTermsSource()
