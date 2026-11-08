"""L16 inventory: SourceUnit enumeration + index build over the pinned snapshot.

Units are the addressable atoms tasks cite in ``source_unit_ids``: one unit per
entry plus one per cross-reference family aggregate. Digests are content hashes
of the canonical record, so a unit id is attributable to pinned bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    IndexManifest, SourceSnapshot, SourceUnit,
)
from benchmark.plugins.source.l16.records import record_sha256
from benchmark.plugins.source.l16.store import _j
from benchmark.plugins.source.l16.uniprot_dat import iter_dat

INDEX_SUBDIR = "indexes"


def iter_source_units(snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
    dat_gz = snapshot.root / "uniprot_sprot.dat.gz"
    for rec in iter_dat(dat_gz):
        yield SourceUnit(
            unit_id=f"uniprot_entry/{rec['accession']}",
            kind="protein_entry",
            payload_ref="uniprot_sprot.dat.gz",
            digest=record_sha256(rec),
            metadata={
                "entry_name": rec["entry_name"],
                "taxon_id": rec["taxon_id"],
                "gene_primary": rec["gene_primary"],
                "pe_level": rec["pe_level"],
                "n_features": len(rec["features"]),
                "n_variants": len(rec["variants"]),
                "n_isoforms": len(rec["isoforms"]),
                "n_go": len(rec["go_refs"]),
                "n_rhea": len(rec["rhea_refs"]),
            },
        )


def build_index(snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest:
    """Build the per-entry digest index (JSONL) + summary counts."""
    out_dir = Path(out_dir) / INDEX_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "uniprot_entry_index.jsonl"
    kind_counts: dict[str, int] = {}
    n = 0
    with index_path.open("w", encoding="utf-8") as fh:
        for unit in iter_source_units(snapshot):
            fh.write(_j({
                "unit_id": unit.unit_id, "kind": unit.kind,
                "digest": unit.digest, **unit.metadata}) + "\n")
            kind_counts[unit.kind] = kind_counts.get(unit.kind, 0) + 1
            n += 1
    return IndexManifest(
        source_id=snapshot.source_id, release_id=snapshot.release_id,
        out_dir=str(out_dir), unit_count=n, kind_counts=kind_counts)
