"""L14 source inventory: census-level index over the pinned ChEMBL 37 SQLite.

The pinned SQLite IS the complete snapshot (24.5M activities): a full
row-level re-materialization into JSONL would duplicate gigabytes for no
verification gain. This module therefore emits a census index:
  * full per-kind counts (computed live from the DB, never copied from
    release notes),
  * a deterministic, evenly-spaced sample of unit ids per kind
    (``rowid`` stride sampling; documented in the index manifest),
  * per-unit digests over the sampled rows.

Task construction reads the SQLite directly through
``benchmark/plugins/source/l14/kb.py``; the sample units here back the
SourcePlugin ``enumerate_units`` surface and cross-lane registry audits.

Unit kinds:
  chembl_molecule        molecule_dictionary (molregno, chembl_id)
  chembl_assay           assays (assay_id, chembl_id)
  chembl_activity        activities (activity_id)
  chembl_target          target_dictionary (tid, chembl_id)
  chembl_document        docs (doc_id, chembl_id)
  chembl_mechanism       drug_mechanism (mec_id)
  chembl_indication      drug_indication (drugind_id)
  chembl_component       component_sequences (component_id, accession)
  chembl_protein_class   protein_classification (protein_class_id)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import IndexManifest, SourceSnapshot, SourceUnit

INDEX_SUBDIR = "indexes"
SAMPLE_CAP_PER_KIND = 2048

KINDS: dict[str, dict[str, str]] = {
    "chembl_molecule": {
        "table": "molecule_dictionary", "pk": "molregno",
        "cols": "molregno, chembl_id, molecule_type, max_phase",
        "id_expr": "chembl_id"},
    "chembl_assay": {
        "table": "assays", "pk": "assay_id",
        "cols": "assay_id, chembl_id, assay_type, tid",
        "id_expr": "chembl_id"},
    "chembl_activity": {
        "table": "activities", "pk": "activity_id",
        "cols": "activity_id, assay_id, molregno, standard_type",
        "id_expr": "CAST(activity_id AS TEXT)"},
    "chembl_target": {
        "table": "target_dictionary", "pk": "tid",
        "cols": "tid, chembl_id, target_type, organism",
        "id_expr": "chembl_id"},
    "chembl_document": {
        "table": "docs", "pk": "doc_id",
        "cols": "doc_id, chembl_id, journal, year",
        "id_expr": "chembl_id"},
    "chembl_mechanism": {
        "table": "drug_mechanism", "pk": "mec_id",
        "cols": "mec_id, molregno, tid, action_type",
        "id_expr": "CAST(mec_id AS TEXT)"},
    "chembl_indication": {
        "table": "drug_indication", "pk": "drugind_id",
        "cols": "drugind_id, molregno, mesh_heading, max_phase_for_ind",
        "id_expr": "CAST(drugind_id AS TEXT)"},
    "chembl_component": {
        "table": "component_sequences", "pk": "component_id",
        "cols": "component_id, accession, organism",
        "id_expr": "CAST(component_id AS TEXT)"},
    "chembl_protein_class": {
        "table": "protein_classification", "pk": "protein_class_id",
        "cols": "protein_class_id, parent_id, pref_name, class_level",
        "id_expr": "CAST(protein_class_id AS TEXT)"},
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _db(snapshot: SourceSnapshot) -> Path:
    from source_registry.fetchers.l14_chembl import db_path
    return db_path(snapshot.root)


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def iter_source_units(snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
    """Yield a deterministic evenly-spaced sample of units per kind."""
    con = _connect(_db(snapshot))
    for kind, spec in KINDS.items():
        total = con.execute(f"SELECT count(*) FROM {spec['table']}").fetchone()[0]
        if total == 0:
            continue
        stride = max(1, total // SAMPLE_CAP_PER_KIND)
        sql = (f"SELECT {spec['cols']} FROM {spec['table']} "
               f"WHERE ({spec['pk']} % {stride}) = 0 ORDER BY {spec['pk']} "
               f"LIMIT {SAMPLE_CAP_PER_KIND}")
        for row in con.execute(sql):
            uid_sql = (f"SELECT {spec['id_expr']} FROM {spec['table']} "
                       f"WHERE {spec['pk']} = ?")
            uid = con.execute(uid_sql, (row[0],)).fetchone()[0]
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
            yield SourceUnit(
                unit_id=f"{kind}/{uid}",
                kind=kind,
                payload_ref=f"sqlite://{spec['table']}/{spec['pk']}={row[0]}",
                digest=_sha256_text(payload),
                metadata={"sampled": True, "stride": stride,
                          "kind_total": total},
            )
    con.close()


def build_index(snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest:
    """Write the census index: full counts + sampled unit shards + manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = _connect(_db(snapshot))
    kind_counts: dict[str, int] = {}
    shards: dict[str, dict[str, Any]] = {}
    for kind, spec in KINDS.items():
        total = con.execute(f"SELECT count(*) FROM {spec['table']}").fetchone()[0]
        kind_counts[kind] = total
        stride = max(1, total // SAMPLE_CAP_PER_KIND) if total else 0
        path = out_dir / f"{kind}.sample.jsonl"
        n = 0
        with path.open("w", encoding="utf-8") as fh:
            if total:
                sql = (f"SELECT {spec['cols']} FROM {spec['table']} "
                       f"WHERE ({spec['pk']} % {stride}) = 0 "
                       f"ORDER BY {spec['pk']} LIMIT {SAMPLE_CAP_PER_KIND}")
                uid_sql = (f"SELECT {spec['id_expr']} FROM {spec['table']} "
                           f"WHERE {spec['pk']} = ?")
                for row in con.execute(sql):
                    uid = con.execute(uid_sql, (row[0],)).fetchone()[0]
                    fh.write(json.dumps({
                        "unit_id": f"{kind}/{uid}",
                        "kind": kind,
                        "payload_ref": f"sqlite://{spec['table']}/"
                                       f"{spec['pk']}={row[0]}",
                        "digest": _sha256_text(json.dumps(
                            row, ensure_ascii=False, sort_keys=True)),
                        "stride": stride,
                    }, sort_keys=True, ensure_ascii=False) + "\n")
                    n += 1
        shards[kind] = {"sampled_units": n, "stride": stride,
                        "sha256": _sha256_text(path.read_text(encoding="utf-8"))
                        if n else ""}
    con.close()
    manifest = {
        "source_id": snapshot.source_id,
        "release_id": snapshot.release_id,
        "out_dir": str(out_dir),
        "unit_count": sum(kind_counts.values()),
        "kind_counts": kind_counts,
        "sample_cap_per_kind": SAMPLE_CAP_PER_KIND,
        "note": "census index: full per-kind counts + deterministic "
                "stride-sampled units; complete truth is the pinned SQLite",
        "shards": shards,
    }
    (out_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IndexManifest(
        source_id=snapshot.source_id, release_id=snapshot.release_id,
        out_dir=str(out_dir), unit_count=manifest["unit_count"],
        kind_counts=kind_counts)
