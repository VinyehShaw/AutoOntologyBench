"""L16 reference path: independent answer recomputation from the XML store.

Every query runs SQL against ``store/l16_uniprot_xml.sqlite`` — the store built
by the independent XML parser — so the reference path differs from the fast
path in BOTH the source bytes/parser (XML vs DAT) and the evaluation engine
(SQLite SQL vs in-memory Python views). verdict PASS iff exact set equality.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

_STORE_CACHE: dict[str, "sqlite3.Connection"] = {}


def _db(root: Path) -> "sqlite3.Connection":
    import sqlite3
    key = str(root)
    if key not in _STORE_CACHE:
        conn = sqlite3.connect(
            f"file:{root / 'store' / 'l16_uniprot_xml.sqlite'}?mode=ro",
            uri=True)
        conn.execute("PRAGMA case_sensitive_like=ON")
        _STORE_CACHE[key] = conn
    return _STORE_CACHE[key]


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reference_answer(root: Path, family_id: str, params: dict[str, Any]
                     ) -> tuple[str, ...]:
    db = _db(Path(root))
    if family_id == "org_proteins_by_gene":
        rows = db.execute(
            "SELECT accession FROM entries WHERE gene_primary=?",
            (params["gene"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_entries_by_taxon_keyword":
        rows = db.execute(
            "SELECT DISTINCT e.entry_name FROM entries e "
            "JOIN keywords k ON k.accession=e.accession "
            "WHERE e.taxon_id=? AND k.keyword=?",
            (params["taxon"], params["keyword"])).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "app_protein_existence_stratum":
        rows = db.execute(
            "SELECT accession FROM entries WHERE taxon_id=? AND pe_level=?",
            (params["taxon"], params["pe"])).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "app_function_evidence_eco":
        rows = db.execute(
            "SELECT DISTINCT f.accession FROM function_ecos f "
            "JOIN entries e ON e.accession=f.accession "
            "WHERE e.taxon_id=? AND f.eco=?",
            (params["taxon"], params["eco"])).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_go_annotation_by_taxon":
        rows = db.execute(
            "SELECT DISTINCT g.accession FROM go_refs g "
            "JOIN entries e ON e.accession=g.accession "
            "WHERE e.taxon_id=? AND g.go_id=?",
            (params["taxon"], params["go"])).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "app_ec_class_members":
        rows = db.execute(
            "SELECT DISTINCT accession FROM ec WHERE ec=? OR ec LIKE ?",
            (params["ec_prefix"], params["ec_prefix"] + ".%")).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_rhea_catalysts":
        rows = db.execute(
            "SELECT accession FROM rhea WHERE rhea_id=?",
            (params["rhea"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_subcellular_location":
        rows = db.execute(
            "SELECT DISTINCT s.accession FROM subcellular s "
            "JOIN entries e ON e.accession=s.accession "
            "WHERE e.taxon_id=? AND s.term=?",
            (params["taxon"], params["location"])).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "rep_feature_boundary_repair":
        rows = db.execute(
            "SELECT begin, end FROM features WHERE accession=? AND ftype=? "
            "AND begin NOT LIKE '%?%' AND begin NOT LIKE '<%' "
            "AND begin NOT LIKE '>%' AND end NOT LIKE '%?%' "
            "AND end NOT LIKE '<%' AND end NOT LIKE '>%'",
            (params["accession"], params["ftype"])).fetchall()
        return tuple(sorted(f"{params['ftype']}|{b}|{e}" for b, e in rows))
    if family_id == "org_disease_associated_proteins":
        rows = db.execute(
            "SELECT accession FROM diseases WHERE name=?",
            (params["disease"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_variant_inventory":
        rows = db.execute(
            "SELECT var_id FROM variants WHERE accession=? AND var_id != ''",
            (params["accession"],)).fetchall()
        return tuple(sorted({r[0] for r in rows}))
    if family_id == "app_isoform_inventory":
        rows = db.execute(
            "SELECT iso_id FROM isoforms WHERE accession=?",
            (params["accession"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_xref_provenance":
        rows = db.execute(
            "SELECT id FROM xrefs WHERE accession=? AND db=?",
            (params["accession"], params["db"])).fetchall()
        return tuple(sorted({r[0] for r in rows}))
    if family_id == "org_literature_provenance":
        rows = db.execute(
            "SELECT accession FROM pubmeds WHERE pubmed_id=?",
            (params["pubmed"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "org_proteome_membership":
        rows = db.execute(
            "SELECT accession FROM xrefs WHERE db='Proteomes' AND id=?",
            (params["proteome"],)).fetchall()
        return tuple(sorted(r[0] for r in rows))
    if family_id == "con_ec_rhea_catalyst_map":
        rows = db.execute(
            "SELECT DISTINCT cp.ec, cp.rhea_id FROM catalytic_pairs cp "
            "JOIN entries e ON e.accession=cp.accession WHERE e.taxon_id=?",
            (params["taxon"],)).fetchall()
        return tuple(sorted(f"{ec}|{rid}" for ec, rid in rows))
    raise ValueError(f"no reference query for family {family_id}")


def verify_reference(root: Path, family_id: str, params: dict[str, Any],
                     expected: tuple[str, ...], meta: dict[str, str]
                     ) -> dict[str, Any]:
    got = reference_answer(root, family_id, params)
    match = set(got) == set(expected)
    return {
        "verdict": "PASS" if match else "FAIL",
        "matches_fast_path": match,
        "engine": "sqlite3 (stdlib) over XML-parser store "
                  + meta.get("parser", "?"),
        "store": "l16_uniprot_xml.sqlite",
        "store_release": meta.get("release_id", ""),
        "store_source_sha256": meta.get("source_sha256", ""),
        "query_family": family_id,
        "query_sha256": _sha({"family": family_id, "params": params}),
        "answer_sha256": _sha(sorted(set(got))),
        "answer_count": len(set(got)),
        "expected_sha256": _sha(sorted(set(expected))),
    }
