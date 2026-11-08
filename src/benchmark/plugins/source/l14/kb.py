"""L14 KB access layer over the pinned ChEMBL 37 SQLite.

Read-only connections (``mode=ro&immutable=1``); every process opens its own
connection. Small lookup needs go through indexed point queries; the heavier
sampling pools used by the task families are aggregate GROUP BY extracts
cached on disk next to the census index (``pool_cache.json``) so every worker
starts fast and every pool is recomputed deterministically from the pinned
bytes (pool sha256 recorded in the build report).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "chembl"
                    / "chembl-37")
POOL_CACHE_NAME = "indexes/pool_cache.json"


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only = ON")
    con.execute("PRAGMA temp_store = MEMORY")
    return con


@dataclass
class L14KB:
    pin_root: Path
    db: Path
    _pools: dict[str, Any] | None = None

    def connection(self) -> sqlite3.Connection:
        return connect(self.db)

    # -- point lookups ------------------------------------------------------
    def molecule(self, con: sqlite3.Connection, molregno: int) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT molregno, chembl_id, pref_name, molecule_type, max_phase "
            "FROM molecule_dictionary WHERE molregno = ?", (molregno,)).fetchone()
        if not row:
            return None
        return {"molregno": row[0], "chembl_id": row[1], "pref_name": row[2],
                "molecule_type": row[3], "max_phase": row[4]}

    def molregno(self, con: sqlite3.Connection, chembl_id: str) -> int | None:
        row = con.execute(
            "SELECT molregno FROM molecule_dictionary WHERE chembl_id = ?",
            (chembl_id,)).fetchone()
        return row[0] if row else None

    def target(self, con: sqlite3.Connection, tid: int) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT tid, chembl_id, pref_name, target_type, organism, tax_id "
            "FROM target_dictionary WHERE tid = ?", (tid,)).fetchone()
        if not row:
            return None
        return {"tid": row[0], "chembl_id": row[1], "pref_name": row[2],
                "target_type": row[3], "organism": row[4], "tax_id": row[5]}

    def assay(self, con: sqlite3.Connection, assay_id: int) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT assay_id, chembl_id, assay_type, assay_organism, tid, "
            "doc_id, confidence_score, description FROM assays "
            "WHERE assay_id = ?", (assay_id,)).fetchone()
        if not row:
            return None
        return {"assay_id": row[0], "chembl_id": row[1], "assay_type": row[2],
                "assay_organism": row[3], "tid": row[4], "doc_id": row[5],
                "confidence_score": row[6], "description": row[7]}

    def document(self, con: sqlite3.Connection, doc_id: int) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT doc_id, chembl_id, journal, year, doi, pubmed_id "
            "FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        if not row:
            return None
        return {"doc_id": row[0], "chembl_id": row[1], "journal": row[2],
                "year": row[3], "doi": row[4], "pubmed_id": row[5]}

    # -- pool cache ----------------------------------------------------------
    def pools(self) -> dict[str, Any]:
        if self._pools is not None:
            return self._pools
        path = self.pin_root / POOL_CACHE_NAME
        if path.is_file():
            self._pools = json.loads(path.read_text(encoding="utf-8"))["pools"]
            return self._pools
        self._pools = build_pool_cache(self.pin_root, self.db,
                                       out_path=path)["pools"]
        return self._pools


def load_kb(pin_root: str | Path) -> L14KB:
    from source_registry.fetchers.l14_chembl import db_path
    root = Path(pin_root)
    return L14KB(pin_root=root, db=db_path(root))


def _rows(con: sqlite3.Connection, sql: str, args: Iterable[Any] = ()):
    return con.execute(sql, tuple(args)).fetchall()


def build_pool_cache(pin_root: Path, db: Path, *,
                     out_path: Path | None = None) -> dict[str, Any]:
    """Aggregate sampling pools; recomputed from the pinned DB (never shipped
    numbers). Only small aggregate maps are kept; row truth stays in SQLite."""
    con = connect(db)
    pools: dict[str, Any] = {}

    pools["parents_with_children"] = [
        {"parent_molregno": r[0], "n_children": r[1]}
        for r in _rows(con, """
            SELECT parent_molregno, count(*) FROM molecule_hierarchy
            GROUP BY parent_molregno HAVING count(*) BETWEEN 3 AND 60
            ORDER BY parent_molregno""")]

    pools["target_measurement"] = [
        {"tid": r[0], "standard_type": r[1], "relation": r[2], "units": r[3],
         "n_molecules": r[4]}
        for r in _rows(con, """
            SELECT a.tid, act.standard_type, act.standard_relation,
                   act.standard_units, count(DISTINCT act.molregno)
            FROM activities act JOIN assays a ON act.assay_id = a.assay_id
            WHERE a.assay_type = 'B' AND a.confidence_score >= 8
              AND a.tid IS NOT NULL
              AND act.standard_type IN ('IC50','Ki','EC50','Kd','Potency')
              AND act.standard_relation = '='
              AND act.standard_units = 'nM'
              AND act.standard_value IS NOT NULL
              AND act.molregno IS NOT NULL
            GROUP BY a.tid, act.standard_type, act.standard_relation,
                     act.standard_units
            HAVING count(DISTINCT act.molregno) BETWEEN 30 AND 20000
            ORDER BY a.tid""")]

    pools["assay_pubunits_mix"] = [
        {"assay_id": r[0], "standard_type": r[1], "n_pub_units": r[2],
         "n_molecules": r[3]}
        for r in _rows(con, """
            SELECT act.assay_id, act.standard_type,
                   count(DISTINCT act.units), count(DISTINCT act.molregno)
            FROM activities act
            WHERE act.standard_units = 'nM' AND act.standard_relation = '='
              AND act.standard_type IN ('IC50','Ki','EC50')
              AND act.units IS NOT NULL
              AND act.molregno IS NOT NULL
              AND act.standard_value IS NOT NULL
            GROUP BY act.assay_id, act.standard_type
            HAVING count(DISTINCT act.units) >= 2
               AND count(DISTINCT act.molregno) BETWEEN 15 AND 400
            ORDER BY act.assay_id""")]

    pools["assay_measurement"] = [
        {"assay_id": r[0], "standard_type": r[1], "units": r[2],
         "n_molecules": r[3]}
        for r in _rows(con, """
            SELECT act.assay_id, act.standard_type, act.standard_units,
                   count(DISTINCT act.molregno)
            FROM activities act
            WHERE act.standard_relation = '='
              AND act.standard_type IN ('IC50','Ki','EC50','Potency')
              AND act.standard_units = 'nM'
              AND act.standard_value IS NOT NULL
              AND act.molregno IS NOT NULL
            GROUP BY act.assay_id, act.standard_type, act.standard_units
            HAVING count(DISTINCT act.molregno) BETWEEN 15 AND 400
            ORDER BY act.assay_id""")]

    pools["target_organism_assays"] = [
        {"tid": r[0], "assay_organism": r[1], "n_assays": r[2]}
        for r in _rows(con, """
            SELECT tid, assay_organism, count(*) FROM assays
            WHERE assay_type = 'B' AND assay_organism IS NOT NULL
              AND tid IS NOT NULL
            GROUP BY tid, assay_organism
            HAVING count(*) BETWEEN 8 AND 4000
            ORDER BY tid""")]

    pools["molecule_assay_type"] = [
        {"molregno": r[0], "standard_type": r[1], "n_assays": r[2]}
        for r in _rows(con, """
            SELECT act.molregno, act.standard_type,
                   count(DISTINCT act.assay_id)
            FROM activities act
            WHERE act.standard_relation = '='
              AND act.standard_type IN ('IC50','Ki','EC50','Kd')
              AND act.standard_units = 'nM'
              AND act.standard_value IS NOT NULL
              AND act.molregno IS NOT NULL
            GROUP BY act.molregno, act.standard_type
            HAVING count(DISTINCT act.assay_id) BETWEEN 6 AND 300
            ORDER BY act.molregno""")]

    # protein classification closure (small hierarchy, walked in Python)
    class_rows = _rows(con, """
        SELECT protein_class_id, parent_id, pref_name, class_level
        FROM protein_classification""")
    children: dict[int, list[int]] = {}
    info: dict[int, dict[str, Any]] = {}
    for cid, parent, name, level in class_rows:
        info[cid] = {"pref_name": name, "class_level": level,
                     "parent_id": parent}
        if parent:
            children.setdefault(parent, []).append(cid)

    def subtree(root: int) -> set[int]:
        seen: set[int] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(children.get(node, ()))
        return seen

    comp2classes: dict[int, set[int]] = {}
    for comp, cid in _rows(con, "SELECT component_id, protein_class_id "
                                "FROM component_class"):
        comp2classes.setdefault(comp, set()).add(cid)
    tid2comps: dict[int, set[int]] = {}
    for tid, comp in _rows(con, "SELECT tid, component_id FROM target_components"):
        tid2comps.setdefault(tid, set()).add(comp)
    human_sp = {r[0] for r in _rows(con, """
        SELECT tid FROM target_dictionary
        WHERE target_type = 'SINGLE PROTEIN' AND organism = 'Homo sapiens'""")}

    class_target_counts: list[dict[str, Any]] = []
    for cid, meta in sorted(info.items()):
        if meta["class_level"] not in (2, 3):
            continue
        sub = subtree(cid)
        tids: set[int] = set()
        for tid in human_sp:
            for comp in tid2comps.get(tid, ()):
                if comp2classes.get(comp, set()) & sub:
                    tids.add(tid)
                    break
        if 6 <= len(tids) <= 400:
            class_target_counts.append({
                "protein_class_id": cid, "pref_name": meta["pref_name"],
                "class_level": meta["class_level"], "n_targets": len(tids)})
    pools["protein_class_targets"] = class_target_counts

    pools["mechanism_targets"] = [
        {"tid": r[0], "action_type": r[1], "n_parents": r[2]}
        for r in _rows(con, """
            SELECT m.tid, m.action_type, count(DISTINCT h.parent_molregno)
            FROM drug_mechanism m
            JOIN molecule_hierarchy h ON h.molregno = m.molregno
            WHERE m.action_type IS NOT NULL
            GROUP BY m.tid, m.action_type
            HAVING count(DISTINCT h.parent_molregno) BETWEEN 4 AND 300
            ORDER BY m.tid""")]

    pools["indication_mesh"] = [
        {"mesh_heading": r[0], "n_parents": r[1]}
        for r in _rows(con, """
            SELECT d.mesh_heading, count(DISTINCT h.parent_molregno)
            FROM drug_indication d
            JOIN molecule_hierarchy h ON h.molregno = d.molregno
            WHERE d.mesh_heading IS NOT NULL AND d.max_phase_for_ind >= 2
            GROUP BY d.mesh_heading
            HAVING count(DISTINCT h.parent_molregno) BETWEEN 4 AND 200
            ORDER BY d.mesh_heading""")]

    pools["uniprot_targets"] = [
        {"accession": r[0], "n_tids": r[1]}
        for r in _rows(con, """
            SELECT c.accession, count(DISTINCT tc.tid)
            FROM component_sequences c
            JOIN target_components tc ON tc.component_id = c.component_id
            JOIN target_dictionary t ON t.tid = tc.tid
            WHERE c.accession IS NOT NULL
              AND t.target_type = 'SINGLE PROTEIN'
              AND t.organism = 'Homo sapiens'
            GROUP BY c.accession
            HAVING count(DISTINCT tc.tid) BETWEEN 1 AND 6
            ORDER BY c.accession""")]

    pools["journal_year_activity"] = [
        {"journal": r[0], "year": r[1], "n_activities": r[2],
         "n_molecules": r[3]}
        for r in _rows(con, """
            SELECT d.journal, d.year, count(*), count(DISTINCT act.molregno)
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            JOIN docs d ON a.doc_id = d.doc_id
            WHERE d.journal IS NOT NULL AND d.year IS NOT NULL
              AND act.standard_relation = '='
              AND act.standard_type IN ('IC50','Ki','EC50')
              AND act.standard_units = 'nM'
              AND act.molregno IS NOT NULL
            GROUP BY d.journal, d.year
            HAVING count(DISTINCT act.molregno) BETWEEN 25 AND 5000
            ORDER BY d.journal, d.year""")]

    pools["target_journal_multihop"] = [
        {"tid": r[0], "journal": r[1], "n_molecules": r[2]}
        for r in _rows(con, """
            SELECT a.tid, d.journal, count(DISTINCT act.molregno)
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            JOIN docs d ON a.doc_id = d.doc_id
            WHERE a.assay_type = 'B' AND a.confidence_score >= 8
              AND d.journal IS NOT NULL
              AND act.standard_relation = '='
              AND act.standard_type = 'IC50'
              AND act.standard_units = 'nM'
              AND act.molregno IS NOT NULL
            GROUP BY a.tid, d.journal
            HAVING count(DISTINCT act.molregno) BETWEEN 8 AND 3000
            ORDER BY a.tid""")]

    pools["assays_per_target_doc"] = [
        {"tid": r[0], "n_assays": r[1]}
        for r in _rows(con, """
            SELECT tid, count(*) FROM assays
            WHERE assay_type = 'B' AND doc_id IS NOT NULL AND tid IS NOT NULL
              AND confidence_score >= 8
            GROUP BY tid HAVING count(*) BETWEEN 12 AND 2000
            ORDER BY tid""")]

    pin_file = pin_root / "chembl_37_db.sha256"
    meta = {
        "built_from_db_sha256": (
            pin_file.read_text().strip() if pin_file.is_file() else None),
        "pools": sorted(pools),
    }
    payload = {"meta": meta, "pools": pools}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                            encoding="utf-8")
    con.close()
    return payload
