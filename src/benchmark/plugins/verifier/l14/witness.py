"""L14 semantic witnesses: counterfactual perturbation evidence per family.

For every candidate we build an in-memory SQLite slice from the pinned DB
(same table names/columns the family predicates read), then compute:
  * governed probe: family answer on the unmutated slice (must EQUAL the
    full-DB reference answer — proves slice completeness),
  * perturbed probe: family answer on the mutated slice (must DIFFER),
  * control probe: a bystander answer/query on both slices (must be
    UNCHANGED).

A candidate is NON_SEMANTIC when the seeded perturbation fails to flip the
governed probe or also moves the control probe. Slice content hash recorded.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from typing import Any, Callable

from benchmark.plugins.verifier.l14 import pyclean

SLICE_COLUMNS: dict[str, list[str]] = {
    "molecule_dictionary": ["molregno", "chembl_id"],
    "molecule_hierarchy": ["molregno", "parent_molregno"],
    "assays": ["assay_id", "chembl_id", "tid", "assay_type",
               "assay_organism", "confidence_score", "doc_id"],
    "activities": ["activity_id", "assay_id", "molregno", "standard_type",
                   "standard_relation", "standard_value", "standard_units"],
    "target_dictionary": ["tid", "chembl_id", "target_type", "organism"],
    "docs": ["doc_id", "journal", "year"],
    "drug_mechanism": ["mec_id", "molregno", "tid", "action_type"],
    "drug_indication": ["drugind_id", "molregno", "mesh_heading",
                        "max_phase_for_ind"],
    "component_sequences": ["component_id", "accession"],
    "target_components": ["tid", "component_id"],
    "component_class": ["component_id", "protein_class_id"],
    "protein_classification": ["protein_class_id", "parent_id"],
}

COL_TYPES = {
    "molregno": "INTEGER", "chembl_id": "TEXT", "parent_molregno": "INTEGER",
    "assay_id": "INTEGER", "tid": "INTEGER", "assay_type": "TEXT",
    "assay_organism": "TEXT", "confidence_score": "INTEGER",
    "doc_id": "INTEGER", "activity_id": "INTEGER", "standard_type": "TEXT",
    "standard_relation": "TEXT", "standard_value": "REAL",
    "standard_units": "TEXT", "target_type": "TEXT", "organism": "TEXT",
    "journal": "TEXT", "year": "INTEGER", "mec_id": "INTEGER",
    "action_type": "TEXT", "drugind_id": "INTEGER", "mesh_heading": "TEXT",
    "max_phase_for_ind": "REAL", "component_id": "INTEGER",
    "accession": "TEXT", "protein_class_id": "INTEGER", "parent_id": "INTEGER",
}

DECOY_DOC_ID = 9_999_999_001
DECOY_PARENT = 9_999_999_002
DECOY_TEXT = "ZZZ_AOB_DECOY"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _one(con, sql, args=()):
    r = con.execute(sql, args).fetchone()
    return r[0] if r else None


def _mk_slice() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for table, cols in SLICE_COLUMNS.items():
        defs = ", ".join(f"{c} {COL_TYPES[c]}" for c in cols)
        con.execute(f"CREATE TABLE {table} ({defs})")
    return con


def _copy(src, dst, table: str, where: str, args: tuple[Any, ...] = ()) -> int:
    cols = SLICE_COLUMNS[table]
    rows = src.execute(
        f"SELECT {', '.join(cols)} FROM {table} WHERE {where}",
        args).fetchall()
    marks = ",".join("?" * len(cols))
    dst.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks})", rows)
    return len(rows)


def _copy_in_chunks(src, dst, table: str, col: str,
                    ids: list[Any]) -> int:
    """IN-list copy chunked below SQLite's variable limit."""
    total = 0
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        total += _copy(src, dst, table, f"{col} IN ({marks})", tuple(chunk))
    return total


def _copy_molecules_for_activities(src, dst) -> None:
    mols = [r[0] for r in dst.execute(
        "SELECT DISTINCT molregno FROM activities WHERE molregno "
        "IS NOT NULL")]
    _copy_in_chunks(src, dst, "molecule_dictionary", "molregno", mols)
    _copy_in_chunks(src, dst, "molecule_hierarchy", "molregno", mols)


def _fingerprint(con) -> str:
    parts = []
    for table in sorted(SLICE_COLUMNS):
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        parts.append(f"{table}:{n}")
    return _sha("|".join(parts))


def _unique_support_activity(slice_con, base_where: str,
                             args: tuple[Any, ...]) -> int | None:
    """activity_id whose molecule has exactly one qualifying activity row
    under ``base_where`` (mutating it provably removes that molecule)."""
    row = slice_con.execute(
        "SELECT activity_id FROM activities WHERE activity_id IN ("
        "  SELECT MIN(activity_id) FROM activities "
        "  WHERE " + base_where + " AND molregno IS NOT NULL "
        "  GROUP BY molregno HAVING count(*) = 1) "
        "ORDER BY activity_id LIMIT 1", args).fetchone()
    return row[0] if row else None


def _weakest_molecule_rows(slice_con, base_where: str,
                           args: tuple[Any, ...]) -> list[int]:
    """All qualifying activity ids of the molecule with the fewest qualifying
    rows — deleting/mutating all of them provably removes that molecule."""
    row = slice_con.execute(
        "SELECT molregno, count(*) c FROM activities "
        "WHERE " + base_where + " AND molregno IS NOT NULL "
        "GROUP BY molregno ORDER BY c, molregno LIMIT 1", args).fetchone()
    if row is None:
        return []
    return [r[0] for r in slice_con.execute(
        "SELECT activity_id FROM activities WHERE " + base_where +
        " AND molregno = ? ORDER BY activity_id", args + (row[0],))]


def _delete_weakest_assay_support(slice_con, base_where: str,
                                  args: tuple[Any, ...]) -> dict | None:
    """Assay-set answers: delete all qualifying rows of the assay with the
    fewest qualifying rows (drops that assay from the answer)."""
    row = slice_con.execute(
        "SELECT assay_id, count(*) c FROM activities "
        "WHERE " + base_where + " AND molregno IS NOT NULL "
        "GROUP BY assay_id ORDER BY c, assay_id LIMIT 1", args).fetchone()
    if row is None:
        return None
    ids = [r[0] for r in slice_con.execute(
        "SELECT activity_id FROM activities WHERE " + base_where +
        " AND assay_id = ?", args + (row[0],))]
    marks = ",".join("?" * len(ids))
    slice_con.execute(f"DELETE FROM activities WHERE activity_id IN "
                      f"({marks})", tuple(ids))
    return {"op": "delete_weakest_assay_rows", "assay_id": row[0],
            "n_rows": len(ids)}


def _tid_of(con, chembl_id):
    return _one(con, "SELECT tid FROM target_dictionary WHERE chembl_id = ?",
                (chembl_id,))


def _molregno(con, chembl_id):
    return _one(con, "SELECT molregno FROM molecule_dictionary "
                     "WHERE chembl_id = ?", (chembl_id,))


def _assay_id(con, chembl_id):
    return _one(con, "SELECT assay_id FROM assays WHERE chembl_id = ?",
                (chembl_id,))


def _control_target(src, tid: int) -> int | None:
    return _one(src, "SELECT tid FROM assays WHERE tid IS NOT NULL "
                     "AND tid != ? GROUP BY tid "
                     "HAVING count(*) BETWEEN 5 AND 500 "
                     "ORDER BY tid LIMIT 1", (tid,))


def _scope_target_with_control(src, slc, tid: int, control_tid: int):
    _copy(src, slc, "target_dictionary", "tid IN (?, ?)", (tid, control_tid))
    _copy(src, slc, "assays", "tid IN (?, ?)", (tid, control_tid))
    aids = [r[0] for r in slc.execute("SELECT assay_id FROM assays")]
    _copy_in_chunks(src, slc, "activities", "assay_id", aids)
    _copy_molecules_for_activities(src, slc)
    dids = [r[0] for r in slc.execute(
        "SELECT DISTINCT doc_id FROM assays WHERE doc_id IS NOT NULL")]
    _copy_in_chunks(src, slc, "docs", "doc_id", dids)


def _slice_parent(src, params, rng):
    parent = _molregno(src, params["parent_chembl_id"])
    if parent is None:
        return None
    ctl = _one(src, "SELECT parent_molregno FROM molecule_hierarchy "
                    "WHERE parent_molregno != ? GROUP BY parent_molregno "
                    "HAVING count(*) >= 3 ORDER BY parent_molregno LIMIT 1",
               (parent,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "molecule_hierarchy", "parent_molregno IN (?, ?)",
          (parent, ctl))
    mols = {r[0] for r in slc.execute(
        "SELECT molregno FROM molecule_hierarchy")}
    mols |= {parent, ctl}
    _copy_in_chunks(src, slc, "molecule_dictionary", "molregno",
                    sorted(mols))

    def mutate(s):
        child = _one(s, "SELECT molregno FROM molecule_hierarchy "
                        "WHERE parent_molregno = ? AND molregno != ? "
                        "ORDER BY molregno LIMIT 1", (parent, parent))
        if child is None:
            return None
        s.execute("UPDATE molecule_hierarchy SET parent_molregno = ? "
                  "WHERE molregno = ?", (DECOY_PARENT, child))
        return {"op": "reparent_child", "molregno": child}

    def control(s):
        return s.execute("SELECT count(*) FROM molecule_hierarchy "
                         "WHERE parent_molregno = ?", (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_molecule(src, params, rng):
    mol = _molregno(src, params["molecule_chembl_id"])
    if mol is None:
        return None
    ctl = _one(src, "SELECT molregno FROM activities WHERE molregno "
                    "IS NOT NULL AND molregno != ? GROUP BY molregno "
                    "HAVING count(*) BETWEEN 5 AND 200 "
                    "ORDER BY molregno LIMIT 1", (mol,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "activities", "molregno IN (?, ?)", (mol, ctl))
    aids = [r[0] for r in slc.execute(
        "SELECT DISTINCT assay_id FROM activities")]
    _copy_in_chunks(src, slc, "assays", "assay_id", aids)
    _copy_molecules_for_activities(src, slc)
    tids = [r[0] for r in slc.execute(
        "SELECT DISTINCT tid FROM assays WHERE tid IS NOT NULL")]
    _copy_in_chunks(src, slc, "target_dictionary", "tid", tids)
    st = params["standard_type"]
    base = ("standard_type = ? AND standard_relation = '=' "
            "AND standard_units = 'nM' AND standard_value IS NOT NULL "
            "AND molregno = ?")

    def mutate(s):
        row = s.execute(
            "SELECT MIN(activity_id), assay_id FROM activities "
            "WHERE " + base + " GROUP BY assay_id HAVING count(*) = 1 "
            "ORDER BY assay_id LIMIT 1", (st, mol)).fetchone()
        if row is not None:
            s.execute("DELETE FROM activities WHERE activity_id = ?",
                      (row[0],))
            return {"op": "delete_unique_support_row",
                    "activity_id": row[0], "assay_id": row[1]}
        return _delete_weakest_assay_support(s, base, (st, mol))

    def control(s):
        return s.execute(
            "SELECT count(*) FROM activities WHERE molregno = ?",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_target(src, params, rng):
    tid = _tid_of(src, params["target_chembl_id"])
    if tid is None:
        return None
    ctl = _control_target(src, tid)
    if ctl is None:
        return None
    slc = _mk_slice()
    _scope_target_with_control(src, slc, tid, ctl)
    return slc, tid, ctl


def _slice_target_binding(src, params, rng):
    built = _slice_target(src, params, rng)
    if built is None:
        return None
    slc, tid, ctl = built
    st, thr = params["standard_type"], params["threshold"]
    base = ("standard_type = ? AND standard_relation = '=' "
            "AND standard_units = 'nM' AND standard_value IS NOT NULL "
            "AND standard_value <= ? AND assay_id IN ("
            "  SELECT assay_id FROM assays WHERE tid = ? "
            "  AND assay_type = 'B' AND confidence_score >= 8)")

    def mutate(s):
        aid = _unique_support_activity(s, base, (st, thr, tid))
        if aid is None:
            ids = _weakest_molecule_rows(s, base, (st, thr, tid))
            if not ids:
                return None
            marks = ",".join("?" * len(ids))
            s.execute("UPDATE activities SET standard_value = "
                      "standard_value * 1000000.0 WHERE activity_id IN ("
                      + marks + ")", tuple(ids))
            return {"op": "rescale_weakest_molecule_rows_x1e6",
                    "n_rows": len(ids)}
        s.execute("UPDATE activities SET standard_value = "
                  "standard_value * 1000000.0 WHERE activity_id = ?",
                  (aid,))
        return {"op": "rescale_value_x1e6", "activity_id": aid}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM activities WHERE assay_id IN ("
            "  SELECT assay_id FROM assays WHERE tid = ?)",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_assay_context(src, params, rng):
    built = _slice_target(src, params, rng)
    if built is None:
        return None
    slc, tid, ctl = built
    org = params["assay_organism"]

    def mutate(s):
        aid = _one(s, "SELECT assay_id FROM assays WHERE tid = ? "
                      "AND assay_type = 'B' AND assay_organism = ? "
                      "ORDER BY assay_id LIMIT 1", (tid, org))
        if aid is None:
            return None
        s.execute("UPDATE assays SET assay_organism = ? WHERE assay_id = ?",
                  (DECOY_TEXT, aid))
        return {"op": "reassign_organism", "assay_id": aid}

    def control(s):
        return s.execute("SELECT count(*) FROM assays WHERE tid = ?",
                         (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_assay(src, params, rng):
    aid = _assay_id(src, params["assay_chembl_id"])
    if aid is None:
        return None
    ctl = _one(src, "SELECT assay_id FROM activities WHERE assay_id != ? "
                    "GROUP BY assay_id HAVING count(*) BETWEEN 5 AND 200 "
                    "ORDER BY assay_id LIMIT 1", (aid,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "assays", "assay_id IN (?, ?)", (aid, ctl))
    _copy(src, slc, "activities", "assay_id IN (?, ?)", (aid, ctl))
    _copy_molecules_for_activities(src, slc)
    tids = [r[0] for r in slc.execute(
        "SELECT DISTINCT tid FROM assays WHERE tid IS NOT NULL")]
    _copy_in_chunks(src, slc, "target_dictionary", "tid", tids)
    return slc, aid, ctl


def _slice_measurement(src, params, rng):
    built = _slice_assay(src, params, rng)
    if built is None:
        return None
    slc, aid, ctl = built
    st = params["standard_type"]
    thr = params.get("threshold")
    base = ("standard_type = ? AND standard_relation = '=' "
            "AND standard_units = 'nM' AND standard_value IS NOT NULL "
            "AND assay_id = ?")
    args: tuple[Any, ...] = (st, aid)
    if thr is not None:
        base += " AND standard_value <= ?"
        args = (st, aid, thr)

    def mutate(s):
        x = _unique_support_activity(s, base, args)
        if x is None:
            ids = _weakest_molecule_rows(s, base, args)
            if not ids:
                return None
            marks = ",".join("?" * len(ids))
            s.execute("UPDATE activities SET standard_relation = '<' "
                      "WHERE activity_id IN (" + marks + ")", tuple(ids))
            return {"op": "flip_weakest_molecule_relations",
                    "n_rows": len(ids)}
        s.execute("UPDATE activities SET standard_relation = '<' "
                  "WHERE activity_id = ?", (x,))
        return {"op": "flip_relation_to_lt", "activity_id": x}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM activities WHERE assay_id = ?",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_target_class(src, params, rng):
    slc = _mk_slice()
    _copy(src, slc, "protein_classification", "1=1")
    _copy(src, slc, "component_class", "1=1")
    _copy(src, slc, "target_components", "1=1")
    _copy(src, slc, "target_dictionary",
          "target_type = 'SINGLE PROTEIN' AND organism = 'Homo sapiens'")
    pcid = params["protein_class_id"]
    children: dict[int, list[int]] = {}
    for c, p in slc.execute("SELECT protein_class_id, parent_id FROM "
                            "protein_classification"):
        if p is not None:
            children.setdefault(p, []).append(c)

    def subtree(root):
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(children.get(n, ()))
        return seen

    sub = subtree(pcid)
    ctl_class = _one(src, "SELECT protein_class_id FROM "
                          "protein_classification WHERE class_level = 2 "
                          "AND protein_class_id != ? ORDER BY "
                          "protein_class_id LIMIT 1", (pcid,))
    if ctl_class is None:
        slc.close()
        return None
    forbidden = sub | subtree(ctl_class)
    outside = [c for (c,) in slc.execute(
        "SELECT protein_class_id FROM protein_classification")
        if c not in forbidden]
    if not outside:
        slc.close()
        return None
    decoy_class = sorted(outside)[len(outside) // 2]

    def mutate(s):
        row = s.execute(
            "SELECT MIN(cc.component_id), tc.tid FROM component_class cc "
            "JOIN target_components tc "
            "  ON tc.component_id = cc.component_id "
            "JOIN target_dictionary td ON td.tid = tc.tid "
            "WHERE cc.protein_class_id IN ("
            + ",".join(str(x) for x in sorted(sub)) + ") "
            "AND td.target_type = 'SINGLE PROTEIN' "
            "AND td.organism = 'Homo sapiens' "
            "GROUP BY tc.tid "
            "HAVING count(DISTINCT cc.component_id) = 1 "
            "ORDER BY tc.tid LIMIT 1").fetchone()
        if row is None:
            return None
        comp, _t = row
        s.execute("UPDATE component_class SET protein_class_id = ? "
                  "WHERE component_id = ? AND protein_class_id IN ("
                  + ",".join(str(x) for x in sorted(sub)) + ")",
                  (decoy_class, comp))
        return {"op": "relabel_component_class", "component_id": comp,
                "to_class": decoy_class}

    def control(s):
        return len(pyclean.answer_org_target_class_members(
            s, {"protein_class_id": ctl_class}))

    return slc, mutate, control


def _slice_journal(src, params, rng):
    j, y = params["journal"], params.get("year")
    cj = _one(src, "SELECT DISTINCT journal FROM docs WHERE journal IS NOT "
                   "NULL AND journal != ? AND year = ? "
                   "ORDER BY journal LIMIT 1", (j, y))
    if cj is None:
        cj = _one(src, "SELECT DISTINCT journal FROM docs WHERE journal "
                       "IS NOT NULL AND journal != ? ORDER BY journal "
                       "LIMIT 1", (j,))
    if cj is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "docs", "journal IN (?, ?) AND year = ?", (j, cj, y))
    dids = [r[0] for r in slc.execute("SELECT doc_id FROM docs")]
    if not dids:
        slc.close()
        return None
    _copy_in_chunks(src, slc, "assays", "doc_id", dids)
    aids = [r[0] for r in slc.execute("SELECT assay_id FROM assays")]
    _copy_in_chunks(src, slc, "activities", "assay_id", aids)
    _copy_molecules_for_activities(src, slc)
    tids = [r[0] for r in slc.execute(
        "SELECT DISTINCT tid FROM assays WHERE tid IS NOT NULL")]
    _copy_in_chunks(src, slc, "target_dictionary", "tid", tids)
    base = ("standard_relation = '=' AND standard_units = 'nM' "
            "AND standard_type IN ('IC50','Ki','EC50') "
            "AND standard_value IS NOT NULL AND molregno IS NOT NULL "
            "AND assay_id IN (SELECT assay_id FROM assays WHERE doc_id IN ("
            "  SELECT doc_id FROM docs WHERE journal = ? AND year = ?))")

    def mutate(s):
        ids = _weakest_molecule_rows(s, base, (j, y))
        if not ids:
            return None
        marks = ",".join("?" * len(ids))
        aids = [r[0] for r in s.execute(
            f"SELECT DISTINCT assay_id FROM activities WHERE "
            f"activity_id IN ({marks})", tuple(ids))]
        s.execute("INSERT INTO docs (doc_id, journal, year) "
                  "VALUES (?, ?, ?)", (DECOY_DOC_ID, DECOY_TEXT, 1900))
        amarks = ",".join("?" * len(aids))
        s.execute(f"UPDATE assays SET doc_id = ? WHERE assay_id IN "
                  f"({amarks})", (DECOY_DOC_ID, *aids))
        return {"op": "relink_documents_to_decoy", "assay_ids": aids,
                "molecule_rows": len(ids)}

    def control(s):
        return len(pyclean.answer_org_journal_year_provenance(
            s, {"journal": cj, "year": y}))

    return slc, mutate, control


def _slice_mechanism(src, params, rng):
    tid = _tid_of(src, params["target_chembl_id"])
    if tid is None:
        return None
    ctl = _one(src, "SELECT DISTINCT tid FROM drug_mechanism WHERE tid != ? "
                    "ORDER BY tid LIMIT 1", (tid,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "drug_mechanism", "tid IN (?, ?)", (tid, ctl))
    _copy(src, slc, "target_dictionary", "tid IN (?, ?)", (tid, ctl))
    mols = [r[0] for r in slc.execute(
        "SELECT DISTINCT molregno FROM drug_mechanism")]
    if mols:
        _copy_in_chunks(src, slc, "molecule_hierarchy", "molregno", mols)
        parents = {r[0] for r in slc.execute(
            "SELECT DISTINCT parent_molregno FROM molecule_hierarchy "
            "WHERE parent_molregno IS NOT NULL")}
        allm = sorted(set(mols) | parents)
        _copy_in_chunks(src, slc, "molecule_dictionary", "molregno", allm)
    at = params["action_type"]

    def mutate(s):
        row = s.execute(
            "SELECT MIN(dm.mec_id) FROM drug_mechanism dm "
            "JOIN molecule_hierarchy h ON h.molregno = dm.molregno "
            "WHERE dm.tid = ? AND dm.action_type = ? "
            "GROUP BY h.parent_molregno HAVING count(*) = 1 "
            "ORDER BY dm.mec_id LIMIT 1", (tid, at)).fetchone()
        if row is None:
            return None
        s.execute("UPDATE drug_mechanism SET action_type = ? "
                  "WHERE mec_id = ?", (DECOY_TEXT, row[0]))
        return {"op": "retype_action", "mec_id": row[0]}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM drug_mechanism WHERE tid = ?",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_indication(src, params, rng):
    mesh = params["mesh_heading"]
    ctl = _one(src, "SELECT DISTINCT mesh_heading FROM drug_indication "
                    "WHERE mesh_heading IS NOT NULL AND mesh_heading != ? "
                    "ORDER BY mesh_heading LIMIT 1", (mesh,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "drug_indication", "mesh_heading IN (?, ?)", (mesh, ctl))
    mols = [r[0] for r in slc.execute(
        "SELECT DISTINCT molregno FROM drug_indication")]
    if mols:
        _copy_in_chunks(src, slc, "molecule_hierarchy", "molregno", mols)
        parents = {r[0] for r in slc.execute(
            "SELECT DISTINCT parent_molregno FROM molecule_hierarchy "
            "WHERE parent_molregno IS NOT NULL")}
        allm = sorted(set(mols) | parents)
        _copy_in_chunks(src, slc, "molecule_dictionary", "molregno", allm)

    def mutate(s):
        row = s.execute(
            "SELECT MIN(d.drugind_id) FROM drug_indication d "
            "JOIN molecule_hierarchy h ON h.molregno = d.molregno "
            "WHERE d.mesh_heading = ? AND d.max_phase_for_ind >= 2 "
            "GROUP BY h.parent_molregno HAVING count(*) = 1 "
            "ORDER BY d.drugind_id LIMIT 1", (mesh,)).fetchone()
        if row is None:
            return None
        s.execute("UPDATE drug_indication SET mesh_heading = ? "
                  "WHERE drugind_id = ?", (DECOY_TEXT, row[0]))
        return {"op": "rekey_mesh_heading", "drugind_id": row[0]}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM drug_indication WHERE mesh_heading = ?",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_uniprot(src, params, rng):
    acc = params["accession"]
    ctl = _one(src, "SELECT DISTINCT accession FROM component_sequences "
                    "WHERE accession IS NOT NULL AND accession != ? "
                    "ORDER BY accession LIMIT 1", (acc,))
    if ctl is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "component_sequences", "accession IN (?, ?)", (acc, ctl))
    comps = [r[0] for r in slc.execute(
        "SELECT component_id FROM component_sequences")]
    _copy_in_chunks(src, slc, "target_components", "component_id", comps)
    tids = [r[0] for r in slc.execute(
        "SELECT DISTINCT tid FROM target_components")]
    _copy_in_chunks(src, slc, "target_dictionary", "tid", tids)

    def mutate(s):
        row = s.execute(
            "SELECT tc.tid, MIN(tc.component_id) FROM target_components tc "
            "JOIN target_dictionary td ON td.tid = tc.tid "
            "WHERE tc.component_id IN ("
            "  SELECT component_id FROM component_sequences "
            "  WHERE accession = ?) "
            "AND td.target_type = 'SINGLE PROTEIN' "
            "AND td.organism = 'Homo sapiens' "
            "GROUP BY tc.tid HAVING count(*) = 1 "
            "ORDER BY tc.tid LIMIT 1", (acc,)).fetchone()
        if row is None:
            return None
        s.execute("DELETE FROM target_components WHERE tid = ? "
                  "AND component_id = ?", (row[0], row[1]))
        return {"op": "rewire_component", "tid": row[0],
                "component_id": row[1]}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM target_components WHERE component_id IN ("
            "  SELECT component_id FROM component_sequences "
            "  WHERE accession = ?)", (ctl,)).fetchone()[0]

    return slc, mutate, control


def _slice_multihop(src, params, rng):
    tid = _tid_of(src, params["target_chembl_id"])
    if tid is None:
        return None
    slc = _mk_slice()
    _copy(src, slc, "target_dictionary", "tid = ?", (tid,))
    _copy(src, slc, "assays", "tid = ?", (tid,))
    aids = [r[0] for r in slc.execute("SELECT assay_id FROM assays")]
    _copy_in_chunks(src, slc, "activities", "assay_id", aids)
    _copy_molecules_for_activities(src, slc)
    dids = [r[0] for r in slc.execute(
        "SELECT DISTINCT doc_id FROM assays WHERE doc_id IS NOT NULL")]
    _copy_in_chunks(src, slc, "docs", "doc_id", dids)
    j = params["journal"]
    cj = _one(slc, "SELECT DISTINCT journal FROM docs WHERE journal IS NOT "
                   "NULL AND journal != ? ORDER BY journal LIMIT 1", (j,))
    if cj is None:
        cj = _one(src, "SELECT DISTINCT journal FROM docs WHERE journal "
                       "IS NOT NULL AND journal != ? ORDER BY journal "
                       "LIMIT 1", (j,))
    if cj is None:
        slc.close()
        return None
    base = ("standard_type = 'IC50' AND standard_relation = '=' "
            "AND standard_units = 'nM' AND standard_value IS NOT NULL "
            "AND molregno IS NOT NULL AND assay_id IN ("
            "  SELECT assay_id FROM assays WHERE tid = ? "
            "  AND assay_type = 'B' AND confidence_score >= 8 "
            "  AND doc_id IN ("
            "    SELECT doc_id FROM docs WHERE journal = ?))")

    def mutate(s):
        ids = _weakest_molecule_rows(s, base, (tid, j))
        if not ids:
            return None
        marks = ",".join("?" * len(ids))
        aids = [r[0] for r in s.execute(
            f"SELECT DISTINCT assay_id FROM activities WHERE "
            f"activity_id IN ({marks})", tuple(ids))]
        s.execute("INSERT INTO docs (doc_id, journal, year) "
                  "VALUES (?, ?, ?)", (DECOY_DOC_ID, DECOY_TEXT, 1900))
        amarks = ",".join("?" * len(aids))
        s.execute(f"UPDATE assays SET doc_id = ? WHERE assay_id IN "
                  f"({amarks})", (DECOY_DOC_ID, *aids))
        return {"op": "relink_documents_to_decoy", "assay_ids": aids,
                "molecule_rows": len(ids)}

    def control(s):
        return len(pyclean.answer_app_multihop_target_journal(
            s, {"target_chembl_id": params["target_chembl_id"],
                "journal": cj}))

    return slc, mutate, control


def _slice_standardization(src, params, rng):
    built = _slice_assay(src, params, rng)
    if built is None:
        return None
    slc, aid, ctl = built
    st, thr = params["standard_type"], params["threshold"]
    base = ("standard_type = ? AND standard_relation = '=' "
            "AND standard_units = 'nM' AND standard_value IS NOT NULL "
            "AND assay_id = ? AND standard_value <= ?")

    def mutate(s):
        x = _unique_support_activity(s, base, (st, aid, thr))
        if x is None:
            ids = _weakest_molecule_rows(s, base, (st, aid, thr))
            if not ids:
                return None
            marks = ",".join("?" * len(ids))
            s.execute("UPDATE activities SET standard_value = "
                      "standard_value * 1000000.0 WHERE activity_id IN ("
                      + marks + ")", tuple(ids))
            return {"op": "rescale_weakest_molecule_rows_x1e6",
                    "n_rows": len(ids)}
        s.execute("UPDATE activities SET standard_value = "
                  "standard_value * 1000000.0 WHERE activity_id = ?", (x,))
        return {"op": "rescale_value_x1e6", "activity_id": x}

    def control(s):
        return s.execute(
            "SELECT count(*) FROM activities WHERE assay_id = ?",
            (ctl,)).fetchone()[0]

    return slc, mutate, control


SLICE_BUILDERS: dict[str, Callable] = {
    "org_parent_salt_members": _slice_parent,
    "app_molecule_assays_of_type": _slice_molecule,
    "app_target_binding_set": _slice_target_binding,
    "org_assay_context_organism": _slice_assay_context,
    "app_measurement_exact_set": _slice_measurement,
    "app_unit_normalized_potency": _slice_measurement,
    "rep_relation_flip_repair": _slice_measurement,
    "rep_standardization_table": _slice_standardization,
    "org_target_class_members": _slice_target_class,
    "org_journal_year_provenance": _slice_journal,
    "app_mechanism_action_type": _slice_mechanism,
    "app_indication_mesh": _slice_indication,
    "org_uniprot_target_map": _slice_uniprot,
    "app_multihop_target_journal": _slice_multihop,
}


def certify(con_src, family_id: str, params: dict[str, Any],
            expected: tuple[str, ...], witness_kind: str,
            seed: int) -> dict[str, Any]:
    """Build slice, run governed/perturbed/control probes."""
    rng = random.Random(seed)
    answer_fn = pyclean.ANSWER_BY_FAMILY[family_id]
    builder = SLICE_BUILDERS[family_id]
    try:
        built = builder(con_src, params, rng)
    except Exception as exc:
        return {"ok": False, "reason": f"slice build failed: {exc}",
                "kind": witness_kind}
    if built is None:
        return {"ok": False, "reason": "no slice scope", "kind": witness_kind}
    slc, mutate, control = built
    expected_set = frozenset(expected)
    try:
        governed_before = answer_fn(slc, params)
        if governed_before != expected_set:
            return {"ok": False,
                    "reason": "slice does not reproduce reference answer",
                    "kind": witness_kind,
                    "governed_slice": len(governed_before),
                    "governed_full": len(expected_set)}
        control_before = control(slc)
        fp = _fingerprint(slc)
        mutation = mutate(slc)
        if mutation is None:
            return {"ok": False, "reason": "no mutation target found",
                    "kind": witness_kind}
        governed_after = answer_fn(slc, params)
        control_after = control(slc)
    finally:
        slc.close()
    ok = (governed_after != expected_set) and (control_after == control_before)
    return {
        "ok": ok,
        "kind": witness_kind,
        "mutation": mutation,
        "governed_correct_count": len(expected_set),
        "governed_perturbed_count": len(governed_after),
        "control_before_sha256": _sha(repr(control_before)),
        "control_after_sha256": _sha(repr(control_after)),
        "control_unchanged": control_after == control_before,
        "slice_fingerprint": fp,
        "reason": "" if ok else "perturbation did not flip governed probe "
                               "or control changed",
    }
