"""L14 clean-parser reference path: pure-Python predicates over raw rows.

Second, structurally independent implementation of every L14 family answer.
SQL is used only to extract raw rows with single-key restrictions (no JOIN
logic, no aggregation, no filtering predicates beyond the scoping key); all
semantic filtering, unit normalization, hierarchy walking and set building is
implemented imperatively here. The fast-path SQL (families.py) and this path
must agree exactly or the candidate is rejected.

Runs against any SQLite database with the L14 slice schema (the pinned DB or
an in-memory witness slice).
"""
from __future__ import annotations

import sqlite3
from typing import Any

UNIT_SCALE_TO_NM = {"nM": 1.0, "uM": 1_000.0, "pM": 0.001, "mM": 1_000_000.0}


def _rows(con: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()):
    return con.execute(sql, args).fetchall()


def _molregno(con, chembl_id):
    r = _rows(con, "SELECT molregno FROM molecule_dictionary "
                   "WHERE chembl_id = ?", (chembl_id,))
    return r[0][0] if r else None


def _tid(con, chembl_id):
    r = _rows(con, "SELECT tid FROM target_dictionary WHERE chembl_id = ?",
              (chembl_id,))
    return r[0][0] if r else None


def _assay_id(con, chembl_id):
    r = _rows(con, "SELECT assay_id FROM assays WHERE chembl_id = ?",
              (chembl_id,))
    return r[0][0] if r else None


def _chembl_map(con, molregnos):
    out = {}
    for m in molregnos:
        r = _rows(con, "SELECT chembl_id FROM molecule_dictionary "
                       "WHERE molregno = ?", (m,))
        if r:
            out[m] = r[0][0]
    return out


def _assay_chembl_map(con, assay_ids):
    out = {}
    for a in assay_ids:
        r = _rows(con, "SELECT chembl_id FROM assays WHERE assay_id = ?",
                  (a,))
        if r:
            out[a] = r[0][0]
    return out


def _activities_for_assays(con, assay_ids):
    if not assay_ids:
        return []
    marks = ",".join("?" * len(assay_ids))
    return _rows(con,
                 "SELECT activity_id, assay_id, molregno, standard_type, "
                 "standard_relation, standard_value, standard_units "
                 "FROM activities WHERE assay_id IN (" + marks + ")",
                 tuple(assay_ids))


def _assays_for_tids(con, tids):
    if not tids:
        return []
    marks = ",".join("?" * len(tids))
    return _rows(con,
                 "SELECT assay_id, chembl_id, tid, assay_type, "
                 "assay_organism, confidence_score, doc_id "
                 "FROM assays WHERE tid IN (" + marks + ")",
                 tuple(tids))


def answer_org_parent_salt_members(con, params):
    parent = _molregno(con, params["parent_chembl_id"])
    if parent is None:
        return frozenset()
    rows = _rows(con, "SELECT molregno FROM molecule_hierarchy "
                      "WHERE parent_molregno = ?", (parent,))
    cmap = _chembl_map(con, [r[0] for r in rows])
    return frozenset(c for c in cmap.values()
                     if c != params["parent_chembl_id"])


def answer_app_molecule_assays_of_type(con, params):
    m = _molregno(con, params["molecule_chembl_id"])
    if m is None:
        return frozenset()
    rows = _rows(con, "SELECT assay_id, standard_type, standard_relation, "
                      "standard_value, standard_units FROM activities "
                      "WHERE molregno = ?", (m,))
    keep = {r[0] for r in rows
            if r[1] == params["standard_type"] and r[2] == "="
            and r[4] == "nM" and r[3] is not None}
    return frozenset(_assay_chembl_map(con, keep).values())


def _binding_assay_rows(con, tid):
    assays = _assays_for_tids(con, [tid])
    good = {a[0]: a for a in assays
            if a[3] == "B" and a[6] is not None and a[6] >= 8}
    return assays, good


def answer_app_target_binding_set(con, params):
    tid = _tid(con, params["target_chembl_id"])
    if tid is None:
        return frozenset()
    _, good = _binding_assay_rows(con, tid)
    acts = _activities_for_assays(con, sorted(good))
    mols = {r[2] for r in acts
            if r[3] == params["standard_type"] and r[4] == "="
            and r[6] == "nM" and r[5] is not None
            and r[2] is not None and float(r[5]) <= params["threshold"]}
    return frozenset(_chembl_map(con, mols).values())


def answer_org_assay_context_organism(con, params):
    tid = _tid(con, params["target_chembl_id"])
    if tid is None:
        return frozenset()
    assays, _ = _binding_assay_rows(con, tid)
    return frozenset(a[1] for a in assays
                     if a[3] == "B" and a[4] == params["assay_organism"])


def _assay_scoped_activities(con, assay_chembl_id):
    aid = _assay_id(con, assay_chembl_id)
    if aid is None:
        return None, []
    return aid, _activities_for_assays(con, [aid])


def answer_app_measurement_exact_set(con, params):
    aid, acts = _assay_scoped_activities(con, params["assay_chembl_id"])
    mols = {r[2] for r in acts
            if r[3] == params["standard_type"] and r[4] == "="
            and r[6] == "nM" and r[5] is not None and r[2] is not None}
    return frozenset(_chembl_map(con, mols).values())


def answer_app_unit_normalized_potency(con, params):
    return _repair_answer(con, params)


def _repair_answer(con, params):
    aid, acts = _assay_scoped_activities(con, params["assay_chembl_id"])
    mols = {r[2] for r in acts
            if r[3] == params["standard_type"] and r[4] == "="
            and r[6] == "nM" and r[5] is not None and r[2] is not None
            and float(r[5]) <= params["threshold"]}
    return frozenset(_chembl_map(con, mols).values())


def answer_rep_relation_flip_repair(con, params):
    return _repair_answer(con, params)


def answer_rep_standardization_table(con, params):
    return _repair_answer(con, params)


def answer_org_target_class_members(con, params):
    children: dict[int, list[int]] = {}
    for cid, parent in _rows(con, "SELECT protein_class_id, parent_id "
                                  "FROM protein_classification"):
        if parent is not None:
            children.setdefault(parent, []).append(cid)
    sub = set()
    stack = [params["protein_class_id"]]
    while stack:
        node = stack.pop()
        if node in sub:
            continue
        sub.add(node)
        stack.extend(children.get(node, ()))
    comp_rows = _rows(con, "SELECT component_id, protein_class_id "
                           "FROM component_class")
    comp_ok = {c for c, cid in comp_rows if cid in sub}
    tc_rows = _rows(con, "SELECT tid, component_id FROM target_components")
    tids = {t for t, c in tc_rows if c in comp_ok}
    out = set()
    for tid, chembl_id, ttype, org in _rows(
            con, "SELECT tid, chembl_id, target_type, organism "
                 "FROM target_dictionary"):
        if tid in tids and ttype == "SINGLE PROTEIN" and org == "Homo sapiens":
            out.add(chembl_id)
    return frozenset(out)


def answer_org_journal_year_provenance(con, params):
    docs = _rows(con, "SELECT doc_id FROM docs WHERE journal = ? "
                      "AND year = ?", (params["journal"], params["year"]))
    if not docs:
        return frozenset()
    marks = ",".join("?" * len(docs))
    assays = _rows(con, "SELECT assay_id FROM assays WHERE doc_id IN ("
                        + marks + ")", tuple(d[0] for d in docs))
    acts = _activities_for_assays(con, [a[0] for a in assays])
    mols = {r[2] for r in acts
            if r[3] in ("IC50", "Ki", "EC50") and r[4] == "="
            and r[6] == "nM" and r[5] is not None and r[2] is not None}
    return frozenset(_chembl_map(con, mols).values())


def answer_app_mechanism_action_type(con, params):
    tid = _tid(con, params["target_chembl_id"])
    if tid is None:
        return frozenset()
    rows = _rows(con, "SELECT molregno, action_type FROM drug_mechanism "
                      "WHERE tid = ?", (tid,))
    mols = {r[0] for r in rows if r[1] == params["action_type"]}
    pchembl, parents = _parent_chembl_map(con, mols)
    return frozenset(pchembl[p] for p in
                     {parents[m] for m in mols} if p in pchembl)


def _parent_chembl_map(con, mols):
    parents = {}
    for m in mols:
        r = _rows(con, "SELECT parent_molregno FROM molecule_hierarchy "
                       "WHERE molregno = ?", (m,))
        parents[m] = r[0][0] if r else m
    wanted = sorted(set(parents.values()))
    out = {}
    for p in wanted:
        r = _rows(con, "SELECT chembl_id FROM molecule_dictionary "
                       "WHERE molregno = ?", (p,))
        if r:
            out[p] = r[0][0]
    return out, parents


def answer_app_indication_mesh(con, params):
    rows = _rows(con, "SELECT molregno, max_phase_for_ind "
                      "FROM drug_indication WHERE mesh_heading = ?",
                 (params["mesh_heading"],))
    mols = {r[0] for r in rows
            if r[1] is not None and float(r[1]) >= 2}
    pchembl, parents = _parent_chembl_map(con, mols)
    return frozenset(pchembl[p] for p in
                     {parents[m] for m in mols} if p in pchembl)


def answer_org_uniprot_target_map(con, params):
    comps = _rows(con, "SELECT component_id FROM component_sequences "
                       "WHERE accession = ?", (params["accession"],))
    if not comps:
        return frozenset()
    marks = ",".join("?" * len(comps))
    tc = _rows(con, "SELECT tid, component_id FROM target_components "
                    "WHERE component_id IN (" + marks + ")",
               tuple(c[0] for c in comps))
    tids = {t for t, _c in tc}
    out = set()
    for tid, chembl_id, ttype, org in _rows(
            con, "SELECT tid, chembl_id, target_type, organism "
                 "FROM target_dictionary"):
        if tid in tids and ttype == "SINGLE PROTEIN" and org == "Homo sapiens":
            out.add(chembl_id)
    return frozenset(out)


def answer_app_multihop_target_journal(con, params):
    tid = _tid(con, params["target_chembl_id"])
    if tid is None:
        return frozenset()
    assays, good = _binding_assay_rows(con, tid)
    docs = {d[0] for d in _rows(con, "SELECT doc_id FROM docs "
                                     "WHERE journal = ?",
                                (params["journal"],))}
    keep_assays = {a[0] for a in assays
                   if a[0] in good and a[6] in docs}
    acts = _activities_for_assays(con, sorted(keep_assays))
    mols = {r[2] for r in acts
            if r[3] == "IC50" and r[4] == "=" and r[6] == "nM"
            and r[5] is not None and r[2] is not None}
    return frozenset(_chembl_map(con, mols).values())


ANSWER_BY_FAMILY = {
    "org_parent_salt_members": answer_org_parent_salt_members,
    "app_molecule_assays_of_type": answer_app_molecule_assays_of_type,
    "app_target_binding_set": answer_app_target_binding_set,
    "org_assay_context_organism": answer_org_assay_context_organism,
    "app_measurement_exact_set": answer_app_measurement_exact_set,
    "app_unit_normalized_potency": answer_app_unit_normalized_potency,
    "rep_relation_flip_repair": answer_rep_relation_flip_repair,
    "rep_standardization_table": answer_rep_standardization_table,
    "org_target_class_members": answer_org_target_class_members,
    "org_journal_year_provenance": answer_org_journal_year_provenance,
    "app_mechanism_action_type": answer_app_mechanism_action_type,
    "app_indication_mesh": answer_app_indication_mesh,
    "org_uniprot_target_map": answer_org_uniprot_target_map,
    "app_multihop_target_journal": answer_app_multihop_target_journal,
}
