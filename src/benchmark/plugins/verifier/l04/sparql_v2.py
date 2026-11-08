"""L04 v2 SPARQL recipes: native rdflib reference queries for the 18 new
families (20K expansion). Same contract as sparql.py: query results are
normalized to the same canonical answer strings as the KB fast path, so
native verification is an independent recomputation, never a re-run of the
fast-path code.

Graph usage:
  G_rhea    — reactions/participants/compounds/equations/flags/ec/citations/
              formula/charge/reaction-level rdfs:subClassOf (is_a) links
  G_chebi   — rdfs:subClassOf closure (property paths)
  G_derived — TSV-carried published mappings (ph73, names, smiles, xrefs)
"""
from __future__ import annotations

import hashlib
from typing import Any

from benchmark.plugins.source.l04.kb import L04KB
from benchmark.plugins.verifier.l04 import plugin as _legacy
from benchmark.plugins.verifier.l04.sparql import PREFIXES, normalize_rows

RH = "http://rdf.rhea-db.org/"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


import json


def _master_with_chebi(chebi: str) -> str:
    return (
        f'?c rh:accession "{chebi}" . ?p rh:compound ?c . ?s rh:contains ?p . '
        f'?rxn rh:side ?s ; rh:accession ?acc . '
        f'FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }} ')


#: families verified by the harness-level set-combination path
_SET_OP: dict[str, str] = {
    "org_masters_isa_neighborhood": "difference",
}

#: families verified by the two-phase entity-resolution path
_TWO_PHASE = {
    "org_masters_by_name_token",
    "org_masters_by_smiles_substring",
    "org_masters_involving_ph73_microspecies",
}

def build_query_v2(family_id: str,
                   params: dict[str, Any]) -> tuple[str, str, str]:
    """Return (graph_set, query_text, normalize) for a v2 family."""
    p = params
    fid = family_id

    if fid == "org_masters_isomer_formula_class":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  ?c rh:formula "{_esc(p["formula"])}" .
  ?p rh:compound ?c . ?s rh:contains ?p .
  ?rxn rh:side ?s ; rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
}}"""
        return "rhea", q, "acc"

    if fid == "org_masters_by_charge_and_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:side ?s2 . ?s2 rh:contains ?p2 . ?p2 rh:compound ?c2 .
  ?c2 rh:charge ?q . FILTER(?q = {int(p["charge"])})
}}"""
        return "rhea", q, "acc"

    if fid == "org_masters_by_citation_band_and_chebi":
        lo, hi = (int(x) for x in p["band"].split("-"))
        # OPTIONAL inside the aggregate: masters with zero citations must
        # produce count 0 or band 0-0 could never match (fast path parity).
        # The aggregate is scoped to the involving masters (see the
        # cardinality family note on rdflib's global-aggregate join).
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {{ SELECT ?rxn (COUNT(DISTINCT ?cit) AS ?n) WHERE {{
       ?c0 rh:accession "{p["chebi"]}" . ?p0 rh:compound ?c0 .
       ?s0 rh:contains ?p0 . ?rxn rh:side ?s0 .
       OPTIONAL {{ ?rxn rh:citation ?cit }}
     }} GROUP BY ?rxn }}
  ?rxn rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  FILTER(?n >= {lo} && ?n <= {hi})
}}"""
        return "rhea", q, "acc"

    if fid in ("org_masters_by_name_token", "org_masters_by_smiles_substring",
               "org_masters_involving_ph73_microspecies"):
        # verified via the two-phase harness (entity resolution on G_derived,
        # then chunked VALUES involving-queries on G_rhea): rdflib does not
        # push outer bindings into a joined BGP, so a single composed query
        # evaluates the unselective join first (minutes); the legacy set-
        # algebra families established the harness-combination precedent.
        raise KeyError(f"{fid} is verified via verify_two_phase_v2")

    if fid == "org_masters_multi_ec":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {{ SELECT ?rxn (COUNT(DISTINCT ?ec) AS ?n) WHERE {{
       ?c0 rh:accession "{p["chebi"]}" . ?p0 rh:compound ?c0 .
       ?s0 rh:contains ?p0 . ?rxn rh:side ?s0 .
       ?rxn rh:ec ?ec .
     }} GROUP BY ?rxn }}
  ?rxn rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  FILTER(?n >= 2)
}}"""
        return "rhea", q, "acc"

    if fid == "org_masters_no_ec":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  FILTER NOT EXISTS {{ ?rxn rh:ec ?ec }}
}}"""
        return "rhea", q, "acc"

    if fid == "org_masters_go_mediated":
        chebi = p["chebi"]
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  ?c rh:accession "{chebi}" . ?p rh:compound ?c . ?s rh:contains ?p .
  ?m1 rh:side ?s . FILTER EXISTS {{ ?m1 rdfs:subClassOf rh:Reaction }}
  {{ ?m1 rdfs:seeAlso ?go }} UNION
  {{ ?m1 rh:directionalReaction|rh:bidirectionalReaction ?ch1 .
     ?ch1 rdfs:seeAlso ?go }}
  FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
  {{ ?m2 rdfs:seeAlso ?go }} UNION
  {{ ?m2 rh:directionalReaction|rh:bidirectionalReaction ?ch2 .
     ?ch2 rdfs:seeAlso ?go }}
  ?m2 rh:accession ?acc .
  FILTER EXISTS {{ ?m2 rdfs:subClassOf rh:Reaction }}
}}"""
        return "rhea", q, "acc"

    if fid == "org_opposite_side_copartners":
        q = PREFIXES + f"""
SELECT DISTINCT ?other WHERE {{
  ?c1 rh:accession "{p["chebi"]}" .
  ?p1 rh:compound ?c1 . ?s1 rh:contains ?p1 .
  ?rxn rh:side ?s1 ; rh:side ?s2 .
  FILTER(STRENDS(STR(?s1), "_L") != STRENDS(STR(?s2), "_L"))
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  ?s2 rh:contains ?p2 . ?p2 rh:compound ?c2 . ?c2 rh:accession ?other .
  FILTER(STRSTARTS(?other, "CHEBI:"))
  FILTER(?other != "{p["chebi"]}")
}}"""
        return "rhea", q, "acc"

    if fid == "app_masters_chebi_both_sides":
        # selective entry: the entity's participant rows; side-role filters
        # applied after the join (rdflib evaluates BGPs left to right)
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  ?cL rh:accession "{p["chebi"]}" . ?pL rh:compound ?cL . ?sL rh:contains ?pL .
  ?rxn rh:side ?sL ; rh:side ?sR ; rh:accession ?acc .
  FILTER(STRENDS(STR(?sL), "_L")) FILTER(STRENDS(STR(?sR), "_R"))
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  ?cR rh:accession "{p["chebi"]}" . ?pR rh:compound ?cR . ?sR rh:contains ?pR .
}}"""
        return "rhea", q, "acc"

    if fid == "app_masters_by_participant_cardinality":
        lo, hi = (int(x) for x in p["band"].split("-"))
        # count only participants whose compound carries a CHEBI accession:
        # the fast-path participant index drops non-ChEBI (polymer) rows.
        # The aggregate is scoped to the involving masters inside the
        # subquery: a global 18.5k-row group keyset hash-joins pathologically
        # under rdflib (measured), the scoped form runs in seconds.
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {{ SELECT ?rxn (COUNT(DISTINCT ?pp) AS ?n) WHERE {{
       ?c0 rh:accession "{p["chebi"]}" . ?p0 rh:compound ?c0 .
       ?s0 rh:contains ?p0 . ?rxn rh:side ?s0 .
       ?rxn rh:side ?ss . ?ss rh:contains ?pp .
       ?pp rh:compound ?ccx . ?ccx rh:accession ?accx .
       FILTER(STRSTARTS(?accx, "CHEBI:"))
     }} GROUP BY ?rxn }}
  ?rxn rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  FILTER(?n >= {lo} && ?n <= {hi})
}}"""
        return "rhea", q, "acc"

    if fid == "rep_go_link_table_repair":
        # reference = gold GO terms of masters involving the entity (the
        # repaired table is exactly the published rdfs:seeAlso GO mapping)
        q = PREFIXES + f"""
SELECT DISTINCT ?go WHERE {{
  {_master_with_chebi(p["chebi"])}
  {{ ?rxn rdfs:seeAlso ?go }} UNION
  {{ ?rxn rh:directionalReaction|rh:bidirectionalReaction ?ch .
     ?ch rdfs:seeAlso ?go }}
  FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
}}"""
        return "rhea", q, "go"

    if fid == "rep_xref_table_repair":
        q = PREFIXES + f"""
SELECT DISTINCT ?x WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn l04d:xref ?x .
  FILTER(STRSTARTS(STR(?x), "{p["db"]}:"))
}}"""
        return "rhea+derived", q, "literal"

    if fid == "rep_status_table_repair":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:status <{RH}{p["status"]}> .
}}"""
        return "rhea", q, "acc"

    raise KeyError(f"no v2 SPARQL recipe for family {fid}")


def _involving_query(chebi: str) -> str:
    return PREFIXES + "SELECT DISTINCT ?acc WHERE {\n" + \
        _master_with_chebi(chebi) + "}"


def _isa_neighborhood_query(chebi: str) -> str:
    return PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  ?c rh:accession "{chebi}" . ?p rh:compound ?c . ?s rh:contains ?p .
  ?m1 rh:side ?s .
  FILTER EXISTS {{ ?m1 rdfs:subClassOf rh:Reaction }}
  ?m1 rh:directionalReaction|rh:bidirectionalReaction ?d1 .
  {{ ?d1 rdfs:subClassOf ?d2 }} UNION {{ ?d2 rdfs:subClassOf ?d1 }}
  ?m2 rh:directionalReaction|rh:bidirectionalReaction ?d2 .
  FILTER EXISTS {{ ?m2 rdfs:subClassOf rh:Reaction }}
  ?m2 rh:accession ?acc .
}}"""


def build_set_queries_v2(family_id: str,
                         params: dict[str, Any]) -> tuple[str, list[str], str]:
    if family_id == "org_masters_isa_neighborhood":
        return ("rhea",
                [_isa_neighborhood_query(params["chebi"]),
                 _involving_query(params["chebi"])],
                "difference")
    raise KeyError(f"no v2 set-algebra recipe for family {family_id}")


# ---------------------------------------------------------------------------
# reference verification (dual-path where the family semantics requires it)
# ---------------------------------------------------------------------------

def _entity_resolution_query(family_id: str, params: dict[str, Any]) -> str:
    """Phase-1 query on G_derived: resolve the ChEBI entity IRIs the family's
    lexical/mapping predicate selects."""
    if family_id == "org_masters_by_name_token":
        return PREFIXES + ("SELECT DISTINCT ?cc WHERE {\n"
                           "  ?cc l04d:rheaName ?nm .\n"
                           f"  FILTER(CONTAINS(LCASE(STR(?nm)), \"{_esc(params['token'])}\"))\n}}")
    if family_id == "org_masters_by_smiles_substring":
        return PREFIXES + ("SELECT DISTINCT ?cc WHERE {\n"
                           "  ?cc l04d:chebiSmiles ?sm .\n"
                           f"  FILTER(CONTAINS(STR(?sm), \"{_esc(params['smiles_sub'])}\"))\n}}")
    if family_id == "org_masters_involving_ph73_microspecies":
        return PREFIXES + ("SELECT DISTINCT ?cc WHERE {\n"
                           f"  chebi:{params['chebi'].split(':')[1]} l04d:ph73Mapping ?cc .\n}}")
    raise KeyError(family_id)


def _involving_values_query(entities: list[str]) -> str:
    """Phase-2 query on G_rhea: master reactions involving any of the given
    ChEBI accessions (legacy con_crosswalk VALUES shape, proven selective)."""
    values = " ".join(f'"{e}"' for e in entities)
    return PREFIXES + f"""SELECT DISTINCT ?acc WHERE {{
  VALUES ?chebi {{ {values} }}
  ?c rh:accession ?chebi . ?p rh:compound ?c . ?s rh:contains ?p .
  ?rxn rh:side ?s ; rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
}}"""


def verify_two_phase_v2(pin_root, family_id: str, params: dict[str, Any],
                        expected: tuple[str, ...], kb: L04KB) -> dict[str, Any]:
    expected_set = set(expected)
    gd = _legacy.combined_graph(pin_root, "derived")
    q1 = _entity_resolution_query(family_id, params)
    ents = sorted({"CHEBI:" + str(r[0]).rsplit("_", 1)[-1]
                   for r in gd.query(q1)})
    gr = _legacy.combined_graph(pin_root, "rhea")
    got: set[str] = set()
    queries = [q1]
    chunk = 25  # ~1s/entity on the 2.05M-triple graph; bounds per-query time
    for i in range(0, len(ents), chunk):
        q2 = _involving_values_query(ents[i:i + chunk])
        queries.append(q2)
        got |= set(normalize_rows("acc", gr.query(q2)))
    match = got == expected_set
    return {
        "verdict": "PASS" if match else "FAIL",
        "matches_fast_path": match,
        "graph_set": "derived+rhea(two-phase)",
        "graph_triples": len(gd) + len(gr),
        "graph_manifest_sha": _legacy._graph_manifest_sha(),
        "resolved_entities": len(ents),
        "query_sha256": _sha(queries),
        "query_text": "\n---\n".join(queries),
        "answer_sha256": _sha(sorted(got)),
        "answer_count": len(got),
        "expected_sha256": _sha(sorted(expected_set)),
        "engine": _legacy._ENGINE,
    }


def verify_reference_v2(pin_root, family_id: str, params: dict[str, Any],
                        expected: tuple[str, ...], kb: L04KB) -> dict[str, Any]:
    """Independent SPARQL recomputation of the v2 fast-path answer."""
    expected_set = set(expected)

    if family_id in _TWO_PHASE:
        return verify_two_phase_v2(pin_root, family_id, params, expected, kb)

    if family_id == "con_ancestor_closure_via_probes":
        # same reference math as the legacy ancestor-closure family: the
        # regime difference (budgeted construction) lives in the public task
        ev = _legacy._verify_closure(pin_root, "ont_chebi_ancestor_closure",
                                     params, expected, kb)
        ev["graph_manifest_sha"] = _legacy._graph_manifest_sha()
        ev["recipe"] = "closure_ancestor (shared reference math)"
        return ev

    if family_id == "org_masters_closure_conjunction":
        # Mirrors the legacy org_reactions_chebi_ancestor evidence shape:
        # descendants are pre-resolved by the rdflib-parsed ChEBI BFS (the
        # independent path; must agree with the ElementTree fast path) and
        # injected as VALUES — a property path over the 4M-triple combined
        # graph is ~40min/candidate under rdflib, the VALUES form seconds.
        parents_b, index_b = _legacy.chebi_structures_rdflib(pin_root)
        universe = sorted(kb.chebi_to_reactions)
        queries: list[str] = []
        sets: list[set[str]] = []
        notes: dict[str, Any] = {}
        agree = True
        g = _legacy.combined_graph(pin_root, "rhea")
        for key in ("ancestor_a", "ancestor_b"):
            anchor = params[key]
            desc_b = sorted(c for c in universe if anchor in index_b.get(c, ()))
            desc_a = sorted(c for c in universe
                            if anchor in kb.chebi_ancestors(c))
            agree = agree and (desc_a == desc_b)
            notes[f"descendants_in_universe[{anchor}]"] = len(desc_b)
            values = " ".join("chebi:" + c.split(":")[1] for c in desc_b)
            q = (
                "PREFIX rh: <http://rdf.rhea-db.org/>\n"
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>\n"
                "SELECT DISTINCT ?acc WHERE {\n"
                f"  VALUES ?class {{ {values} }}\n"
                "  ?c rh:chebi ?class . ?p rh:compound ?c ."
                " ?s rh:contains ?p .\n"
                "  ?rxn rh:side ?s ; rh:accession ?acc .\n"
                "  FILTER EXISTS { ?rxn rdfs:subClassOf rh:Reaction }\n}")
            queries.append(q)
            sets.append(set(normalize_rows("acc", g.query(q))))
        got = sets[0] & sets[1]
        match = got == expected_set
        ok = match and agree
        return {
            "verdict": "PASS" if ok else "FAIL",
            "matches_fast_path": match,
            "dual_path_closure_agree": agree,
            "graph_set": "rhea",
            "graph_triples": len(g),
            "graph_manifest_sha": _legacy._graph_manifest_sha(),
            "set_operation": "intersection",
            "query_sha256": _sha(queries),
            "query_text": "\n---\n".join(queries),
            "operand_counts": [len(s) for s in sets],
            "answer_sha256": _sha(sorted(got)),
            "answer_count": len(got),
            "expected_sha256": _sha(sorted(expected_set)),
            "engine": _legacy._ENGINE,
            **notes,
        }

    if family_id in _SET_OP:
        graph_set, queries, op = build_set_queries_v2(family_id, params)
        g = _legacy.combined_graph(pin_root, graph_set)
        sets = [set(normalize_rows("acc", g.query(q))) for q in queries]
        got = sets[0] - sets[1] if op == "difference" else sets[0] & sets[1]
        match = got == expected_set
        return {
            "verdict": "PASS" if match else "FAIL",
            "matches_fast_path": match,
            "graph_set": graph_set,
            "graph_triples": len(g),
            "graph_manifest_sha": _legacy._graph_manifest_sha(),
            "set_operation": op,
            "query_sha256": _sha(queries),
            "query_text": "\n---\n".join(queries),
            "operand_counts": [len(s) for s in sets],
            "answer_sha256": _sha(sorted(got)),
            "answer_count": len(got),
            "expected_sha256": _sha(sorted(expected_set)),
            "engine": _legacy._ENGINE,
        }

    graph_set, query, norm = build_query_v2(family_id, params)
    g = _legacy.combined_graph(pin_root, graph_set)
    results = g.query(query)
    got = normalize_rows(norm, results)
    match = set(got) == expected_set
    return {
        "verdict": "PASS" if match else "FAIL",
        "matches_fast_path": match,
        "graph_set": graph_set,
        "graph_triples": len(g),
        "graph_manifest_sha": _legacy._graph_manifest_sha(),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "query_text": query,
        "answer_sha256": _sha(sorted(set(got))),
        "answer_count": len(set(got)),
        "expected_sha256": _sha(sorted(expected_set)),
        "engine": _legacy._ENGINE,
    }
