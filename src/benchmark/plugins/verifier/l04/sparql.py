"""L04 SPARQL recipes: one verification query per family over the pinned graphs.

Query results are normalized to the same canonical answer strings as the KB
fast path (``benchmark.plugins.task_family.l04.families``), so native
verification is an independent recomputation of the declared answer, not a
re-run of the same code.

Graph usage per family:
  G_rhea            — reactions/participants/compounds/equations/flags/ec/seeAlso
  G_chebi           — rdfs:subClassOf closure (property paths; owlrl for the
                      dual-engine family)
  G_derived         — TSV-carried mappings (ph73; names/smiles when needed)
"""
from __future__ import annotations

from typing import Any

RH = "http://rdf.rhea-db.org/"

PREFIXES = """PREFIX rh: <http://rdf.rhea-db.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX l04d: <http://autoontobench/ns/l04/derived/>
PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
"""

MASTER_FILTER = "?rxn rdfs:subClassOf rh:Reaction ."


def _master_with_chebi(chebi: str) -> str:
    # pattern order matters for rdflib (left-to-right evaluation): the bound
    # accession literal is the selective entry point; the rest are index walks.
    # The master-class check MUST be a FILTER EXISTS: as a trailing BGP pattern
    # rdflib 7.6 materializes the (subClassOf rh:Reaction) triple set against
    # every candidate row (quadratic blowup: >10min for heavy entities, vs ~1s
    # as an existence check; measured on CHEBI:15377, 6749 participant rows).
    return (
        f'?c rh:accession "{chebi}" . ?p rh:compound ?c . ?s rh:contains ?p . '
        f'?rxn rh:side ?s ; rh:accession ?acc . '
        f'FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }} ')


# The dispatch below is built explicitly per family (no string templating of
# untrusted content: params are validated ChEBI/RHEA/EC identifiers).

def build_query(family_id: str, params: dict[str, Any]) -> tuple[str, str, str]:
    """Return (graph_set, query_text, normalize) where normalize names the
    row->string projection used to canonicalize results."""
    fid = family_id
    p = params

    if fid == "org_reactions_involving_chebi":
        q = PREFIXES + "SELECT DISTINCT ?acc WHERE {\n" + \
            _master_with_chebi(p["chebi"]) + "}"
        return "rhea", q, "acc"

    if fid == "org_participant_rows_involving_chebi":
        # selective-first: enter via the entity's participant rows, then project
        # every participant row of those master reactions.
        q = PREFIXES + f"""
SELECT DISTINCT ?acc ?s ?p ?c ?chebi WHERE {{
  ?cE rh:accession "{p["chebi"]}" . ?pE rh:compound ?cE .
  ?sE rh:contains ?pE . ?rxn rh:side ?sE .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  ?rxn rh:accession ?acc ; rh:side ?s .
  ?s rh:contains ?p . ?p rh:compound ?c . ?c rh:accession ?chebi .
  FILTER(STRSTARTS(?chebi, "CHEBI:"))
}}"""
        return "rhea", q, "participant_row"

    if fid in ("org_reactions_chebi_conjunction", "org_reactions_chebi_difference"):
        # Set-algebra families are verified via build_set_queries (two selective
        # SPARQL runs combined by the harness); a single nested FILTER query on
        # the 2M-triple graph is two orders of magnitude slower under rdflib.
        raise KeyError(f"{fid} is verified via build_set_queries")

    if fid in ("org_reactions_chebi_ancestor",):
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  ?rxn rdfs:subClassOf rh:Reaction ; rh:accession ?acc ; rh:side ?s .
  ?s rh:contains ?p . ?p rh:compound ?c . ?c rh:chebi ?class .
  ?class rdfs:subClassOf+ chebi:{p["ancestor"].split(":")[1]} .
}}"""
        return "rhea+chebi", q, "acc"

    if fid == "org_go_terms_of_reactions_involving_chebi":
        # GO seeAlso links live on masters and (7 rows in release 142) on
        # directional children; the TSV fast path groups by MASTER_ID, so the
        # SPARQL path unions master + children to match that grouping exactly.
        q = PREFIXES + f"""
SELECT DISTINCT ?go WHERE {{
  {_master_with_chebi(p["chebi"])}
  {{ ?rxn rdfs:seeAlso ?go }} UNION
  {{ ?rxn rh:directionalReaction|rh:bidirectionalReaction ?ch .
     ?ch rdfs:seeAlso ?go }}
  FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
}}"""
        return "rhea", q, "go"

    if fid == "org_reactions_by_go_term_and_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  {{ ?rxn rdfs:seeAlso <http://purl.obolibrary.org/obo/GO_{p["go"].split(":")[1]}> }} UNION
  {{ ?rxn rh:directionalReaction|rh:bidirectionalReaction ?ch .
     ?ch rdfs:seeAlso <http://purl.obolibrary.org/obo/GO_{p["go"].split(":")[1]}> }}
}}"""
        return "rhea", q, "acc"

    if fid == "org_chebi_copartners":
        q = PREFIXES + f"""
SELECT DISTINCT ?other WHERE {{
  ?c1 rh:accession "{p["chebi"]}" .
  ?p1 rh:compound ?c1 . ?s1 rh:contains ?p1 .
  ?rxn rh:side ?s1 ; rh:side ?s2 .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
  ?s2 rh:contains ?p2 . ?p2 rh:compound ?c2 . ?c2 rh:accession ?other .
  FILTER(STRSTARTS(?other, "CHEBI:"))
  FILTER(?other != "{p["chebi"]}")
}}"""
        return "rhea", q, "acc"

    if fid == "org_reactions_by_ec_class_and_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:ec ?ec .
  FILTER(STRSTARTS(STR(?ec), "http://purl.uniprot.org/enzyme/{p["ec_class"]}."))
}}"""
        return "rhea", q, "acc"

    if fid == "org_xrefs_of_reactions_involving_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?x WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn l04d:xref ?x .
  FILTER(STRSTARTS(STR(?x), "{p["db"]}:"))
}}"""
        return "rhea+derived", q, "literal"

    if fid in ("app_directional_children_of_master", "rep_direction_assignment_repair"):
        # LR/RL roles are not stated on the master->child links; they are
        # recovered independently from the pinned equation texts under Rhea's
        # writing convention (release 142 writes NO "<=" arrows): the LR child
        # repeats the master equation's left-hand side, the RL child swaps the
        # sides. Cross-checked against the rhea-directions.tsv roles used by
        # the fast path; masters whose children break the convention reject.
        q = PREFIXES + f"""
SELECT ?code ?child WHERE {{ {{
  <{RH}{p["master"].split(":")[1]}> rh:directionalReaction ?d ; rh:equation ?meq .
  ?d rh:accession ?child ; rh:equation ?eq .
  BIND(STRBEFORE(STR(?meq), " = ") AS ?mL)
  BIND(STRAFTER(STR(?meq), " = ") AS ?mR)
  BIND(STRBEFORE(STR(?eq), " =>") AS ?cL)
  BIND(IF(?cL = ?mL, "LR", IF(?cL = ?mR, "RL", "UNKNOWN")) AS ?code)
  FILTER(?code != "UNKNOWN")
}} UNION {{
  <{RH}{p["master"].split(":")[1]}> rh:bidirectionalReaction ?b .
  ?b rh:accession ?child .
  BIND("BI" AS ?code)
}} }}"""
        return "rhea", q, "child_assignment"

    if fid == "app_transport_reactions_involving_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:isTransport ?t . FILTER(?t = true)
}}"""
        return "rhea", q, "acc"

    if fid == "app_unbalanced_reactions_involving_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:isChemicallyBalanced ?b . FILTER(?b = false)
}}"""
        return "rhea", q, "acc"

    if fid == "app_reaction_equations_by_ec":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc ?eq WHERE {{
  ?rxn rh:ec <http://purl.uniprot.org/enzyme/{p["ec"]}> ;
       rh:accession ?acc ; rh:equation ?eq .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
}}"""
        return "rhea", q, "equation_row"

    if fid == "app_reactions_by_status_and_chebi":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  {_master_with_chebi(p["chebi"])}
  ?rxn rh:status <{RH}{p["status"]}> .
}}"""
        return "rhea", q, "acc"

    if fid == "app_side_participants_of_reaction":
        num = p["rhea"].split(":")[1]
        q = PREFIXES + f"""
SELECT DISTINCT ?p ?c ?chebi WHERE {{
  <{RH}{num}_{p["side"]}> rh:contains ?p .
  ?p rh:compound ?c . ?c rh:accession ?chebi .
  FILTER(STRSTARTS(?chebi, "CHEBI:"))
}}"""
        return "rhea", q, "side_row"

    if fid == "rep_participant_side_flip_repair":
        num = p["rhea"].split(":")[1]
        q = PREFIXES + f"""
SELECT ?side ?p ?c ?chebi WHERE {{
  VALUES ?sideIri {{ <{RH}{num}_L> <{RH}{num}_R> }}
  ?sideIri rh:contains ?p . ?p rh:compound ?c . ?c rh:accession ?chebi .
  FILTER(STRSTARTS(?chebi, "CHEBI:"))
  BIND(STRAFTER(STR(?sideIri), "_") AS ?side)
}}"""
        return "rhea", q, "flip_row"

    if fid == "rep_ph73_mapping_repair":
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  chebi:{p["chebi"].split(":")[1]} l04d:ph73Mapping ?major .
  ?c rh:chebi ?major .
  ?p rh:compound ?c . ?s rh:contains ?p .
  ?rxn rh:side ?s ; rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
}}"""
        return "rhea+derived", q, "acc"

    if fid == "ont_chebi_ancestor_closure":
        q = PREFIXES + f"""
SELECT DISTINCT ?anc WHERE {{
  chebi:{p["chebi"].split(":")[1]} rdfs:subClassOf+ ?anc .
  FILTER(STRSTARTS(STR(?anc), "http://purl.obolibrary.org/obo/CHEBI_"))
}}"""
        return "chebi", q, "chebi_class"

    if fid == "ont_chebi_common_ancestors_of_reaction":
        return "special:common_ancestors", "", ""

    if fid in ("ont_chebi_descendants_in_rhea", "ont_chebi_closure_dual_engine"):
        q = PREFIXES + f"""
SELECT DISTINCT ?class WHERE {{
  ?p rh:compound ?c . ?c rh:chebi ?class .
  ?class rdfs:subClassOf+ chebi:{p["ancestor"].split(":")[1]} .
}}"""
        return "rhea+chebi", q, "chebi_class"

    if fid == "con_crosswalk_via_budgeted_probes":
        values = " ".join(f'"{c}"' for c in sorted(p["chebi_set"]))
        q = PREFIXES + f"""
SELECT DISTINCT ?acc WHERE {{
  VALUES ?chebi {{ {values} }}
  ?c rh:accession ?chebi . ?pp rh:compound ?c . ?s rh:contains ?pp .
  ?rxn rh:side ?s ; rh:accession ?acc .
  FILTER EXISTS {{ ?rxn rdfs:subClassOf rh:Reaction }}
}}"""
        return "rhea", q, "acc"

    if fid == "con_reaction_profile_assembly":
        num = p["master"].split(":")[1]
        q = PREFIXES + f"""
SELECT ?row WHERE {{ {{
  <{RH}{num}> rh:directionalReaction ?d ; rh:equation ?meq .
  ?d rh:accession ?child ; rh:equation ?eq .
  BIND(STRBEFORE(STR(?meq), " = ") AS ?mL)
  BIND(STRAFTER(STR(?meq), " = ") AS ?mR)
  BIND(STRBEFORE(STR(?eq), " =>") AS ?cL)
  BIND(IF(?cL = ?mL, "LR", IF(?cL = ?mR, "RL", "UNKNOWN")) AS ?code)
  FILTER(?code != "UNKNOWN")
  BIND(CONCAT("child|", ?code, "|", ?child) AS ?row)
}} UNION {{
  <{RH}{num}> rh:bidirectionalReaction ?b . ?b rh:accession ?child .
  BIND(CONCAT("child|BI|", ?child) AS ?row)
}} UNION {{
  <{RH}{num}> l04d:xref ?x . BIND(CONCAT("xref|", ?x) AS ?row)
}} UNION {{
  <{RH}{num}> rdfs:seeAlso ?go .
  FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
  BIND(CONCAT("go|GO:", STRAFTER(STR(?go), "GO_")) AS ?row)
}} UNION {{
  <{RH}{num}> rh:side ?s . ?s rh:contains ?pp .
  ?pp rh:compound ?c . ?c rh:accession ?chebi .
  FILTER(STRSTARTS(?chebi, "CHEBI:"))
  BIND(CONCAT("participant|", STRAFTER(STR(?s), "_"), "|", STR(?pp), "|",
              STR(?c), "|", ?chebi) AS ?row)
}} }}"""
        return "rhea+derived", q, "row_literal"

    if fid in ("rep_crosswalk_single_flip", "rep_crosswalk_multi_swap"):
        # The reference answer is defined under the REPAIRED artifact; the
        # repaired crosswalk is exactly the published compound->ChEBI mapping
        # (rh:accession on the compound node in the pinned Rhea RDF), so the
        # native reference query runs against G_rhea directly.
        q = PREFIXES + "SELECT DISTINCT ?acc WHERE {\n" + \
            _master_with_chebi(p["chebi"]) + "}"
        return "artifact+rhea", q, "acc"

    raise KeyError(f"no SPARQL recipe for family {fid}")


def _involving_query(chebi: str) -> str:
    return PREFIXES + "SELECT DISTINCT ?acc WHERE {\n" + \
        _master_with_chebi(chebi) + "}"


def build_set_queries(family_id: str,
                      params: dict[str, Any]) -> tuple[str, list[str], str]:
    """Two selective per-entity SPARQL runs plus the declared set operation.

    rdflib evaluates nested FILTER (NOT) EXISTS over the 2M-triple Rhea graph
    without a selective entry point; splitting the join into two selective
    queries keeps native SPARQL evidence while the harness applies only the
    set operation declared by the family (intersection / difference).
    """
    if family_id == "org_reactions_chebi_conjunction":
        return ("rhea",
                [_involving_query(params["chebi_a"]),
                 _involving_query(params["chebi_b"])],
                "intersection")
    if family_id == "org_reactions_chebi_difference":
        return ("rhea",
                [_involving_query(params["chebi_a"]),
                 _involving_query(params["chebi_b"])],
                "difference")
    raise KeyError(f"no set-algebra recipe for family {family_id}")


# ---------------------------------------------------------------------------
# result normalization
# ---------------------------------------------------------------------------

def normalize_rows(kind: str, results) -> tuple[str, ...]:
    out: set[str] = set()
    for row in results:
        if kind in ("acc",):
            out.add(str(row[0]))
        elif kind == "go":
            out.add("GO:" + str(row[0]).rsplit("GO_", 1)[-1])
        elif kind == "literal":
            out.add(str(row[0]))
        elif kind == "participant_row":
            acc, side_iri, part, comp, chebi = (str(x) for x in row)
            side = side_iri.rsplit("_", 1)[-1]
            out.add("|".join([acc, side, part, comp, chebi]))
        elif kind == "child_assignment":
            out.add(f"{row[0]}={row[1]}")
        elif kind == "child_assignment_lr_only":
            out.add(f"{row[0]}={row[1]}")
        elif kind == "equation_row":
            out.add(f"{row[0]} | {row[1]}")
        elif kind == "side_row":
            out.add("|".join(str(x) for x in row))
        elif kind == "flip_row":
            out.add("|".join(str(x) for x in row))
        elif kind == "chebi_class":
            out.add("CHEBI:" + str(row[0]).rsplit("_", 1)[-1])
        elif kind == "participant_classes":
            out.add("CHEBI:" + str(row[0]).rsplit("_", 1)[-1])
        elif kind == "row_literal":
            out.add(str(row[0]))
        else:
            raise KeyError(kind)
    return tuple(sorted(out))
