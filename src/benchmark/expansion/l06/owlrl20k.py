"""L06 20K-expansion independent owlrl leg (leg 2).

Reuses the legacy axiom graph builder
(``benchmark.plugins.verifier.l06.owlrl_leg.build_closed_graph``), which
already injects every axiom the new mechanisms need:

  - BFO_0000050 (part_of)  owl:TransitiveProperty                    (R2)
  - propertyChainAxiom (is_a ∘ part_of) ⊑ part_of                    (R3)
  - propertyChainAxiom (part_of ∘ is_a) ⊑ part_of                    (R4)
  - pos/neg_regulates rdfs:subPropertyOf regulates                   (R5)
  - propertyChainAxiom (is_a ∘ R) ⊑ R and (R ∘ is_a) ⊑ R for
    R ∈ {regulates, positively_regulates, negatively_regulates}      (R6/R7)
  - regulates is NOT transitive                                       (R8)

New read operators (each must equal the leg-1 operator in
``fragment20k.py`` exactly, or the task is rejected PROFILE_UNSUPPORTED):

  posregulators_owlrl / negregulators_owlrl   subjects of entailed
      (s, positively/negatively_regulates, seed)
  generic_only_regulators_owlrl   regulates-subjects minus both polarity
      subject sets (paths using no polarity edge)
  dual_regulators_owlrl   regulates-subjects(a) ∩ regulates-subjects(b)
  partof_isa_overlap_owlrl   part_of ancestors ∩ is_a ancestors of seed
  reg_subject_descendants_owlrl   fragment terms with an entailed outgoing
      regulation triple, restricted to isa-descendants of the root
  exp_annotation_closure_owlrl / evidence_discriminated_owlrl
      evidence-gated propagated closures (evidence split recomputed from
      the pinned GAF by the caller — never taken from the stored AST)
  term_annotated_genes_owlrl / exp_term_genes_owlrl
      reverse join: per candidate gene, does any direct annotation
      propagate to the term in the closed graph
  signature_rows_owlrl   polarity-aware closure rows (isa / partof /
      reg / reg_pos / reg_neg) read off the closed graph — the dual-leg
      check that the repair-signature quantity itself is owlrl-correct
      for the terms a repair must restore.
"""
from __future__ import annotations

from typing import Any, Iterable

import rdflib
from rdflib import URIRef
from rdflib.namespace import RDFS

from benchmark.plugins.verifier.l06.owlrl_leg import (
    GO_NS,
    REL_NAME_TO_IRI,
    _g,
    _u,
    build_closed_graph,
    isa_ancestors_owlrl,
    partof_ancestors_owlrl,
    regulators_owlrl,
)

ISA = RDFS.subClassOf
POSREG = URIRef(GO_NS + REL_NAME_TO_IRI["positively_regulates"])
NEGREG = URIRef(GO_NS + REL_NAME_TO_IRI["negatively_regulates"])
REG = URIRef(GO_NS + REL_NAME_TO_IRI["regulates"])


def _subjects(g: rdflib.Graph, pred: URIRef, obj: str,
              universe: set[str]) -> frozenset[str]:
    out = {_g(s) for s in g.subjects(pred, _u(obj))}
    out.discard(obj)
    return frozenset(x for x in out if x in universe)


def posregulators_owlrl(g: rdflib.Graph, seed: str,
                        universe: set[str]) -> frozenset[str]:
    return _subjects(g, POSREG, seed, universe)


def negregulators_owlrl(g: rdflib.Graph, seed: str,
                        universe: set[str]) -> frozenset[str]:
    return _subjects(g, NEGREG, seed, universe)


def generic_only_regulators_owlrl(g: rdflib.Graph, seed: str,
                                  universe: set[str]) -> frozenset[str]:
    allr = _subjects(g, REG, seed, universe)
    return allr - _subjects(g, POSREG, seed, universe) \
        - _subjects(g, NEGREG, seed, universe)


def dual_regulators_owlrl(g: rdflib.Graph, a: str, b: str,
                          universe: set[str]) -> frozenset[str]:
    return _subjects(g, REG, a, universe) & _subjects(g, REG, b, universe)


def partof_isa_overlap_owlrl(g: rdflib.Graph, seed: str,
                             universe: set[str]) -> frozenset[str]:
    return partof_ancestors_owlrl(g, seed, universe) \
        & isa_ancestors_owlrl(g, seed, universe)


def reg_subject_descendants_owlrl(g: rdflib.Graph, root: str,
                                  universe: set[str],
                                  isa_children: dict[str, list[str]]
                                  ) -> frozenset[str]:
    """Fragment terms carrying an entailed outgoing regulation triple
    (any polarity, object in the universe), restricted to the isa
    descendants of ``root`` (R6 subject-side inheritance is exactly what
    the chain axioms materialize)."""
    from benchmark.plugins.source.l06.closure import descendants
    have_edge: set[str] = set()
    for pred in (REG, POSREG, NEGREG):
        for s, o in g.subject_objects(pred):
            s_id, o_id = _g(s), _g(o)
            if s_id in universe and o_id in universe:
                have_edge.add(s_id)
    return frozenset(have_edge & descendants(root, isa_children))


def _propagated(g: rdflib.Graph, terms: Iterable[str],
                universe: set[str]) -> set[str]:
    out: set[str] = set()
    for t in terms:
        if t not in universe:
            continue
        out.add(t)
        out.update(isa_ancestors_owlrl(g, t, universe))
        out.update(partof_ancestors_owlrl(g, t, universe))
    return out


def exp_annotation_closure_owlrl(g: rdflib.Graph, exp_terms: Iterable[str],
                                 universe: set[str]) -> frozenset[str]:
    return frozenset(_propagated(g, exp_terms, universe))


def evidence_discriminated_owlrl(g: rdflib.Graph, exp_terms: Iterable[str],
                                 other_terms: Iterable[str],
                                 universe: set[str]) -> frozenset[str]:
    return frozenset(_propagated(g, exp_terms, universe)
                     - _propagated(g, other_terms, universe))


def _gene_propagates_to(g: rdflib.Graph, term: str, direct: Iterable[str],
                        universe: set[str]) -> bool:
    for t in direct:
        if t not in universe:
            continue
        if (t == term or term in isa_ancestors_owlrl(g, t, universe)
                or term in partof_ancestors_owlrl(g, t, universe)):
            return True
    return False


def term_annotated_genes_owlrl(g: rdflib.Graph, term: str,
                               direct_by_gene: dict[str, Iterable[str]],
                               universe: set[str]) -> frozenset[str]:
    return frozenset(gene for gene, direct in direct_by_gene.items()
                     if _gene_propagates_to(g, term, direct, universe))


def exp_term_genes_owlrl(g: rdflib.Graph, term: str,
                         exp_direct_by_gene: dict[str, Iterable[str]],
                         universe: set[str]) -> frozenset[str]:
    return frozenset(gene for gene, direct in exp_direct_by_gene.items()
                     if _gene_propagates_to(g, term, direct, universe))


def signature_rows_owlrl(g: rdflib.Graph, focus: list[str],
                         universe: set[str]) -> dict[str, dict[str, list[str]]]:
    """Polarity-aware closure rows read off the closed graph; must equal
    ``fragment20k.closure_signature20k`` restricted to the same focus."""
    out: dict[str, dict[str, list[str]]] = {}
    for t in focus:
        out[t] = {
            "isa": sorted(isa_ancestors_owlrl(g, t, universe)),
            "partof": sorted(partof_ancestors_owlrl(g, t, universe)),
            "reg": sorted(regulators_owlrl(g, t, universe)),
            "reg_pos": sorted(posregulators_owlrl(g, t, universe)),
            "reg_neg": sorted(negregulators_owlrl(g, t, universe)),
        }
    return out


def owlrl_agreement20k(ast: dict[str, Any], frag: dict[str, Any],
                       reference: frozenset[str],
                       evidence: dict[str, Any] | None = None
                       ) -> tuple[bool, str]:
    """Leg 2 for every closure-computable 20K kind. Structural depth and
    repair kinds return (True, 'owlrl_not_applicable') — repairs instead get
    their dual-leg via ``signature_rows_owlrl`` on the reference fragment
    (see verify20k.prove_exists)."""
    kind = ast["kind"]
    u = set(frag["universe"])
    if kind == "partof_depth_band" or kind in {
            "repair_partof_edge_deletion", "repair_spurious_isa_edge",
            "repair_regulation_polarity_swap", "repair_spurious_partof_edge"}:
        return True, "owlrl_not_applicable"
    from benchmark.plugins.verifier.l06.fragment import fragment_edges
    g = build_closed_graph(fragment_edges(frag))
    if kind == "posregulators":
        return posregulators_owlrl(g, ast["seed"], u) == reference, "owlrl_posreg"
    if kind == "negregulators":
        return negregulators_owlrl(g, ast["seed"], u) == reference, "owlrl_negreg"
    if kind == "generic_only_regulators":
        return generic_only_regulators_owlrl(g, ast["seed"], u) == reference, \
            "owlrl_generic_only"
    if kind == "dual_regulators":
        return dual_regulators_owlrl(g, ast["a"], ast["b"], u) == reference, \
            "owlrl_dual_reg"
    if kind == "partof_isa_overlap":
        return partof_isa_overlap_owlrl(g, ast["seed"], u) == reference, \
            "owlrl_partof_isa"
    if kind == "reg_subject_descendants":
        got = reg_subject_descendants_owlrl(g, ast["root"], u,
                                            frag["isa_children"])
        return got == reference, "owlrl_reg_subject_desc"
    if kind == "exp_annotation_closure":
        ev = (evidence or {}).get(ast["gene"], {})
        got = exp_annotation_closure_owlrl(g, ev.get("exp", ()), u)
        return got == reference, "owlrl_exp_closure"
    if kind == "evidence_discriminated":
        ev = (evidence or {}).get(ast["gene"], {})
        got = evidence_discriminated_owlrl(g, ev.get("exp", ()),
                                           ev.get("other", ()), u)
        return got == reference, "owlrl_evidence_discriminated"
    if kind in ("term_annotated_genes", "exp_term_genes"):
        ev_all = {g_: (evidence or {}).get(g_, {}).get(
            "exp" if kind == "exp_term_genes" else "all", ())
            for g_ in ast["candidate_genes"]}
        if kind == "term_annotated_genes":
            got = term_annotated_genes_owlrl(g, ast["term"], ev_all, u)
        else:
            got = exp_term_genes_owlrl(g, ast["term"], ev_all, u)
        return got == reference, f"owlrl_{kind}"
    return True, "owlrl_not_applicable"
