"""L06 independent owlrl leg: W3C-rules entailment on task fragments.

This is the second, independent verification path required by the build order
("独立原生验证，重算证据"). The fragment edge list is materialized as an RDF
graph (is_a -> rdfs:subClassOf; relations -> their OBO property IRIs), axiom
triples are injected to encode the declared profile go-rel-closure/1:

  - BFO_0000050 (part_of) typed owl:TransitiveProperty                 (R2)
  - propertyChainAxiom (is_a ∘ part_of) ⊑ part_of                      (R3)
  - propertyChainAxiom (part_of ∘ is_a) ⊑ part_of                      (R4)
  - pos/neg_regulates rdfs:subPropertyOf regulates                     (R5)
  - propertyChainAxiom (is_a ∘ R) ⊑ R and (R ∘ is_a) ⊑ R for R in
    {regulates, positively_regulates, negatively_regulates}            (R6/R7)

``owlrl.DeductiveClosure(owlrl.OWLRL_Semantics)`` then materializes the
entailed triples; answers are read off the closed graph. is_a ancestors use
RDFS subClassOf transitivity (R1). Regulates gets NO TransitiveProperty (R8).

The leg answers the same questions as ``fragment.py`` and the two must agree
exactly; any disagreement is a PROFILE_UNSUPPORTED rejection.
"""
from __future__ import annotations

from typing import Any

import owlrl
import rdflib
from rdflib import URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

GO_NS = "http://purl.obolibrary.org/obo/"
REL_NAME_TO_IRI = {
    "part_of": "BFO_0000050",
    "has_part": "BFO_0000051",
    "occurs_in": "BFO_0000066",
    "regulates": "RO_0002211",
    "negatively_regulates": "RO_0002212",
    "positively_regulates": "RO_0002213",
    "capable_of": "RO_0002215",
    "capable_of_part_of": "RO_0002216",
}
ISA = RDFS.subClassOf


def _u(go_id: str) -> URIRef:
    return URIRef(GO_NS + go_id.replace(":", "_"))


def _g(uri: URIRef) -> str:
    s = str(uri)
    return s[len(GO_NS):].replace("_", ":", 1) if s.startswith(GO_NS) else s


def build_closed_graph(edges: list[tuple[str, str, str]]) -> rdflib.Graph:
    g = rdflib.Graph()
    partof = URIRef(GO_NS + REL_NAME_TO_IRI["part_of"])
    reg = URIRef(GO_NS + REL_NAME_TO_IRI["regulates"])
    posreg = URIRef(GO_NS + REL_NAME_TO_IRI["positively_regulates"])
    negreg = URIRef(GO_NS + REL_NAME_TO_IRI["negatively_regulates"])
    g.add((partof, RDF.type, OWL.TransitiveProperty))
    g.add((posreg, RDFS.subPropertyOf, reg))
    g.add((negreg, RDFS.subPropertyOf, reg))
    for prop, chain in (
        (partof, (ISA, partof)),
        (partof, (partof, ISA)),
        (reg, (ISA, reg)), (reg, (reg, ISA)),
        (posreg, (ISA, posreg)), (posreg, (posreg, ISA)),
        (negreg, (ISA, negreg)), (negreg, (negreg, ISA)),
    ):
        lst = rdflib.BNode()
        Collection(g, lst, list(chain))
        g.add((prop, OWL.propertyChainAxiom, lst))
    for s, p, o in edges:
        pred = ISA if p == "is_a" else URIRef(GO_NS + REL_NAME_TO_IRI[p])
        g.add((_u(s), pred, _u(o)))
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    return g


def isa_ancestors_owlrl(g: rdflib.Graph, seed: str, universe: set[str]) -> frozenset[str]:
    out = {_g(o) for o in g.transitive_objects(_u(seed), ISA)}
    out.discard(seed)
    return frozenset(x for x in out if x in universe)


def partof_ancestors_owlrl(g: rdflib.Graph, seed: str, universe: set[str]) -> frozenset[str]:
    """R1-R4 part_of ancestors: the chain axioms make entailed part_of(seed,o)
    exactly the declared profile set."""
    pred = URIRef(GO_NS + REL_NAME_TO_IRI["part_of"])
    out = {_g(o) for o in g.objects(_u(seed), pred)}
    out.discard(seed)
    return frozenset(x for x in out if x in universe)


def regulators_owlrl(g: rdflib.Graph, seed: str, universe: set[str]) -> frozenset[str]:
    """R5-R7 regulators of seed: entailed (s, regulates, seed) with subProperty
    rollup and both is_a propagation chains; regulates is not transitive."""
    pred = URIRef(GO_NS + REL_NAME_TO_IRI["regulates"])
    out = {_g(s) for s in g.subjects(pred, _u(seed))}
    out.discard(seed)
    return frozenset(x for x in out if x in universe)
