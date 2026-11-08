"""BYOK native verifier plugin (example 2): rdflib-based independent re-proof.

prove_exists recomputes the subclass closure from the pinned snapshot and
requires the recorded expected answer to match (build-time answer is re-derived
from the real source, never trusted blindly). certify_semantic_relevance
removes one rdfs:subClassOf edge feeding the answer and requires the closure to
change — a source-native semantic witness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness)

FILE_NAME = "dublin_core_terms.ttl"


def _closure(snapshot_root: Path, class_iri: str, *, drop_edge: bool = False) -> list[str]:
    from rdflib import Graph, RDFS, URIRef

    graph = Graph()
    graph.parse(str(snapshot_root / FILE_NAME), format="turtle")
    parent = URIRef(class_iri)
    if drop_edge:
        for sub in list(graph.subjects(RDFS.subClassOf, parent)):
            graph.remove((sub, RDFS.subClassOf, parent))
            break
    return sorted({str(s) for s in graph.transitive_subjects(RDFS.subClassOf, parent)
                   if s != parent})


class DcmiClosureVerifier:
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ev = grounded_task.native_evidence
        actual = _closure(Path(ev["snapshot_root"]), ev["class_iri"])
        expected = list(ev["expected_subclasses"])
        if not actual:
            return ProofOrCounterexample(exists=False, reason="closure empty on real source")
        if actual != expected:
            return ProofOrCounterexample(exists=False,
                                         reason="recomputed closure != recorded answer")
        return ProofOrCounterexample(exists=True, evidence={
            "answer_size": len(actual), "engine": "rdflib transitive_subjects(RDFS.subClassOf)"})

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        ev = task.native_evidence
        submitted = submitted_state_or_answer
        if isinstance(submitted, dict) and "subclasses" in submitted:
            submitted = submitted["subclasses"]
        if not isinstance(submitted, list):
            return NativeVerdict(valid=False, detail={"reason": "expected list of IRIs"})
        actual = _closure(Path(ev["snapshot_root"]), ev["class_iri"])
        return NativeVerdict(valid=sorted(str(s) for s in submitted) == actual,
                             detail={"expected_size": len(actual)})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ev = task.native_evidence
        full = _closure(Path(ev["snapshot_root"]), ev["class_iri"])
        cut = _closure(Path(ev["snapshot_root"]), ev["class_iri"], drop_edge=True)
        return SemanticWitness(ok=cut != full, witness={
            "full_size": len(full), "after_edge_removal": len(cut),
            "kind": "closure-changes-when-subclass-edge-removed"})


PLUGIN = DcmiClosureVerifier()
