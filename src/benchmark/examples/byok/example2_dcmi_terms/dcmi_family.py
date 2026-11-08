"""BYOK task-family plugin (example 2): ontology closure tasks over DCMI terms.

Family `dcmi-subclass-closure`: given the pinned DCMI terms ontology and a
class, enumerate the exact set of its transitive subclasses in the ontology.
Grounding computes the reference answer with rdflib and stores it only in the
private oracle; the public task carries the class IRI and the snapshot ref.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask, SourceSnapshot

FAMILIES = ("dcmi-subclass-closure", "dcmi-subproperty-closure")
PROFILE = "rdf-rdfs/1.1"


def _closures(snapshot_root: Path, predicate: Any) -> list[tuple[str, list[str]]]:
    from rdflib import Graph

    graph = Graph()
    graph.parse(str(snapshot_root / "dublin_core_terms.ttl"), format="turtle")
    parents: dict[Any, set[str]] = {}
    for sub, _, parent in graph.triples((None, predicate, None)):
        parents.setdefault(parent, set()).add(sub)
    out: list[tuple[str, list[str]]] = []
    for parent in sorted(parents, key=str):
        closure = sorted({str(s) for s in graph.transitive_subjects(predicate, parent)
                          if s != parent})
        if closure:
            out.append((str(parent), closure))
    return out


class DcmiClosureFamily:
    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    def propose(self, snapshot: SourceSnapshot, *, family: str, seed: int,
                constraints: dict[str, Any]) -> Iterator[CandidateTaskAST]:
        from rdflib import RDFS

        count = int(constraints.get("count", 8))
        if family == "dcmi-subclass-closure":
            predicate, relation = RDFS.subClassOf, "rdfs:subClassOf"
        elif family == "dcmi-subproperty-closure":
            predicate, relation = RDFS.subPropertyOf, "rdfs:subPropertyOf"
        else:
            return
        closures = _closures(Path(snapshot.root), predicate)
        offset = seed % len(closures) if closures else 0
        produced = 0
        for i, (iri, closure) in enumerate(closures):
            if i < offset or produced >= count:
                continue
            local = iri.rsplit("/", 1)[-1]
            yield CandidateTaskAST(
                lane_id="BYOK", family_id=family, base_goal_id=f"{family}-{local}",
                instance_id=f"{family}-{local}-{seed}",
                regime=str(constraints.get("regime", "application")),
                semantic_profile=PROFILE,
                ast={"root_iri": iri, "relation": relation,
                     "expected_closure": closure,
                     "snapshot_root": str(snapshot.root), "seed": seed},
                source_unit_ids=(f"dcmi-terms/{local}",))
            produced += 1

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        iri = candidate.ast["root_iri"]
        relation = candidate.ast["relation"]
        kind = "classes" if relation == "rdfs:subClassOf" else "properties"
        public = {
            "user_goal": (
                f"Using the pinned DCMI Metadata Terms ontology (dublin_core_terms.ttl), "
                f"list the complete set of {kind} that are transitive {relation} "
                f"descendants of <{iri}> as declared in that file. Return the sorted "
                f"list of full IRIs; do not include the root itself."),
            "ontology_ref": "external_knowledge/pinned/dcmi-terms/dcmi-20120614/dublin_core_terms.ttl",
            "root_iri": iri, "relation": relation,
            "allowed_tools": ["rdflib"],
            "budget": {"probes": 2, "actions": 4},
        }
        return GroundedTask(
            ast=candidate, public=public,
            oracle_ref=f"oracle://byok/{candidate.instance_id}",
            native_evidence={"expected_closure": candidate.ast["expected_closure"],
                             "root_iri": iri, "relation": relation,
                             "snapshot_root": candidate.ast["snapshot_root"]})


PLUGIN = DcmiClosureFamily()
