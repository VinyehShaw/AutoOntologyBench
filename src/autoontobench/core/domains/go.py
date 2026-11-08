"""Gene Ontology domain adapter — Review 5B workstream W3 (scientific sources).

Track: ``logical_ontology``. This module WRAPS the existing frozen GO slice
(``data/erule/logical/go/``, built from ``references/sources/gene_ontology/go-basic.obo``
release 2026-07-26, sha256 b08d45b2…, CC-BY-4.0) as a NativeDomain skeleton.

LOUD TRACK DECLARATION (carried from audit/GO_AUDIT.md): the decision contract is
boolean entailment of declared competency queries (``rdfs:subClassOf`` /
``goslice:partOf`` between GO term pairs) under OWL2-RL/RDFS closure. GO is NOT a
knockout or phenotype oracle; no binary action-legality decision is defined, measured,
or claimed anywhere in this track. Nothing here measures or implies phenotype
prediction, gene essentiality, or action legality.

Reference truth is DUAL-ENGINE: owlrl 7.6.2 (``DeductiveClosure(OWLRL_Semantics)``)
and Apache Jena 6.2.0 ``arq`` (SPARQL property paths) must agree on every label;
disagreement raises and the query is excluded, never hidden. Answer semantics are
three-valued: ``true`` / ``false`` / ``unknown`` (unknown = a queried term is absent
from the declared 288-term slice).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.protocols import (
    NativeDecision,
    NativeState,
    NativeStateAndFeedback,
    PublicObservation,
)

#: Review-5 G0 (2026-10-26): the pre-repack GO world root is quarantined; the
#: evaluator-side native-truth ontology is the sealed private copy (byte-identical,
#: sha256 a45f155a…c2df — see audit/review5/QUARANTINE_EXECUTION.md).
GO_TTL = Path("worlds/private/erule-go-logical-true-s0/expert_reference_w3c/ontology.ttl")
ARQ = Path("/opt/jena/apache-jena-6.2.0/bin/arq")
GO_OBO = "http://purl.obolibrary.org/obo/GO_"
GOSLICE = "urn:erule:go:slice#"
PROFILE = "go-entailment/1"

RELATIONS: dict[str, tuple[str, str]] = {
    "subClassOf": ("http://www.w3.org/2000/01/rdf-schema#subClassOf", "rdfs"),
    "partOf": (GOSLICE + "partOf", "goslice"),
}

Answer = Literal["true", "false", "unknown"]


@dataclass(frozen=True)
class EntailmentQuery:
    child: str      # e.g. "GO:0007093"
    relation: str   # key of RELATIONS
    parent: str     # e.g. "GO:1903047"


class DualEngineDisagreement(RuntimeError):
    """owlrl and Jena disagree on an entailment label; query must be excluded."""


class JenaRunnerError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _go_iri(term: str) -> str:
    return GO_OBO + term.replace("GO:", "")


def _terms_present(ttl_path: Path, terms: tuple[str, ...]) -> dict[str, bool]:
    import rdflib

    graph = rdflib.Graph()
    graph.parse(ttl_path, format="turtle")
    subjects = {str(s) for s in graph.subjects()}
    return {t: _go_iri(t) in subjects for t in terms}


def entail_owlrl(ttl_path: Path, query: EntailmentQuery) -> bool:
    """Engine A: owlrl OWL2-RL closure, then direct triple containment."""
    import rdflib
    from owlrl import DeductiveClosure, OWLRL_Semantics

    graph = rdflib.Graph()
    graph.parse(ttl_path, format="turtle")
    DeductiveClosure(OWLRL_Semantics, rdfs_closure=True, axiomatic_triples=False).expand(graph)
    pred = rdflib.URIRef(RELATIONS[query.relation][0])
    return (
        rdflib.URIRef(_go_iri(query.child)), pred, rdflib.URIRef(_go_iri(query.parent))
    ) in graph


def entail_jena(ttl_path: Path, query: EntailmentQuery) -> bool:
    """Engine B: Apache Jena arq SPARQL ASK with transitive property path."""
    pred, _prefix = RELATIONS[query.relation]
    sparql = (
        "PREFIX go: <http://purl.obolibrary.org/obo/GO_>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX goslice: <urn:erule:go:slice#>\n"
        f"ASK {{ go:{query.child.replace('GO:', '')} <{pred}>+ go:{query.parent.replace('GO:', '')} . }}\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rq", delete=False) as fh:
        fh.write(sparql)
        query_path = fh.name
    env = dict(os.environ)
    if shutil.which("java") is None:
        # pinned JDK lives in the dedicated conda env (same setup as audit/logs/go)
        env["JAVA_HOME"] = "/opt/conda/envs/jena"
        env["PATH"] = "/opt/conda/envs/jena/bin:" + env.get("PATH", "")
    proc = subprocess.run(
        [str(ARQ), "--data", str(ttl_path), "--query", query_path],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if proc.returncode != 0:
        raise JenaRunnerError(f"arq rc={proc.returncode}: {proc.stderr[-300:]}")
    lines = [ln.strip().lower() for ln in proc.stdout.splitlines() if ln.strip()]
    if set(lines) & {"true", "yes"}:
        return True
    if set(lines) & {"false", "no"}:
        return False
    raise JenaRunnerError(f"unparseable arq output: {proc.stdout[-300:]!r}")


def reference_entailment(ttl_path: Path, query: EntailmentQuery) -> tuple[Answer, dict[str, Any]]:
    """Three-valued dual-certified label: unknown if a term is outside the slice."""
    presence = _terms_present(ttl_path, (query.child, query.parent))
    if not all(presence.values()):
        return "unknown", {"terms_present": presence, "engines": "membership-check (both trivially agree)"}
    owl = entail_owlrl(ttl_path, query)
    jena = entail_jena(ttl_path, query)
    if owl != jena:
        raise DualEngineDisagreement(
            f"{query.child} {query.relation} {query.parent}: owlrl={owl} jena={jena} — exclude with reproducer"
        )
    label: Answer = "true" if owl else "false"
    return label, {
        "terms_present": presence,
        "owlrl": owl,
        "jena": jena,
        "engines_agree": True,
        "ttl_sha256": _sha256_bytes(ttl_path.read_bytes()),
    }


class GoDomain:
    """Logical-ontology NativeDomain: the only mutating action is ``submit_answer``."""

    def __init__(self, ttl_path: Path = GO_TTL):
        self._ttl = ttl_path

    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        instance = json.loads(Path(spec.initial_state_ref).read_text())
        return NativeState(
            payload={"task_id": spec.task_id, "query": instance["query"],
                     "submitted_answer": None, "terminal": False}
        )

    def observe(self, state: NativeState, rights: set[str]) -> PublicObservation:
        return PublicObservation(payload={
            "task_id": state.payload["task_id"],
            "query": state.payload["query"],
            "track": "logical_ontology (NOT a knockout/phenotype oracle)",
            "terminal": state.payload["terminal"],
        })

    def legal_action(self, state: NativeState, action: dict[str, Any]) -> NativeDecision:
        if action.get("type") != "submit_answer":
            return NativeDecision(legal=None, detail={"note": "unknown action type"})
        ok = action.get("answer") in ("true", "false", "unknown")
        return NativeDecision(legal=ok, detail={} if ok else {"reason": "answer must be true|false|unknown"})

    def step(self, state: NativeState, action: dict[str, Any]) -> NativeStateAndFeedback:
        if state.payload["terminal"]:
            return NativeStateAndFeedback(state, {"error": "already terminal"}, 0.0)
        decision = self.legal_action(state, action)
        if decision.legal is not True:
            return NativeStateAndFeedback(state, {"rejected": decision.detail}, 0.0)
        new = NativeState(payload={**state.payload, "submitted_answer": action["answer"],
                                   "terminal": True})
        return NativeStateAndFeedback(new, {"accepted": True}, 1.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        query = state.payload["query"]
        label, detail = reference_entailment(
            self._ttl,
            EntailmentQuery(child=query["child"], relation=query["relation"], parent=query["parent"]),
        )
        submitted = state.payload["submitted_answer"] if submitted_answer is None else submitted_answer
        return GoalResult(
            goal_reached=submitted == label,
            evidence={
                "reference_label": label,
                "submitted_label": submitted,
                "dual_engine": detail,
            },
        )
