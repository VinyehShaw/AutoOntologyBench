"""Rhea + ChEBI domain adapter — Review 5B workstream W3 (scientific sources).

Profile: ``linked-data/1`` (analytical track). A task is an exact-set query over the
frozen Rhea release 142 (2026-09-02) pinned under ``sources/rhea/`` with ChEBI release
256 (2026-10-06, ``sources/chebi/``) as the linked chemical-entity reference.

Identity truth comes ONLY from published cross-reference IDs (``rh:accession`` /
``rh:chebi`` in the frozen Rhea RDF); names are never used for identity. Ground truth
is the dual-certified participant index built by TWO independent implementations
(rdflib vs stdlib ElementTree; see
``data/erule/review5/scientific/rhea_index_builder.py``); ``load_certified_index``
refuses an uncertified or disagreeing index.

No live SPARQL endpoints. No biological efficacy or reaction-outcome prediction.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.protocols import (
    NativeDecision,
    NativeState,
    NativeStateAndFeedback,
    PublicObservation,
)

RheaId = NewType("RheaId", str)  # e.g. "RHEA:10000" (master reaction)
ChebiId = NewType("ChebiId", str)  # e.g. "CHEBI:16459"
CompoundIri = NewType("CompoundIri", str)  # e.g. "http://rdf.rhea-db.org/Compound_4808"

INDEX_DIR = Path("data/erule/review5/scientific/rhea_index")
PROFILE = "linked-data/1"


class IndexCertificationError(RuntimeError):
    """The certified index is missing, corrupt, or not dual-certified."""


class MalformedAction(ValueError):
    """Submitted action/answer does not match the analytical-track schema."""


@dataclass(frozen=True)
class ParticipantRecord:
    rhea_id: RheaId
    side: Literal["L", "R"]
    participant: str
    compound: CompoundIri
    chebi: ChebiId


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _num(rhea_id: RheaId) -> int:
    return int(rhea_id.split(":")[1])


@dataclass(frozen=True)
class ParticipantIndex:
    """Dual-certified participant index with derived lookups."""

    records: tuple[ParticipantRecord, ...]
    manifest_sha256: str

    def reactions_involving(self, chebi: ChebiId) -> tuple[RheaId, ...]:
        return tuple(sorted({r.rhea_id for r in self.records if r.chebi == chebi}, key=_num))

    def participants_of(self, rhea_ids: frozenset[RheaId]) -> tuple[ParticipantRecord, ...]:
        return tuple(r for r in self.records if r.rhea_id in rhea_ids)

    def participant_rows_involving(self, chebi: ChebiId) -> tuple[str, ...]:
        """Two-stage linked query: reactions selected by `chebi`, then ALL participant
        rows of those reactions with each participant's linked ChEBI identity."""
        selected = frozenset(self.reactions_involving(chebi))
        return tuple(sorted({
            participant_row(r) for r in self.records if r.rhea_id in selected
        }))

    def chebi_universe(self) -> frozenset[ChebiId]:
        return frozenset(r.chebi for r in self.records)

    def gold_crosswalk(self) -> dict[CompoundIri, ChebiId]:
        """The certified compound -> ChEBI mapping (one ChEBI per compound)."""
        out: dict[CompoundIri, ChebiId] = {}
        for r in self.records:
            prior = out.setdefault(r.compound, r.chebi)
            if prior != r.chebi:
                raise IndexCertificationError(
                    f"compound {r.compound} carries two ChEBI links: {prior} vs {r.chebi}")
        return out


def participant_row(r: ParticipantRecord) -> str:
    """Canonical linked-metadata row: reaction | side | participant | compound | chebi."""
    return "|".join([r.rhea_id, r.side, r.participant, r.compound, r.chebi])


def expected_answer(index: ParticipantIndex, query: dict[str, Any]) -> tuple[str, ...]:
    """The FROZEN native query over the certified index (single definition used by
    verify_terminal, the task builder and the qualification checks)."""
    qtype = query.get("type")
    if qtype == "reactions_involving_chebi":
        return tuple(str(r) for r in index.reactions_involving(ChebiId(query["chebi"])))
    if qtype == "participant_rows_involving_chebi":
        return index.participant_rows_involving(ChebiId(query["chebi"]))
    raise MalformedAction(f"unsupported query type {qtype!r}")


def load_certified_index(index_dir: Path = INDEX_DIR) -> ParticipantIndex:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise IndexCertificationError(f"missing index manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("paths_agree"):
        raise IndexCertificationError("index manifest: dual paths did not agree")
    payload = (index_dir / "participant_index.jsonl.gz").read_bytes()
    with gzip.open(index_dir / "participant_index.jsonl.gz", "rt") as fh:
        decompressed = fh.read().encode()
    if manifest.get("index_sha256") != _sha256_bytes(decompressed):
        raise IndexCertificationError("index payload sha256 mismatch vs manifest")
    records: list[ParticipantRecord] = []
    for line in decompressed.decode().splitlines():
            row = json.loads(line)
            records.append(
                ParticipantRecord(RheaId(row["rhea_id"]), row["side"], row["participant"],
                                  CompoundIri(row["compound"]), ChebiId(row["chebi"]))
            )
    return ParticipantIndex(tuple(records), manifest["manifest_sha256"])


# ---- linked-data/1 fixed consumer: submitted crosswalk Phi over public facts --------


def public_facts(records: tuple[ParticipantRecord, ...]) -> tuple[dict[str, str], ...]:
    """Agent-visible raw facts: reaction structure + compound IRIs. NO ChEBI links."""
    return tuple(
        {"rhea_id": r.rhea_id, "side": r.side, "participant": r.participant,
         "compound": r.compound}
        for r in records
    )


def load_crosswalk(artifact_bytes: bytes) -> dict[CompoundIri, ChebiId]:
    """Parse the submitted Phi (crosswalk) at the boundary; reject anything else."""
    payload = json.loads(artifact_bytes.decode())
    if payload.get("profile") != PROFILE or not isinstance(payload.get("mapping"), dict):
        raise MalformedAction("crosswalk must be {profile: linked-data/1, mapping: {...}}")
    return {CompoundIri(k): ChebiId(v) for k, v in payload["mapping"].items()}


def consumer_reactions_involving(
    facts: tuple[dict[str, str], ...], crosswalk: dict[CompoundIri, ChebiId], chebi: ChebiId
) -> tuple[RheaId, ...]:
    """Fixed generic join: submitted Phi supplies compound->ChEBI; structure is raw fact."""
    hits = {
        RheaId(f["rhea_id"]) for f in facts if crosswalk.get(CompoundIri(f["compound"])) == chebi
    }
    return tuple(sorted(hits, key=_num))


def consumer_participant_rows(
    facts: tuple[dict[str, str], ...], crosswalk: dict[CompoundIri, ChebiId], chebi: ChebiId
) -> tuple[str, ...]:
    """Two-stage consumer join: select reactions via Phi, then emit every participant
    row of the selected reactions with each row's ChEBI identity taken from Phi.
    Rows whose compound is unmapped in Phi are omitted (identity unknown)."""
    selected = {
        f["rhea_id"] for f in facts
        if crosswalk.get(CompoundIri(f["compound"])) == chebi
    }
    rows = set()
    for f in facts:
        if f["rhea_id"] not in selected:
            continue
        linked = crosswalk.get(CompoundIri(f["compound"]))
        if linked is None:
            continue
        rows.add("|".join([f["rhea_id"], f["side"], f["participant"], f["compound"], linked]))
    return tuple(sorted(rows))


def consumer_answer(
    facts: tuple[dict[str, str], ...], crosswalk: dict[CompoundIri, ChebiId], query: dict[str, Any]
) -> tuple[str, ...]:
    """Fixed generic consumer for the linked-data/1 profile: the submitted Phi is the
    ONLY source of ChEBI identity; structure comes from the raw public facts."""
    qtype = query.get("type")
    if qtype == "reactions_involving_chebi":
        return tuple(str(r) for r in consumer_reactions_involving(
            facts, crosswalk, ChebiId(query["chebi"])))
    if qtype == "participant_rows_involving_chebi":
        return consumer_participant_rows(facts, crosswalk, ChebiId(query["chebi"]))
    raise MalformedAction(f"unsupported query type {qtype!r}")


def swap_crosswalk(
    crosswalk: dict[CompoundIri, ChebiId], source: ChebiId, target: ChebiId
) -> dict[CompoundIri, ChebiId]:
    """Targeted semantic fault: every compound mapped to `source` is remapped to `target`."""
    return {c: (target if v == source else v) for c, v in crosswalk.items()}


# ---- NativeDomain adapter (analytical track) ----------------------------------------


class RheaChebiDomain:
    """Analytical NativeDomain: the only mutating action is ``submit_answer``."""

    def __init__(self, index: ParticipantIndex):
        self._index = index

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
            "index_records": len(self._index.records),
            "terminal": state.payload["terminal"],
        })

    def legal_action(self, state: NativeState, action: dict[str, Any]) -> NativeDecision:
        if action.get("type") != "submit_answer":
            return NativeDecision(legal=None, detail={"note": "unknown action type"})
        answer = action.get("answer")
        ok = isinstance(answer, list) and all(isinstance(x, str) for x in answer)
        return NativeDecision(legal=ok, detail={} if ok else {"reason": "answer must be list[str]"})

    def step(self, state: NativeState, action: dict[str, Any]) -> NativeStateAndFeedback:
        if state.payload["terminal"]:
            return NativeStateAndFeedback(state, {"error": "already terminal"}, 0.0)
        decision = self.legal_action(state, action)
        if decision.legal is not True:
            return NativeStateAndFeedback(state, {"rejected": decision.detail}, 0.0)
        new = NativeState(payload={**state.payload,
                                   "submitted_answer": sorted(set(action["answer"])),
                                   "terminal": True})
        return NativeStateAndFeedback(new, {"accepted": True}, 1.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        query = state.payload["query"]
        expected = expected_answer(self._index, query)
        raw = state.payload["submitted_answer"] if submitted_answer is None else submitted_answer
        if raw is None:
            # T5.6 reset probe: no answer submitted is goal_reached=False, not malformed.
            return GoalResult(
                goal_reached=False,
                evidence={"reason": "no answer submitted",
                          "expected_count": len(set(expected)),
                          "reference_engine": "dual-certified-index",
                          "index_manifest_sha256": self._index.manifest_sha256},
            )
        if not (isinstance(raw, list) and all(isinstance(x, str) for x in raw)):
            raise MalformedAction("submitted answer must be a list of strings")
        submitted = sorted(set(raw))
        expected_set, got_set = set(expected), set(submitted)
        missing, extra = sorted(expected_set - got_set), sorted(got_set - expected_set)
        return GoalResult(
            goal_reached=expected_set == got_set,
            evidence={
                "expected_count": len(expected_set),
                "submitted_count": len(got_set),
                "missing_sample": missing[:5],
                "extra_sample": extra[:5],
                "expected_sha256": _sha256_bytes(json.dumps(sorted(expected_set)).encode()),
                "submitted_sha256": _sha256_bytes(json.dumps(sorted(got_set)).encode()),
                "reference_engine": "dual-certified-index",
                "index_manifest_sha256": self._index.manifest_sha256,
            },
        )
