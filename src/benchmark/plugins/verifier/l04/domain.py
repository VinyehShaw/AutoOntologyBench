"""L04 evaluator-side domain: analytical exact-set tasks + artifact-repair tasks.

Mirrors the semantics of ``autoontobench.core.domains.rhea_chebi.RheaChebiDomain``
(reset/step/verify_terminal discrimination) for the wider L04 family set. Never
imports any ontology consumer; the reference answer lives evaluator-side only.

Action schema:
  analytical/construction: {"type": "submit_answer", "answer": list[str]}
  repair:                  {"type": "submit_artifact", "artifact": {...}} followed by
                           {"type": "submit_answer", "answer": list[str]} where the
                           answer must equal the downstream result computed through
                           the submitted artifact (verifier recomputes it natively).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from autoontobench.contracts.models import GoalResult


class MalformedAction(ValueError):
    pass


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class L04State:
    task_id: str
    query: dict[str, Any]
    submitted_answer: tuple[str, ...] | None
    submitted_artifact: dict[str, Any] | None
    terminal: bool


class L04Domain:
    """Exact-set terminal semantics for all L04 families."""

    def reset(self, task_id: str, query: dict[str, Any]) -> L04State:
        return L04State(task_id=task_id, query=query, submitted_answer=None,
                        submitted_artifact=None, terminal=False)

    def step(self, state: L04State, action: dict[str, Any]) -> tuple[L04State, dict[str, Any]]:
        if state.terminal:
            return state, {"error": "already terminal"}
        atype = action.get("type")
        if atype == "submit_answer":
            answer = action.get("answer")
            if not (isinstance(answer, list) and all(isinstance(x, str) for x in answer)):
                return state, {"rejected": "answer must be list[str]"}
            new = L04State(state.task_id, state.query, tuple(sorted(set(answer))),
                           state.submitted_artifact, True)
            return new, {"accepted": True}
        if atype == "submit_artifact":
            artifact = action.get("artifact")
            if not isinstance(artifact, dict):
                return state, {"rejected": "artifact must be a dict"}
            new = L04State(state.task_id, state.query, state.submitted_answer,
                           artifact, False)
            return new, {"accepted": True}
        return state, {"rejected": f"unknown action type {atype!r}"}

    def verify_terminal(
        self,
        state: L04State,
        expected_answer: tuple[str, ...],
        *,
        reference_engine: str,
        evidence_graph_sha256: str,
    ) -> GoalResult:
        if state.submitted_answer is None:
            return GoalResult(goal_reached=False, evidence={
                "reason": "no answer submitted",
                "expected_count": len(set(expected_answer)),
                "reference_engine": reference_engine,
                "evidence_graph_sha256": evidence_graph_sha256,
            })
        expected_set = set(expected_answer)
        got_set = set(state.submitted_answer)
        return GoalResult(
            goal_reached=expected_set == got_set,
            evidence={
                "expected_count": len(expected_set),
                "submitted_count": len(got_set),
                "missing_sample": sorted(expected_set - got_set)[:5],
                "extra_sample": sorted(got_set - expected_set)[:5],
                "expected_sha256": _sha(sorted(expected_set)),
                "submitted_sha256": _sha(sorted(got_set)),
                "reference_engine": reference_engine,
                "evidence_graph_sha256": evidence_graph_sha256,
            },
        )
