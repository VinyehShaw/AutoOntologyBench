"""L14 evaluator-side domain: analytical exact-set tasks + artifact-repair.

Mirrors the semantics of ``autoontobench.core.domains`` for the L14 family
set. Never imports any ontology consumer; the reference answer lives
evaluator-side only.

Action schema:
  analytical/organization: {"type": "submit_answer", "answer": list[str]}
  repair:                  {"type": "submit_artifact", "artifact": {...}} then
                           {"type": "submit_answer", "answer": list[str]}; the
                           answer must equal the downstream result computed
                           through the submitted artifact (recomputed natively
                           by comparing the artifact's corrected rows against
                           the expected correction map).
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
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class L14State:
    task_id: str
    query: dict[str, Any]
    submitted_answer: tuple[str, ...] | None
    submitted_artifact: dict[str, Any] | None
    terminal: bool


class L14Domain:
    """Exact-set terminal semantics for all L14 families."""

    def reset(self, task_id: str, query: dict[str, Any]) -> L14State:
        return L14State(task_id=task_id, query=query, submitted_answer=None,
                        submitted_artifact=None, terminal=False)

    def step(self, state: L14State,
             action: dict[str, Any]) -> tuple[L14State, dict[str, Any]]:
        if state.terminal:
            return state, {"error": "already terminal"}
        atype = action.get("type")
        if atype == "submit_answer":
            answer = action.get("answer")
            if not (isinstance(answer, list)
                    and all(isinstance(x, str) for x in answer)):
                return state, {"rejected": "answer must be list[str]"}
            new = L14State(state.task_id, state.query,
                           tuple(sorted(set(answer))),
                           state.submitted_artifact, True)
            return new, {"accepted": True}
        if atype == "submit_artifact":
            artifact = action.get("artifact")
            if not isinstance(artifact, dict):
                return state, {"rejected": "artifact must be a dict"}
            new = L14State(state.task_id, state.query,
                           state.submitted_answer, artifact, False)
            return new, {"accepted": True}
        return state, {"rejected": f"unknown action type {atype!r}"}

    def verify_terminal(
        self,
        state: L14State,
        expected_answer: tuple[str, ...],
        *,
        reference_engine: str,
        evidence_db_sha256: str,
        expected_artifact: dict[str, Any] | None = None,
    ) -> GoalResult:
        if state.submitted_answer is None:
            return GoalResult(goal_reached=False, evidence={
                "reason": "no answer submitted",
                "expected_count": len(set(expected_answer)),
                "reference_engine": reference_engine,
                "evidence_db_sha256": evidence_db_sha256,
            })
        expected_set = set(expected_answer)
        got_set = set(state.submitted_answer)
        artifact_ok = True
        artifact_detail: dict[str, Any] = {}
        if expected_artifact is not None:
            if state.submitted_artifact is None:
                artifact_ok = False
                artifact_detail = {"reason": "repair task: no artifact"}
            else:
                exp_fixes = expected_artifact.get("corrected_rows", {})
                got_rows = (state.submitted_artifact.get("rows")
                            if isinstance(state.submitted_artifact, dict)
                            else None) or {}
                mismatches = []
                for key, want in exp_fixes.items():
                    got = got_rows.get(key)
                    if got != want:
                        mismatches.append(key)
                artifact_ok = not mismatches
                artifact_detail = {
                    "expected_corrections": len(exp_fixes),
                    "mismatch_count": len(mismatches),
                    "mismatch_sample": mismatches[:5],
                }
        ok = expected_set == got_set and artifact_ok
        return GoalResult(
            goal_reached=ok,
            evidence={
                "expected_count": len(expected_set),
                "submitted_count": len(got_set),
                "missing_sample": sorted(expected_set - got_set)[:5],
                "extra_sample": sorted(got_set - expected_set)[:5],
                "expected_sha256": _sha(sorted(expected_set)),
                "submitted_sha256": _sha(sorted(got_set)),
                "artifact_check": artifact_detail,
                "reference_engine": reference_engine,
                "evidence_db_sha256": evidence_db_sha256,
            },
        )
