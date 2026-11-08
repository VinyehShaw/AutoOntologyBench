"""Scripted mock model for G0-gated M1 (owner: W6).

G0 rule: no real Researcher Evaluation in M1 — every run uses this scripted
``MockModel``; NO DashScope/API calls (W5 owns the API adapters). A real
model later plugs in through the same duck-typed ``AgentModel`` surface:
``next_action(observation) -> {"tool": ..., "args": {...}}``.

A ``MockModel`` consumes a fixed script of tool calls; when the script is
exhausted it emits ``submit_final`` with the configured answer. It also
doubles as the deterministic replay vehicle: a frozen ``action_sequence``
IS the script, so replay = run with the same script, same seed, same spec.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

MOCK_MODEL_ID = "mock-scripted/0.1"


class MockModel:
    """Duck-typed AgentModel: deterministic, offline, fully logged."""

    provider = "mock"
    model_exact_id = MOCK_MODEL_ID

    def __init__(
        self,
        script: Sequence[Mapping[str, Any]],
        *,
        final_answer: Any = None,
    ):
        self._script = [dict(step) for step in script]
        self._final_answer = (
            final_answer if final_answer is not None else {"summary": "mock script complete"}
        )
        self._cursor = 0
        self.calls_made = 0

    def next_action(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._cursor < len(self._script):
            call = dict(self._script[self._cursor])
            self._cursor += 1
        else:
            call = {"tool": "submit_final", "args": {"answer": self._final_answer}}
        self.calls_made += 1
        return call


def replay_model(action_sequence: Sequence[Mapping[str, Any]], *, final_answer: Any = None) -> MockModel:
    """Deterministic replay mode: the frozen action sequence is the script."""
    return MockModel(action_sequence, final_answer=final_answer)
