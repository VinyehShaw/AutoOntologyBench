"""Mechanistic controls (owner: W6; Review-5B §6 E1/E2).

Controls are data: each names an artifact source key and its expected
relation to gold. They are mechanistic instrument checks, NOT crippled
substitutes for DIRECT. ``artifact_key`` resolves through the harness's
``artifact_sources`` map (fixture-provided in M1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    artifact_key: str
    expected_relation: str  # expected decision behaviour vs gold on governed steps
    note: str


CONTROLS: dict[str, ControlSpec] = {
    "vocab_only": ControlSpec(
        "vocab_only",
        artifact_key="vocab_only",
        expected_relation="uncovered_unknown",
        note="vocabulary present, rules absent: governed actions must be UNKNOWN (uncovered)",
    ),
    "no_op": ControlSpec(
        "no_op",
        artifact_key="no_op",
        expected_relation="uncovered_unknown_or_allow",
        note="empty/irrelevant shapes: must never change a governed decision vs no-artifact",
    ),
    "equiv_rewrite": ControlSpec(
        "equiv_rewrite",
        artifact_key="equiv_rewrite",
        expected_relation="same_as_gold",
        note="semantic-equivalent rewrite: zero decision changes vs gold on all steps",
    ),
    "corrupt": ControlSpec(
        "corrupt",
        artifact_key="corrupt",
        expected_relation="differs_on_governed_only",
        note="fault-injected: must differ from gold on governed steps, never on unaffected",
    ),
    "repaired": ControlSpec(
        "repaired",
        artifact_key="repaired",
        expected_relation="same_as_gold",
        note="targeted repair of corrupt: must restore gold decisions on governed steps",
    ),
}
