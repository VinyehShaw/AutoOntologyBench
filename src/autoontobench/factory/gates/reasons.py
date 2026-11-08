"""Stable qualification gate + rejection reason codes (owner: S11).

Reason codes are part of the audit contract (FINAL-BUILD-ORDER-v2 §5.5.2 lists
the minimum taxonomy). Codes are frozen strings: once emitted into a build's
`rejections.jsonl` they must keep parsing, so extend — never rename.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Gate(str, Enum):
    """Ordered qualification gates (cheap -> expensive, §Appendix A)."""

    SCHEMA = "schema"
    NATIVE_VERIFIED = "native_verified"
    SOLVABLE = "solvable"
    NONTRIVIAL = "nontrivial"
    SEMANTIC_RELEVANCE = "semantic_relevance"
    LEAKAGE_CLEAN = "leakage_clean"
    DEDUP = "dedup"
    RELEASE_ELIGIBLE = "release_eligible"


# Gate order for the status ladder; index = ladder depth reached.
GATE_LADDER: tuple[Gate, ...] = (
    Gate.SCHEMA,
    Gate.NATIVE_VERIFIED,
    Gate.SOLVABLE,
    Gate.NONTRIVIAL,
    Gate.SEMANTIC_RELEVANCE,
    Gate.LEAKAGE_CLEAN,
    Gate.DEDUP,
    Gate.RELEASE_ELIGIBLE,
)


class Reason(str, Enum):
    """Stable rejection / hold codes. `OK` marks a passed gate."""

    OK = "OK"
    # schema / structural
    SCHEMA_INVALID = "SCHEMA_INVALID"
    SOURCE_MISSING = "SOURCE_MISSING"
    # native verification
    NATIVE_ENGINE_MISSING = "NATIVE_ENGINE_MISSING"
    PROFILE_UNSUPPORTED = "PROFILE_UNSUPPORTED"
    PROOF_MISSING = "PROOF_MISSING"
    NON_DETERMINISTIC = "NON_DETERMINISTIC"
    # solvability / triviality
    UNSOLVABLE = "UNSOLVABLE"
    TRIVIAL = "TRIVIAL"
    # semantics
    NON_SEMANTIC = "NON_SEMANTIC"
    SEMANTIC_WITNESS_MISSING = "SEMANTIC_WITNESS_MISSING"
    # leakage / security
    LEAKAGE = "LEAKAGE"
    ISOLATION_VIOLATION = "ISOLATION_VIOLATION"
    # dedup
    DUPLICATE = "DUPLICATE"
    # licensing / release
    LICENSE_HOLD = "LICENSE_HOLD"
    REDISTRIBUTION_RESTRICTED = "REDISTRIBUTION_RESTRICTED"


@dataclass(frozen=True)
class GateVerdict:
    """Per-task verdict for one gate. `evidence_refs` point at real artifacts."""

    gate: Gate
    passed: bool
    reason: Reason
    evidence_refs: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "passed": self.passed,
            "reason": self.reason.value,
            "evidence_refs": list(self.evidence_refs),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TaskDecision:
    """Full pipeline outcome for one task: ladder depth + first failing verdict."""

    task_id: str
    lane_id: str
    status: str  # ladder label, e.g. CANDIDATE / SOLVABLE / QUALIFIED / RELEASE_ELIGIBLE
    verdicts: tuple[GateVerdict, ...]
    reject_reason: Reason = Reason.OK

    @property
    def qualified(self) -> bool:
        return self.reject_reason is Reason.OK and self.status in (
            "QUALIFIED",
            "RELEASE_ELIGIBLE",
        )
