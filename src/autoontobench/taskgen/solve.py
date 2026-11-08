"""T2/T3 solver stage: goal-first construction with replay-verified solutions.

Design (Implementation section 4 T2): sample goal + solution plan backwards from a
solved end state (BFS/A* over the native transition model), pick a non-goal start,
replay through the independent native environment, store the plan privately as an
EXISTENCE PROOF (never model-visible). Needs NativeDomain (W2/W3/W4) -> stubbed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoontobench.core.protocols import NativeDomain


@dataclass(frozen=True)
class SolutionProof:
    """Private existence proof. Lives under the private root only."""
    plan: tuple[dict[str, Any], ...]
    min_actions: int
    replay_ok: bool
    native_terminal_sha: str


@dataclass(frozen=True)
class SolutionResult:
    solved: bool
    proof: SolutionProof | None
    reason: str = ""


def solve_blueprint(domain: NativeDomain | None, blueprint: dict[str, Any],
                    seed: int) -> SolutionResult:
    raise NotImplementedError(
        "NOT_IMPLEMENTED_DEP: solve_blueprint needs NativeDomain replay "
        "(W2/W3/W4); lands with the first native domain (M2)")
