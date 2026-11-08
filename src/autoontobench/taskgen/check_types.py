"""T5 check types shared by every check module (Implementation section 4, step T5).

Split out of `validate.py` so the check modules stay import-cycle free and each
file remains small enough to review. Status semantics (binding):

- ``PASS``                check ran on real evidence and held.
- ``FAIL``                check ran and found a defect -> task REJECTED(reason).
- ``NOT_IMPLEMENTED_DEP`` check cannot honestly run yet; ``dep`` names the exact
                          missing dependency. Never a silent PASS, never a 0.
- ``NOT_APPLICABLE``      the task's own claim does not require this check today
                          (e.g. T5.7/T5.14 on an ``ontology_contract=native-only/0``
                          task). The ``reason`` must name the trigger that would
                          re-activate the check as REQUIRED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from autoontobench.contracts.models import WordingCandidate
from autoontobench.taskgen.registry import SourceRegistry

if TYPE_CHECKING:  # avoid import cycle: native_check never imports check types
    from autoontobench.taskgen.native_check import NativeEvidence

REPO_ROOT = Path(__file__).resolve().parents[2]


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_IMPLEMENTED_DEP = "NOT_IMPLEMENTED_DEP"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: CheckStatus
    reason: str = ""
    dep: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckContext:
    """Everything a T5 check may consult. Absent pieces force honest markers."""

    registry: SourceRegistry | None = None
    repo_root: Path = REPO_ROOT
    blueprint: Mapping[str, Any] | None = None  # W2 blueprint record (goal/qc/plan ref)
    wording: WordingCandidate | None = None     # T4 DashScope paraphrase (optional)
    native: NativeEvidence | None = None        # replay/reset evidence (per blueprint)
    seen_goal_shas: Mapping[str, str] = field(default_factory=dict)  # sha -> base goal id


def dep(name: str, dependency: str) -> CheckResult:
    return CheckResult(check=name, status=CheckStatus.NOT_IMPLEMENTED_DEP, dep=dependency)
