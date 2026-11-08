"""T5 deterministic qualification checklist (Implementation section 4, step T5).

Fifteen checks live in `checks_native.py` (1,2,4,5,6,15), `checks_public.py`
(3,8,9,10,11,12,13) and `checks_o.py` (7,14), ordered below exactly as in the
spec. Every check returns PASS / FAIL / NOT_IMPLEMENTED_DEP / NOT_APPLICABLE —
never a silent pass. Status ladder (T5):

    CANDIDATE -> NATIVE_VERIFIED -> TASK_FEASIBLE -> SEMANTICALLY_GROUNDED
    -> LEAK_FREE -> QUALIFIED,  or REJECTED(reason) / PENDING while required
    deps are missing.

A rung is reached iff all of its checks PASS or are NOT_APPLICABLE. A FAIL
anywhere rejects the task; a NOT_IMPLEMENTED_DEP on a required check caps the
ladder below that rung and the decision stays PENDING. `TaskQualificationValidator`
caches native replay evidence per blueprint (paraphrases share one certified task).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from autoontobench.contracts.models import TaskSpecV5, WordingCandidate
from autoontobench.taskgen import checks_native, checks_o, checks_public
from autoontobench.taskgen.check_types import (
    CheckContext, CheckResult, CheckStatus, REPO_ROOT,
)
from autoontobench.taskgen.dedupe import goal_sha
from autoontobench.taskgen.native_check import NativeEvidence, gather_native_evidence
from autoontobench.taskgen.registry import SourceRegistry

#: the 15 T5 checks in spec order (Implementation §4 T5)
ALL_CHECKS = (
    checks_native.check_provenance_license,              # 1
    checks_native.check_native_reference_executable,     # 2
    checks_public.check_wellformed_typed_input,          # 3
    checks_native.check_native_goal_reachable,           # 4
    checks_native.check_independent_gold_check,          # 5
    checks_native.check_not_solved_at_reset,             # 6
    checks_o.check_controlled_o_content_sensitivity,     # 7
    checks_public.check_leak_free,                       # 8
    checks_public.check_wording_entails_certified_goal,  # 9
    checks_public.check_no_generated_answer_exposed,     # 10
    checks_public.check_action_length_calibrated,        # 11
    checks_public.check_originality_dedup,               # 12
    checks_public.check_declared_track_and_split,        # 13
    checks_o.check_operator_profile_consumed_as_authored,  # 14
    checks_native.check_public_docs_parity,              # 15
)

#: rung -> the T5 checks that gate it (Implementation §4 T5 status ladder)
LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CANDIDATE", ("wellformed_typed_input",)),
    ("NATIVE_VERIFIED", ("provenance_license", "native_reference_executable",
                         "independent_gold_check")),
    ("TASK_FEASIBLE", ("native_goal_reachable", "not_solved_at_reset")),
    ("SEMANTICALLY_GROUNDED", ("wording_entails_certified_goal",
                               "no_generated_answer_exposed",
                               "action_length_calibrated", "originality_dedup",
                               "declared_track_and_split")),
    ("LEAK_FREE", ("leak_free", "public_docs_parity")),
    ("QUALIFIED", ("controlled_o_content_sensitivity",
                   "operator_profile_consumed_as_authored")),
)


@dataclass(frozen=True)
class QualificationReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def decision(self) -> str:
        if any(r.status is CheckStatus.FAIL for r in self.results):
            return "REJECTED"
        if any(r.status is CheckStatus.NOT_IMPLEMENTED_DEP for r in self.results):
            return "PENDING"
        return "QUALIFIED"

    @property
    def ladder_status(self) -> str:
        """Highest rung whose gating checks all PASS / NOT_APPLICABLE."""
        if self.decision == "REJECTED":
            return "REJECTED"
        by_check = {r.check: r.status for r in self.results}
        highest = "CANDIDATE"
        for rung, gates in LADDER:
            if all(by_check.get(c) in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE)
                   for c in gates):
                highest = rung
            else:
                break
        return highest

    @property
    def rejection_reasons(self) -> list[str]:
        return [f"{r.check}: {r.reason}" for r in self.results
                if r.status is CheckStatus.FAIL]

    @property
    def pending_deps(self) -> list[str]:
        return [f"{r.check} (dep: {r.dep})" for r in self.results
                if r.status is CheckStatus.NOT_IMPLEMENTED_DEP]

    @property
    def not_applicable_checks(self) -> list[str]:
        return [f"{r.check}: {r.reason}" for r in self.results
                if r.status is CheckStatus.NOT_APPLICABLE]


class TaskQualificationValidator:
    def __init__(self, registry: SourceRegistry | None = None, *,
                 repo_root: Path = REPO_ROOT, log_root: Path | None = None) -> None:
        self.registry = registry
        self.repo_root = Path(repo_root)
        self.log_root = Path(log_root) if log_root else (
            self.repo_root / "data" / "erule" / "review5" / "taskfactory" / "logs" / "qualify")
        self.seen_goal_shas: dict[str, str] = {}
        self._native_cache: dict[str, NativeEvidence] = {}

    def native_evidence(
        self, spec: TaskSpecV5, blueprint: Mapping[str, Any] | None
    ) -> NativeEvidence:
        """Replay/reset evidence, computed once per blueprint_id (paraphrase share)."""
        key = str((blueprint or {}).get("blueprint_id") or spec.task_id)
        if key not in self._native_cache:
            self._native_cache[key] = gather_native_evidence(
                spec, blueprint, log_root=self.log_root)
        return self._native_cache[key]

    def validate(self, spec: TaskSpecV5, *,
                 blueprint: Mapping[str, Any] | None = None,
                 wording: WordingCandidate | None = None,
                 native: NativeEvidence | None = None) -> QualificationReport:
        if native is None and blueprint is not None:
            native = self.native_evidence(spec, blueprint)
        ctx = CheckContext(registry=self.registry, repo_root=self.repo_root,
                           blueprint=blueprint, wording=wording, native=native,
                           seen_goal_shas=self.seen_goal_shas)
        results = [check(spec, ctx) for check in ALL_CHECKS]
        return QualificationReport(results=results)

    def validate_batch(
        self,
        items: Iterable[tuple[TaskSpecV5, Mapping[str, Any] | None, WordingCandidate | None]],
    ) -> list[QualificationReport]:
        reports = []
        for spec, blueprint, wording in items:
            report = self.validate(spec, blueprint=blueprint, wording=wording)
            if report.decision != "REJECTED":
                base_id = str((blueprint or {}).get("blueprint_id") or spec.task_id)
                self.seen_goal_shas.setdefault(
                    goal_sha(checks_public.spec_goal_payload(spec, blueprint)), base_id)
            reports.append(report)
        return reports
