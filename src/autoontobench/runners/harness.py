"""Researcher harness for Review-5 (owner: W6). Implements the frozen
``ResearcherHarness`` protocol (``run(PublicTask, model, arm, budget)``).

One harness instance binds: a ``NativeDomain`` (independent native truth), a
``TaskSpecV5`` (server-side; private refs never leave it), an ontology
consumer, and the arm configs. ``run`` drives the model loop:

    reset -> [model sees observation -> model emits tool call -> session
    executes + logs] until submit_final or budget exhaustion -> seal the run.

Deterministic replay mode (Review-5B §6 E2): pass a ``MockModel`` scripted
with the frozen ``action_sequence`` over a fresh ``reset(spec, seed)``; the
E2 O-swap hook lives in the session (``forced_artifact_bytes``), so replay
with gold/wrong/repaired O isolates ``decision_changes_due_to_O``.

The harness never scores. Scoring (native verifier + unsafe guard) lives in
``autoontobench.scoring.task_success`` and consumes the sealed run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import TaskSpecV5
from autoontobench.core.ontology_consumer import OntologyConsumer
from autoontobench.core.policy_hooks import (
    DEFAULT_FILE_POLICY,
    ActionAllowlist,
    FileEditPolicy,
)
from autoontobench.core.protocols import (
    Budget,
    NativeDomain,
    PublicTask,
    SealedRun,
)
from autoontobench.runners.arm_config import ArmConfig
from autoontobench.runners.tool_api import ToolSession
from autoontobench.runners.tool_log import canonical_sha256


class ResearcherHarnessImpl:
    """Frozen-protocol harness. ``model`` is any duck-typed AgentModel
    (``next_action(observation) -> tool call``); M1 runs only MockModel."""

    def __init__(
        self,
        *,
        domain: NativeDomain,
        spec: TaskSpecV5,
        arms: Mapping[str, ArmConfig],
        ontology_tool: OntologyConsumer | None = None,
        artifact_sources: Mapping[str, Mapping[str, str]] | None = None,
        action_allowlist: ActionAllowlist | None = None,
        file_policy: FileEditPolicy = DEFAULT_FILE_POLICY,
        rights: frozenset[str] | None = None,
        public_root: Path | None = None,
        run_log_dir: Path | None = None,
        seed: int = 0,
    ):
        self._domain = domain
        self._spec = spec
        self._arms = dict(arms)
        self._ontology_tool = ontology_tool
        self._artifact_sources = dict(artifact_sources or {})
        self._action_allowlist = action_allowlist
        self._file_policy = file_policy
        self._rights = set(rights or set())
        self._public_root = public_root
        self._run_log_dir = run_log_dir
        self._seed = seed

    def run(
        self, task_public_spec: PublicTask, model: Any, arm: str, budget: Budget
    ) -> SealedRun:
        if task_public_spec.task_id != self._spec.task_id:
            raise ValueError(
                "public task %r does not match harness spec %r"
                % (task_public_spec.task_id, self._spec.task_id)
            )
        config = self._arms[arm]
        run_id = "dev--%s--%s--seed%d" % (self._spec.task_id, arm, self._seed)
        initial = self._domain.reset(self._spec, self._seed)
        initial_sha = canonical_sha256(initial.payload)
        log_path = (
            self._run_log_dir / run_id / "tool_log.jsonl" if self._run_log_dir else None
        )
        initial_files = (
            self._artifact_sources.get(config.initial_artifact)
            if config.initial_artifact is not None
            else None
        )
        allowlist = self._action_allowlist or _allowlist_from_domain(self._domain)
        session = ToolSession(
            domain=self._domain,
            spec=self._spec,
            state=initial,
            rights=self._rights,
            budget=budget,
            action_allowlist=allowlist,
            file_policy=self._file_policy,
            ontology=self._ontology_tool,
            # profile id verbatim; non-ontology reps are honest reject points (see consumer docstring)
            ontology_profile=config.representation,
            tool_allowlist=config.tool_allowlist,
            initial_files=initial_files,
            artifact_editable=config.artifact_editable,
            public_root=self._public_root,
            run_id=run_id,
            log_path=log_path,
        )
        started = time.monotonic()
        hard_cap = budget.total_tool_calls + 4  # session enforces; this is the loop fuse
        while not session.submitted and session.meter.tool_calls <= hard_cap:
            observation = session.model_observation()
            call = model.next_action(observation)
            session.execute(call)
        wall_seconds = time.monotonic() - started
        fail_reason = None
        if not session.submitted:
            fail_reason = "BUDGET_EXCEEDED" if session.meter.tool_calls >= budget.total_tool_calls else "RUNNER_TIMEOUT"
        terminal_sha = canonical_sha256(session.state.payload)
        header = {
            "row_type": "run_header",
            "run_id": run_id,
            "task_id": self._spec.task_id,
            "arm": arm,
            "seed": self._seed,
            "model_exact_id": getattr(model, "model_exact_id", None),
            "provider": getattr(model, "provider", "unknown"),
            "initial_state_sha": initial_sha,
            "action_allowlist": sorted(allowlist.kinds),
            "unsafe_action_count": session.unsafe_action_count,
            "tool_calls": session.meter.tool_calls,
            "oracle_probes": session.meter.oracle_probes,
            "actions": session.meter.actions,
            "sealed_ontology_sha": session.sealed_ontology_sha,
            "forced_artifact": session.has_forced_artifact,
            "wall_seconds": round(wall_seconds, 3),
            "fail_reason": fail_reason,
            "arm_parity_note": config.parity_note,
        }
        rows = [header] + [row.to_dict() for row in session.rows]
        status = "SEALED" if session.submitted else "FAILED_TO_SUBMIT"
        return SealedRun(
            run_id=run_id,
            task_id=self._spec.task_id,
            arm=arm,
            rows=rows,
            terminal_state_sha=terminal_sha,
            final_answer=session.final_answer,
            status=status,
        )


def _allowlist_from_domain(domain: Any) -> ActionAllowlist:
    kinds = getattr(domain, "ACTION_KINDS", None)
    if not kinds:
        raise ValueError(
            "no action allowlist supplied and the domain exposes no ACTION_KINDS"
        )
    return ActionAllowlist(kinds=frozenset(kinds))
