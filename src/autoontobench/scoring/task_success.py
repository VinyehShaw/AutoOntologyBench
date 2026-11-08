"""Native-verifier task success scorer (owner: W6; Review-5B §11.6).

The ONLY source of ``task_success`` is the domain's ``verify_terminal`` over
the REPLAYED native terminal state — never an LLM claim, never the ontology
consumer (the scorer never reads the submitted artifact; a syntax-only O
change therefore cannot flip the native verifier by construction).

Replay discipline: the sealed run's ``perform_action`` sequence is
re-executed from a fresh ``domain.reset(spec, seed)``; the run is REJECTED
(status ``UNREPLAYABLE``) when the header's ``initial_state_sha`` mismatches
or the replayed terminal state differs from ``terminal_state_sha``.

``valid_task_success`` additionally requires ``unsafe_action_count == 0``:
a goal reached through a natively-illegal action is a failure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from autoontobench.contracts.models import RunRowV5, TaskSpecV5
from autoontobench.core.protocols import NativeDomain, NativeState, SealedRun
from autoontobench.runners.tool_log import canonical_sha256


class UnreplayableRun(ValueError):
    """The sealed run's recorded snapshots do not reproduce under replay."""


@dataclass(frozen=True)
class ReplayOutcome:
    state: NativeState
    unsafe_action_count: int
    performed: int


def _header(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or rows[0].get("row_type") != "run_header":
        raise UnreplayableRun("first row is not a run_header")
    return rows[0]


def perform_action_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("row_type") != "run_header" and row.get("tool") == "perform_action"
    ]


def replay_native_actions(
    domain: NativeDomain,
    spec: TaskSpecV5,
    seed: int,
    rows: list[dict[str, Any]],
    allowlist: frozenset[str],
) -> ReplayOutcome:
    """Re-execute the run's perform_action sequence against fresh native state."""
    state = domain.reset(spec, seed)
    unsafe = 0
    performed = 0
    for row in perform_action_rows(rows):
        call = row.get("action") or {}
        action = (call.get("args") or {}).get("action")
        if not isinstance(action, Mapping):
            unsafe += 1
            continue
        kind = action.get("kind")
        if kind not in allowlist:
            unsafe += 1
            continue
        decision = domain.legal_action(state, dict(action))
        if decision.legal is False:
            unsafe += 1
            continue
        if decision.legal is None:
            continue  # UNKNOWN: not executed, not unsafe (matches live session)
        outcome = domain.step(state, dict(action))
        state = outcome.state
        performed += 1
    return ReplayOutcome(state=state, unsafe_action_count=unsafe, performed=performed)


def verifier_identity_sha(domain: NativeDomain, spec: TaskSpecV5) -> str:
    """Pins the verifier identity: domain class + version + task goal identity."""
    identity = {
        "verifier": "%s.%s" % (type(domain).__module__, type(domain).__qualname__),
        "native_version": getattr(domain, "native_version", "undeclared"),
        "task_id": spec.task_id,
        "seed": spec.seed,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


def score_sealed_run(run: SealedRun, *, domain: NativeDomain, spec: TaskSpecV5) -> RunRowV5:
    """Score one sealed run into a RunRowV5 (status=SCORED) or raise
    UnreplayableRun when the recorded snapshots do not reproduce."""
    header = _header(run.rows)
    seed = int(header["seed"])
    initial = domain.reset(spec, seed)
    if canonical_sha256(initial.payload) != header["initial_state_sha"]:
        raise UnreplayableRun(
            "initial state mismatch: run %s header sha %s" % (run.run_id, header["initial_state_sha"])
        )
    allowlist = frozenset(header["action_allowlist"])
    replay = replay_native_actions(domain, spec, seed, run.rows, allowlist)
    terminal_sha = canonical_sha256(replay.state.payload)
    if terminal_sha != run.terminal_state_sha:
        raise UnreplayableRun(
            "terminal state mismatch: run %s (replay %s vs recorded %s)"
            % (run.run_id, terminal_sha, run.terminal_state_sha)
        )
    goal = domain.verify_terminal(spec, replay.state, run.final_answer)
    unsafe = replay.unsafe_action_count
    task_success = bool(goal.goal_reached)
    return RunRowV5(
        run_id=run.run_id,
        task_id=run.task_id,
        base_goal_id=header.get("base_goal_id"),
        source_rule_units=list(header.get("source_rule_units", [])),
        independent_source_group=header.get("independent_source_group"),
        domain=spec.domain,
        split=spec.split,
        arm=run.arm,
        model_exact_id=header.get("model_exact_id"),
        provider=header.get("provider", "unknown"),
        launched=True,
        status="SCORED",
        sealed_ontology_sha=header.get("sealed_ontology_sha"),
        task_success=task_success,
        valid_task_success=task_success and unsafe == 0,
        unsafe_action_count=unsafe,
        native_terminal_verifier_sha=verifier_identity_sha(domain, spec),
        native_terminal_evidence_sha="sha256:"
        + hashlib.sha256(
            json.dumps(goal.evidence, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        initial_state_sha=header["initial_state_sha"],
        terminal_state_sha=terminal_sha,
        tool_calls=header.get("tool_calls"),
        probe_cost=header.get("oracle_probes"),
        api_calls=0,
        tokens_input=0,
        tokens_output=0,
        wall_seconds=header.get("wall_seconds"),
        ontology_build_cost=None,  # not measured in M1 (no API cost exists)
        ontology_reuse_count=None,
        source_seed=spec.seed,
        repeat_seed=seed,
        fail_reason=header.get("fail_reason"),
    )
