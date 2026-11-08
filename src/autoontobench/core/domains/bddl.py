"""BEHAVIOR-1K BDDL symbolic-goal domain adapter (W4, Review 5B).

Track: stateful_action over a SYMBOLIC literal state, profile ``bddl-goal/1``.
Label: SYMBOLIC_BDDL only — never EMBODIED. Physics/vision/TSR tracks require the
OmniGibson simulator and are BLOCKED (see BLOCKED_EMBODIED_REASON).

Native truth: the ``bddl`` package (pip 3.6.0) parses the pinned problem file and
evaluates its :goal formula through its own TrivialBackend over a symbolic predicate
state. ``verify_terminal`` rebuilds that evaluator from the state's literal list, so
the check is independent of any agent-side bookkeeping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.protocols import (
    NativeDecision,
    NativeState,
    NativeStateAndFeedback,
    PermissionSet,
    PublicObservation,
    TypedAction,
)

PROFILE = "bddl-goal/1"
LABEL = "SYMBOLIC_BDDL"

BLOCKED_EMBODIED_REASON = (
    "EMBODIED track BLOCKED: requires OmniGibson simulator (multi-GB assets: BEHAVIOR-1K "
    "scene/IG objects dataset + NVIDIA Isaac Sim / omni.isaac runtime), a GPU-capable EGL "
    "rendering context, and the BEHAVIOR challenge action-primitive stack. Not installable "
    "in this CPU container within the M1 timebox; symbolic BDDL goal-logic track only."
)


class NotRunnableYet(RuntimeError):
    """Raised when this adapter's native dependency is unavailable; message carries the blocker."""


def _require_bddl() -> Any:
    try:
        import bddl  # type: ignore
        from bddl import activity  # type: ignore
        from bddl.trivial_backend import (  # type: ignore
            TrivialBackend,
            TrivialGenericObject,
            TrivialSimulator,
        )
    except ImportError as exc:
        raise NotRunnableYet("BLOCKED: bddl not installed (pip install bddl==3.6.0)") from exc
    return bddl, activity, TrivialBackend, TrivialGenericObject, TrivialSimulator


def _load_problem(content: str) -> Any:
    """Parse pinned BDDL content. bddl 3.6.0 ships domains named ``igibson``/``omnigibson``;

    the pinned problem0.bddl declares ``(:domain behavior-1k)``. All 8 predicates used by
    problem0 exist verbatim in the omnigibson domain (verified 2026-10-25), so only the
    domain NAME is aliased in memory; the on-disk artifact and predicate set are untouched.
    """
    bddl, activity, _tb, _tgo, _ts = _require_bddl()
    aliased = content.replace("(:domain behavior-1k)", "(:domain omnigibson)")
    return activity.Conditions(
        "behavior_activity_w4", 0, "omnigibson", predefined_problem=aliased
    )


class BddlSymbolicGoalDomain:
    """NativeDomain for BDDL goal-predicate checking over a symbolic literal state.

    Actions: ``assert_literal`` / ``retract_literal`` mutate only the symbolic state
    (a synthetic transition model, labelled as such); goal satisfaction is evaluated
    natively by bddl. No physics is modelled or claimed.
    """

    profile = PROFILE
    label = LABEL

    def __init__(self) -> None:
        _require_bddl()

    def _conds(self, spec: TaskSpecV5) -> Any:
        cfg_path = Path(spec.initial_state_ref)
        if not cfg_path.exists():
            raise FileNotFoundError(f"domain config missing: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        content = Path(cfg["bddl_path"]).read_text()
        return _load_problem(content)

    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        _bddl, activity, _tb, _tgo, _ts = _require_bddl()
        conds = self._conds(spec)
        literals = [list(lit) for lit in conds.parsed_initial_conditions]
        scope = activity.get_object_scope(conds)
        return NativeState(payload={
            "literals": literals,
            "objects": sorted(scope),
            "goal": str(conds.parsed_goal_conditions),
            "seed": seed,
            "label": LABEL,
        })

    def observe(self, state: NativeState, rights: PermissionSet) -> PublicObservation:
        return PublicObservation(payload={
            "objects": state.payload["objects"],
            "goal": state.payload["goal"],
            "n_literals": len(state.payload["literals"]),
            "label": LABEL,
        })

    def legal_action(self, state: NativeState, action: TypedAction) -> NativeDecision:
        atype = action.get("type")
        if atype not in {"assert_literal", "retract_literal"}:
            return NativeDecision(legal=False, detail={"reason": f"unknown action {atype!r}"})
        lit = action.get("literal")
        if not isinstance(lit, list) or not lit or not all(isinstance(x, str) for x in lit):
            return NativeDecision(legal=False, detail={"reason": "literal must be list[str]"})
        known = set(state.payload["objects"])
        for obj in lit[1:]:
            if obj not in known:
                return NativeDecision(legal=None, detail={"reason": f"unknown object {obj}"})
        return NativeDecision(legal=True, detail={"label": LABEL})

    def step(self, state: NativeState, action: TypedAction) -> NativeStateAndFeedback:
        decision = self.legal_action(state, action)
        if decision.legal is not True:
            return NativeStateAndFeedback(state=state, feedback={"error": decision.detail}, cost=0.0)
        literals = [list(lit) for lit in state.payload["literals"]]
        target = list(action["literal"])
        negated = ["not", target]
        if action["type"] == "assert_literal":
            literals = [l for l in literals if l != negated]
            if target not in literals:
                literals.append(target)
        else:
            literals = [l for l in literals if l != target]
            if negated not in literals:
                literals.append(negated)
        new_payload = {**state.payload, "literals": literals}
        return NativeStateAndFeedback(
            state=NativeState(payload=new_payload),
            feedback={"applied": action["type"], "n_literals": len(literals)},
            cost=1.0,
        )

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        _bddl, activity, TrivialBackend, TrivialGenericObject, TrivialSimulator = _require_bddl()
        conds = self._conds(spec)
        scope = activity.get_object_scope(conds)
        sim = TrivialSimulator()
        for name in scope:
            scope[name] = TrivialGenericObject(name, sim)
        sim.set_state(state.payload["literals"])
        goals = activity.get_goal_conditions(
            conds, TrivialBackend(), scope, generate_ground_options=False
        )
        satisfied, detail = activity.evaluate_goal_conditions(goals)
        state_sha = hashlib.sha256(
            json.dumps(state.payload["literals"], sort_keys=True).encode()
        ).hexdigest()
        return GoalResult(
            goal_reached=bool(satisfied),
            evidence={
                "state_sha256": state_sha,
                "goal_detail": str(detail)[:500],
                "label": LABEL,
            },
        )


class BddlEmbodiedDomain:
    """Placeholder for the EMBODIED track. Always raises with the recorded blocker."""

    profile = "embodied-action/1"

    def __init__(self) -> None:
        raise NotRunnableYet(BLOCKED_EMBODIED_REASON)
