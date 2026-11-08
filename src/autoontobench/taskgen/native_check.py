"""Native re-verification evidence feeding T5 checks 2/4/5/6 (W8 M3).

Replays the blueprint's private solution plan through the W2 ``GatekeeperDomain``
(every step must be accepted by pinned-OPA ``legal_action``) and independently
re-runs ``verify_terminal`` on the reset state and on the terminal state. Runs
ONCE per blueprint: paraphrase wordings share the underlying certified task, so
``TaskQualificationValidator`` caches evidence by ``blueprint_id``.

The domain is evaluator-side ground truth and never imports OntologyTool
(protocols.py HARD RULE). Oracle/step logs route to the caller-supplied
``log_root`` (W8-owned tree); sibling workstream log trees are untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import TaskSpecV5

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NativeEvidence:
    """Replay/reset probe results for one blueprint. ``available=False`` with a
    written ``reason`` whenever the evidence cannot honestly be produced."""

    available: bool
    reason: str = ""
    plan_len: int | None = None
    goal_reached_after_plan: bool | None = None
    goal_reached_at_reset: bool | None = None
    terminal_state_sha256: str | None = None
    evidence_sha256: str | None = None
    stored_terminal_sha256: str | None = None
    matches_stored_evidence: bool | None = None
    min_action_count: int | None = None
    nontrivial: bool | None = None
    interacting_units: int | None = None
    log_root: str | None = None


def _resolve(ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else REPO_ROOT / p


def load_private_plan(blueprint: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    ref = blueprint.get("private_solution_plan_path")
    if not ref:
        return None
    path = _resolve(str(ref))
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = payload.get("plan")
    return plan if isinstance(plan, list) else None


SUPPORTED_DOMAINS = frozenset({"gatekeeper", "rhea", "chebi"})

ANALYTICAL_SOURCE_SETS = (frozenset({"rhea", "chebi"}),)


def _cached_rhea_index(index_dir: str) -> Any:
    if index_dir not in _RHEA_INDEX_CACHE:
        from autoontobench.core.domains.rhea_chebi import load_certified_index

        _RHEA_INDEX_CACHE[index_dir] = load_certified_index(Path(index_dir))
    return _RHEA_INDEX_CACHE[index_dir]


_RHEA_INDEX_CACHE: dict[str, Any] = {}


def domain_for(source_id: str, log_root: Path, *, spec: Any | None = None) -> Any | None:
    """NativeDomain dispatch. Only sources with a landed W2/W3/W4 domain resolve;
    anything else returns None so checks degrade to NOT_IMPLEMENTED_DEP."""
    if source_id == "gatekeeper":
        from autoontobench.core.domains.gatekeeper import GatekeeperDomain

        return GatekeeperDomain(log_root=Path(log_root))
    if source_id in ("rhea", "chebi"):
        from autoontobench.core.domains.rhea_chebi import RheaChebiDomain

        index_dir = (
            getattr(spec, "private_native_truth_ref", None)
            or "data/erule/review5/scientific/rhea_index"
        )
        return RheaChebiDomain(_cached_rhea_index(str(_resolve(str(index_dir)))))
    return None


def _gather_analytical_evidence(
    spec: TaskSpecV5, blueprint: Mapping[str, Any], plan: list[dict[str, Any]],
    stored_terminal: str | None, meta: dict[str, Any],
) -> NativeEvidence:
    """linked-data/1 (Rhea+ChEBI): the plan is a single submit_answer carrying the
    dual-path-certified expected answer; verify_terminal must discriminate reset
    (no answer -> not reached) from terminal (exact set equality -> reached)."""
    domain = domain_for("rhea", Path(str(meta["log_root"])), spec=spec)
    state = domain.reset(spec, seed=spec.seed)
    reset_result = domain.verify_terminal(spec, state, None)
    for i, action in enumerate(plan):
        decision = domain.legal_action(state, action)
        if decision.legal is not True:
            return NativeEvidence(
                False, f"plan step {i} rejected by native legality: {decision.detail}",
                plan_len=len(plan), goal_reached_at_reset=bool(reset_result.goal_reached),
                stored_terminal_sha256=stored_terminal, **meta)
        state = domain.step(state, action).state
    terminal = domain.verify_terminal(spec, state, None)
    terminal_sha = hashlib.sha256(
        json.dumps(state.payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    evidence_sha = hashlib.sha256(
        json.dumps(terminal.evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return NativeEvidence(
        True, "answer replayed through RheaChebiDomain over the dual-certified index",
        plan_len=len(plan),
        goal_reached_after_plan=bool(terminal.goal_reached),
        goal_reached_at_reset=bool(reset_result.goal_reached),
        terminal_state_sha256=terminal_sha,
        evidence_sha256=evidence_sha,
        stored_terminal_sha256=stored_terminal,
        matches_stored_evidence=(
            None if stored_terminal is None else terminal_sha == stored_terminal),
        **meta,
    )


def gather_native_evidence(
    spec: TaskSpecV5, blueprint: Mapping[str, Any] | None, *, log_root: Path
) -> NativeEvidence:
    meta = {
        "min_action_count": blueprint.get("min_action_count") if blueprint else None,
        "nontrivial": (blueprint.get("qc") or {}).get("nontrivial") if blueprint else None,
        "interacting_units": blueprint.get("interacting_units") if blueprint else None,
        "log_root": str(log_root),
    }
    if blueprint is None:
        return NativeEvidence(False, "no W2 blueprint record supplied", **meta)
    source_ids = sorted({r.source_id for r in spec.source_refs})
    plan = load_private_plan(blueprint)
    if plan is None:
        return NativeEvidence(False, "private solution plan missing/unreadable", **meta)
    stored_terminal = (blueprint.get("replay") or {}).get("terminal_state_sha256")
    if frozenset(source_ids) in ANALYTICAL_SOURCE_SETS:
        return _gather_analytical_evidence(spec, blueprint, plan, stored_terminal, meta)
    if source_ids != ["gatekeeper"]:
        return NativeEvidence(
            False, f"no NativeDomain dispatch for source_ids={source_ids}", **meta
        )
    domain = domain_for("gatekeeper", Path(log_root))
    state = domain.reset(spec, seed=spec.seed)
    reset_result = domain.verify_terminal(spec, state, None)
    for i, action in enumerate(plan):
        decision = domain.legal_action(state, action)
        if not decision.legal:
            return NativeEvidence(
                False,
                f"plan step {i} rejected by native legality: {decision.detail}",
                plan_len=len(plan),
                goal_reached_at_reset=bool(reset_result.goal_reached),
                stored_terminal_sha256=stored_terminal,
                **meta,
            )
        state = domain.step(state, action).state
    terminal = domain.verify_terminal(spec, state, None)
    terminal_sha = terminal.evidence.get("terminal_state_sha256")
    return NativeEvidence(
        True,
        "plan replayed through GatekeeperDomain; every step pinned-OPA legal",
        plan_len=len(plan),
        goal_reached_after_plan=bool(terminal.goal_reached),
        goal_reached_at_reset=bool(reset_result.goal_reached),
        terminal_state_sha256=terminal_sha,
        evidence_sha256=terminal.evidence.get("evidence_sha256"),
        stored_terminal_sha256=stored_terminal,
        matches_stored_evidence=(
            None if stored_terminal is None else terminal_sha == stored_terminal
        ),
        **meta,
    )
