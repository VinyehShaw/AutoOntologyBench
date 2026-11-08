"""BioModels SBML numeric domain adapter (W4, Review 5B).

Track: TaskTrack.numeric, profile ``sbml-executable/1``. IN-SILICO ONLY:
simulation results are never biological/wet-lab ground truth (Shared Contract §C).

Native truth: libSBML parses the pinned model; RoadRunner (CVODE) integrates it with
pinned tolerances. ``verify_terminal`` re-runs the solver from the source file and
compares against the submitted answer — it never trusts a cached agent-side result.
"""
from __future__ import annotations

import hashlib
import json
import os
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

PROFILE = "sbml-executable/1"
DEFAULT_ABS_TOL = 1e-12
DEFAULT_REL_TOL = 1e-8


class NotRunnableYet(RuntimeError):
    """Raised when this adapter's native dependency is unavailable; message carries the blocker."""


def _require_native() -> tuple[Any, Any]:
    try:
        import libsbml  # type: ignore
    except ImportError as exc:
        raise NotRunnableYet(
            "BLOCKED: python-libsbml not installed (pip install python-libsbml==5.21.2)"
        ) from exc
    try:
        import roadrunner  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise NotRunnableYet(
            "BLOCKED: libroadrunner not importable (pip install libroadrunner==2.10.0; "
            "needs LD_LIBRARY_PATH=<env>/lib for libpython3.11.so.1.0)"
        ) from exc
    return libsbml, roadrunner


def _load_config(spec: TaskSpecV5) -> dict[str, Any]:
    cfg_path = Path(spec.initial_state_ref)
    if not cfg_path.exists():
        raise FileNotFoundError(f"domain config missing: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _parse_model(sbml_path: str) -> dict[str, Any]:
    libsbml, _ = _require_native()
    doc = libsbml.readSBML(sbml_path)
    if doc.getNumErrors() > 0:
        raise ValueError(f"SBML parse errors in {sbml_path}: {doc.getNumErrors()}")
    m = doc.getModel()
    params: dict[str, float] = {p.getId(): p.getValue() for p in m.getListOfParameters()}
    for r in m.getListOfReactions():
        kl = r.getKineticLaw()
        if kl is not None:
            for p in kl.getListOfParameters():
                params[f"{r.getId()}.{p.getId()}"] = p.getValue()
    return {
        "model_id": m.getId(),
        "species": {s.getId(): s.getInitialConcentration() for s in m.getListOfSpecies()},
        "reactions": [r.getId() for r in m.getListOfReactions()],
        "parameters": params,
        "sbml_sha256": hashlib.sha256(Path(sbml_path).read_bytes()).hexdigest(),
    }


def _simulate(sbml_path: str, t_end: float, points: int,
              abs_tol: float, rel_tol: float) -> dict[str, Any]:
    _, roadrunner = _require_native()
    rr = roadrunner.RoadRunner(Path(sbml_path).read_text())
    integ = rr.getIntegrator()
    integ.setSetting("absolute_tolerance", abs_tol)
    integ.setSetting("relative_tolerance", rel_tol)
    result = rr.simulate(0, t_end, points)
    colnames = list(result.colnames)
    final = {colnames[i]: float(result[-1, i]) for i in range(len(colnames))}
    import numpy as np

    arr = np.asarray(result, dtype=float)
    return {
        "colnames": colnames,
        "final": final,
        "result_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "t_end": t_end,
        "points": points,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "in_silico": True,
    }


class BiomodelsSbmlDomain:
    """NativeDomain for pinned SBML models. All actions operate on the typed state;

    simulation is the only state-changing action and is fully deterministic
    (CVODE + pinned tolerances; verified bit-identical across runs in M1 smoke).
    """

    profile = PROFILE

    def __init__(self) -> None:
        self._libsbml, self._roadrunner = _require_native()

    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        cfg = _load_config(spec)
        model = _parse_model(cfg["sbml_path"])
        payload = {
            "config": {
                "sbml_path": cfg["sbml_path"],
                "t_end": float(cfg.get("t_end", 500)),
                "points": int(cfg.get("points", 501)),
                "abs_tol": float(cfg.get("abs_tol", DEFAULT_ABS_TOL)),
                "rel_tol": float(cfg.get("rel_tol", DEFAULT_REL_TOL)),
            },
            "model": model,
            "simulation": None,
            "seed": seed,
        }
        return NativeState(payload=payload)

    def observe(self, state: NativeState, rights: PermissionSet) -> PublicObservation:
        model = state.payload["model"]
        return PublicObservation(payload={
            "model_id": model["model_id"],
            "species_ids": sorted(model["species"]),
            "reaction_ids": model["reactions"],
            "parameter_ids": sorted(model["parameters"]),
            "track": "numeric",
            "label": "IN_SILICO",
        })

    def legal_action(self, state: NativeState, action: TypedAction) -> NativeDecision:
        atype = action.get("type")
        if atype == "simulate":
            return NativeDecision(legal=True, detail={"native": "roadrunner/cvode"})
        if atype == "submit_answer":
            ok = isinstance(action.get("values"), dict) and bool(action["values"])
            return NativeDecision(legal=ok, detail={"requires": "values: {colname: float}"})
        return NativeDecision(legal=False, detail={"reason": f"unknown action type {atype!r}"})

    def step(self, state: NativeState, action: TypedAction) -> NativeStateAndFeedback:
        decision = self.legal_action(state, action)
        if not decision.legal:
            return NativeStateAndFeedback(state=state, feedback={"error": decision.detail}, cost=0.0)
        if action["type"] == "simulate":
            cfg = state.payload["config"]
            sim = _simulate(cfg["sbml_path"], cfg["t_end"], cfg["points"],
                            cfg["abs_tol"], cfg["rel_tol"])
            new_payload = {**state.payload, "simulation": sim}
            return NativeStateAndFeedback(
                state=NativeState(payload=new_payload),
                feedback={"simulated": True, "result_sha256": sim["result_sha256"]},
                cost=1.0,
            )
        return NativeStateAndFeedback(state=state, feedback={"submitted": True}, cost=0.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        if not isinstance(submitted_answer, dict) or "values" not in submitted_answer:
            return GoalResult(goal_reached=False, evidence={"reason": "no values submitted"})
        cfg = state.payload["config"]
        truth = _simulate(cfg["sbml_path"], cfg["t_end"], cfg["points"],
                          cfg["abs_tol"], cfg["rel_tol"])
        rtol = float(os.environ.get("AOB_SBML_ANSWER_RTOL", "1e-4"))
        mismatches: dict[str, dict[str, float]] = {}
        for col, claimed in submitted_answer["values"].items():
            actual = truth["final"].get(col)
            if actual is None:
                mismatches[col] = {"claimed": float(claimed), "actual": float("nan")}
                continue
            denom = max(abs(actual), 1e-30)
            err = abs(float(claimed) - actual) / denom
            if err > rtol:
                mismatches[col] = {"claimed": float(claimed), "actual": actual}
        return GoalResult(
            goal_reached=not mismatches,
            evidence={
                "truth_result_sha256": truth["result_sha256"],
                "answer_rtol": rtol,
                "mismatches": mismatches,
                "in_silico": True,
            },
        )
