"""L07 deterministic SBML simulation wrapper (libroadrunner CVODE).

Lane protocol (pinned, recorded in every evidence payload): CVODE with
abs_tol=1e-12, rel_tol=1e-8, t_end=100, points=201. ``simulate`` is pure with
respect to (sbml_bytes, protocol); ``simulate_checked`` runs the integrator
twice and requires bit-identical result bytes — that double-run is the lane's
determinism check and both hashes land in evidence.

libroadrunner needs ``libpython3.11.so.1.0`` on the loader path; the conda
base env provides it, so we preload it explicitly (RTLD_GLOBAL) instead of
requiring LD_LIBRARY_PATH at process start.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

T_END = 100.0
POINTS = 201
ABS_TOL = 1e-12
REL_TOL = 1e-8
PROTOCOL = {"integrator": "cvode", "abs_tol": ABS_TOL, "rel_tol": REL_TOL,
            "t_end": T_END, "points": POINTS}

_ROADRUNNER: Any = None


class SimulationError(RuntimeError):
    pass


def load_roadrunner() -> Any:
    global _ROADRUNNER
    if _ROADRUNNER is None:
        libpython = Path("/opt/conda/lib/libpython3.11.so.1.0")
        if libpython.is_file():
            ctypes.CDLL(str(libpython), mode=ctypes.RTLD_GLOBAL)
        try:
            import roadrunner
            roadrunner.Logger.disableConsoleLogging()
        except ImportError as exc:
            raise SimulationError(
                "libroadrunner not importable (libpython3.11.so.1.0 missing)"
            ) from exc
        _ROADRUNNER = roadrunner
    return _ROADRUNNER


def _col_of(cols: list[str], sid: str) -> str | None:
    for cand in (f"[{sid}]", sid):
        if cand in cols:
            return cand
    return None


def trajectory_features(values: np.ndarray) -> dict[str, float]:
    """Deterministic per-column features used by query/organization families."""
    first = float(values[0])
    last = float(values[-1])
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    imax = int(np.argmax(values))
    peak_interior = bool(0 < imax < len(values) - 1)
    ref = max(abs(first), abs(last), 1e-30)
    peak_prominence = (vmax - max(first, last)) / ref if peak_interior else 0.0
    diffs = np.diff(values)
    tol = 1e-9 * max(abs(vmax), abs(vmin), 1e-30)
    monotone_up = bool(np.all(diffs >= -tol))
    monotone_down = bool(np.all(diffs <= tol))
    return {
        "first": first, "last": last, "min": vmin, "max": vmax, "span": span,
        "argmax_index": imax, "peak_interior": peak_interior,
        "peak_prominence_rel": peak_prominence,
        "monotone_up": monotone_up, "monotone_down": monotone_down,
        "variation_rel": span / ref,
    }


def classify_shape(feat: dict[str, float]) -> str:
    """Lane's deterministic trajectory-shape classes (used by family f03)."""
    if feat["variation_rel"] < 1e-6:
        return "flat"
    if feat["peak_interior"] and feat["peak_prominence_rel"] > 0.05:
        return "peaked"
    if feat["monotone_up"]:
        return "increasing"
    if feat["monotone_down"]:
        return "decreasing"
    return "other"


def simulate(sbml: str, *, t_end: float = T_END, points: int = POINTS,
             abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> dict[str, Any]:
    roadrunner = load_roadrunner()
    try:
        rr = roadrunner.RoadRunner(sbml)
    except Exception as exc:
        raise SimulationError(f"roadrunner load failed: {exc.__class__.__name__}") from exc
    integ = rr.getIntegrator()
    integ.setSetting("absolute_tolerance", abs_tol)
    integ.setSetting("relative_tolerance", rel_tol)
    try:
        result = rr.simulate(0, t_end, points)
    except Exception as exc:
        raise SimulationError(f"cvode failed: {exc.__class__.__name__}") from exc
    arr = np.asarray(result, dtype=float)
    cols = list(result.colnames)
    if not np.all(np.isfinite(arr)):
        raise SimulationError("non-finite trajectory")
    return {"colnames": cols, "array": arr}


def simulate_checked(sbml: str, species_ids: list[str], **kw: Any) -> dict[str, Any]:
    """Double-run + bit-identical check; returns finals/features per species."""
    a = simulate(sbml, **kw)
    b = simulate(sbml, **kw)
    sha_a = hashlib.sha256(a["array"].tobytes()).hexdigest()
    sha_b = hashlib.sha256(b["array"].tobytes()).hexdigest()
    if sha_a != sha_b:
        raise SimulationError(f"non-deterministic trajectory: {sha_a[:12]} != {sha_b[:12]}")
    cols: list[str] = a["colnames"]
    arr: np.ndarray = a["array"]
    per_species: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for sid in species_ids:
        col = _col_of(cols, sid)
        if col is None:
            missing.append(sid)
            continue
        values = arr[:, cols.index(col)]
        feat = trajectory_features(values)
        feat["shape"] = classify_shape(feat)
        per_species[sid] = feat
    return {
        "result_sha256": sha_a,
        "colnames": cols,
        "species": per_species,
        "missing_species": missing,
        "finals": {sid: per_species[sid]["last"] for sid in per_species},
    }
