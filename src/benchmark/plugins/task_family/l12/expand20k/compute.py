"""Shared native-answer computation for the L12 20K expansion.

Single source of truth used by BOTH the generators (reference answers) and
the verifier (independent recompute), so exact-artifact equality can never
drift between generation and verification. Operations ``dfg`` / ``variants``
/ ``endpoints`` / ``nonconformant_trace_set`` delegate to the legacy
``propose._compute_answer`` (registry-independent, unchanged); the new
operations ``footprints`` and ``footprints_conf`` are implemented here.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT
from benchmark.plugins.task_family.l12.propose import _compute_answer as _legacy_compute

NEW_EXACT_OPS = ("footprints", "footprints_conf")

_FP_MODEL_CACHE: dict[str, Any] = {}
_FP_CONF_CACHE: dict[tuple[str, str, int], list[list[str]]] = {}


def model_footprints(net_rel: str) -> dict[str, Any]:
    """Footprints of a PNML net (reachability-graph variant), cached per net."""
    if net_rel not in _FP_MODEL_CACHE:
        RT._load()
        fp_alg = RT._mods["fp_alg"]
        net, im, fm = RT.net(net_rel)
        _FP_MODEL_CACHE[net_rel] = fp_alg.apply(net, im, fm)
    return _FP_MODEL_CACHE[net_rel]


def fp_conf_violations(traces: list[list[str]], net_rel: str) -> list[list[str]]:
    """Per-trace footprints-conformance violations (sorted string lists).

    Uses the pm4py LOG_MODEL conformance variant over trace-by-trace log
    footprints vs the net's model footprints: one violation set per trace
    (empty == conformant). Violations are stringified exactly as pm4py
    renders them; the answer contract documents this normalization.
    """
    RT._load()
    fp_alg = RT._mods["fp_alg"]
    fp_conf = RT._mods["fp_conf"]
    mfp = model_footprints(net_rel)
    tfp = fp_alg.apply(RT.event_log(traces), variant=fp_alg.Variants.TRACE_BY_TRACE)
    res = fp_conf.apply(tfp, mfp)
    return [sorted(str(v) for v in s) for s in res]


def fp_conf_violations_cached(log_rel: str, net_rel: str, n: int) -> list[list[str]]:
    key = (log_rel, net_rel, n)
    if key not in _FP_CONF_CACHE:
        traces = RT.traces(log_rel)
        proj = engine.project_first_n(traces, n)
        _FP_CONF_CACHE[key] = fp_conf_violations(proj, net_rel)
    return _FP_CONF_CACHE[key]


def _normalize_footprints(fp: dict[str, Any]) -> dict[str, Any]:
    """Deterministic JSON-safe footprints: sets of pairs keep pair direction
    (sequence semantics), sets of strings sort as strings, the dfg Counter
    becomes a sorted [a, b, count] list. (The legacy engine.footprints_log
    helper splits string elements into characters and leaks tuple keys —
    never fired because no legacy family used footprints.)"""
    out: dict[str, Any] = {}
    for k, v in fp.items():
        if isinstance(v, (set, frozenset)):
            elems = list(v)
            if elems and all(isinstance(e, (tuple, list)) for e in elems):
                out[k] = sorted([str(e[0]), str(e[1])] for e in elems)
            else:
                out[k] = sorted(str(e) for e in elems)
        elif isinstance(v, dict):
            items = list(v.items())
            if items and all(isinstance(kk, tuple) and len(kk) == 2 for kk, _ in items):
                out[k] = sorted([[str(kk[0]), str(kk[1])], int(vv)]
                                for kk, vv in items)
            else:
                out[k] = sorted([[str(kk), vv] for kk, vv in items])
        else:
            out[k] = v
    return out


def footprints_answer(traces: list[list[str]]) -> dict[str, Any]:
    RT._load()
    fp_alg = RT._mods["fp_alg"]
    fp = fp_alg.apply(RT.event_log(traces), variant=fp_alg.Variants.ENTIRE_EVENT_LOG)
    return _normalize_footprints(fp)


def compute_answer(
    operation: str, log_rel: str, n: int, net_rel: str | None
) -> dict[str, Any]:
    """Exact-artifact answer for a first-N projection (single source of truth)."""
    if operation == "footprints":
        traces = RT.traces(log_rel)
        proj = engine.project_first_n(traces, n)
        ans = footprints_answer(proj)
        ans["traces_total"] = len(proj)
        return ans
    if operation == "footprints_conf":
        assert net_rel is not None
        viol = fp_conf_violations_cached(log_rel, net_rel, n)
        return {
            "violating_trace_indices": [i for i, v in enumerate(viol) if v],
            "violations_by_trace": {str(i): v for i, v in enumerate(viol) if v},
            "traces_total": len(viol),
        }
    return _legacy_compute(operation, log_rel, n, net_rel)


def nontrivial(operation: str, ans: dict[str, Any]) -> bool:
    """Nontriviality floor per operation (legacy floors + new ops)."""
    if operation == "footprints":
        return len(ans.get("activities", [])) >= 2 and len(ans.get("sequence", [])) >= 1
    if operation == "footprints_conf":
        return 1 <= len(ans.get("violating_trace_indices", [])) < ans.get("traces_total", 0)
    if operation == "dfg":
        return len(ans["edges"]) >= 3
    if operation == "variants":
        return len(ans["variants"]) >= 2
    if operation == "endpoints":
        return len(ans["activity_counts"]) >= 2
    if operation == "nonconformant_trace_set":
        return 1 <= len(ans["nonconformant_trace_indices"]) < ans["traces_total"]
    return False


def answer_key(ans: dict[str, Any]) -> str:
    """Semantic identity of an exact answer (ignores the projection size)."""
    return engine.sha256_json({k: v for k, v in ans.items() if k != "traces_total"})
