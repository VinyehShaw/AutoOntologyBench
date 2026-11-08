"""Proposers/instantiators for the L07 20K-candidate mechanisms.

Each base goal is one (family, model) pair; five instances per base goal are
deterministic variations (a sampled universe, or a pinned initial-value ladder).
All truth values are recomputed from pinned SBML bytes by the native verifier;
the proposal-time simulations here only screen for a nontrivial, solver-stable
answer.
"""
from __future__ import annotations

from random import Random
from typing import Any

from benchmark.plugins.source.l07.sbml_edit import ParameterEdit
from benchmark.plugins.source.l07.simulate import SimulationError
from benchmark.plugins.task_family.l07.goals_sim import (
    CANON_MULTS, INSTANCE_LADDERS,
)
from benchmark.plugins.task_family.l07.support import SimRunner, rel_diff, sample_universe

RETENTION_MIN_GAP = 1.10
PULSE_MIN_GAP = 1.50
PEAK_UNIVERSE = 7
PEAK_K = 3
UNIVERSE_SIZE = 6
DEGREE_UNIVERSE = 8
INIT_EFFECT_RTOL = 0.05


def _features(sim: SimRunner, model_id: str, pool: list[str]) -> dict[str, dict[str, Any]]:
    return sim.sim(model_id, tuple(sorted(pool)))["species"]


def _initial_values(struct: Any) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for sid in struct.floating_species:
        info = struct.species_info(sid)
        if info.initial_amount is not None:
            value, field = info.initial_amount, "initial_amount"
        elif info.initial_concentration is not None:
            value, field = info.initial_concentration, "initial_concentration"
        else:
            continue
        if value and abs(value) > 0:
            out.append((sid, float(value), field))
    return out


def _proper_set(values: set[str], universe: list[str]) -> bool:
    return bool(values) and len(values) < len(universe)


# -- f01 retention_set ---------------------------------------------------------

def retention_of(feat: dict[str, Any]) -> float | None:
    vmax = feat["max"]
    if vmax <= 0:
        return None
    return feat["last"] / vmax


def propose_retention_set(sim: SimRunner, model_id: str,
                          rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    pool = struct.floating_species[:12]
    if len(pool) < UNIVERSE_SIZE:
        return None
    feats = _features(sim, model_id, pool)
    ratios = {s: r for s in pool if (r := retention_of(feats[s])) is not None}
    if len(ratios) < UNIVERSE_SIZE:
        return None
    positive = sorted({v for v in ratios.values() if 0 < v <= 1.0000001})
    gaps = [(a, b) for a, b in zip(positive, positive[1:])
            if a > 0 and b <= 1.0000001 and b / a >= RETENTION_MIN_GAP]
    if not gaps:
        return None
    a, b = max(gaps, key=lambda ab: ab[1] / ab[0])
    threshold = (a * b) ** 0.5
    answer = {s for s, r in ratios.items() if r >= threshold}
    if not _proper_set(answer, list(ratios)):
        return None
    if len(answer) < 2 or len(ratios) - len(answer) < 2:
        return None
    return {
        "goal_params": {"model_id": model_id, "retention_threshold": threshold},
        "ratios": ratios,
        "pool": sorted(ratios),
    }


def instantiate_retention_set(base: dict[str, Any], rng: Random,
                              size: int) -> dict[str, Any] | None:
    ratios = base["ratios"]
    threshold = base["goal_params"]["retention_threshold"]
    for _ in range(32):
        universe = sample_universe(rng, base["pool"], size)
        if universe is None:
            return None
        answer = {s for s in universe if ratios[s] >= threshold}
        if _proper_set(answer, universe):
            return {"universe_ids": universe}
    return None


# -- f02 peak_time_ranking -----------------------------------------------------

def _peak_key(feat: dict[str, Any], sid: str) -> tuple[int, float]:
    # Species id is the verifier's final tie-break, not part of the uniqueness
    # key: universes must be tie-free on (peak index, final value) so the
    # top-3 answer cannot depend on an arbitrary id order.
    return (int(feat["argmax_index"]), float(feat["last"]))


def propose_peak_time_ranking(sim: SimRunner, model_id: str,
                              rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    pool = struct.floating_species[:12]
    if len(pool) < PEAK_UNIVERSE:
        return None
    feats = _features(sim, model_id, pool)
    keys = {s: _peak_key(feats[s], s) for s in pool}
    if len(set(keys.values())) < PEAK_UNIVERSE:
        return None
    return {"goal_params": {"model_id": model_id, "k": PEAK_K},
            "keys": keys, "pool": sorted(pool)}


def instantiate_peak_time_ranking(base: dict[str, Any], rng: Random,
                                  size: int) -> dict[str, Any] | None:
    keys = base["keys"]
    pool = base["pool"]
    for _ in range(64):
        universe = sample_universe(rng, pool, size)
        if universe is None:
            return None
        if len({keys[s] for s in universe}) == len(universe):
            return {"universe_ids": universe}
    return None


# -- f03 pulse_set -------------------------------------------------------------

def propose_pulse_set(sim: SimRunner, model_id: str,
                      rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    pool = struct.floating_species[:12]
    if len(pool) < UNIVERSE_SIZE:
        return None
    feats = _features(sim, model_id, pool)
    maxima = {s: feats[s]["max"] for s in pool}
    positive = sorted({v for v in maxima.values() if v > 0})
    gaps = [(a, b) for a, b in zip(positive, positive[1:])
            if a > 0 and b / a >= PULSE_MIN_GAP]
    if not gaps:
        return None
    a, b = max(gaps, key=lambda ab: ab[1] / ab[0])
    level = (a * b) ** 0.5
    answer = {s for s in pool if maxima[s] >= level and feats[s]["last"] < level}
    if not _proper_set(answer, pool):
        return None
    if len(answer) < 2 or len(pool) - len(answer) < 2:
        return None
    return {
        "goal_params": {"model_id": model_id, "level": level},
        "maxima": maxima,
        "last": {s: feats[s]["last"] for s in pool},
        "pool": sorted(pool),
    }


def instantiate_pulse_set(base: dict[str, Any], rng: Random,
                          size: int) -> dict[str, Any] | None:
    level = base["goal_params"]["level"]
    maxima, last = base["maxima"], base["last"]
    for _ in range(32):
        universe = sample_universe(rng, base["pool"], size)
        if universe is None:
            return None
        answer = {s for s in universe if maxima[s] >= level and last[s] < level}
        if _proper_set(answer, universe):
            return {"universe_ids": universe}
    return None


# -- shared initial-condition probing -----------------------------------------

def _init_edit(gp: dict[str, Any], value: float) -> ParameterEdit:
    return ParameterEdit("", float(value), kind="species_initial",
                         species_id=gp["species"])


def _init_outcomes(sim: SimRunner, model_id: str, species: str,
                   target: str, v0: float,
                   multipliers: tuple[float, ...]) -> list[float] | None:
    finals: list[float] = []
    for mult in multipliers:
        try:
            res = sim.sim(model_id, (target,),
                          ParameterEdit("", v0 * mult, kind="species_initial",
                                        species_id=species))
        except (SimulationError, ValueError):
            return None
        finals.append(res["finals"][target])
    return finals


def _initial_candidates(gp: dict[str, Any], inst_seq: int) -> list[float]:
    ladder = INSTANCE_LADDERS[inst_seq % len(INSTANCE_LADDERS)]
    return [gp["initial_value"] * mult for mult in ladder]


# -- f04 init_threshold_set ----------------------------------------------------

def propose_init_threshold_set(sim: SimRunner, model_id: str,
                               rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    pool = struct.floating_species
    if len(pool) < 3:
        return None
    candidates = _initial_values(struct)
    rng.shuffle(candidates)
    for sid, v0, field in candidates[:6]:
        probes = sorted(rng.sample(pool, min(3, len(pool))))
        for target in probes:
            outcomes = _init_outcomes(sim, model_id, sid, target, v0, CANON_MULTS)
            if outcomes is None:
                continue
            if rel_diff(min(outcomes), max(outcomes)) < INIT_EFFECT_RTOL:
                continue
            try:
                baseline = sim.sim(model_id, (target,))["finals"][target]
            except (SimulationError, ValueError):
                continue
            if baseline == 0:
                continue
            ok = True
            for inst_seq in range(len(INSTANCE_LADDERS)):
                cands = [v0 * m for m in INSTANCE_LADDERS[inst_seq]]
                finals = _init_outcomes(sim, model_id, sid, target, v0,
                                        INSTANCE_LADDERS[inst_seq])
                if finals is None:
                    ok = False
                    break
                accepted = [v for v, f in zip(cands, finals) if f >= baseline]
                if not accepted or len(accepted) == len(cands):
                    ok = False
                    break
            if ok:
                return {"goal_params": {
                    "model_id": model_id, "species": sid, "field": field,
                    "target_species": target, "target": float(baseline),
                    "initial_value": float(v0),
                }}
    return None


def instantiate_init_threshold_set(base: dict[str, Any], inst_seq: int) -> dict[str, Any]:
    gp = base["goal_params"]
    cands = _initial_candidates(gp, inst_seq)
    return {"candidates": cands, "probe_species": [gp["target_species"]]}


# -- f05 init_window_set -------------------------------------------------------

def propose_init_window_set(sim: SimRunner, model_id: str,
                            rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    pool = struct.floating_species
    if len(pool) < 3:
        return None
    candidates = _initial_values(struct)
    rng.shuffle(candidates)
    for sid, v0, field in candidates[:6]:
        probes = sorted(rng.sample(pool, min(3, len(pool))))
        for target in probes:
            outcomes = _init_outcomes(sim, model_id, sid, target, v0, CANON_MULTS)
            if outcomes is None:
                continue
            ordered = sorted(set(outcomes))
            if len(ordered) < 3:
                continue
            lo = (ordered[0] + ordered[1]) / 2.0
            hi = (ordered[1] + ordered[2]) / 2.0
            if not lo < hi or outcomes[-1] <= lo or outcomes[0] >= hi:
                continue
            ok = True
            for inst_seq in range(len(INSTANCE_LADDERS)):
                cands = [v0 * m for m in INSTANCE_LADDERS[inst_seq]]
                finals = _init_outcomes(sim, model_id, sid, target, v0,
                                        INSTANCE_LADDERS[inst_seq])
                if finals is None:
                    ok = False
                    break
                accepted = [v for v, f in zip(cands, finals) if lo <= f <= hi]
                if not accepted or len(accepted) == len(cands):
                    ok = False
                    break
            if ok:
                return {"goal_params": {
                    "model_id": model_id, "species": sid, "field": field,
                    "target_species": target,
                    "window_low": float(lo), "window_high": float(hi),
                    "initial_value": float(v0),
                }}
    return None


def instantiate_init_window_set(base: dict[str, Any], inst_seq: int) -> dict[str, Any]:
    gp = base["goal_params"]
    cands = _initial_candidates(gp, inst_seq)
    return {"candidates": cands, "probe_species": [gp["target_species"]]}


# -- f06 reaction_degree_query -------------------------------------------------

def reaction_degrees(struct: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in struct.reactions:
        species = {s for s, _ in r.reactants}
        species.update(s for s, _ in r.products)
        species.update(r.modifiers)
        out[r.rid] = len(species)
    return out


def propose_reaction_degree_query(sim: SimRunner, model_id: str,
                                  rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    degrees = reaction_degrees(struct)
    if len(degrees) < DEGREE_UNIVERSE:
        return None
    values = sorted(set(degrees.values()))
    for k in values:
        answer = {r for r, d in degrees.items() if d >= k}
        if 1 <= len(answer) <= len(degrees) - 4:
            return {
                "goal_params": {"model_id": model_id, "degree_threshold": int(k)},
                "degrees": degrees,
                "pool": sorted(degrees),
            }
    return None


def instantiate_reaction_degree_query(base: dict[str, Any], rng: Random,
                                      size: int) -> dict[str, Any] | None:
    degrees = base["degrees"]
    k = base["goal_params"]["degree_threshold"]
    hits = sorted(r for r, d in degrees.items() if d >= k)
    misses = sorted(set(degrees) - set(hits))
    if not hits or len(misses) < 4:
        return None
    take = min(len(hits), 4)
    for _ in range(24):
        universe = sorted(rng.sample(hits, take)
                          + rng.sample(misses, min(len(misses), size - take)))
        if len(universe) < 5:
            continue
        answer = {r for r in universe if degrees[r] >= k}
        if _proper_set(answer, universe):
            return {"universe_reaction_ids": universe}
    return None