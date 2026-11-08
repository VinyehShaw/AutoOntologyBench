"""L07 simulation-backed base-goal proposers (families f01-f07, f10).

All numbers come from the lane's pinned CVODE protocol through ``SimRunner``
(cached at proposal time; independently recomputed by the verifier at
qualification time). Instance ladders / corruption factors are fixed
deterministic tables so a rebuild reproduces byte-identical candidates.
"""
from __future__ import annotations

import re
from itertools import combinations
from random import Random
from typing import Any

from benchmark.plugins.source.l07.sbml_edit import ParameterEdit, all_reaction_mentions
from benchmark.plugins.source.l07.simulate import SimulationError
from benchmark.plugins.task_family.l07.support import SimRunner, rel_diff, sample_universe

CANON_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
INSTANCE_LADDERS: tuple[tuple[float, ...], ...] = (
    (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    (0.2, 0.4, 1.0, 2.0, 5.0, 10.0),
    (0.3, 0.6, 1.2, 2.5, 5.0, 10.0),
    (0.5, 1.0, 1.5, 3.0, 6.0, 12.0),
    (0.15, 0.35, 0.7, 1.5, 3.0, 6.0),
)
CORRUPT_FACTORS = (0.1, 0.25, 0.5, 2.0, 10.0)
EFFECT_FACTOR = 3.0
EFFECT_RTOL = 0.05


def _finals(sim: SimRunner, model_id: str) -> dict[str, float]:
    rec = sim.index["models"][model_id]
    return {sid: f["last"] for sid, f in (rec["screening"].get("species") or {}).items()}


def _shapes(sim: SimRunner, model_id: str) -> dict[str, str]:
    rec = sim.index["models"][model_id]
    return {sid: f["shape"] for sid, f in (rec["screening"].get("species") or {}).items()}


def swap_species_in_formula(formula: str, pairs: list[tuple[str, str]]) -> str:
    """Simultaneous whole-word swap of species-id pairs in an L3 formula."""
    mapping: dict[str, str] = {}
    for a, b in pairs:
        mapping[a], mapping[b] = b, a
    pattern = re.compile(r"\b(%s)\b" % "|".join(
        re.escape(s) for s in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group(1)], formula)


# -- f01 threshold_set ---------------------------------------------------------

def propose_threshold_set(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    """Cut the species at the widest multiplicative gap in final values.

    Choosing the cut between two well-separated value clusters (rather than the
    median of all positives) keeps the answer away from the solver noise floor
    and makes the threshold counterfactual (x0.5 / x2) actually change the
    answer instead of flipping numerical dust. Values below ``1e-9 * max`` are
    treated as numerical zeros and never anchor the cut.
    """
    finals = _finals(sim, model_id)
    positive = sorted(v for v in finals.values() if v > 0)
    if len(positive) < 8:
        return None
    floor = positive[-1] * 1e-9
    vals = sorted({v for v in positive if v >= floor})
    if len(vals) < 2:
        return None
    best: tuple[float, float, float] | None = None
    for lo, hi in zip(vals, vals[1:]):
        ratio = hi / lo if lo > 0 else float("inf")
        if best is None or ratio > best[0]:
            best = (ratio, lo, hi)
    if best is None or best[0] < 1.5:
        return None
    threshold = (best[1] * best[2]) ** 0.5
    if sum(1 for v in finals.values() if v >= threshold) < 2:
        return None
    if sum(1 for v in finals.values() if v < threshold) < 2:
        return None
    return {"goal_params": {"model_id": model_id, "threshold": threshold},
            "finals": finals}


def instantiate_threshold_set(base: dict[str, Any], rng: Random,
                              size: int) -> dict[str, Any] | None:
    finals = base["finals"]
    threshold = base["goal_params"]["threshold"]
    for _ in range(24):
        universe = sample_universe(rng, sorted(finals), size)
        if universe is None:
            return None
        answer = [s for s in universe if finals[s] >= threshold]
        if answer and len(answer) < len(universe):
            return {"universe_ids": universe}
    return None


# -- f02 ranking ----------------------------------------------------------------

def _tie_free(finals: dict[str, float], ids: list[str], rtol: float = 1e-4) -> bool:
    vals = sorted(finals[s] for s in ids)
    return all(rel_diff(a, b) > rtol for a, b in zip(vals, vals[1:]))


def propose_ranking(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    finals = _finals(sim, model_id)
    if len(finals) < 7:
        return None
    pool = sorted(finals)
    for _ in range(24):
        if _tie_free(finals, rng.sample(pool, 7)):
            return {"goal_params": {"model_id": model_id, "k": 3}, "finals": finals}
    return None


def instantiate_ranking(base: dict[str, Any], rng: Random, size: int) -> dict[str, Any] | None:
    finals = base["finals"]
    for _ in range(40):
        universe = sample_universe(rng, sorted(finals), size)
        if universe and _tie_free(finals, universe):
            return {"universe_ids": universe}
    return None


# -- f03 shape_org --------------------------------------------------------------

def propose_shape_org(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    shapes = _shapes(sim, model_id)
    pool = [s for s, sh in shapes.items() if sh != "other"]
    if len(pool) < 8 or len({shapes[s] for s in pool}) < 2:
        return None
    return {"goal_params": {"model_id": model_id}, "shapes": shapes, "pool": pool}


def instantiate_shape_org(base: dict[str, Any], rng: Random, size: int) -> dict[str, Any] | None:
    shapes, pool = base["shapes"], base["pool"]
    for _ in range(40):
        universe = sample_universe(rng, pool, size)
        if universe and len({shapes[s] for s in universe}) >= 2:
            return {"universe_ids": universe}
    return None


# -- shared parameter probing ---------------------------------------------------

def _canonical_probe(sim: SimRunner, model_id: str, pid: str, p0: float,
                     floating: list[str]) -> dict[float, dict[str, float]] | None:
    """Finals per canonical multiplier; None when any rung fails to simulate."""
    out: dict[float, dict[str, float]] = {}
    for mult in CANON_MULTS:
        try:
            res = sim.sim(model_id, tuple(floating), ParameterEdit(pid, p0 * mult))
        except (SimulationError, ValueError):
            return None
        out[mult] = res["finals"]
    return out


def _param_candidates(sim: SimRunner, model_id: str, rng: Random,
                      max_params: int = 4
                      ) -> list[tuple[str, float, dict[float, dict[str, float]]]]:
    struct = sim.struct(model_id)
    params = [p for p in struct.settable_parameters
              if abs(struct.parameter_value(p)) > 0]
    rng.shuffle(params)
    floating = struct.floating_species[:12]
    out: list[tuple[str, float, dict[float, dict[str, float]]]] = []
    for pid in params[:max_params]:
        p0 = struct.parameter_value(pid)
        probe = _canonical_probe(sim, model_id, pid, p0, floating)
        if probe is None:
            continue
        spread = max(rel_diff(probe[CANON_MULTS[0]][s], probe[CANON_MULTS[-1]][s])
                     for s in floating)
        if spread > 0.01:
            out.append((pid, p0, probe))
    return out


# -- f04 param_threshold --------------------------------------------------------

def propose_param_threshold(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    for pid, p0, probe in _param_candidates(sim, model_id, rng):
        species = sorted(probe[1.0])
        rng.shuffle(species)
        for target in species[:4]:
            outcomes = [probe[m][target] for m in CANON_MULTS]
            lo, hi = min(outcomes), max(outcomes)
            if rel_diff(lo, hi) <= 0.05:
                continue
            return {"goal_params": {"model_id": model_id, "parameter": pid,
                                    "target_species": target,
                                    "target": (lo + hi) / 2.0,
                                    "original_value": p0}}
    return None


def instantiate_param_threshold(sim: SimRunner, base: dict[str, Any],
                                inst_seq: int) -> dict[str, Any] | None:
    gp = base["goal_params"]
    ladder = [gp["original_value"] * m for m in INSTANCE_LADDERS[inst_seq % 5]]
    finals = []
    for v in ladder:
        try:
            res = sim.sim(gp["model_id"], (gp["target_species"],),
                          ParameterEdit(gp["parameter"], v))
        except (SimulationError, ValueError):
            return None
        finals.append(res["finals"][gp["target_species"]])
    answer = [v for v, f in zip(ladder, finals) if f >= gp["target"]]
    if not answer or len(answer) == len(ladder):
        return None
    return {"candidates": ladder}


# -- f10 order_inversion --------------------------------------------------------

def propose_order_inversion(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    for pid, p0, probe in _param_candidates(sim, model_id, rng):
        floating = sorted(probe[1.0])
        pairs = list(combinations(floating, 2))
        rng.shuffle(pairs)
        for a, b in pairs[:6]:
            fa0, fb0 = probe[1.0][a], probe[1.0][b]
            if not (fa0 > fb0 and rel_diff(fa0, fb0) >= 0.05):
                continue
            inverted = [m for m in CANON_MULTS if probe[m][a] < probe[m][b]]
            kept = [m for m in CANON_MULTS if probe[m][a] > probe[m][b]]
            if inverted and kept:
                return {"goal_params": {"model_id": model_id, "parameter": pid,
                                        "species_a": a, "species_b": b,
                                        "original_value": p0}}
    return None


def instantiate_order_inversion(sim: SimRunner, base: dict[str, Any],
                                inst_seq: int) -> dict[str, Any] | None:
    gp = base["goal_params"]
    ladder = [gp["original_value"] * m for m in INSTANCE_LADDERS[inst_seq % 5]]
    probes = (gp["species_a"], gp["species_b"])
    answer = []
    for v in ladder:
        try:
            res = sim.sim(gp["model_id"], probes, ParameterEdit(gp["parameter"], v))
        except (SimulationError, ValueError):
            return None
        if res["finals"][gp["species_a"]] < res["finals"][gp["species_b"]]:
            answer.append(v)
    if not answer or len(answer) == len(ladder):
        return None
    return {"candidates": ladder}


# -- f05 / f06 repair -----------------------------------------------------------

def _effect_probe(sim: SimRunner, model_id: str, edit: ParameterEdit,
                  probes: tuple[str, ...]) -> dict[str, Any] | None:
    """Corrupted-vs-reference probe check: at least one probe species must move."""
    try:
        ref = sim.sim(model_id, probes)
        broken = sim.sim(model_id, probes, edit)
    except (SimulationError, ValueError):
        return None
    moved = {s: rel_diff(ref["finals"][s], broken["finals"][s]) for s in probes}
    if not any(d >= EFFECT_RTOL for d in moved.values()):
        return None
    return {"probe_rel_diff": moved}


def propose_param_repair(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    params = [p for p in struct.settable_parameters
              if abs(struct.parameter_value(p)) > 0]
    rng.shuffle(params)
    probes = tuple(sorted(rng.sample(struct.floating_species,
                                     min(3, len(struct.floating_species)))))
    for pid in params[:6]:
        p0 = struct.parameter_value(pid)
        eff = _effect_probe(sim, model_id, ParameterEdit(pid, p0 * EFFECT_FACTOR), probes)
        if eff is not None:
            return {"goal_params": {"model_id": model_id, "parameter": pid,
                                    "probe_species": list(probes)}}
    return None


def propose_init_repair(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    candidates = []
    for sid in struct.floating_species:
        info = struct.species_info(sid)
        v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
        if v0 is not None and abs(v0) > 0:
            field = "initial_amount" if info.initial_amount is not None else "initial_concentration"
            candidates.append((sid, v0, field))
    rng.shuffle(candidates)
    pool = list(struct.floating_species)
    for sid, v0, field in candidates[:6]:
        probes = tuple(sorted(rng.sample(pool, min(3, len(pool)))))
        eff = _effect_probe(
            sim, model_id,
            ParameterEdit("", v0 * EFFECT_FACTOR, kind="species_initial", species_id=sid),
            probes)
        if eff is not None:
            return {"goal_params": {"model_id": model_id, "species": sid, "field": field,
                                    "probe_species": list(probes)}}
    return None


# -- f07 law_repair ---------------------------------------------------------------

def _swap_variants(mentions: list[str]) -> list[list[tuple[str, str]]]:
    pairs = list(combinations(mentions, 2))
    singles = [[p] for p in pairs]
    doubles = [[p, q] for p, q in combinations(pairs, 2)
               if not set(p) & set(q)]
    return [list(v) for v in singles + doubles]


def propose_law_repair(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    raw = sim.raw(model_id)
    reactions = [r for r in struct.reactions if r.kinetic_formula]
    rng.shuffle(reactions)
    mentions_by = all_reaction_mentions(raw)
    floating = set(struct.floating_species)
    probes = tuple(sorted(rng.sample(struct.floating_species,
                                     min(5, len(struct.floating_species)))))
    for r in reactions:
        mentions = [m for m in mentions_by.get(r.rid, []) if m in floating]
        if len(mentions) < 3:
            continue
        variants = _swap_variants(mentions)
        if len(variants) < 5:
            continue
        formula = r.kinetic_formula or ""
        effective: list[list[tuple[str, str]]] = []
        for variant in variants:
            corrupted = swap_species_in_formula(formula, variant)
            if corrupted == formula:
                continue
            eff = _effect_probe(
                sim, model_id,
                ParameterEdit("", 0.0, kind="kinetic_law", reaction_id=r.rid,
                              formula=corrupted),
                probes)
            if eff is not None:
                effective.append(variant)
            if len(effective) >= 5:
                break
        if len(effective) >= 5:
            return {"goal_params": {"model_id": model_id, "reaction": r.rid,
                                    "probe_species": list(probes)},
                    "variants": [sorted(p for p in v) for v in effective]}
    return None
