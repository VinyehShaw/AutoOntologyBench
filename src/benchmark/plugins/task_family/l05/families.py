"""L05 task families: 16 structural families over the RDKit hierarchy + ChEBI corpus.

Each family exposes ``propose_params(rng, pool, ctx)`` returning the *base goal*
parameters (group labels / SMARTS / ranges — everything except the universe
seed), or None when the draw cannot satisfy the family's structural
preconditions (caller retries with the next rng draw). Universe sampling and
mutation operators live in ``sample.py``; gates live in the verifier plugin.

Regime rationale: L05 has no native state-transition semantics, so families
are ``application`` (analytical query), ``repair`` (broken SMARTS vs golden
behavior) and ``organization`` (aligning the RDKit group library — source 1 —
onto ChEBI molecule entities — source 2). No ``construction`` family: the
closed 52-group library admits no honest hidden-structure discovery claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

REGIME_APPLICATION = "application"
REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    complexity: str          # low | medium | high (structural proxy)
    base_goals: int
    instances_per_goal: int
    min_answer: int          # qualification needs |answer| >= this on the universe
    description: str


FAMILY_TABLE: tuple[FamilySpec, ...] = (
    FamilySpec("l05.fg_membership_exact_set", REGIME_APPLICATION, "low", 15, 6, 4,
               "exact membership set of one functional group"),
    FamilySpec("l05.fg_conjunction2", REGIME_APPLICATION, "low", 15, 6,
               6, "molecules containing ALL of two groups"),
    FamilySpec("l05.fg_conjunction3", REGIME_APPLICATION, "medium", 12, 6,
               5, "molecules containing ALL of three groups"),
    FamilySpec("l05.fg_conjunction_negation4", REGIME_APPLICATION, "high", 12, 5,
               4, "G1 AND G2 AND NOT G3 AND NOT G4 precise filtering"),
    FamilySpec("l05.fg_disjunction3", REGIME_APPLICATION, "medium", 10, 6,
               8, "molecules containing ANY of three groups"),
    FamilySpec("l05.fg_xor_exactly_one", REGIME_APPLICATION, "high", 10, 6,
               4, "molecules containing exactly one of three groups"),
    FamilySpec("l05.fg_negation_band", REGIME_APPLICATION, "medium", 11, 6,
               8, "heavy-atom band AND NOT a common group"),
    FamilySpec("l05.fg_hierarchy_descendant_union", REGIME_APPLICATION, "medium",
               10, 6, 6, "union over all descendants of a parent group"),
    FamilySpec("l05.fg_hierarchy_parent_not_child", REGIME_APPLICATION, "high",
               10, 5, 3, "parent group matched but a chosen child group NOT matched"),
    FamilySpec("l05.fg_match_count_range", REGIME_APPLICATION, "medium", 11, 6,
               4, "substructure match count of a group within [lo, hi]"),
    FamilySpec("l05.fg_profile_exact", REGIME_APPLICATION, "high", 12, 5,
               3, "exact boolean profile over four groups"),
    FamilySpec("l05.fg_heavy_atom_band_with_group", REGIME_APPLICATION, "low", 14, 6,
               6, "heavy atoms in band AND contains group"),
    FamilySpec("l05.smarts_repair_syntax", REGIME_REPAIR, "high", 12, 6,
               4, "syntax-broken SMARTS (flat library) vs golden match set"),
    FamilySpec("l05.smarts_repair_semantic_wrongset", REGIME_REPAIR, "high", 12, 6,
               4, "valid-but-wrong SMARTS (atom swap) vs golden match set"),
    FamilySpec("l05.smarts_repair_overgeneral", REGIME_REPAIR, "high", 12, 5,
               4, "constraint-stripped SMARTS matches a strict superset"),
    FamilySpec("l05.org_align_group_molecule_sets", REGIME_ORGANIZATION, "medium",
               12, 6, 3, "re-align deranged group-label -> molecule-set table"),
)

BY_ID = {f.family_id: f for f in FAMILY_TABLE}


def q_group(g: str) -> dict:
    return {"op": "group", "group": g}


def q_and(*args: dict) -> dict:
    return {"op": "and", "args": list(args)}


def q_or(*args: dict) -> dict:
    return {"op": "or", "args": list(args)}


def q_not(a: dict) -> dict:
    return {"op": "not", "arg": a}


def q_xor(*args: dict) -> dict:
    return {"op": "xor", "args": list(args)}


def _pick(rng: Random, items: list[str], k: int) -> list[str]:
    return rng.sample(items, k)


def propose_query(spec: FamilySpec, rng: Random, pool: "GroupPool") -> dict | None:
    """Base-goal parameters for query families; None = draw failed, retry."""
    fid = spec.family_id
    if fid == "l05.fg_membership_exact_set":
        g = rng.choice(pool.frequent)
        return {"query": q_group(g), "require_interaction": False}
    if fid == "l05.fg_conjunction2":
        g1, g2 = _pick(rng, pool.frequent, 2)
        return {"query": q_and(q_group(g1), q_group(g2)), "require_interaction": True}
    if fid == "l05.fg_conjunction3":
        g1, g2, g3 = _pick(rng, pool.frequent, 3)
        return {"query": q_and(q_group(g1), q_group(g2), q_group(g3)),
                "require_interaction": True}
    if fid == "l05.fg_conjunction_negation4":
        g1, g2, g3, g4 = _pick(rng, pool.mixed, 4)
        return {"query": q_and(q_group(g1), q_group(g2), q_not(q_group(g3)),
                               q_not(q_group(g4))), "require_interaction": True}
    if fid == "l05.fg_disjunction3":
        g1, g2, g3 = _pick(rng, pool.medium, 3)
        return {"query": q_or(q_group(g1), q_group(g2), q_group(g3)),
                "require_interaction": False}
    if fid == "l05.fg_xor_exactly_one":
        g1, g2, g3 = _pick(rng, pool.medium, 3)
        return {"query": q_xor(q_group(g1), q_group(g2), q_group(g3)),
                "require_interaction": False}
    if fid == "l05.fg_negation_band":
        g = rng.choice(pool.frequent)
        lo, hi = rng.choice([(6, 20), (10, 30), (15, 40)])
        return {"query": q_and({"op": "heavy_band", "lo": lo, "hi": hi},
                               q_not(q_group(g))), "require_interaction": False}
    if fid == "l05.fg_hierarchy_descendant_union":
        p = rng.choice(pool.parents_multi)
        return {"query": {"op": "descendants_of", "group": p},
                "require_interaction": False}
    if fid == "l05.fg_hierarchy_parent_not_child":
        p = rng.choice(pool.parents_with_children)
        child = rng.choice(pool.children_of(p))
        return {"query": q_and(q_group(p), q_not(q_group(child))),
                "require_interaction": True}
    if fid == "l05.fg_match_count_range":
        g = rng.choice(pool.count_variable)
        lo = rng.choice([1, 2])
        hi = lo + rng.choice([1, 2, 3])
        return {"query": {"op": "count_range", "group": g, "lo": lo, "hi": hi},
                "require_interaction": False}
    if fid == "l05.fg_profile_exact":
        g1, g2, g3, g4 = _pick(rng, pool.mixed, 4)
        return {"query": {"op": "profile", "pos": [g1, g2], "neg": [g3, g4]},
                "require_interaction": True}
    if fid == "l05.fg_heavy_atom_band_with_group":
        g = rng.choice(pool.frequent)
        lo, hi = rng.choice([(5, 15), (8, 24), (12, 36)])
        return {"query": q_and({"op": "heavy_band", "lo": lo, "hi": hi}, q_group(g)),
                "require_interaction": True}
    raise ValueError(f"not a query family: {fid}")
