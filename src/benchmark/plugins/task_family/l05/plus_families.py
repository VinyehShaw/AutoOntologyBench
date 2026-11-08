"""20K L05 expansion families: descriptor bands, dual counts, new repair modes.

These families extend the legacy 16 L05 families without renaming or
re-parameterising them:

* molecule axis  — RDKit descriptor vectors (rings, rotatable bonds, HBD/HBA,
  charge, stereo centers, element composition, MW, sp3 fraction);
* query axis     — descriptor bands, descriptor+group conjunction, dual
  substructure-count ranges, and parent-minus-descendants filtering;
* repair axis    — over-specific rule (strict subset defect), hierarchy
  branch confusion (parent/child definition), and negation-predicate flip.

Every family is a distinct mechanism, not a parameter variation of a legacy
family. Family ids are prefixed ``l05plus.`` and instance/base ids use the
``l05x-`` / ``l05p-`` namespaces so they cannot collide with legacy ids.
"""
from __future__ import annotations

import re
from random import Random

from benchmark.plugins.task_family.l05.families import (
    REGIME_APPLICATION, REGIME_REPAIR, FamilySpec,
)

PLUS_FAMILY_TABLE: tuple[FamilySpec, ...] = (
    FamilySpec("l05plus.prop_band_dual", REGIME_APPLICATION, "medium", 10, 5, 8,
               "joint band over two independent RDKit molecule descriptors"),
    FamilySpec("l05plus.element_composition_band", REGIME_APPLICATION, "low", 8, 5, 8,
               "element-composition band (N/O/S/halogen atom count)"),
    FamilySpec("l05plus.count_range_dual", REGIME_APPLICATION, "high", 8, 5, 5,
               "two simultaneous substructure match-count ranges"),
    FamilySpec("l05plus.prop_band_with_group", REGIME_APPLICATION, "medium", 8, 5, 6,
               "descriptor band AND functional-group membership conjunction"),
    FamilySpec("l05plus.smarts_repair_overspecific", REGIME_REPAIR, "high", 8, 5, 4,
               "valid but over-specific SMARTS that misses true positives"),
    FamilySpec("l05plus.smarts_repair_wrong_branch", REGIME_REPAIR, "high", 8, 5, 4,
               "functional-group rule replaced by a parent/child branch definition"),
    FamilySpec("l05plus.smarts_repair_exclusion_flip", REGIME_REPAIR, "high", 6, 4, 4,
               "negation predicate removed from an exclusion-bearing rule"),
)

PLUS_BY_ID = {f.family_id: f for f in PLUS_FAMILY_TABLE}
PLUS_QUERY_FAMILIES = frozenset(
    f.family_id for f in PLUS_FAMILY_TABLE if f.regime == REGIME_APPLICATION)
PLUS_REPAIR_FAMILIES = frozenset(
    f.family_id for f in PLUS_FAMILY_TABLE if f.regime == REGIME_REPAIR)
PLUS_FAMILIES = frozenset(PLUS_BY_ID)

PROP_PAIRS: tuple[tuple[str, str], ...] = (
    ("rings", "rotatable_bonds"),
    ("aromatic_rings", "hbd"),
    ("aliphatic_rings", "hba"),
    ("saturated_rings", "hetero_atoms"),
    ("hbd", "hba"),
    ("rings", "hetero_atoms"),
    ("rotatable_bonds", "formal_charge"),
    ("stereo_centers", "rings"),
    ("fraction_csp3", "rings"),
    ("mw", "aromatic_rings"),
)
ELEMENTS: tuple[str, ...] = ("N", "O", "S", "Cl", "Br", "F", "P")
_NEGATION = re.compile(r"!")
_BRACKET_ATOM = re.compile(r"\[[^\]]+\]")
_BARE_ATOM = re.compile(r"(?<![A-Za-z\[])(Cl|Br|[BCNOPSFIcnops])(?![A-Za-z])")
_OVERSPECIFIC_SUFFIXES = (";R", ";!R", ";H1", ";H0", ";X3", ";D2", ";+0")


def _quantile(values: list[int], frac: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, int(frac * (len(values) - 1))))
    return values[idx]


def _band_from_dist(dist: list[int], rng: Random, *, min_width: int = 1) -> tuple[int, int]:
    a = rng.uniform(0.05, 0.55)
    b = min(0.98, a + rng.uniform(0.08, 0.35))
    lo = _quantile(dist, a)
    hi = _quantile(dist, b)
    if hi - lo < min_width:
        hi = lo + min_width
    return lo, hi


def propose_plus_query(spec: FamilySpec, rng: Random, *, pool, dist) -> dict | None:
    """Return base-goal parameters for a 20K application family."""
    fid = spec.family_id
    if fid == "l05plus.prop_band_dual":
        p1, p2 = rng.choice(PROP_PAIRS)
        if rng.random() < 0.5:
            p1, p2 = p2, p1
        lo1, hi1 = _band_from_dist(dist(p1), rng)
        lo2, hi2 = _band_from_dist(dist(p2), rng)
        return {"query": {"op": "and", "args": [
            {"op": "prop_band", "prop": p1, "lo": lo1, "hi": hi1},
            {"op": "prop_band", "prop": p2, "lo": lo2, "hi": hi2},
        ]}, "require_interaction": False}
    if fid == "l05plus.element_composition_band":
        usable = [e for e in ELEMENTS
                  if sum(1 for v in dist(f"elem_{e}") if v > 0) >= 500]
        if not usable:
            return None
        element = rng.choice(usable)
        nonzero = [v for v in dist(f"elem_{element}") if v > 0]
        lo, hi = _band_from_dist(nonzero, rng)
        return {"query": {"op": "element_band", "element": element,
                          "lo": lo, "hi": hi}, "require_interaction": False}
    if fid == "l05plus.count_range_dual":
        if len(pool.count_variable) < 2:
            return None
        g1, g2 = rng.sample(pool.count_variable, 2)
        lo1 = rng.choice([1, 2])
        lo2 = rng.choice([1, 2])
        return {"query": {"op": "count_range_dual", "groups": [g1, g2],
                          "ranges": [[lo1, lo1 + rng.choice([1, 2, 3])],
                                     [lo2, lo2 + rng.choice([1, 2, 3])]]},
                "require_interaction": False}
    if fid == "l05plus.prop_band_with_group":
        if not pool.mixed:
            return None
        group = rng.choice(pool.mixed)
        props = ("rings", "aromatic_rings", "rotatable_bonds", "hbd", "hba",
                 "hetero_atoms", "mw", "fraction_csp3")
        prop = rng.choice(props)
        lo, hi = _band_from_dist(dist(prop), rng)
        return {"query": {"op": "prop_band_with_group", "prop": prop, "lo": lo,
                          "hi": hi, "group": group}, "require_interaction": True}
    raise ValueError(f"not a plus query family: {fid}")


def mutate_overspecific(smarts: str, rng: Random) -> str | None:
    """Add one atom constraint; downstream screening requires a strict subset."""
    candidates: list[str] = []
    for match in _BRACKET_ATOM.finditer(smarts):
        token = match.group(0)
        for suffix in _OVERSPECIFIC_SUFFIXES:
            candidates.append(smarts[:match.end() - 1] + suffix + smarts[match.end() - 1:])
    for match in _BARE_ATOM.finditer(smarts):
        atom = match.group(0)
        for suffix in _OVERSPECIFIC_SUFFIXES:
            candidates.append(smarts[:match.start()] + f"[{atom}{suffix}]"
                              + smarts[match.end():])
    rng.shuffle(candidates)
    for candidate in candidates:
        if candidate == smarts or smarts in candidate:
            continue
        from rdkit import Chem
        if Chem.MolFromSmarts(candidate) is not None:
            return candidate
    return None


def mutate_wrong_branch(smarts: str, label: str, ctx, rng: Random) -> str | None:
    """Replace a rule with its hierarchy parent/child definition (wrong branch)."""
    parent = label.rsplit(".", 1)[0] if "." in label else None
    children = sorted(ctx.children.get(label, ()))
    alternatives = ([parent] if parent else []) + children
    if not alternatives:
        return None
    rng.shuffle(alternatives)
    for alt in alternatives:
        broken = ctx.smarts.get(alt)
        if broken and broken != smarts and smarts not in broken:
            return broken
    return None


def mutate_exclusion_flip(smarts: str, rng: Random) -> str | None:
    """Remove a negation operator from an exclusion predicate."""
    positions = [m.start() for m in _NEGATION.finditer(smarts)]
    rng.shuffle(positions)
    for i in positions:
        broken = smarts[:i] + smarts[i + 1:]
        if broken == smarts or smarts in broken:
            continue
        from rdkit import Chem
        if Chem.MolFromSmarts(broken) is not None:
            return broken
    return None