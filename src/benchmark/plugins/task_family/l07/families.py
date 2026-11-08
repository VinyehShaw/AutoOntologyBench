"""L07 family table: 10 task families over pinned BioModels SBML snapshots.

Regimes follow the build order's repair / organization / construction axes plus
application queries. Complexity split: 2 low families (20%), 4 medium (40%),
4 high (40%). Every family is natively verified by libsbml structure checks
and/or the pinned CVODE protocol (see verifier/l07).
"""
from __future__ import annotations

from dataclasses import dataclass

PROFILE = "sbml-curated/1"
LANE = "L07"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    kind: str
    regime: str
    complexity: str
    base_goals: int
    instances_per_goal: int
    min_universe: int
    universe_size: int
    needs_sim: bool
    min_floating_species: int
    min_reactions: int
    min_settable_params: int
    description: str


FAMILY_TABLE: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l07-f01-threshold-set-query",
        "threshold_set", "application", "low",
        8, 5, 4, 6, True, 6, 1, 0,
        ("Exact set of universe species whose final (t=100, pinned "
         "CVODE protocol) value is >= the given threshold."),
    ),
    FamilySpec(
        "l07-f02-final-ranking-query",
        "ranking", "application", "medium",
        8, 5, 5, 7, True, 7, 1, 0,
        ("Top-3 universe species by final value, exact order "
         "(tie-free universe enforced natively)."),
    ),
    FamilySpec(
        "l07-f03-trajectory-shape-organization",
        "shape_org", "organization", "medium",
        8, 5, 6, 8, True, 8, 1, 0,
        ("Organize universe species into the lane's deterministic "
         "trajectory-shape classes (flat/increasing/decreasing/peaked)."),
    ),
    FamilySpec(
        "l07-f04-parameter-threshold-search",
        "param_threshold", "construction", "high",
        8, 5, 1, 1, True, 2, 1, 1,
        ("Pick the subset of candidate values for a named global "
         "parameter that drive a target species' final value to/above "
         "a target; verified by re-simulation per candidate."),
    ),
    FamilySpec(
        "l07-f05-parameter-value-repair",
        "param_repair", "repair", "medium",
        8, 5, 2, 3, True, 3, 1, 1,
        ("A global parameter in the provided SBML was corrupted "
         "(scaled). Restore its value so the probe species' final "
         "values match the stated reference."),
    ),
    FamilySpec(
        "l07-f06-initial-value-repair",
        "init_repair", "repair", "medium",
        8, 5, 2, 3, True, 3, 1, 0,
        ("A species' initial amount/concentration in the provided "
         "SBML was corrupted. Restore it so probe finals match the "
         "stated reference."),
    ),
    FamilySpec(
        "l07-f07-kinetic-law-repair",
        "law_repair", "repair", "high",
        8, 5, 2, 3, True, 3, 2, 0,
        ("Two species identifiers in a reaction's kinetic-law "
         "formula were swapped. Repair the formula so probe finals "
         "match the stated reference."),
    ),
    FamilySpec(
        "l07-f08-network-role-query",
        "role_query", "application", "low",
        8, 5, 5, 8, False, 2, 8, 0,
        ("Exact set of universe reactions satisfying a structural "
         "relation (consumes/produces/modulates) to a target species."),
    ),
    FamilySpec(
        "l07-f09-species-role-organization",
        "species_role_org", "organization", "high",
        8, 5, 8, 10, False, 10, 3, 0,
        ("Organize universe species into deterministic reaction-graph "
         "roles (boundary/source/sink/intermediate/catalyst/isolated)."),
    ),
    FamilySpec(
        "l07-f10-steady-ordering-construction",
        "order_inversion", "construction", "high",
        8, 5, 2, 2, True, 2, 1, 1,
        ("Pick the subset of candidate parameter values that INVERT "
         "the final-value ordering of a named species pair."),
    ),
)

BY_ID = {f.family_id: f for f in FAMILY_TABLE}
