"""L07 20K-candidate family table: new mechanisms beyond the legacy 8K lane.

The legacy L07 lane (``families.py``, dataset ``autoontobench-bank-8000-rc1``)
has ten families over the first capable BioModels ids. This module defines a
disjoint set of mechanisms used only by the 20K candidate expansion:

* ``retention_set``        - trajectory retention (final / max) set query
* ``peak_time_ranking``    - trajectory peak-time ranking
* ``pulse_set``            - transient-pulse (max crosses a level, final below)
* ``init_threshold_set``   - initial-condition threshold construction
* ``init_window_set``      - initial-condition window construction
* ``reaction_degree_query``- reaction-star degree set query (structural)

Every family has a distinct semantic predicate, so instances cannot be
produced by renaming or by re-parameterising a legacy family. Model pools are
additionally filtered to exclude every model used by the legacy lane.
"""
from __future__ import annotations

from benchmark.plugins.task_family.l07.families import FamilySpec

LANE = "L07"
PROFILE = "sbml-curated/1"

FAMILY_TABLE_20K: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l07x-f01-retention-set",
        "retention_set", "application", "medium",
        18, 5, 6, 6, True, 6, 1, 0,
        ("Return the exact subset of universe species whose trajectory "
         "retention (final value / maximum value) is at least the stated "
         "ratio under the pinned CVODE protocol."),
    ),
    FamilySpec(
        "l07x-f02-peak-time-ranking",
        "peak_time_ranking", "application", "high",
        18, 5, 7, 7, True, 7, 1, 0,
        ("Rank the top-3 universe species by the time index of their "
         "trajectory maximum (latest first; final value then species id "
         "break ties) under the pinned protocol."),
    ),
    FamilySpec(
        "l07x-f03-pulse-set",
        "pulse_set", "application", "medium",
        18, 5, 6, 6, True, 6, 1, 0,
        ("Return the exact subset of universe species whose trajectory "
         "maximum reaches the stated level but whose final value falls "
         "back below it (transient pulses) under the pinned protocol."),
    ),
    FamilySpec(
        "l07x-f04-init-threshold-search",
        "init_threshold_set", "construction", "high",
        19, 5, 2, 3, True, 3, 1, 0,
        ("Pick the exact subset of candidate initial values for one "
         "species that drive a target species' final value to or above "
         "its value at the original initial condition; verified by "
         "re-simulation per candidate."),
    ),
    FamilySpec(
        "l07x-f05-init-window-search",
        "init_window_set", "construction", "high",
        19, 5, 2, 3, True, 3, 1, 0,
        ("Pick the exact subset of candidate initial values for one "
         "species that hold a target species' final value inside a stated "
         "two-sided window; verified by re-simulation per candidate."),
    ),
    FamilySpec(
        "l07x-f06-reaction-degree-query",
        "reaction_degree_query", "organization", "low",
        18, 5, 5, 8, False, 2, 8, 0,
        ("Parse the provided SBML reaction graph and return the exact set "
         "of universe reactions whose star degree (distinct species among "
         "reactants, products and modifiers) is at least the threshold."),
    ),
)

BY_ID_20K = {spec.family_id: spec for spec in FAMILY_TABLE_20K}

#: Kinds introduced by this expansion. Used by the collision audit to prove
#: that no legacy kind was reused verbatim.
NEW_KINDS_20K = tuple(spec.kind for spec in FAMILY_TABLE_20K)
LEGACY_KINDS = (
    "threshold_set", "ranking", "shape_org", "param_threshold",
    "param_repair", "init_repair", "law_repair", "role_query",
    "species_role_org", "order_inversion",
)
SIM_KINDS_20K = (
    "retention_set", "peak_time_ranking", "pulse_set",
    "init_threshold_set", "init_window_set",
)
STRUCT_KINDS_20K = ("reaction_degree_query",)