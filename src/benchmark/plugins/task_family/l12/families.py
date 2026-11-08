"""L12 task families: 10 structural families over the pinned pm4py snapshot.

Solver-grounded reverse generation (build order §4.1), pm4py-native edition:
  authentic artifact (gold net/log pair replaying at fitness 1.0, or a real log)
  -> deterministic mutator corrupts a trace / the model / asks for an exact audit
  -> native check (strict token replay, log skeleton, DECLARE, DFG/variants)
     proves the initial state broken and the reference solution/answer correct
All acceptance decisions are pm4py-native; mutators are proposals validated by
the engine at generation time and independently re-proven by the verifier.
Every task is a complete user goal (a conformant trace/model, or an exact
multi-element audit artifact), never a single fit/not-fit guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LANE_ID = "L12"
SEMANTIC_PROFILE = "pm4py-conformance-offline/1"
GENERATOR_VERSION = "l12-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"
REGIME_CONSTRUCTION = "construction"
REGIME_APPLICATION = "application"

GOAL_REPAIR_TRACE = "repair_trace"
GOAL_REPAIR_MODEL = "repair_model"
GOAL_EXACT_ARTIFACT = "exact_artifact"

INSTANCES_PER_GOAL = 5

#: Complexity tiers by events-in-scope (see engine.complexity_of). Families
#: target ~20/40/40 low/medium/high via artifact choice, recorded honestly.
@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    goal_kind: str
    operation: str  # native check / exact-answer operation
    intent: str
    bases: tuple[dict[str, Any], ...]

    @property
    def total(self) -> int:
        return len(self.bases) * INSTANCES_PER_GOAL


#: (pair_id, trace seed) base goals for trace-repair families over gold pairs.
def _trace_bases(seeds: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
    pairs = ("running_example", "roadtraffic50", "a12", "a22", "a32")
    return tuple(
        {"pair": pairs[i % len(pairs)], "trace_seed": s}
        for i, s in enumerate(seeds)
    )


def _projection_bases(logs: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return tuple({"log": lg} for lg in logs)


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l12-f01-trace-order-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE, "token_replay_strict",
        "Restore the event order of the corrupted trace so it replays perfectly on the reference Petri net.",
        _trace_bases((11, 23, 37, 41, 53, 67, 71)),
    ),
    FamilySpec(
        "l12-f02-trace-event-restore", REGIME_REPAIR, GOAL_REPAIR_TRACE, "token_replay_strict",
        "Reinsert the missing event at the right position so the trace replays perfectly on the reference Petri net.",
        _trace_bases((13, 29, 31, 43, 59, 61, 79)),
    ),
    FamilySpec(
        "l12-f03-trace-foreign-purge", REGIME_REPAIR, GOAL_REPAIR_TRACE, "token_replay_strict",
        "Purge the spurious (foreign or misplaced) event from the trace so it replays perfectly on the reference Petri net.",
        _trace_bases((17, 19, 47, 73, 89, 97, 101)),
    ),
    FamilySpec(
        "l12-f04-model-transition-relabel", REGIME_REPAIR, GOAL_REPAIR_MODEL, "token_replay_strict",
        "Restore the obscured transition label in the Petri net so the entire reference log replays perfectly again.",
        tuple(
            {"pair": p, "label_seed": s}
            for p, s in (
                ("running_example", 5), ("roadtraffic50", 7), ("a12", 11),
                ("a22", 13), ("a32", 17), ("a22", 19), ("a32", 23),
            )
        ),
    ),
    FamilySpec(
        "l12-f05-dfg-reconstruction", REGIME_CONSTRUCTION, GOAL_EXACT_ARTIFACT, "dfg",
        "Reconstruct the exact directly-follows graph (edges with counts, start and end activities) of the referenced log projection.",
        _projection_bases(
            ("running_example", "roadtraffic50", "reviewing", "receipt", "a12", "a22", "a32")
        ),
    ),
    FamilySpec(
        "l12-f06-variant-partition", REGIME_ORGANIZATION, GOAL_EXACT_ARTIFACT, "variants",
        "Partition the referenced log projection into its exact process variants (activity sequences with their trace indices).",
        _projection_bases(
            ("running_example", "roadtraffic50", "reviewing", "receipt", "a12", "a22", "a32")
        ),
    ),
    FamilySpec(
        "l12-f07-endpoint-frequency-audit", REGIME_APPLICATION, GOAL_EXACT_ARTIFACT, "endpoints",
        "Report the exact start/end activity distributions and per-activity occurrence counts of the referenced log projection.",
        _projection_bases(
            ("running_example", "roadtraffic50", "reviewing", "receipt", "a12", "a22", "a32")
        ),
    ),
    FamilySpec(
        "l12-f08-skeleton-conformance-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE, "log_skeleton",
        "Repair the corrupted trace so it satisfies every log-skeleton constraint mined from the reference log.",
        _trace_bases((103, 107, 109, 113, 127, 131, 137)),
    ),
    FamilySpec(
        "l12-f09-declare-conformance-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE, "declare",
        "Repair the corrupted trace so it satisfies every DECLARE constraint mined from the reference log.",
        _trace_bases((139, 149, 151, 157, 163, 167, 173)),
    ),
    FamilySpec(
        "l12-f10-nonconformant-trace-isolation", REGIME_APPLICATION, GOAL_EXACT_ARTIFACT,
        "nonconformant_trace_set",
        "Identify exactly which traces of the noisy log projection fail strict token-based replay on the reference model.",
        tuple(
            {"noisy_pair": p}
            for p in (
                "a12_n05", "a12_n10", "a22_n05", "a22_n10", "a32_n05", "a32_n10",
                "receipt_onevar",
            )
        ),
    ),
)

FAMILY_BY_ID: dict[str, FamilySpec] = {f.family_id: f for f in FAMILIES}


def list_families() -> list[str]:
    return [f.family_id for f in FAMILIES]
