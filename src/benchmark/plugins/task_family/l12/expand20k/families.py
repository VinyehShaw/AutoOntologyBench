"""L12 20K expansion task families: 14 new families over new source units.

Design (800 expected candidates, 160 base goals, instances_per_goal=5):

| family | regime | goal_kind | operation | bases | tasks |
|---|---|---|---|---|---:|
| l12-f11-distant-swap-repair | repair | repair_trace | token_replay_strict | 2 pairs x 2 seeds = 4 | 20 |
| l12-f12-segment-reversal-repair | repair | repair_trace | token_replay_strict | 4 | 20 |
| l12-f13-event-relocation-repair | repair | repair_trace | token_replay_strict | 4 | 20 |
| l12-f14-duplicate-event-purge | repair | repair_trace | token_replay_strict | 4 | 20 |
| l12-f15-double-event-restore | repair | repair_trace | token_replay_strict | 4 | 20 |
| l12-f16-dual-transition-relabel | repair | repair_model | token_replay_strict | 2 pairs x 4 label seeds = 8 | 40 |
| l12-f17-skeleton-repair-newlogs | repair | repair_trace | log_skeleton | 18 logs | 90 |
| l12-f18-declare-repair-newlogs | repair | repair_trace | declare | 18 logs | 90 |
| l12-f19-dfg-audit-newlogs | construction | exact_artifact | dfg | 18 logs | 90 |
| l12-f20-variant-audit-newlogs | organization | exact_artifact | variants | 18 logs | 90 |
| l12-f21-endpoint-audit-newlogs | application | exact_artifact | endpoints | 18 logs | 90 |
| l12-f22-nonconformant-isolation-newpairs | application | exact_artifact | nonconformant_trace_set | 18 noisy pairs | 90 |
| l12-f23-footprints-audit | organization | exact_artifact | footprints | 18 logs | 90 |
| l12-f24-footprints-conf-isolation | application | exact_artifact | footprints_conf | 6 noisy pairs | 30 |

New mechanisms vs legacy: five new corruption modes (distant swap, segment
reversal, relocation, duplicate purge, double deletion), dual-label model
repair, log-skeleton/DECLARE repair over log-only bases (legacy mined only
from gold-pair logs), and two genuinely new native operations (log footprints
audit; footprints conformance isolation — a different conformance theory from
token replay). All bases reference NEW source units (see registry.py).
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.task_family.l12.families import (
    FamilySpec,
    REGIME_APPLICATION,
    REGIME_CONSTRUCTION,
    REGIME_ORGANIZATION,
    REGIME_REPAIR,
    GOAL_EXACT_ARTIFACT,
    GOAL_REPAIR_MODEL,
    GOAL_REPAIR_TRACE,
)
from benchmark.plugins.task_family.l12.expand20k.registry import (
    NEW_GOLD_PAIRS,
    NEW_LOGS,
    NEW_NOISY_PAIRS,
    FP_CONF_PAIRS,
    SEED_BASE,
)

INSTANCES_PER_GOAL = 5

_GOLD_KEYS = tuple(sorted(NEW_GOLD_PAIRS))  # ("roadtraffic100", "roadtraffic10000")
_NET_KEYS = tuple(sorted({v["net"] for v in NEW_GOLD_PAIRS.values()}))
_LOG_KEYS = tuple(sorted(NEW_LOGS))
_NOISY_KEYS = tuple(sorted(NEW_NOISY_PAIRS))
_FP_CONF_KEYS = tuple(sorted(FP_CONF_PAIRS))


def _trace_bases() -> tuple[dict[str, Any], ...]:
    """(net, slice) bases for corruption families; instances take distinct
    offsets into the per-NET viable-corruption stream (slice*5 + inst).
    One net (roadtraffic.pnml, shared by the two new gold pairs) x 4 slices
    = 4 base goals = 20 tasks per corruption family."""
    return tuple(
        {"net": net, "slice": s}
        for net in _NET_KEYS
        for s in range(4)
    )


def _relabel_bases() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"pair": pair, "slice": s}
        for pair in _GOLD_KEYS
        for s in range(4)
    )


def _log_bases() -> tuple[dict[str, Any], ...]:
    return tuple({"log": k, "trace_seed": SEED_BASE + 2000 + i}
                 for i, k in enumerate(_LOG_KEYS))


def _noisy_bases() -> tuple[dict[str, Any], ...]:
    return tuple({"noisy_pair": k} for k in _NOISY_KEYS)


def _fp_conf_bases() -> tuple[dict[str, Any], ...]:
    return tuple({"noisy_pair": k} for k in _FP_CONF_KEYS)


FAMILIES_20K: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l12-f11-distant-swap-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "token_replay_strict",
        "Restore the event order of the corrupted trace (two distant positions were "
        "transposed) so it replays perfectly on the reference Petri net.",
        _trace_bases(),
    ),
    FamilySpec(
        "l12-f12-segment-reversal-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "token_replay_strict",
        "Restore the event order of the corrupted trace (a contiguous segment was "
        "reversed) so it replays perfectly on the reference Petri net.",
        _trace_bases(),
    ),
    FamilySpec(
        "l12-f13-event-relocation-repair", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "token_replay_strict",
        "Restore the event order of the corrupted trace (one event was moved to the "
        "end) so it replays perfectly on the reference Petri net.",
        _trace_bases(),
    ),
    FamilySpec(
        "l12-f14-duplicate-event-purge", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "token_replay_strict",
        "Purge the duplicated event from the trace so it replays perfectly on the "
        "reference Petri net.",
        _trace_bases(),
    ),
    FamilySpec(
        "l12-f15-double-event-restore", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "token_replay_strict",
        "Reinsert the two missing events at the right positions so the trace "
        "replays perfectly on the reference Petri net.",
        _trace_bases(),
    ),
    FamilySpec(
        "l12-f16-dual-transition-relabel", REGIME_REPAIR, GOAL_REPAIR_MODEL,
        "token_replay_strict",
        "Restore the two obscured transition labels in the Petri net so the entire "
        "reference log replays perfectly again.",
        _relabel_bases(),
    ),
    FamilySpec(
        "l12-f17-skeleton-repair-newlogs", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "log_skeleton",
        "Repair the corrupted trace so it satisfies every log-skeleton constraint "
        "mined from the reference log.",
        _log_bases(),
    ),
    FamilySpec(
        "l12-f18-declare-repair-newlogs", REGIME_REPAIR, GOAL_REPAIR_TRACE,
        "declare",
        "Repair the corrupted trace so it satisfies every DECLARE constraint mined "
        "from the reference log.",
        _log_bases(),
    ),
    FamilySpec(
        "l12-f19-dfg-audit-newlogs", REGIME_CONSTRUCTION, GOAL_EXACT_ARTIFACT, "dfg",
        "Reconstruct the exact directly-follows graph (edges with counts, start and "
        "end activities) of the referenced log projection.",
        tuple({"log": k} for k in _LOG_KEYS),
    ),
    FamilySpec(
        "l12-f20-variant-audit-newlogs", REGIME_ORGANIZATION, GOAL_EXACT_ARTIFACT,
        "variants",
        "Partition the referenced log projection into its exact process variants "
        "(activity sequences with their trace indices).",
        tuple({"log": k} for k in _LOG_KEYS),
    ),
    FamilySpec(
        "l12-f21-endpoint-audit-newlogs", REGIME_APPLICATION, GOAL_EXACT_ARTIFACT,
        "endpoints",
        "Report the exact start/end activity distributions and per-activity "
        "occurrence counts of the referenced log projection.",
        tuple({"log": k} for k in _LOG_KEYS),
    ),
    FamilySpec(
        "l12-f22-nonconformant-isolation-newpairs", REGIME_APPLICATION,
        GOAL_EXACT_ARTIFACT, "nonconformant_trace_set",
        "Identify exactly which traces of the noisy log projection fail strict "
        "token-based replay on the reference model.",
        _noisy_bases(),
    ),
    FamilySpec(
        "l12-f23-footprints-audit", REGIME_ORGANIZATION, GOAL_EXACT_ARTIFACT,
        "footprints",
        "Report the exact behavioral footprints of the referenced log projection "
        "(activities, always-precedes sequence relations, parallel relations, start "
        "and end activities).",
        tuple({"log": k} for k in _LOG_KEYS),
    ),
    FamilySpec(
        "l12-f24-footprints-conf-isolation", REGIME_APPLICATION, GOAL_EXACT_ARTIFACT,
        "footprints_conf",
        "Identify exactly which traces of the noisy log projection violate the "
        "reference model's footprints constraints (behavioral-profile conformance), "
        "with the violating relations per trace.",
        _fp_conf_bases(),
    ),
)

FAMILY_BY_ID_20K: dict[str, FamilySpec] = {f.family_id: f for f in FAMILIES_20K}


def expected_totals() -> dict[str, int]:
    return {
        f.family_id: len(f.bases) * INSTANCES_PER_GOAL for f in FAMILIES_20K
    }
