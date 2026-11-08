"""L03 20K-expansion task families: 8 new mechanisms over TCK + Camunda pools.

Genuinely new mechanisms (not renames of the 8K families f01..f08):

- f09 tck-scenario-output: TCK test scenario evaluation with a *dual oracle*
  (aob-dmn-static must agree with the pinned TCK expectation; disagreements
  are honest rejects). 8K f01 has no second oracle.
- f10 suite-conformance-audit: identify the one test case in a visible suite
  whose expected value is inconsistent with the table. The tampered value is
  always a value that also appears as a true expectation elsewhere in the
  suite, so value-frequency shortcuts fail and the table must be evaluated.
- f11 output-entry-repair: single *output* cell corrupted (f03 repairs input
  entries; the output-entry mutation class changes what matched rules return,
  not which rules match).
- f12 aggregation-repair: COLLECT aggregation slot corrupted (f04 repairs the
  hit-policy slot; aggregation semantics SUM/MIN/MAX/COUNT/none is a
  different slot with different repair reasoning).
- f13 rule-deletion-repair: a whole rule was removed (structural mutation;
  f03/f04/f06 are single-cell/slot mutations). Restoring behavior may admit
  non-identical tables - judged by full-universe behavioral equality.
- f14 partial-input-rule-audit: three-valued rule classification under a
  partial input assignment (one variable unknown): which rules match under
  EVERY completion of the missing variable vs under SOME completion. No 8K
  family evaluates under incomplete information.
- f15 chain-root-synthesis: construct the *root* input assignment that makes
  the terminal decision return a target output (cross-table construction;
  8K construction f08 is single-table only).
- f16 ordered-policy-output: produce the exact ordered output list under
  RULE ORDER / OUTPUT ORDER hit policies (ordering semantics; f07 is COLLECT
  accumulation, f02 is unordered matched-rule-id audit).

Regimes: application (f09/f10/f14/f16), repair (f11/f12/f13),
construction (f15). Organization remains N/A for this lane (no distinct
upstream sources of aligned rule views; splitting one file is forbidden).
"""
from __future__ import annotations

from dataclasses import dataclass

LANE_ID = "L03"
SEMANTIC_PROFILE = "camunda-dmn-1.1-table-static/1"
GENERATOR_VERSION = "l03-families-20k/1.0.0"

REGIME_REPAIR = "repair"
REGIME_APPLICATION = "application"
REGIME_CONSTRUCTION = "construction"

GOAL_BUDGETS = {
    "low": {"probes": 4, "actions": 8},
    "medium": {"probes": 6, "actions": 12},
    "high": {"probes": 8, "actions": 16},
}


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    goal_kind: str
    intent: str
    target_base_goals: int
    target_instances: int


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l03-f09-tck-scenario-output", REGIME_APPLICATION, "tck_output",
        "Evaluate the pinned TCK decision table for the TCK test scenario's typed inputs and produce the exact output the TCK expects (dual-oracle conformance subset).",
        19, 49,
    ),
    FamilySpec(
        "l03-f10-suite-conformance-audit", REGIME_APPLICATION, "suite_audit",
        "One test case in the visible suite has an expected value inconsistent with the decision table; identify exactly which case id it is.",
        22, 112,
    ),
    FamilySpec(
        "l03-f11-output-entry-repair", REGIME_REPAIR, "output_repair",
        "Exactly one output-entry cell of the decision table was corrupted (matched rules return wrong values); repair that single cell so the whole case suite passes.",
        20, 100,
    ),
    FamilySpec(
        "l03-f12-aggregation-repair", REGIME_REPAIR, "aggregation_repair",
        "The COLLECT decision table's aggregation was changed and returns wrong aggregate results; restore the aggregation so every case in the suite matches the intended result.",
        11, 55,
    ),
    FamilySpec(
        "l03-f13-rule-deletion-repair", REGIME_REPAIR, "rule_deletion_repair",
        "One whole rule was deleted from the decision table; restore the table's full intended behavior for every case in the suite.",
        30, 92,
    ),
    FamilySpec(
        "l03-f14-partial-input-rule-audit", REGIME_APPLICATION, "partial_input_audit",
        "Given a partial input assignment (one input variable unknown), classify the table's rules into those that match under every completion of the missing variable and those that match under some completion (three-valued partial evaluation).",
        24, 100,
    ),
    FamilySpec(
        "l03-f15-chain-root-synthesis", REGIME_CONSTRUCTION, "chain_synthesis",
        "Construct a root input assignment for the decision chain that makes the terminal decision return the target output (cross-table input synthesis).",
        20, 42,
    ),
    FamilySpec(
        "l03-f16-ordered-policy-output", REGIME_APPLICATION, "ordered_output",
        "Compute the exact ordered output list of the RULE ORDER / OUTPUT ORDER decision table for the given input case (ordering-policy semantics, not COLLECT accumulation).",
        13, 66,
    ),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

FAMILY_QUOTA: dict[str, tuple[int, int]] = {
    f.family_id: (f.target_base_goals, f.target_instances) for f in FAMILIES
}
