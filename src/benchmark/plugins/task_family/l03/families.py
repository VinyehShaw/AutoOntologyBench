"""L03 task families: 8 structural families over the pinned Camunda DMN corpus.

Solver-grounded reverse generation (build order §4.1): every candidate is
generated from the parsed decision tables together with its native reference
verdict; the verifier independently recomputes from the same pinned runtime.
Families are distinguished by mechanism (single table vs DRG chain vs
aggregation), goal shape (exact output / rule audit / repair / synthesis) and
artifact mutation class - not by renames.

Regime coverage is honest: repair (f03/f04/f06), application (f01/f02/f05/f07),
construction (f08). The organization regime is N/A for this lane: the corpus
holds no genuinely distinct upstream sources of aligned DMN rule views, and
splitting one file in two is forbidden by the build order.
"""
from __future__ import annotations

from dataclasses import dataclass

LANE_ID = "L03"
SEMANTIC_PROFILE = "camunda-dmn-1.1-table-static/1"
GENERATOR_VERSION = "l03-families/1.0.0"

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
        "l03-f01-exact-decision-output", REGIME_APPLICATION, "exact_output",
        "Evaluate the pinned DMN decision table for the given typed inputs and produce the exact decision output value.",
        10, 50,
    ),
    FamilySpec(
        "l03-f02-rule-match-audit", REGIME_APPLICATION, "rule_audit",
        "Produce the exact ordered set of decision-table rule ids that match the given typed inputs under the table's hit policy.",
        10, 50,
    ),
    FamilySpec(
        "l03-f03-rule-entry-repair", REGIME_REPAIR, "rule_repair",
        "The decision table returns wrong outputs for some cases; repair the single faulty rule condition so the whole case suite passes.",
        10, 50,
    ),
    FamilySpec(
        "l03-f04-hit-policy-repair", REGIME_REPAIR, "hit_policy_repair",
        "The decision table's hit policy was changed and now returns wrong-shaped results; restore the hit policy so every case in the suite matches the intended result.",
        10, 50,
    ),
    FamilySpec(
        "l03-f05-cross-table-chain-eval", REGIME_APPLICATION, "chain_eval",
        "Evaluate the terminal decision of a multi-table DMN requirements graph, propagating intermediate decision outputs, and produce its exact output.",
        10, 50,
    ),
    FamilySpec(
        "l03-f06-cross-table-binding-repair", REGIME_REPAIR, "binding_repair",
        "A decision no longer receives its required decision's output (broken variable binding); restore the cross-table binding so the decision chain evaluates to the intended results.",
        10, 50,
    ),
    FamilySpec(
        "l03-f07-collect-aggregation-result", REGIME_APPLICATION, "collect_result",
        "Compute the exact COLLECT result (list or aggregate) of the decision table for inputs that match multiple rules.",
        10, 50,
    ),
    FamilySpec(
        "l03-f08-hidden-rule-input-synthesis", REGIME_CONSTRUCTION, "input_synthesis",
        "Only the table schema, rule outputs and a probe budget are visible; construct a typed input assignment that makes the decision return the target output.",
        10, 50,
    ),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}


def complexity_of_table(n_rules: int, n_inputs: int, n_outputs: int = 1) -> str:
    if n_rules > 10 or n_inputs >= 3 or n_outputs >= 3:
        return "high"
    if n_rules >= 5 or n_inputs == 2 or n_outputs == 2:
        return "medium"
    return "low"


def complexity_of_chain(n_decisions: int) -> str:
    if n_decisions >= 3:
        return "high"
    return "medium"
