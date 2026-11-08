"""L10 20K-expansion task families: 12 NEW structural families (f14..f25).

Expansion build ``aob-20000-candidate-dev1`` (dataset_id
``autoontobench-bank-20000-candidate-dev1``, seed base 20261101). These are
genuinely new mechanisms beyond the legacy 13 families (f01..f13 of
``aob-8000-rc1``): no renamed legacy family, no parameter-only variant of a
legacy defect kind. Legacy modules are untouched; this module only *reuses*
their shared tables/budgets/disclaimer.

All tasks remain strictly symbolic (static structure checks and closed-world
goal evaluation over ``:init`` atom sets). No simulator, no physics, no
embodied execution is involved or claimed.

New family mechanisms (each a distinct operation absent from f01..f13):
  organization (exact inventories / closures / partitions):
    f14 quantifier-inventory — exact [head, var, category] list of the
       outermost quantifiers of a selected sub-goal (scope extraction)
    f15 conjunct-shape-census — exact conjunct-index -> root-shape tag map
       (expression-shape classification)
    f16 conjunct-reference-closure — exact conjunct-index -> referenced-
       instance-set closure (direct ?instance refs + quantifier domains)
    f17 goal-polarity-audit — exact index set of goal leaves under negative
       polarity (negation-depth analysis)
    f22 predicate-join — exact goal-predicate -> partner-activity init-atom
       list mapping (cross-activity join on the predicate key)
    f23 room-occupancy — exact room -> instance-set grouping projected from
       shuffled ``inroom`` atoms (value-keyed grouping, not predicate-keyed)
    f24 obligation-provenance — exact obligation-atom -> generating-conjunct
       index list partition (obligation provenance)
  application (exact audit artifacts over perturbed problems):
    f20 init-contradiction-audit — exact atom set asserted both positively
       and negatively after one injected negation (consistency checking)
    f21 init-duplicate-audit — exact duplicated atom set after injected
       duplicate atoms (multiset counting)
  construction (criterion-exact, both legs):
    f18 discriminating-state — build a ground-atom state satisfying
       sub-goal A while falsifying conjunct B (state separating two goals;
       the dual direction of legacy f10)
    f19 weakening-goal — build a goal expression satisfied under BOTH S1
       and S2 (and false under the empty state) where the reference goal
       separates S1/S2 (non-separating construction)
"""
from __future__ import annotations

from dataclasses import dataclass

from benchmark.plugins.task_family.l10.families import (
    ALLOWED_TOOLS,
    BUDGETS,
    INSTANCES_PER_GOAL,
    LANE_ID,
    REGIME_APPLICATION,
    REGIME_CONSTRUCTION,
    REGIME_ORGANIZATION,
    SEMANTIC_PROFILE,
    SYMBOLIC_ONLY_NOTE,
)

GENERATOR_VERSION = "l10-families-20k/1.0.0"

#: 12 new families x 5 base goals x 5 instances = 300 new tasks (L10 quota
#: from configs/lanes_20000.yaml: target_total 900 - legacy 600 = 300).
NEW_BASE_GOAL_QUOTA = {
    "l10-f14-quantifier-inventory-audit": 5,
    "l10-f15-conjunct-shape-census": 5,
    "l10-f16-conjunct-reference-closure": 5,
    "l10-f17-goal-polarity-audit": 5,
    "l10-f18-discriminating-state-construction": 5,
    "l10-f19-weakening-goal-construction": 5,
    "l10-f20-init-contradiction-audit": 5,
    "l10-f21-init-duplicate-audit": 5,
    "l10-f22-cross-activity-predicate-join": 5,
    "l10-f23-room-occupancy-organization": 5,
    "l10-f24-obligation-provenance-partition": 5,
    "l10-f25-predicate-usage-matrix": 5,
}


@dataclass(frozen=True)
class FamilySpec20K:
    family_id: str
    regime: str
    goal_kind: str
    intent: str
    base_goals: int

    @property
    def instances(self) -> int:
        return self.base_goals * INSTANCES_PER_GOAL


NEW_FAMILIES: tuple[FamilySpec20K, ...] = (
    FamilySpec20K(
        "l10-f14-quantifier-inventory-audit", REGIME_ORGANIZATION,
        "quant_inventory",
        "List every outermost quantifier of the selected sub-goal in textual "
        "order as an exact [head, variable, iterable-category] triple list.", 5),
    FamilySpec20K(
        "l10-f15-conjunct-shape-census", REGIME_ORGANIZATION,
        "shape_census",
        "Classify every selected goal conjunct by its root expression shape "
        "(predicate atom vs operator head); submit the exact conjunct-index "
        "-> shape-tag mapping.", 5),
    FamilySpec20K(
        "l10-f16-conjunct-reference-closure", REGIME_ORGANIZATION,
        "reference_closure",
        "For every selected goal conjunct, compute the exact set of declared "
        "object instances it references (direct ?instance references plus "
        "every quantifier iterable domain).", 5),
    FamilySpec20K(
        "l10-f17-goal-polarity-audit", REGIME_ORGANIZATION,
        "polarity_audit",
        "Audit the selected goal leaves by negation polarity: submit the "
        "exact sorted index set of leaves that sit under an odd number of "
        "negations.", 5),
    FamilySpec20K(
        "l10-f18-discriminating-state-construction", REGIME_CONSTRUCTION,
        "discriminating_state",
        "Construct a ground-atom state over the declared objects that "
        "satisfies sub-goal A while falsifying conjunct B (a state that "
        "separates two goals).", 5),
    FamilySpec20K(
        "l10-f19-weakening-goal-construction", REGIME_CONSTRUCTION,
        "weakening_goal",
        "Construct a goal expression that, unlike the given reference goal, "
        "is satisfied under BOTH closed-world states S1 and S2 (and is false "
        "under the empty state).", 5),
    FamilySpec20K(
        "l10-f20-init-contradiction-audit", REGIME_APPLICATION,
        "contradiction_audit",
        "One init atom has been additionally asserted negated; find the "
        "exact set of atoms asserted both positively and negatively by "
        ":init (consistency audit).", 5),
    FamilySpec20K(
        "l10-f21-init-duplicate-audit", REGIME_APPLICATION,
        "duplicate_audit",
        "Duplicate atoms have been injected into :init; submit the exact "
        "sorted set of atom strings that occur more than once (multiset "
        "audit).", 5),
    FamilySpec20K(
        "l10-f22-cross-activity-predicate-join", REGIME_ORGANIZATION,
        "predicate_join",
        "Join every predicate used by activity A's goal against activity "
        "B's initial state: exact predicate -> sorted B-init-atom-list "
        "mapping (empty list when B's init uses none).", 5),
    FamilySpec20K(
        "l10-f23-room-occupancy-organization", REGIME_ORGANIZATION,
        "room_grouping",
        "The shuffled ``inroom`` atoms of a BDDL problem are given; "
        "organize them into the exact room -> sorted-instance-set mapping.", 5),
    FamilySpec20K(
        "l10-f24-obligation-provenance-partition", REGIME_ORGANIZATION,
        "provenance_partition",
        "Expand the selected sub-goal to ground obligation atoms and "
        "attribute each atom to the conjuncts that generate it: exact "
        "atom -> sorted generating-conjunct-index list.", 5),
    FamilySpec20K(
        "l10-f25-predicate-usage-matrix", REGIME_ORGANIZATION,
        "predicate_usage",
        "For every domain predicate the problem uses, submit the exact "
        "predicate -> sorted section list (``init`` and/or ``goal``) usage "
        "matrix.", 5),
)

NEW_FAMILY_BY_ID = {f.family_id: f for f in NEW_FAMILIES}

assert len(NEW_FAMILIES) == 12
assert sum(f.base_goals for f in NEW_FAMILIES) == 60
assert sum(f.instances for f in NEW_FAMILIES) == 300
assert not (set(NEW_FAMILY_BY_ID) & {
    "l10-f01-goal-predicate-repair", "l10-f02-goal-arity-repair",
    "l10-f03-goal-binding-repair", "l10-f04-init-relation-repair",
    "l10-f05-quantifier-category-repair", "l10-f06-obligation-atom-query",
    "l10-f07-conjunct-truth-audit", "l10-f08-init-relation-organization",
    "l10-f09-objects-declaration-audit", "l10-f10-goal-construction",
    "l10-f11-minimal-state-construction", "l10-f12-cross-activity-category-join",
    "l10-f13-multi-defect-structure-audit",
}), "new family ids must not collide with legacy family ids"

__all__ = [
    "FamilySpec20K", "NEW_FAMILIES", "NEW_FAMILY_BY_ID",
    "NEW_BASE_GOAL_QUOTA", "GENERATOR_VERSION", "INSTANCES_PER_GOAL",
    "ALLOWED_TOOLS", "BUDGETS", "LANE_ID", "SEMANTIC_PROFILE",
    "SYMBOLIC_ONLY_NOTE",
]
