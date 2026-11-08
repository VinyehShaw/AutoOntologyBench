"""L10 task families: 13 structural families over the pinned BEHAVIOR-1K BDDL.

All tasks are strictly symbolic (static predicate/arity/binding checks and
closed-world goal evaluation over ``:init`` atom sets). No simulator, no
physics, no embodied execution is involved or claimed.

Family mechanisms (each a distinct operation/goal/logic structure):
  repair (exact fragment restoration, gold = pinned original bytes):
    f01 goal-predicate   — one goal leaf's predicate replaced by an unknown token
    f02 goal-arity       — one goal leaf's last argument dropped
    f03 goal-binding     — one goal ``?instance`` reference replaced by an
                           undeclared variant of the same name
    f04 init-relation    — one :init atom's predicate replaced by an unknown token
    f05 quantifier-cat   — one quantifier iterable category replaced by an
                           undeclared token (restorable via the ``?var`` prefix
                           convention)
  application (exact multi-element audit artifacts):
    f06 obligation-query — exact ground obligation-atom set of an and/forall
                           sub-goal (leg A expansion; leg B upstream evaluation)
    f07 conjunct-truth   — exact set of goal conjuncts true under a perturbed
                           closed-world init (both legs)
    f13 defect-audit     — exact (section, code) finding multiset of a
                           multi-defect corrupted problem (leg A checker)
  organization (exact mappings/groupings):
    f08 init-grouping    — group init atoms by predicate (exact mapping)
    f09 objects-audit    — exact set of instances whose declared category
                           violates the instance-naming convention
    f12 category-join    — exact mapping goal-referenced category -> partner
                           activity's instances of that category
  construction (criterion-exact, both legs):
    f10 goal-construction— build a goal expression true in state S1, false in S2
    f11 minimal-state    — build a minimal ground-atom state satisfying a goal
"""
from __future__ import annotations

from dataclasses import dataclass

LANE_ID = "L10"
SEMANTIC_PROFILE = "behavior-bddl-symbolic/1"
GENERATOR_VERSION = "l10-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"
REGIME_CONSTRUCTION = "construction"
REGIME_APPLICATION = "application"

INSTANCES_PER_GOAL = 5

#: Base-goal quotas: 13 families, 120 base goals, 5 instances each -> 600.
BASE_GOAL_QUOTA = {
    "l10-f01-goal-predicate-repair": 10,
    "l10-f02-goal-arity-repair": 9,
    "l10-f03-goal-binding-repair": 9,
    "l10-f04-init-relation-repair": 9,
    "l10-f05-quantifier-category-repair": 9,
    "l10-f06-obligation-atom-query": 10,
    "l10-f07-conjunct-truth-audit": 9,
    "l10-f08-init-relation-organization": 9,
    "l10-f09-objects-declaration-audit": 9,
    "l10-f10-goal-construction": 9,
    "l10-f11-minimal-state-construction": 9,
    "l10-f12-cross-activity-category-join": 9,
    "l10-f13-multi-defect-structure-audit": 10,
}

BUDGETS = {
    "low": {"actions": 6, "probes": 4},
    "medium": {"actions": 10, "probes": 8},
    "high": {"actions": 16, "probes": 12},
}

ALLOWED_TOOLS = {
    REGIME_REPAIR: ["bddl_text_edit", "bddl_domain_reference"],
    REGIME_APPLICATION: ["bddl_static_analysis"],
    REGIME_ORGANIZATION: ["bddl_static_analysis"],
    REGIME_CONSTRUCTION: ["bddl_text_edit"],
}

SYMBOLIC_ONLY_NOTE = (
    "Evaluation is symbolic only: BDDL s-expression structure and closed-world "
    "semantics over the given :init atom set. No simulator, physics, or embodied "
    "execution is involved; nothing here claims embodied task success."
)


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    goal_kind: str
    intent: str
    base_goals: int

    @property
    def instances(self) -> int:
        return self.base_goals * INSTANCES_PER_GOAL


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l10-f01-goal-predicate-repair", REGIME_REPAIR, "repair_predicate",
        "Restore the goal atom's predicate token (replaced by an unknown token) "
        "so the problem passes every domain check and the goal matches the "
        "activity's intended semantics.", 10),
    FamilySpec(
        "l10-f02-goal-arity-repair", REGIME_REPAIR, "repair_arity",
        "Restore the missing argument of the goal atom whose argument list was "
        "truncated, so predicate arity and goal structure are valid again.", 9),
    FamilySpec(
        "l10-f03-goal-binding-repair", REGIME_REPAIR, "repair_binding",
        "Restore the goal's object reference (replaced by an undeclared variant "
        "name) so every goal argument binds to a declared object instance.", 9),
    FamilySpec(
        "l10-f04-init-relation-repair", REGIME_REPAIR, "repair_init_predicate",
        "Restore the :init relation's predicate token (replaced by an unknown "
        "token) so the initial state is well-formed again.", 9),
    FamilySpec(
        "l10-f05-quantifier-category-repair", REGIME_REPAIR, "repair_quantifier_category",
        "Restore the quantifier's iterable category (replaced by an undeclared "
        "token); the ?variable naming convention identifies the declared "
        "category.", 9),
    FamilySpec(
        "l10-f06-obligation-atom-query", REGIME_APPLICATION, "obligation_query",
        "Compute the exact set of ground atoms the sub-goal obligates (unique "
        "for and/forall goals): every atom that must hold for the goal to be "
        "satisfied.", 10),
    FamilySpec(
        "l10-f07-conjunct-truth-audit", REGIME_APPLICATION, "conjunct_truth_audit",
        "Audit which goal conjuncts already hold under the given closed-world "
        "initial state; return the exact index set of true conjuncts.", 9),
    FamilySpec(
        "l10-f08-init-relation-organization", REGIME_ORGANIZATION, "init_grouping",
        "Organize the initial-state atoms by relation type: produce the exact "
        "predicate -> atom-list grouping.", 9),
    FamilySpec(
        "l10-f09-objects-declaration-audit", REGIME_ORGANIZATION, "objects_audit",
        "Audit the shuffled object declarations and return the exact set of "
        "instances whose declared category violates the instance-naming "
        "convention (instance = category + '_' + index).", 9),
    FamilySpec(
        "l10-f10-goal-construction", REGIME_CONSTRUCTION, "goal_construction",
        "Construct a BDDL goal expression over the declared objects that is "
        "satisfied in state S1 but not in state S2.", 9),
    FamilySpec(
        "l10-f11-minimal-state-construction", REGIME_CONSTRUCTION, "minimal_state",
        "Construct a minimal set of ground atoms over the declared objects that "
        "satisfies the given goal (removing any single atom must break it).", 9),
    FamilySpec(
        "l10-f12-cross-activity-category-join", REGIME_ORGANIZATION, "category_join",
        "Join the goal's referenced categories against a partner activity's "
        "object declarations: exact category -> partner-instance-set mapping.", 9),
    FamilySpec(
        "l10-f13-multi-defect-structure-audit", REGIME_APPLICATION, "defect_audit",
        "Audit the corrupted problem and return the exact multiset of "
        "(section, finding-code) defects produced by the domain's static "
        "checks.", 10),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

assert sum(f.base_goals for f in FAMILIES) == 120
assert len(FAMILIES) == 13
