"""L06 task families: 17 structural families over the pinned GO + GOA snapshot.

Regime rationale (Build-order §3.4): GO's native semantics are subsumption +
relation closures, so families are ``application`` (exact answer-set queries
over a GO-derived fragment), ``repair`` (a corrupted fragment whose reference
closure must be restored), and ``organization`` (GAF annotation tables joined
onto the ontology hierarchy — two genuinely different upstream sources:
GO Consortium ontology release vs EMBL-EBI GOA annotation archive).
No ``construction`` family: the pinned snapshot is closed; there is no honest
hidden-structure discovery claim.

``min_answer`` / ``max_fraction`` are qualification gates applied to the
reference answer computed *within the task fragment/universe*.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    min_answer: int
    description: str


FAMILY_TABLE: tuple[FamilySpec, ...] = (
    FamilySpec("l06.isa_ancestors_exact", REGIME_APPLICATION, "low", 12, 5, 3,
               "exact is_a ancestor set of a term within the fragment"),
    FamilySpec("l06.isa_descendants_exact", REGIME_APPLICATION, "low", 12, 5, 3,
               "exact is_a descendant set of a term within the fragment"),
    FamilySpec("l06.isa_common_ancestors", REGIME_APPLICATION, "medium", 10, 5, 2,
               "shared is_a ancestors of two terms"),
    FamilySpec("l06.isa_lowest_common_ancestor", REGIME_APPLICATION, "high", 10, 5, 1,
               "exact lowest-common-ancestor set of two terms"),
    FamilySpec("l06.partof_ancestors_closure", REGIME_APPLICATION, "medium", 12, 5, 2,
               "part_of closure ancestors under go-rel-closure/1 (R1-R4)"),
    FamilySpec("l06.regulates_closure", REGIME_APPLICATION, "medium", 12, 5, 2,
               "regulators of a term under go-rel-closure/1 (R5-R7, non-transitive)"),
    FamilySpec("l06.cross_isa_partof_intersection", REGIME_APPLICATION, "high", 10, 5, 2,
               "is_a descendants of A that also stand in part_of closure to B"),
    FamilySpec("l06.branch_difference", REGIME_APPLICATION, "medium", 10, 5, 3,
               "descendants of A minus descendants of its sub-branch B"),
    FamilySpec("l06.depth_band_terms", REGIME_APPLICATION, "low", 12, 5, 5,
               "terms at fragment-local is_a depth within [lo, hi]"),
    FamilySpec("l06.multi_parent_terms", REGIME_APPLICATION, "high", 10, 5, 2,
               "terms with >=2 distinct is_a parents inside the fragment"),
    FamilySpec("l06.repair_isa_parent_swap", REGIME_REPAIR, "high", 12, 5, 1,
               "one is_a parent edge swapped for a wrong parent; restore the reference closure"),
    FamilySpec("l06.repair_missing_isa_edge", REGIME_REPAIR, "high", 12, 5, 1,
               "one is_a edge removed; restore the lost entailments"),
    FamilySpec("l06.repair_relation_type_confusion", REGIME_REPAIR, "high", 10, 5, 1,
               "part_of edge relabelled as regulates (or vice versa); restore the predicate"),
    FamilySpec("l06.repair_orphan_reconnect", REGIME_REPAIR, "high", 10, 5, 1,
               "all is_a parents of a term removed; reconnect the true parent set"),
    FamilySpec("l06.org_align_gene_aspect_annotations", REGIME_ORGANIZATION, "medium", 10, 5, 3,
               "re-align a deranged gene -> direct-annotation term-set table (GAF x GO)"),
    FamilySpec("l06.org_annotation_scope_closure", REGIME_ORGANIZATION, "medium", 8, 5, 3,
               "full propagated annotation set of a gene under go-rel-closure/1 (GAF + GO)"),
    FamilySpec("l06.org_shared_ancestors_multigene", REGIME_ORGANIZATION, "high", 8, 5, 2,
               "non-root biological-process ancestors shared by three genes (GAF + GO)"),
)

BY_ID = {f.family_id: f for f in FAMILY_TABLE}

QUOTA_QUALIFIED = 900
QUOTA_BASE_GOALS = 180
EXPECTED_INSTANCES = sum(f.base_goals * f.instances_per_goal for f in FAMILY_TABLE)

FRAGMENT_CAP = {"low": 220, "medium": 350, "high": 500}
