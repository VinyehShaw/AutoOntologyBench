"""L06 20K-expansion task families: 15 NEW mechanisms over the same pinned
GO + GOA snapshot (gene-ontology@go-2026-08-05+goa238).

Every family below is a genuinely new mechanism relative to the legacy 8K
L06 families (``benchmark/plugins/task_family/l06/families.py``); none is a
rename or parameter-only variant:

  NEW relation semantics (sub-property polarity — legacy only used the
  rolled-up ``regulates`` view, never the polarity sub-properties):
    l06.posreg_closure            positively_regulates-only regulators (R6/R7, R=pos)
    l06.negreg_closure            negatively_regulates-only regulators (R6/R7, R=neg)
    l06.reg_polarity_exclusive    regulators reaching the target ONLY via the
                                  generic regulates relation (no polarity edge)
  NEW query compositions / operators:
    l06.reg_dual_target           regulators of BOTH a and b (reg∩reg) over a
                                  dual-cone universe
    l06.partof_isa_redundant_ancestors  terms that are simultaneously is_a
                                  ancestors AND part_of-closure ancestors
                                  (partonomy/subsumption overlap)
    l06.reg_subject_descendants   descendants of a root that stand in the
                                  regulates relation (asserted or R6-entailed)
                                  to some fragment term
    l06.partof_depth_band         terms whose isa∪part_of (partonomy-aware)
                                  fragment depth lies in [lo, hi] — legacy
                                  depth_band was is_a-only
  NEW repair tamper mechanisms (legacy only swapped/removed is_a edges and
  relabelled part_of↔regulates):
    l06.repair_partof_edge_deletion        a part_of edge is DELETED (legacy
                                           never deleted relation edges)
    l06.repair_spurious_isa_edge           a SPURIOUS is_a edge is ADDED
                                           (legacy never added edges)
    l06.repair_regulation_polarity_swap    positively↔negatively_regulates
                                           relabel (polarity confusion; needs
                                           the polarity-aware closure signature)
    l06.repair_spurious_partof_edge        a SPURIOUS part_of edge is ADDED
  NEW annotation-evidence mechanisms (legacy GAF joins ignored evidence codes
  and only queried gene→term direction, aspect P):
    l06.org_exp_evidence_closure            evidence-code-gated propagated
                                            annotation closure (EXP/IDA/...)
    l06.org_evidence_discriminated_ancestors  ancestors of experimentally
                                            evidenced annotations that are NOT
                                            ancestors of any non-experimental
                                            annotation of the same gene
    l06.org_term_annotated_genes            REVERSE lookup: which candidate
                                            genes are propagated-annotated to
                                            a term (term→genes, isa+partof
                                            downward universe)
    l06.org_exp_genes_for_term              reverse lookup gated by
                                            experimental evidence codes

Quotas sum to 1350 = 15 families x base_goals x 5 instances; +1 top-up
instance (deterministic next ``inst_seq`` of a filled base goal, mirroring
the legacy top-up mechanism) reaches the lane target of 1351.
"""
from __future__ import annotations

from dataclasses import dataclass

REGIME_APPLICATION = "application"
REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"

DATASET_ID_20K = "autoontobench-bank-20000-candidate-dev1"
SEED_20K = 20261101
PROFILE = "go-rel-closure/1"   # same declared profile; new operators are
                               # profile-consistent extensions documented in
                               # fragment20k.py and owlrl20k.py


@dataclass(frozen=True)
class FamilySpec20k:
    family_id: str
    regime: str
    complexity: str          # low | medium | high (structural proxy)
    base_goals: int
    instances_per_goal: int
    min_answer: int
    description: str


FAMILY_TABLE_20K: tuple[FamilySpec20k, ...] = (
    # ------------------------------------------------ application (610)
    FamilySpec20k("l06.posreg_closure", REGIME_APPLICATION, "medium", 18, 5, 1,
                  "exact set of terms that positively regulate the seed under "
                  "go-rel-closure/1 (R6/R7 with R=positively_regulates only)"),
    FamilySpec20k("l06.negreg_closure", REGIME_APPLICATION, "medium", 18, 5, 1,
                  "exact set of terms that negatively regulate the seed under "
                  "go-rel-closure/1 (R6/R7 with R=negatively_regulates only)"),
    FamilySpec20k("l06.reg_polarity_exclusive", REGIME_APPLICATION, "high", 17, 5, 1,
                  "regulators of the seed that reach it ONLY via the generic "
                  "regulates relation (no polarity edge on the path)"),
    FamilySpec20k("l06.reg_dual_target", REGIME_APPLICATION, "high", 18, 5, 2,
                  "terms that regulate BOTH target a AND target b under "
                  "go-rel-closure/1 (regulation intersection)"),
    FamilySpec20k("l06.partof_isa_redundant_ancestors", REGIME_APPLICATION, "medium", 18, 5, 2,
                  "terms that are simultaneously is_a ancestors AND part_of "
                  "closure ancestors of the seed (partonomy/subsumption overlap)"),
    FamilySpec20k("l06.reg_subject_descendants", REGIME_APPLICATION, "medium", 17, 5, 3,
                  "descendants of the root that stand in the regulates relation "
                  "(asserted or R6-entailed) to some fragment term"),
    FamilySpec20k("l06.partof_depth_band", REGIME_APPLICATION, "low", 16, 5, 5,
                  "terms whose partonomy-aware (is_a ∪ part_of) fragment depth "
                  "lies within [lo, hi] inclusive"),
    # ---------------------------------------------------- repair (375)
    FamilySpec20k("l06.repair_partof_edge_deletion", REGIME_REPAIR, "high", 19, 5, 1,
                  "a part_of edge was deleted from the fragment; restore it so "
                  "the reference closure signature is reproduced"),
    FamilySpec20k("l06.repair_spurious_isa_edge", REGIME_REPAIR, "high", 19, 5, 1,
                  "a spurious is_a edge was ADDED to the fragment; identify and "
                  "remove it to restore the reference closure signature"),
    FamilySpec20k("l06.repair_regulation_polarity_swap", REGIME_REPAIR, "high", 19, 5, 1,
                  "a regulation edge was relabelled with the wrong polarity "
                  "(positively↔negatively regulates); restore the predicate"),
    FamilySpec20k("l06.repair_spurious_partof_edge", REGIME_REPAIR, "high", 18, 5, 1,
                  "a spurious part_of edge was ADDED to the fragment; identify "
                  "and remove it to restore the reference closure signature"),
    # ---------------------------------------------- organization (365+1)
    FamilySpec20k("l06.org_exp_evidence_closure", REGIME_ORGANIZATION, "medium", 19, 5, 3,
                  "experimentally evidenced direct annotations of a gene plus "
                  "every is_a/part_of propagated ancestor (GAF evidence-code "
                  "join x GO closure)"),
    FamilySpec20k("l06.org_evidence_discriminated_ancestors", REGIME_ORGANIZATION, "high", 18, 5, 2,
                  "ancestors of the gene's experimentally evidenced annotations "
                  "that are NOT ancestors of any non-experimental annotation of "
                  "the same gene (evidence-class discrimination)"),
    FamilySpec20k("l06.org_term_annotated_genes", REGIME_ORGANIZATION, "medium", 19, 5, 2,
                  "reverse GAFxGO join: which candidate genes carry a direct or "
                  "is_a/part_of-propagated annotation to the given term"),
    FamilySpec20k("l06.org_exp_genes_for_term", REGIME_ORGANIZATION, "high", 17, 5, 2,
                  "reverse GAFxGO join gated by evidence code: which candidate "
                  "genes carry an experimentally evidenced (direct or "
                  "propagated) annotation to the given term"),
)

BY_ID_20K = {f.family_id: f for f in FAMILY_TABLE_20K}

QUOTA_QUALIFIED_20K = sum(
    f.base_goals * f.instances_per_goal for f in FAMILY_TABLE_20K)  # 1350

# +1 deterministic top-up instance (next inst_seq of one filled base goal)
# -> lane target 1351 (configs/lanes_20000.yaml L06 new_needed).
TOP_UP_20K: dict[str, int] = {"l06.org_term_annotated_genes": 1}

TARGET_NEW_20K = QUOTA_QUALIFIED_20K + sum(TOP_UP_20K.values())  # 1351

FRAGMENT_CAP_20K = {"low": 220, "medium": 350, "high": 500}

# Experimental (wet-lab / high-throughput) GAF evidence codes; the
# evidence-gated families use exactly this set, recorded in the public view.
EXPERIMENTAL_EVIDENCE_CODES: tuple[str, ...] = (
    "EXP", "IDA", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP",
)
