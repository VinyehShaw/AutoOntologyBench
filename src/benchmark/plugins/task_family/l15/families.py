"""L15 family table: 15 task families over pinned Reactome release-98.

Truth sources are the pinned local snapshot only (never the live API):
- path B: BioPAX per-pathway OWL files parsed with rdflib (graph truth);
- path M: official mapping TSVs parsed into SQLite (independent second path);
- path S: homo_sapiens SBML (spot cross-check for reaction sides).

Regimes follow the build order's repair / organization / construction axes plus
application queries. Every family is natively verified by recomputation from
the pinned store; families whose relation exists in both paths require
path agreement (differential_agree) before qualification.
"""
from __future__ import annotations

from dataclasses import dataclass

PROFILE = "reactome-biopax/1"
LANE = "L15"
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
SEED = 20261101


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    kind: str
    regime: str
    complexity: str
    base_goals: int
    instances_per_goal: int
    min_answer: int
    max_answer: int
    truth_path: str          # "B" (BioPAX), "M" (mapping), "B+M" (both, agree)
    description: str


FAMILY_TABLE: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l15-f01-pathway-hierarchy", "pathway_children", "organization", "low",
        20, 5, 2, 30, "B+M",
        ("Exact set of direct child pathways/events of a parent pathway, "
         "reconstructed from the pathway hierarchy (relation file vs BioPAX "
         "pathwayComponent must agree)."),
    ),
    FamilySpec(
        "l15-f02-pathway-events", "pathway_events", "organization", "medium",
        20, 5, 3, 40, "B+M",
        ("Exact set of ReactionLikeEvents directly contained in a pathway "
         "(BioPAX pathwayComponent), excluding nested sub-pathway members."),
    ),
    FamilySpec(
        "l15-f03-reaction-inputs", "reaction_inputs", "application", "low",
        20, 5, 1, 12, "B+M",
        ("Exact set of input (left) PhysicalEntities of a reaction; catalysts, "
         "regulators and outputs are excluded; complexes count as one entity."),
    ),
    FamilySpec(
        "l15-f04-reaction-outputs", "reaction_outputs", "application", "low",
        20, 5, 1, 12, "B+M",
        ("Exact set of output (right) PhysicalEntities of a reaction."),
    ),
    FamilySpec(
        "l15-f05-direction-repair", "direction_repair", "repair", "high",
        20, 5, 1, 10, "B",
        ("A public reaction record had its input/output sides swapped for one "
         "participant. Repair the artifact so sides match the pinned record."),
    ),
    FamilySpec(
        "l15-f06-catalyst", "reaction_catalyst", "application", "medium",
        20, 5, 1, 6, "B+M",
        ("Exact catalyst PhysicalEntity set of a reaction (BioPAX catalysis "
         "controllers), distinguishing catalysis from regulation."),
    ),
    FamilySpec(
        "l15-f07-regulation", "reaction_regulation", "organization", "medium",
        20, 5, 1, 10, "B+M",
        ("Exact set of (regulator, controlType) pairs of a reaction, keeping "
         "ACTIVATION/INHIBITION distinct."),
    ),
    FamilySpec(
        "l15-f08-complex-membership", "complex_members", "organization", "low",
        20, 5, 2, 20, "B",
        ("Exact component set of a Complex (one level), preserving component "
         "identity (no flattening to reference molecules)."),
    ),
    FamilySpec(
        "l15-f09-compartment-repair", "compartment_repair", "repair", "medium",
        20, 5, 2, 12, "B",
        ("Entity records had compartment assignments corrupted (merged to one "
         "compartment). Restore each entity's compartment."),
    ),
    FamilySpec(
        "l15-f10-uniprot-mapping", "uniprot_events", "application", "low",
        20, 5, 2, 25, "M",
        ("Exact set of Reactome event stable ids a UniProt accession maps to "
         "at all levels (mapping file truth), name fields never used as "
         "identity."),
    ),
    FamilySpec(
        "l15-f11-modification-distinction", "modified_forms", "organization", "high",
        20, 5, 2, 15, "B",
        ("Given a reference protein, the exact set of distinct modified "
         "PhysicalEntity forms (phosphorylation etc.) in the pathway slice; "
         "unmodified and modified forms must not be merged."),
    ),
    FamilySpec(
        "l15-f12-cross-reference", "xref_consistency", "application", "medium",
        20, 5, 2, 25, "B+M",
        ("Exact set of external accessions (ChEBI/Ensembl/NCBI per task) "
         "mapping to a given event; mapping file and BioPAX xref must agree."),
    ),
    FamilySpec(
        "l15-f13-species-provenance", "species_provenance", "organization", "medium",
        20, 5, 2, 20, "M",
        ("Exact set of species in which a same-named event exists (all-levels "
         "mapping), keeping human-experiment vs inferred-from-model-organism "
         "provenance distinct."),
    ),
    FamilySpec(
        "l15-f14-pmid-provenance", "reaction_pmids", "application", "low",
        20, 5, 1, 15, "M",
        ("Exact PMID set supporting a reaction (ReactionPMIDS.txt truth)."),
    ),
    FamilySpec(
        "l15-f15-multihop", "multihop_chain", "construction", "high",
        20, 5, 1, 20, "B+M",
        ("Two-hop query: all pathways containing at least one reaction whose "
         "catalyst is a given UniProt accession (mapping + hierarchy closure)."),
    ),
    FamilySpec(
        "l15-f16-complex-pathway", "complex_pathways", "application", "low",
        20, 5, 1, 15, "M",
        ("Exact set of pathways in which a given complex (R-ALL/R-HSA id) "
         "participates, from the official Complex-to-Pathway mapping; complex "
         "identity is not flattened to components."),
    ),
    FamilySpec(
        "l15-f17-entity-literature", "entity_pmids", "application", "low",
        20, 5, 1, 20, "M",
        ("Exact PMID set cited for a UniProt accession's Reactome physical "
         "entity (UniProt_Entity_LiteratureReferences), keeping PE identity "
         "separate from the bare accession."),
    ),
    FamilySpec(
        "l15-f18-disease-flag", "disease_subset", "organization", "low",
        20, 5, 1, 12, "M",
        ("From a candidate pathway list, the exact subset flagged as disease "
         "pathways (HumanDiseasePathways.txt) — identity by stable id."),
    ),
    FamilySpec(
        "l15-f19-top-level-roots", "top_level_subset", "organization", "low",
        20, 5, 1, 12, "M",
        ("From a candidate pathway list, the exact subset that are top-level "
         "(no parent in the pathway hierarchy relation)."),
    ),
    FamilySpec(
        "l15-f20-pathway-membership", "reaction_pathways", "application", "medium",
        20, 5, 2, 8, "M",
        ("Exact set of pathways in which a reaction is a direct member "
         "(reaction-to-pathway mapping); ancestor pathways reached only through "
         "the hierarchy must not be included."),
    ),
    FamilySpec(
        "l15-f21-coreaction-pairs", "coreaction_pairs", "construction", "high",
        20, 5, 1, 20, "M",
        ("Exact set of accessions appearing as inputs in reactions where the "
         "given accession has the catalyst role (reaction-role self-join)."),
    ),
)

BY_ID = {f.family_id: f for f in FAMILY_TABLE}
assert len(FAMILY_TABLE) == 21
#: f16-f21 are mapping-only families that exist to keep the lane buildable at
#: full scale when the BioPAX archive cannot be acquired (see REPORT.md §2);
#: the primary 15 families are preferred whenever the BioPAX store exists.
PRIMARY_FAMILY_IDS = tuple(f.family_id for f in FAMILY_TABLE[:15])
FALLBACK_FAMILY_IDS = (
    "l15-f01-pathway-hierarchy", "l15-f10-uniprot-mapping",
    "l15-f12-cross-reference", "l15-f13-species-provenance",
    "l15-f14-pmid-provenance", "l15-f15-multihop",
    "l15-f16-complex-pathway", "l15-f17-entity-literature",
    "l15-f18-disease-flag", "l15-f19-top-level-roots",
    "l15-f20-pathway-membership", "l15-f21-coreaction-pairs",
)
