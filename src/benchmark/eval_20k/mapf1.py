"""Mapping primitives for the offline 20K MapF1 metric.

A mapping is a set of ``(key, value)`` pairs.  This module holds the family
table (which families expose a real mapping on both sides), the decoders for
the two encodings in the 20K payloads, and the TP/FP/FN aggregation.  It does
no I/O, so it is safe to import and test without any run artifacts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FamilySpec:
    """How one mapping family encodes its prediction and oracle reference."""

    lane_id: str
    family_id: str
    answer_keys: tuple[str, ...]
    gold_path: tuple[str, ...]
    pair_separator: str = ""
    note: str = ""


#: Families with a genuine key/value mapping on both sides.  ``gold_path`` is
#: a path inside the oracle payload; the gold is always the oracle's own
#: reference, never recomputed from source material.
APPLICABLE_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("L05", "l05.org_align_group_molecule_sets",
               ("alignment",), ("reference_submission", "alignment"),
               note="label -> molecule-id set"),
    FamilySpec("L06", "l06.org_align_gene_aspect_annotations",
               ("alignment",), ("reference_submission", "alignment"),
               note="gene -> GO-term set"),
    FamilySpec("L07", "l07-f03-trajectory-shape-organization",
               ("assignment",), ("reference_submission", "assignment"),
               note="species -> shape class"),
    FamilySpec("L07", "l07-f09-species-role-organization",
               ("assignment",), ("reference_submission", "assignment"),
               note="species -> role"),
    FamilySpec("L12", "l12-f04-model-transition-relabel",
               ("mapping",), ("reference_terminal", "mapping"),
               note="transition id -> label"),
    FamilySpec("L12", "l12-f16-dual-transition-relabel",
               ("mapping",), ("reference_terminal", "mapping"),
               note="transition id -> label"),
    FamilySpec("L16", "l16.con_ec_rhea_catalyst_map", (),
               ("expected_answer",), pair_separator="|",
               note="EC -> RHEA pairs, encoded 'EC|RHEA'"),
    FamilySpec("L04", "l04.app_directional_children_of_master", (),
               ("expected_answer",), pair_separator="=",
               note="direction code -> RHEA, encoded 'CODE=RHEA:id'"),
    FamilySpec("L04", "l04.rep_direction_assignment_repair", (),
               ("expected_answer",), pair_separator="=",
               note="direction code -> RHEA, encoded 'CODE=RHEA:id'"),
)

#: Mapping-like families that still fail the both-sides-mapping requirement.
NOT_APPLICABLE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("L10", "l10-f12-cross-activity-category-join",
     "oracle exposes no reference mapping; the verifier derives the gold "
     "mapping from pinned BDDL sources at verify time"),
    ("L10", "l10-f22-cross-activity-predicate-join",
     "oracle exposes no reference mapping; the verifier derives the gold "
     "mapping from pinned BDDL sources at verify time"),
    ("L10", "l10-f08-init-relation-organization",
     "oracle exposes no reference grouping; the verifier derives it from "
     "pinned BDDL sources at verify time"),
    ("L10", "l10-f15-conjunct-shape-census",
     "oracle exposes no reference census mapping; the verifier derives it "
     "from pinned BDDL sources at verify time"),
    ("L10", "l10-f16-conjunct-reference-closure",
     "oracle exposes no reference closure mapping; the verifier derives it "
     "from pinned BDDL sources at verify time"),
    ("L10", "l10-f24-obligation-provenance-partition",
     "oracle exposes no reference partition mapping; the verifier derives it "
     "from pinned BDDL sources at verify time"),
    ("L10", "l10-f25-predicate-usage-matrix",
     "oracle exposes no reference usage matrix; the verifier derives it from "
     "pinned BDDL sources at verify time"),
    ("L13", "l13-f06-cross-api-schema-mapping",
     "prediction is a migrated schema instance, not a mapping; the oracle's "
     "field mapping is a constraint on that instance"),
    ("L14", "l14.org_uniprot_target_map",
     "oracle answer is a flat value set for one key supplied in params, not a "
     "mapping output"),
    ("L15", "l15-f10-uniprot-mapping",
     "oracle and prediction are flat id sets; the family name is the only "
     "mapping-like signal"),
    ("L15", "l15-f21-coreaction-pairs",
     "oracle and prediction are flat accession sets, not key/value pairs"),
    ("L09", "l09-f09-conceptmap-source-membership-repair",
     "oracle stores only reference_artifact_sha256, not a reference ConceptMap "
     "mapping to compare against"),
    ("L04", "l04.con_reaction_profile_assembly",
     "oracle answer encodes fixed-width 'child|DIR|RHEA' records with an "
     "ambiguous key/value split"),
    ("L04", "l04.org_participant_rows_involving_chebi",
     "oracle answer encodes fixed-width reaction/participant rows with an "
     "ambiguous key/value split"),
)

DEFINITIONS: dict[str, str] = {
    "mapping": "a set of (key, value) pairs; list values expand to one pair per element",
    "normalization": "keys and values are whitespace-stripped; no case folding; duplicate pairs collapse",
    "applicability": "scored only when the prediction payload AND the oracle reference both decode to a mapping with >=1 gold pair",
    "micro_f1": "2*TP/(2*TP+FP+FN) over all scored pairs",
    "macro_f1": "unweighted mean of per-family micro-F1 over families with >=1 scored task",
    "macro_f1_unit": "family",
    "coverage": "scored_tasks / oracle_mapping_tasks (excluded prediction rows never enter TP/FP/FN)",
}

SPECS = {(s.lane_id, s.family_id): s for s in APPLICABLE_FAMILIES}


def _norm(value: str) -> str:
    return value.strip()


def _container(value: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if isinstance(value.get(key), dict):
            return value[key]
    return None


def decode_mapping(value: Any, spec: FamilySpec) -> tuple[frozenset | None, str]:
    """Decode one side into a pair set, or ``(None, reason)`` for N/A."""
    if spec.pair_separator:
        if not isinstance(value, list):
            return None, "not_a_pair_list"
        pairs: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, str) or spec.pair_separator not in item:
                return None, "not_a_pair_list"
            key, val = item.split(spec.pair_separator, 1)
            if not key.strip() or not val.strip():
                return None, "not_a_pair_list"
            pairs.add((_norm(key), _norm(val)))
        return frozenset(pairs), ""
    node = _container(value, spec.answer_keys)
    if node is None:
        return None, "not_a_mapping_object"
    pairs = set()
    for key, raw in node.items():
        values = raw if isinstance(raw, list) else [raw]
        if not values or not all(isinstance(v, str) for v in values):
            return None, "not_a_mapping_object"
        for val in values:
            pairs.add((_norm(str(key)), _norm(val)))
    return frozenset(pairs), ""


def score_pairs(gold: frozenset, pred: frozenset) -> dict[str, Any]:
    """TP/FP/FN and P/R/F1 for two pair sets (empty denominators are null)."""
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    denom = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / len(pred) if pred else None,
        "recall": tp / len(gold) if gold else None,
        "f1": (2 * tp / denom) if denom else None,
    }


@dataclass(frozen=True)
class TaskMapping:
    gold: frozenset | None
    pred: frozenset | None
    reason: str = ""


def new_bucket() -> dict[str, Any]:
    return {"family_tasks": 0, "oracle_mapping_tasks": 0, "scored_tasks": 0,
            "tp": 0, "fp": 0, "fn": 0, "excluded": Counter()}


def add_task(bucket: dict[str, Any], task: TaskMapping) -> None:
    bucket["family_tasks"] += 1
    if task.gold is not None:
        bucket["oracle_mapping_tasks"] += 1
    if task.gold is None or task.pred is None:
        bucket["excluded"][task.reason] += 1
        return
    scored = score_pairs(task.gold, task.pred)
    bucket["scored_tasks"] += 1
    for key in ("tp", "fp", "fn"):
        bucket[key] += scored[key]


def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    denom = 2 * tp + fp + fn
    return {
        "family_tasks": bucket["family_tasks"],
        "oracle_mapping_tasks": bucket["oracle_mapping_tasks"],
        "scored_tasks": bucket["scored_tasks"],
        "coverage": (bucket["scored_tasks"] / bucket["oracle_mapping_tasks"]
                     if bucket["oracle_mapping_tasks"] else None),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "f1": (2 * tp / denom) if denom else None,
        "excluded": dict(sorted(bucket["excluded"].items())),
    }