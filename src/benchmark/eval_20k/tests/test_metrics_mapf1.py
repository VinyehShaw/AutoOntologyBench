"""Focused tests for the offline mapping micro/macro-F1 metric."""
from __future__ import annotations

from benchmark.eval_20k.mapf1 import (
    APPLICABLE_FAMILIES,
    TaskMapping,
    add_task,
    decode_mapping,
    finalize,
    new_bucket,
    score_pairs,
)

FAMILY = {s.family_id: s for s in APPLICABLE_FAMILIES}


def _pairs(*items):
    return frozenset(items)


def test_normal_mapping_scores_micro_f1_one():
    spec = FAMILY["l12-f04-model-transition-relabel"]
    gold, reason = decode_mapping(
        {"mapping": {"t1": "Create Fine", "t2": "Payment"}}, spec)
    pred, reason2 = decode_mapping(
        {"mapping": {"t1": "Create Fine", "t2": "Payment"}}, spec)
    assert reason == reason2 == ""
    assert gold == _pairs(("t1", "Create Fine"), ("t2", "Payment"))
    scored = score_pairs(gold, pred)
    assert scored == {"tp": 2, "fp": 0, "fn": 0,
                      "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_partial_mapping_counts_wrong_value_as_fp_and_fn():
    gold = _pairs(("a", "1"), ("b", "2"), ("c", "3"))
    pred = _pairs(("a", "1"), ("b", "X"), ("d", "4"))
    scored = score_pairs(gold, pred)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 2, 2)
    assert scored["precision"] == 1 / 3
    assert scored["recall"] == 1 / 3
    assert scored["f1"] == 2 / 6


def test_list_valued_mapping_expands_to_one_pair_per_element():
    spec = FAMILY["l05.org_align_group_molecule_sets"]
    pairs, reason = decode_mapping(
        {"alignment": {"Nitro": ["CHEBI:1", "CHEBI:2"], "Acid": "CHEBI:3"}},
        spec)
    assert reason == ""
    assert pairs == _pairs(
        ("Nitro", "CHEBI:1"), ("Nitro", "CHEBI:2"), ("Acid", "CHEBI:3"))


def test_pair_encoded_mapping_decodes():
    spec = FAMILY["l16.con_ec_rhea_catalyst_map"]
    pairs, reason = decode_mapping(["1.1.1.1|RHEA:1", " 2.2.2.2 | RHEA:2 "], spec)
    assert reason == ""
    assert pairs == _pairs(("1.1.1.1", "RHEA:1"), ("2.2.2.2", "RHEA:2"))


def test_flat_set_is_na_never_zero_filled():
    spec = FAMILY["l16.con_ec_rhea_catalyst_map"]
    pairs, reason = decode_mapping(["RHEA:1", "RHEA:2"], spec)
    assert pairs is None
    assert reason == "not_a_pair_list"

    bucket = new_bucket()
    add_task(bucket, TaskMapping(None, None, "not_a_pair_list"))
    result = finalize(bucket)
    assert result["scored_tasks"] == 0
    assert result["tp"] == result["fp"] == result["fn"] == 0
    assert result["f1"] is None
    assert result["coverage"] is None  # no oracle mapping tasks either
    assert result["excluded"] == {"not_a_pair_list": 1}


def test_prediction_error_is_excluded_but_gold_still_counted():
    bucket = new_bucket()
    add_task(bucket, TaskMapping(_pairs(("a", "1")), None, "prediction_error"))
    result = finalize(bucket)
    assert result["oracle_mapping_tasks"] == 1
    assert result["scored_tasks"] == 0
    assert result["coverage"] == 0.0
    assert result["tp"] == 0 and result["fn"] == 0
    assert result["excluded"] == {"prediction_error": 1}