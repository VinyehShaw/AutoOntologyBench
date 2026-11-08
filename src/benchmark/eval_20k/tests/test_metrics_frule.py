"""Focused tests for the offline balanced F_rule primitives."""
from __future__ import annotations

import json

import pytest

from benchmark.eval_20k.frule import (
    add_task,
    classify,
    finalize,
    lane_status,
    negative_witness,
    new_bucket,
    positive_witness,
    self_check_ok,
)


def _score(outcomes: list[str]) -> dict:
    bucket = new_bucket()
    for outcome in outcomes:
        add_task(bucket, outcome)
    return finalize(bucket)


def test_balanced_formula_uses_tpr_and_tnr() -> None:
    row = _score(["positive"] * 8 + ["negative"] * 2
                 + ["positive"] * 9 + ["negative"] * 1)
    assert (row["tp"], row["fn"], row["tn"], row["fp"]) == (17, 3, 17, 3)
    assert row["tpr"] == pytest.approx(17 / 20)
    assert row["tnr"] == pytest.approx(17 / 20)
    assert row["balanced_accuracy"] == pytest.approx(0.85)
    assert row["coverage"] == pytest.approx(1.0)


def test_positive_witness_choice_scores_tp_and_tn() -> None:
    row = _score(["positive"])
    assert (row["tp"], row["tn"], row["fn"], row["fp"]) == (1, 1, 0, 0)
    assert row["balanced_accuracy"] == 1.0


def test_negative_witness_choice_scores_fn_and_fp() -> None:
    row = _score(["negative"])
    assert (row["tp"], row["tn"], row["fn"], row["fp"]) == (0, 0, 1, 1)
    assert row["balanced_accuracy"] == 0.0


def test_unknown_is_excluded_not_counted_as_either_class() -> None:
    row = _score(["positive", "positive", "unknown"])
    assert row["known"] == 2
    assert row["unknown"] == 1
    assert row["tp"] == 2 and row["fn"] == 0
    assert row["tpr"] == 1.0
    assert row["tnr"] == 1.0
    assert row["balanced_accuracy"] == 1.0
    assert row["coverage"] == pytest.approx(2 / 3)


def test_no_known_classification_is_na_not_zero() -> None:
    row = _score(["unknown", "unknown"])
    assert row["balanced_accuracy"] is None
    assert row["tpr"] is None
    assert row["tnr"] is None
    assert row["status"] == "N/A"


def test_unavailable_witness_is_blocked_and_na() -> None:
    bucket = new_bucket()
    add_task(bucket, "blocked", "oracle_negative_witness_missing")
    row = finalize(bucket)
    assert row["tasks"] == 1
    assert row["eligible"] == 0
    assert row["balanced_accuracy"] is None
    assert row["status"] == "N/A"
    assert row["blocked"] == {"oracle_negative_witness_missing": 1}


def test_classify_matches_exact_witnesses_only() -> None:
    positive = {"answer_ids": ["a", "b"]}
    negative = {"answer_ids": ["b"]}
    assert classify(positive, positive, negative)[0] == "positive"
    assert classify(negative, positive, negative)[0] == "negative"
    assert classify({"answer_ids": ["c"]}, positive, negative)[0] == "unknown"
    assert classify(None, positive, negative)[0] == "missing"
    assert classify({"_error": "boom"}, positive, negative)[0] == "error"


def test_l15_uses_stored_mutation_submission() -> None:
    payload = {
        "reference_submission": {"answer_ids": ["R1", "R2"]},
        "mutation_submission": {"answer_ids": ["R1"]},
        "self_check": {"positive": True, "negative_must_fail": False,
                       "mutation_differs": True},
    }
    assert self_check_ok("L15", payload) == (True, "")
    assert positive_witness("L15", payload)[0] == payload["reference_submission"]
    assert negative_witness("L15", payload)[0] == payload["mutation_submission"]


def test_self_check_rejects_unvalidated_negative() -> None:
    bad_l06 = {"self_check": {"positive": True, "negative_valid": True}}
    assert self_check_ok("L06", bad_l06) == (False, "oracle_self_check_failed")
    good_l07 = {"self_check": {"positive": True, "negative_must_fail": False}}
    assert self_check_ok("L07", good_l07) == (True, "")


def test_l08_repair_negative_uses_stored_broken_rule() -> None:
    payload = {
        "broken_rule": {"op": ">=", "threshold": 3.0,
                        "lookback_days": 180, "min_count": 2},
        "reference_submission": {"op": "<=", "threshold": 3.5,
                                 "lookback_days": 120, "min_count": 3},
    }
    negative, reason = negative_witness("L08", payload)
    assert reason == ""
    assert negative == {"op": ">=", "threshold": 3.0,
                        "lookback_days": 180, "min_count": 2}


def test_l08_query_negative_uses_evidence_panel() -> None:
    payload = {"reference_submission": {"answer_ids": ["p1", "p2"]}}
    evidence = {"patients": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"},
                             {"id": "p4"}]}
    negative, reason = negative_witness("L08", payload, evidence=evidence)
    assert reason == ""
    assert negative == {"answer_ids": ["p3", "p4"]}


def test_blocked_lane_has_explicit_reason() -> None:
    status, reason = lane_status("L04")
    assert status == "BLOCKED_PERTURBATION_ONLY"
    assert "no independent rejected answer" in reason
    status, _ = lane_status("L15")
    assert status == "ELIGIBLE"


def test_positive_equals_negative_is_not_a_valid_pair() -> None:
    # The scorer must not treat a degenerate pair as evidence.
    assert json.dumps({"answer_ids": ["a"]}, sort_keys=True) == json.dumps(
        {"answer_ids": ["a"]}, sort_keys=True)
    payload = {
        "reference_submission": {"answer_ids": ["a"]},
        "mutation_submission": {"answer_ids": ["a"]},
        "self_check": {"positive": True, "negative_must_fail": False,
                       "mutation_differs": True},
    }
    assert positive_witness("L15", payload)[0] == negative_witness(
        "L15", payload)[0]

def test_build_report_keeps_blocked_lane_out_of_denominator(tmp_path) -> None:
    from benchmark.eval_20k.frule_report import build_report

    run = tmp_path / "runs" / "eval_20k_full2-g1-testmodel_abc"
    oracle_root = tmp_path / "oracle"
    (oracle_root / "l15" / "oracle").mkdir(parents=True)
    (oracle_root / "l04" / "oracle").mkdir(parents=True)
    l15_payload = {
        "reference_submission": {"answer_ids": ["R1", "R2"]},
        "mutation_submission": {"answer_ids": ["R1"]},
        "self_check": {"positive": True, "negative_must_fail": False,
                       "mutation_differs": True},
    }
    l04_payload = {"expected_answer": ["GO:1"], "witness": {"kind": "x"}}
    l15_path = oracle_root / "l15" / "oracle" / "l15.json"
    l04_path = oracle_root / "l04" / "oracle" / "l04.json"
    l15_path.write_text(json.dumps(l15_payload), encoding="utf-8")
    l04_path.write_text(json.dumps(l04_payload), encoding="utf-8")
    run.mkdir(parents=True)
    rows = [
        {"task_id": "t15", "lane_id": "L15", "family_id": "l15.f",
         "oracle_path": str(l15_path)},
        {"task_id": "t04", "lane_id": "L04", "family_id": "l04.f",
         "oracle_path": str(l04_path)},
    ]
    preds = [{"task_id": "t15",
              "answer": {"answer_ids": ["R1", "R2"]}},
             {"task_id": "t04", "answer": {"answer": ["GO:1"]}}]
    (run / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
    report = build_report(
        [{"group": "g1", "model": "testmodel", "path": str(run)}],
        oracle_roots=(str(oracle_root),))
    overall = report["models"]["testmodel"]["overall"]
    assert overall["tasks"] == 2
    assert overall["eligible"] == 1
    assert overall["tp"] == 1 and overall["tn"] == 1
    assert overall["balanced_accuracy"] == 1.0
    assert report["models"]["testmodel"]["per_lane"]["L04"]["overall"][
        "balanced_accuracy"] is None
    assert report["models"]["testmodel"]["per_lane"]["L15"]["overall"][
        "balanced_accuracy"] == 1.0
