"""Split-drop arithmetic and the stratified family-leave-one-out proxy."""
from __future__ import annotations

import pytest

from benchmark.eval_20k.metrics_ood import (
    TaskOutcome,
    compute_split_drop,
    family_leave_one_out,
)


def _outcome(task_id, lane, family, success, status="SCORED",
             lineage=None):
    return TaskOutcome(
        task_id=task_id, lane_id=lane, family_id=family,
        lineage_group=lineage or family, status=status, success=success)


def test_split_drop_uses_iid_as_reference_and_keeps_denominators():
    outcomes = [
        _outcome("i1", "L", "f", True), _outcome("i2", "L", "f", False),
        _outcome("c1", "L", "f", False), _outcome("c2", "L", "f", False),
        _outcome("t1", "L", "f", True), _outcome("t2", "L", "f", False),
        _outcome("m1", "L", "f", False, status="MISSING"),
    ]
    splits = {"i1": "iid", "i2": "iid", "c1": "comp", "c2": "comp",
              "t1": "transfer", "t2": "transfer", "m1": "transfer"}

    result = compute_split_drop(outcomes, splits)

    assert result["per_split"]["iid"] == {
        "tasks": 2, "scored": 2, "success": 1, "rate": 0.5}
    assert result["per_split"]["comp"]["rate"] == 0.0
    assert result["per_split"]["transfer"]["scored"] == 2
    assert result["ood_drops"]["comp"] == pytest.approx(0.5)
    assert result["ood_drops"]["transfer"] == pytest.approx(0.0)
    assert result["unlabelled_tasks"] == 0


def test_split_drop_reports_unlabelled_and_missing_iid():
    result = compute_split_drop(
        [_outcome("a", "L", "f", True)], {})

    assert result["unlabelled_tasks"] == 1
    assert result["ood_drops"]["comp"] is None


def test_family_leave_one_out_is_lane_stratified_and_excludes_unscored():
    outcomes = [
        _outcome("a1", "L1", "A", True), _outcome("a2", "L1", "A", False),
        _outcome("a3", "L1", "A", False, status="MISSING"),
        _outcome("b1", "L1", "B", True),
        _outcome("d1", "L2", "D", False),
        _outcome("e1", "L2", "E", False), _outcome("e2", "L2", "E", False),
        _outcome("f1", "L3", "F", True),
    ]

    result = family_leave_one_out(outcomes, unit="family_id")

    assert result["units_evaluated"] == 4
    assert result["tasks_evaluated"] == 6
    assert result["excluded_tasks"] == 1
    assert result["tasks_not_evaluable"] == 1
    assert result["unweighted_mean_drop"] == pytest.approx(0.0)
    assert result["task_weighted_mean_drop"] == pytest.approx(0.5 / 6)
    assert result["macro_lane_mean_drop"] == pytest.approx(0.0)
    assert result["per_lane"]["L3"]["evaluated"] is False
    records = {row["unit"]: row for row in result["records"]}
    assert records["A"]["drop"] == pytest.approx(0.5)
    assert records["B"]["drop"] == pytest.approx(-0.5)
    assert records["D"]["drop"] == pytest.approx(0.0)


def test_family_leave_one_out_can_group_by_lineage():
    outcomes = [
        _outcome("a1", "L1", "A", True, lineage="G1"),
        _outcome("b1", "L1", "B", False, lineage="G2"),
    ]

    result = family_leave_one_out(outcomes, unit="lineage_group")

    assert result["unit"] == "lineage_group"
    assert result["units_evaluated"] == 2
    assert {row["unit"] for row in result["records"]} == {"G1", "G2"}


def test_family_leave_one_out_rejects_unknown_unit():
    with pytest.raises(ValueError, match="family_id or lineage_group"):
        family_leave_one_out([], unit="lane_id")