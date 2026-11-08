"""Aggregation math: TSR, V-TSR, unsafe, cost, breakdowns."""
from __future__ import annotations

from benchmark.eval_20k.scoring import (
    STATUS_ERROR,
    STATUS_MISSING,
    STATUS_NOT_SUPPORTED,
    STATUS_SCORED,
    TaskResult,
    aggregate,
)


def _r(task_id, lane="L05", family="l05.f", status=STATUS_SCORED,
       success=True, valid=True, unsafe=0, unchecked=False, costs=None):
    return TaskResult(
        task_id=task_id, lane_id=lane, family_id=family, status=status,
        task_success=success if status == STATUS_SCORED else None,
        valid_task_success=valid if status == STATUS_SCORED and not unchecked else None,
        unsafe_action_count=unsafe if status == STATUS_SCORED and not unchecked else None,
        unsafe_unchecked=unchecked, costs=costs or {},
    )


def test_aggregate_counts_and_rates():
    results = [
        _row("t1", success=True),
        _row("t2", success=False),
        _row("t3", status=STATUS_MISSING),
        _row("t4", status=STATUS_NOT_SUPPORTED),
        _row("t5", status=STATUS_ERROR),
    ]
    metrics = aggregate(results)
    overall = metrics["overall"]
    assert overall["scored"] == 2
    assert overall["success"] == 1
    assert overall["missing"] == 1
    assert overall["not_supported"] == 1
    assert overall["error"] == 1
    # TSR: missing counts as failure; not_supported/error excluded
    assert overall["tsr"] == 1 / 3
    assert overall["tsr_denominator"] == 3


def _row(task_id, **kw):
    return _r(task_id, **kw)


def test_v_tsr_excludes_unchecked_actions():
    results = [
        _row("t1", success=True),
        _row("t2", success=True, unchecked=True),  # actions submitted, unchecked
    ]
    overall = aggregate(results)["overall"]
    assert overall["unsafe_unchecked"] == 1
    assert overall["v_tsr"] == 1.0  # only t1 in the V-TSR denominator
    assert overall["verified_success"] == 1


def test_unsafe_rate_and_counts():
    results = [
        _row("t1", success=True, valid=False, unsafe=2),
        _row("t2", success=True),
    ]
    overall = aggregate(results)["overall"]
    assert overall["unsafe_rows"] == 1
    assert overall["unsafe_action_count"] == 2
    assert overall["unsafe_rate"] == 0.5
    assert overall["v_tsr"] == 0.5


def test_cost_totals_and_coverage():
    results = [
        _row("t1", costs={"tokens_input": 100.0, "tokens_output": 5.0}),
        _row("t2", costs={"tokens_input": 50.0}),
    ]
    overall = aggregate(results)["overall"]
    assert overall["costs"]["tokens_input"]["total"] == 150.0
    assert overall["costs"]["tokens_input"]["rows"] == 2
    assert overall["costs"]["tokens_output"]["rows"] == 1


def test_per_lane_and_family_breakdown():
    results = [
        _r("t1", lane="L05", family="l05.a", success=True),
        _r("t2", lane="L05", family="l05.b", success=False),
        _r("t3", lane="L12", family="l12.a", success=True),
    ]
    metrics = aggregate(results)
    assert metrics["per_lane"]["L05"]["tsr"] == 0.5
    assert metrics["per_lane"]["L12"]["tsr"] == 1.0
    assert metrics["per_family"]["l05.a"]["tsr"] == 1.0
    assert metrics["per_family"]["l05.b"]["tsr"] == 0.0


def test_empty_bucket_rates_are_none():
    overall = aggregate([])["overall"]
    assert overall["tsr"] is None
    assert overall["v_tsr"] is None
    assert overall["unsafe_rate"] is None
