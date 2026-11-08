"""CPS math: normal division, zero denominator, missing fields, aggregation."""
from __future__ import annotations

from benchmark.eval_20k.metrics_cost import (
    aggregate_runs,
    compute_cps,
    summarize_run,
)


def _costs(*, tokens_in=100.0, tokens_out=50.0, api_calls=3.0,
           wall_seconds=9.0) -> dict:
    return {
        "tokens_input": {"total": tokens_in, "rows": 1},
        "tokens_output": {"total": tokens_out, "rows": 1},
        "api_calls": {"total": api_calls, "rows": 1},
        "wall_seconds": {"total": wall_seconds, "rows": 1},
    }


def test_compute_cps_normal_case():
    out = compute_cps(_costs(), verified_success=5)

    assert out["cps_tokens"] == 30.0
    assert out["cost_per_valid_success"] == 30.0
    assert out["cps_tokens_denominator"] == 5
    assert out["api_calls_per_success"] == 0.6
    assert out["wall_seconds_per_success"] == 1.8
    assert out["null_reasons"] == {}


def test_compute_cps_zero_denominator_is_null_not_zero():
    out = compute_cps(_costs(), verified_success=0)

    assert out["cps_tokens"] is None
    assert out["cost_per_valid_success"] is None
    assert out["cps_tokens_denominator"] == 0
    assert out["api_calls_per_success"] is None
    assert out["wall_seconds_per_success"] is None
    assert out["null_reasons"]["cps_tokens"] == "verified_success is zero"
    assert out["null_reasons"]["api_calls_per_success"] == (
        "verified_success is zero")


def test_compute_cps_missing_cost_field_is_null_with_reason():
    costs = _costs()
    del costs["tokens_output"]

    out = compute_cps(costs, verified_success=5)

    assert out["cps_tokens"] is None
    assert "tokens_output" in out["null_reasons"]["cps_tokens"]
    # Diagnostics with present inputs remain computable.
    assert out["api_calls_per_success"] == 0.6
    assert out["wall_seconds_per_success"] == 1.8


def _report(verified_success: int, tokens_in: float, tokens_out: float) -> dict:
    return {
        "predictions": {"model": "m1"},
        "metrics": {
            "overall": {
                "verified_success": verified_success,
                "tasks": 10,
                "scored": 10,
                "success": verified_success,
                "costs": {
                    "tokens_input": {"total": tokens_in, "rows": 10},
                    "tokens_output": {"total": tokens_out, "rows": 10},
                    "api_calls": {"total": 10.0, "rows": 10},
                    "wall_seconds": {"total": 20.0, "rows": 10},
                },
            },
        },
    }


def test_aggregate_runs_preserves_denominators():
    runs = [
        summarize_run(_report(2, 100.0, 40.0),
                      "eval_20k_full2-g1-m1_aaa"),
        summarize_run(_report(3, 200.0, 60.0),
                      "eval_20k_full2-g1-m1_bbb"),
    ]

    out = aggregate_runs(runs)

    assert out["verified_success"] == 5
    assert out["tasks"] == 20
    assert out["costs"]["tokens_input"]["total"] == 300.0
    assert out["costs"]["tokens_output"]["total"] == 100.0
    assert out["cps_tokens"] == 80.0
    assert out["cps_tokens_denominator"] == 5
    assert out["null_reasons"] == {}