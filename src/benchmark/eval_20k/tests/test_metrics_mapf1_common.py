from __future__ import annotations

import pytest

from benchmark.eval_20k.metrics_mapf1_common import COMMON_FAMILIES, build_report, compute_common7


def _row(tp: int, fp: int, fn: int, tasks: int = 10, scored: int = 10) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "family_tasks": tasks,
        "scored_tasks": scored,
    }


def _family_block(**overrides: dict) -> dict:
    data = {family: _row(1, 0, 0) for family in COMMON_FAMILIES}
    data.update(overrides)
    return data


def test_common7_aggregates_only_the_common_families() -> None:
    per_family = {
        family: _row(0, 0, 0, tasks=10, scored=0)
        for family in COMMON_FAMILIES
    }
    per_family[COMMON_FAMILIES[0]] = _row(2, 1, 1, tasks=10, scored=4)
    per_family[COMMON_FAMILIES[1]] = _row(0, 0, 1, tasks=10, scored=1)
    per_family["L04/not-common"] = _row(100, 0, 0)
    result = compute_common7(per_family)
    assert result["family_count"] == 7
    assert result["tasks"] == 70
    assert result["scored_tasks"] == 5
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert result["micro_f1"] == pytest.approx(4 / 7)


def test_common7_micro_and_macro_are_consistent() -> None:
    per_family = {
        family: _row(4, 0, 0, tasks=10, scored=10)
        for family in COMMON_FAMILIES
    }
    result = compute_common7(per_family)
    assert result["coverage"] == 1.0
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 1.0


def test_common7_zero_denominator_is_none() -> None:
    per_family = {family: _row(0, 0, 0, tasks=0, scored=0) for family in COMMON_FAMILIES}
    result = compute_common7(per_family)
    assert result["micro_f1"] is None
    assert result["macro_f1"] is None
    assert result["coverage"] is None


def test_common7_missing_family_is_rejected() -> None:
    per_family = _family_block()
    per_family.pop(COMMON_FAMILIES[3])
    with pytest.raises(KeyError):
        compute_common7(per_family)


def test_build_report_preserves_original_macro() -> None:
    block = {
        "overall": {"macro_f1": 0.25, "scored_families": 9},
        "per_family": _family_block(),
    }
    report = build_report({"models": {"m": block}})
    assert report["models"]["m"]["original_9_family_macro_f1"] == 0.25
    assert report["models"]["m"]["macro_f1"] == 1.0