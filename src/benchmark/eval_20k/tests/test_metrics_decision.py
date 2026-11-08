"""Decision-cell arithmetic and alignment for the offline FRR/MRR audit."""
from __future__ import annotations

import pytest

from benchmark.eval_20k.metrics_decision import (
    Decision,
    DecisionCell,
    compute_decision_metrics,
    decision_cells_from_maps,
    unwrap_decisions,
    unwrap_reference_decisions,
)


def test_decision_cells_align_truth_keys_and_mark_unknown():
    truth = {"o1": {"p1": "ALLOW", "p2": "DENY"}, "o2": {"p1": "ALLOW"}}
    predicted = {"o1": {"p1": "DENY"}, "o2": {"p1": "ALLOW"},
                 "o3": {"p1": "DENY"}}

    result = decision_cells_from_maps("t1", truth, predicted)

    cells = {cell.cell_key: cell for cell in result.cells}
    assert cells[("o1", "p1")].predicted is Decision.DENY
    assert cells[("o1", "p2")].predicted is None
    assert cells[("o2", "p1")].predicted is Decision.ALLOW
    assert result.spurious_cells == (("o3", "p1"),)
    assert result.predicted_cell_count == 3
    assert result.malformed_prediction is False


def test_decision_metrics_keep_unknown_out_of_known_denominators():
    cells = [
        DecisionCell("t", ("o", "a"), Decision.ALLOW, Decision.ALLOW),
        DecisionCell("t", ("o", "b"), Decision.ALLOW, Decision.DENY),
        DecisionCell("t", ("o", "c"), Decision.ALLOW, None),
        DecisionCell("t", ("o", "d"), Decision.DENY, Decision.ALLOW),
        DecisionCell("t", ("o", "e"), Decision.DENY, Decision.DENY),
        DecisionCell("t", ("o", "f"), Decision.DENY, None),
    ]

    metrics = compute_decision_metrics(cells)

    assert metrics["legal_cells"] == 3
    assert metrics["illegal_cells"] == 3
    assert metrics["false_rejections"] == 1
    assert metrics["missed_rejections"] == 1
    assert metrics["unknown_legal_cells"] == 1
    assert metrics["unknown_illegal_cells"] == 1
    assert metrics["frr_known"] == pytest.approx(1 / 2)
    assert metrics["mrr_known"] == pytest.approx(1 / 2)
    assert metrics["frr_unknown_as_blocked"] == pytest.approx(2 / 3)
    assert metrics["mrr_unknown_as_passed"] == pytest.approx(2 / 3)
    assert metrics["known_coverage"] == pytest.approx(4 / 6)


def test_invalid_truth_decision_is_rejected():
    with pytest.raises(ValueError, match="ALLOW or DENY"):
        decision_cells_from_maps("t", {"o": {"p": "MAYBE"}}, None)


def test_malformed_prediction_becomes_unknown_not_a_rejection():
    result = decision_cells_from_maps(
        "t", {"o": {"p": "ALLOW"}}, {"o": "not-a-policy-map"})

    assert result.malformed_prediction is True
    assert result.cells[0].predicted is None


def test_empty_decision_set_has_no_rates():
    metrics = compute_decision_metrics([])

    assert metrics["frr_known"] is None
    assert metrics["mrr_known"] is None
    assert metrics["known_coverage"] is None


def test_unwrap_handles_wrapped_direct_and_error_answers():
    wrapped = {"decisions": {"o": {"p": "ALLOW"}}}
    assert unwrap_decisions(wrapped) == {"o": {"p": "ALLOW"}}
    assert unwrap_decisions({"_error": "api_error"}) is None
    assert unwrap_reference_decisions(wrapped) == {"o": {"p": "ALLOW"}}
    assert unwrap_reference_decisions({"o": {"p": "DENY"}}) == {
        "o": {"p": "DENY"}}