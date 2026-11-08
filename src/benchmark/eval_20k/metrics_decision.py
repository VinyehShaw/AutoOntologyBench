"""Offline per-decision FRR/MRR arithmetic for the 20K audit.

The paper's false-rejection rate (FRR) and missed-rejection rate (MRR) are
defined over *consumer decision steps*: a ground-truth-legal step that the
consumer blocks (FRR) and a ground-truth-illegal step that the consumer waves
through (MRR).  The 20K harness is answer-only; it does not record consumer
decision traces.  This module therefore does **not** compute paper FRR/MRR.
It provides the small, auditable arithmetic used by the two places where the
20K data do carry per-decision labels:

* the L02 ``exact_decision_set`` family, whose oracle publishes an
  ``ALLOW``/``DENY`` cell for every ``(object, policy)`` pair and whose
  predictions carry the same cells;
* any future decision-label export with the same shape.

The caller decides whether those cells are a faithful proxy.  Every count is
kept, including unknown/missing predicted cells and spurious predicted cells,
so a rate can never be reported without its denominator.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class Decision(StrEnum):
    """The only two values a published decision cell may take."""

    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass(frozen=True, slots=True)
class DecisionCell:
    """One ground-truth decision cell aligned to one prediction.

    ``predicted is None`` means the model supplied no usable decision for that
    cell (missing row, error marker, missing object/policy, or malformed
    value).  It is an unknown/abstain, never silently treated as a rejection.
    """

    task_id: str
    cell_key: tuple[str, str]
    truth: Decision
    predicted: Decision | None


@dataclass(frozen=True, slots=True)
class DecisionMapResult:
    """Aligned cells plus the prediction-side bookkeeping."""

    cells: tuple[DecisionCell, ...]
    spurious_cells: tuple[tuple[str, str], ...]
    predicted_cell_count: int
    malformed_prediction: bool


def parse_decision(value: object) -> Decision:
    """Parse an ``ALLOW``/``DENY`` cell value or raise :class:`ValueError`."""
    if not isinstance(value, str):
        raise ValueError("decision must be a string, got %r" % (type(value),))
    try:
        return Decision(value)
    except ValueError as exc:
        raise ValueError("decision must be ALLOW or DENY, got %r" % value) from exc


def _flatten_map(value: object, *, where: str) -> dict[tuple[str, str], Decision]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % where)
    cells: dict[tuple[str, str], Decision] = {}
    for object_key, policies in value.items():
        if not isinstance(object_key, str) or not isinstance(policies, Mapping):
            raise ValueError("%s[%r] must map policy ids to decisions"
                             % (where, object_key))
        for policy_key, decision in policies.items():
            if not isinstance(policy_key, str):
                raise ValueError("%s[%r] has a non-string policy key"
                                 % (where, object_key))
            cells[(object_key, policy_key)] = parse_decision(decision)
    return cells


def _count_cells(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    count = 0
    for policies in value.values():
        if isinstance(policies, Mapping):
            count += len(policies)
    return count


def unwrap_decisions(answer: object) -> Mapping[str, object] | None:
    """Return the ``decisions`` payload from a submission answer, if present."""
    if not isinstance(answer, Mapping):
        return None
    decisions = answer.get("decisions")
    return decisions if isinstance(decisions, Mapping) else None


def unwrap_reference_decisions(reference_answer: object) -> Mapping[str, object] | None:
    """Return the decision map from an oracle ``reference_answer``.

    L02 wraps the map as ``{"decisions": {...}}``; L01 publishes the map
    directly.  Returning the direct map is deliberate: the caller can then
    detect that its keys are oracle-only identifiers rather than public
    policy handles.
    """
    if not isinstance(reference_answer, Mapping):
        return None
    nested = reference_answer.get("decisions")
    if isinstance(nested, Mapping):
        return nested
    return reference_answer


def decision_cells_from_maps(
    task_id: str,
    truth: Mapping[str, object],
    predicted: Mapping[str, object] | None,
) -> DecisionMapResult:
    """Align a truth decision map with a predicted decision map.

    Truth cells are the grading universe.  A predicted cell that cannot be
    parsed is treated as unknown and flags ``malformed_prediction``; a predicted
    cell whose key is absent from truth is reported as spurious and never
    enters a denominator.
    """
    truth_flat = _flatten_map(truth, where="truth")
    predicted_flat: dict[tuple[str, str], Decision] | None
    malformed = False
    if predicted is None:
        predicted_flat = None
    else:
        try:
            predicted_flat = _flatten_map(predicted, where="prediction")
        except ValueError:
            predicted_flat = None
            malformed = True
    cells = tuple(
        DecisionCell(
            task_id=task_id,
            cell_key=key,
            truth=truth_flat[key],
            predicted=(predicted_flat.get(key) if predicted_flat else None),
        )
        for key in sorted(truth_flat)
    )
    spurious = tuple(sorted(set(predicted_flat or {}) - set(truth_flat)))
    return DecisionMapResult(
        cells=cells,
        spurious_cells=spurious,
        predicted_cell_count=_count_cells(predicted),
        malformed_prediction=malformed,
    )


def compute_decision_metrics(cells: Iterable[DecisionCell]) -> dict[str, object]:
    """Compute FRR/MRR-style rates with explicit known/unknown denominators.

    ``*_known`` rates exclude cells whose prediction is unknown.  The
    ``*_unknown_as_*`` rates are the conservative sensitivity: an unknown legal
    cell is treated as blocked, and an unknown illegal cell as waved through.
    Both are reported; neither is called a paper FRR/MRR.
    """
    legal = illegal = false_rejections = missed_rejections = 0
    unknown_legal = unknown_illegal = 0
    for cell in cells:
        match cell.truth:
            case Decision.ALLOW:
                legal += 1
                match cell.predicted:
                    case None:
                        unknown_legal += 1
                    case Decision.DENY:
                        false_rejections += 1
                    case Decision.ALLOW:
                        pass
                    case _ as unreachable:
                        assert_never(unreachable)
            case Decision.DENY:
                illegal += 1
                match cell.predicted:
                    case None:
                        unknown_illegal += 1
                    case Decision.ALLOW:
                        missed_rejections += 1
                    case Decision.DENY:
                        pass
                    case _ as unreachable:
                        assert_never(unreachable)
            case _ as unreachable:
                assert_never(unreachable)

    known_legal = legal - unknown_legal
    known_illegal = illegal - unknown_illegal
    total = legal + illegal
    known = known_legal + known_illegal
    return {
        "legal_cells": legal,
        "illegal_cells": illegal,
        "known_legal_cells": known_legal,
        "known_illegal_cells": known_illegal,
        "unknown_legal_cells": unknown_legal,
        "unknown_illegal_cells": unknown_illegal,
        "false_rejections": false_rejections,
        "missed_rejections": missed_rejections,
        "frr_known": (false_rejections / known_legal) if known_legal else None,
        "mrr_known": (missed_rejections / known_illegal) if known_illegal else None,
        "frr_unknown_as_blocked": (
            (false_rejections + unknown_legal) / legal if legal else None),
        "mrr_unknown_as_passed": (
            (missed_rejections + unknown_illegal) / illegal if illegal else None),
        "known_coverage": (known / total) if total else None,
    }