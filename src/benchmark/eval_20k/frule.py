"""Offline balanced F_rule primitives for the 20K v3 benchmark.

F_rule is defined *before* any model is scored as::

    balanced F_rule = (TPR + TNR) / 2

over eligible rule units, where a rule unit is a ``(lane, family)`` pair.  A
task is eligible only when its oracle exposes both

* a positive witness: the oracle reference answer, and
* an independent negative witness: a verifier-provided rejected answer.

The negative witness is never invented here from the model output.  It is the
stored ``mutation_submission`` where the oracle carries one (L15), or the
lane's own build-time negative-submission helper re-run on the same oracle AST
(L05/L06/L07/L08), admitted only when the oracle's ``self_check`` records that
the positive witness passed and the negative witness was rejected.

Each eligible task contributes one observation per witness.  Choosing the
positive answer is a TP on the positive witness and a TN on the negative
witness; choosing the negative answer is FN/FP.  A different answer is
UNKNOWN and is excluded from TPR/TNR (never counted as either class); coverage
reports how much of the eligible set was classified.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

SCHEMA = "aob-frule/1.0"

#: Lanes with an oracle reference + a verifier-provided rejected answer.
ELIGIBLE_LANES: tuple[str, ...] = ("L05", "L06", "L07", "L08", "L15")

#: Lane status for every 20K lane, with the precise blocking reason.
LANE_STATUS: dict[str, tuple[str, str]] = {
    "L01": ("BLOCKED_NO_WITNESS",
            "oracle exposes guards + reference_answer only; no witness, "
            "self_check, or independent negative submission"),
    "L02": ("BLOCKED_NO_WITNESS",
            "oracle exposes guards + reference_answer only; no witness, "
            "self_check, or independent negative submission"),
    "L03": ("BLOCKED_NO_WITNESS",
            "oracle exposes reference only; no witness, self_check, or "
            "independent negative submission"),
    "L04": ("BLOCKED_PERTURBATION_ONLY",
            "oracle witness certifies a perturbed artifact (counts/hashes) "
            "but exposes no independent rejected answer or self_check"),
    "L05": ("ELIGIBLE",
            "reference_submission + verifier wrong_submission under "
            "self_check negative_must_fail=true"),
    "L06": ("ELIGIBLE",
            "reference_submission + lane negative-submission helper under "
            "self_check negative_valid=false"),
    "L07": ("ELIGIBLE",
            "reference_submission + lane negative-submission helper under "
            "self_check negative_must_fail=false"),
    "L08": ("ELIGIBLE",
            "reference_submission + lane negative-submission helper under "
            "self_check negative_rejected=true"),
    "L09": ("BLOCKED_NO_REFERENCE",
            "oracle stores guards/slots/reference_artifact_sha256 only; the "
            "reference content is rebuilt from the pinned source and no "
            "negative witness is stored"),
    "L10": ("BLOCKED_NO_REFERENCE",
            "oracle stores ast/candidate only; no oracle reference answer, "
            "witness, or self_check is stored"),
    "L11": ("BLOCKED_NO_WITNESS",
            "oracle exposes no reference answer, witness, self_check, or "
            "negative submission"),
    "L12": ("BLOCKED_NO_WITNESS",
            "oracle exposes reference_answer but no witness, self_check, or "
            "independent negative submission"),
    "L13": ("BLOCKED_CONTRACT_RELAXATION",
            "oracle witness is a relaxed contract/header bad value, not a "
            "rejected answer; no self_check or stored negative submission"),
    "L14": ("BLOCKED_PERTURBATION_ONLY",
            "oracle witness certifies a perturbed artifact (counts/hashes) "
            "but exposes no independent rejected answer or self_check"),
    "L15": ("ELIGIBLE",
            "reference_submission + stored mutation_submission under "
            "self_check negative_must_fail=false, mutation_differs=true"),
    "L16": ("BLOCKED_PERTURBATION_ONLY",
            "oracle witness stores only the perturbed-answer SHA/count; no "
            "negative answer content or self_check is stored"),
}


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def lane_status(lane_id: str) -> tuple[str, str]:
    return LANE_STATUS.get(lane_id, ("BLOCKED_UNKNOWN_LANE",
                                     "lane has no F_rule witness policy"))


def positive_witness(lane_id: str, payload: dict[str, Any]) -> tuple[Any, str]:
    """Return the oracle reference answer, or a blocking reason."""
    ref = payload.get("reference_submission")
    if isinstance(ref, dict):
        return ref, ""
    answer = payload.get("reference_answer")
    if isinstance(answer, list):
        return {"answer_ids": list(answer)}, ""
    return None, "oracle_reference_missing"


def self_check_ok(lane_id: str, payload: dict[str, Any]) -> tuple[bool, str]:
    check = payload.get("self_check")
    if not isinstance(check, dict):
        return False, "oracle_self_check_missing"
    if lane_id == "L05":
        ok = check.get("positive") is True and check.get("negative_must_fail") is True
    elif lane_id == "L06":
        ok = check.get("positive") is True and check.get("negative_valid") is False
    elif lane_id == "L07":
        ok = check.get("positive") is True and check.get("negative_must_fail") is False
    elif lane_id == "L08":
        ok = (check.get("positive_pass") is True
              and check.get("negative_rejected") is True)
    elif lane_id == "L15":
        ok = (check.get("positive") is True
              and check.get("negative_must_fail") is False
              and check.get("mutation_differs") is True)
    else:
        return False, "oracle_lane_not_eligible"
    return (True, "") if ok else (False, "oracle_self_check_failed")


def _grounded(ast: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(ast=SimpleNamespace(ast=ast))


class L06NegativeBuilder:
    """Re-run L06's own build-time negative helper (never reimplemented)."""

    def __init__(self) -> None:
        self._dispatch: Any = None

    def _get_dispatch(self) -> Any:
        if self._dispatch is None:
            from benchmark.eval_20k.adapters.l06 import (
                _L06Dispatch, _resolve_index_dir,
            )
            index = _resolve_index_dir({})
            if index is None:
                raise RuntimeError("L06 index not found")
            self._dispatch = _L06Dispatch(index)
        return self._dispatch

    def negative(self, payload: dict[str, Any]) -> Any:
        from benchmark.expansion.l06.verify20k import induce

        ast = payload["ast"]
        dispatch = self._get_dispatch()
        verifier = dispatch.verifier_for(ast)
        frag = induce(ast["universe_ids"], verifier.graph)
        from benchmark.eval_20k.adapters.l06 import EXPANSION_KINDS
        if ast.get("kind") in EXPANSION_KINDS:
            from benchmark.expansion.l06.build20k import _wrong_submission20k
            return _wrong_submission20k(_grounded(ast), verifier, frag)
        from benchmark.plugins.task_family.l06.build import _wrong_submission
        return _wrong_submission(_grounded(ast), verifier, frag)


def negative_witness(lane_id: str, payload: dict[str, Any], *,
                     evidence: dict[str, Any] | None = None,
                     l05_adapter: Any = None,
                     l06_builder: L06NegativeBuilder | None = None
                     ) -> tuple[Any, str]:
    """Return an independent verifier-provided rejected answer, or a reason."""
    try:
        if lane_id == "L15":
            neg = payload.get("mutation_submission")
            return (neg, "") if neg is not None else (None, "mutation_missing")
        if lane_id == "L07":
            from benchmark.plugins.task_family.l07.build_20k import (
                _wrong_submission,
            )
            pos, reason = positive_witness(lane_id, payload)
            if pos is None:
                return None, reason
            return _wrong_submission(None, _grounded(payload["ast"]), pos), ""
        if lane_id == "L06":
            builder = l06_builder or L06NegativeBuilder()
            return builder.negative(payload), ""
        if lane_id == "L05":
            if l05_adapter is None:
                from benchmark.eval_20k.lanes import get_adapter
                l05_adapter = get_adapter("L05")
            return l05_adapter.wrong_answer(payload), ""
        if lane_id == "L08":
            return _l08_negative(payload, evidence), ""
    except Exception as exc:  # native helper failure is data, not a crash
        return None, "negative_reconstruction_failed:%s:%s" % (
            type(exc).__name__, str(exc)[:120])
    return None, "oracle_lane_not_eligible"


def _l08_negative(payload: dict[str, Any],
                  evidence: dict[str, Any] | None) -> dict[str, Any]:
    broken = payload.get("broken_rule")
    if isinstance(broken, dict):
        return {"op": broken["op"], "threshold": float(broken["threshold"]),
                "lookback_days": int(broken["lookback_days"]),
                "min_count": int(broken["min_count"])}
    pos, reason = positive_witness("L08", payload)
    if pos is None:
        raise ValueError(reason)
    answer = list(pos.get("answer_ids") or [])
    patients = (evidence or {}).get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("L08 evidence patients missing")
    all_ids = [str(p.get("id")) for p in patients if p.get("id")]
    wrong = sorted(set(all_ids) - set(answer))[: max(1, len(answer))]
    if not wrong:
        wrong = answer[: len(answer) // 2]
    return {"answer_ids": wrong}


def classify(prediction: Any, positive: Any, negative: Any) -> tuple[str, str]:
    """Classify one prediction against the two rule-specific witnesses."""
    if prediction is None:
        return "missing", "prediction_missing"
    if isinstance(prediction, dict) and "_error" in prediction:
        return "error", "prediction_error"
    key = canonical(prediction)
    if key == canonical(positive):
        return "positive", ""
    if negative is not None and key == canonical(negative):
        return "negative", ""
    return "unknown", "neither_witness"


def new_bucket() -> dict[str, Any]:
    return {
        "tasks": 0, "eligible": 0, "blocked": {}, "missing_prediction": 0,
        "prediction_error": 0, "unknown": 0,
        "tp": 0, "fn": 0, "tn": 0, "fp": 0,
    }


def add_task(bucket: dict[str, Any], outcome: str,
             reason: str = "") -> None:
    bucket["tasks"] += 1
    if outcome == "blocked":
        bucket["blocked"][reason] = bucket["blocked"].get(reason, 0) + 1
        return
    bucket["eligible"] += 1
    if outcome == "missing":
        bucket["missing_prediction"] += 1
    elif outcome == "error":
        bucket["prediction_error"] += 1
    elif outcome == "unknown":
        bucket["unknown"] += 1
    elif outcome == "positive":
        bucket["tp"] += 1
        bucket["tn"] += 1
    elif outcome == "negative":
        bucket["fn"] += 1
        bucket["fp"] += 1
    else:  # pragma: no cover - programming error
        raise ValueError("unknown outcome %r" % (outcome,))


def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    tp, fn = bucket["tp"], bucket["fn"]
    tn, fp = bucket["tn"], bucket["fp"]
    known = tp + fn  # == tn + fp under one prediction per task
    eligible = bucket["eligible"]
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    balanced = ((tpr + tnr) / 2.0
                if tpr is not None and tnr is not None else None)
    out = dict(bucket)
    out.update({
        "known": known,
        "selected_positive": tp,
        "selected_negative": fn,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": balanced,
        "coverage": (known / eligible if eligible else None),
        "status": "SCORED" if balanced is not None else "N/A",
    })
    return out