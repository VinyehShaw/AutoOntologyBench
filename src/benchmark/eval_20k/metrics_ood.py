"""Offline OOD split arithmetic and a family-held-out proxy for the 20K audit.

Paper OOD needs official ``dev`` / ``iid`` / ``comp`` / ``transfer`` split
labels.  The 20K public release has ``split == "test"`` on every row, so paper
OOD is blocked.  This module provides two things:

* :func:`compute_split_drop` — the exact arithmetic that would be used if an
  official split existed.  It keeps per-split denominators and reports the
  iid-vs-ood drop as ``iid_rate - split_rate`` (positive = OOD worse).
* :func:`family_leave_one_out` — a clearly labelled *proxy*: for each family
  (or lineage group) in a lane, hold that family out and score it against the
  remaining families of the same lane.  Grouping uses only ``lane_id`` and
  ``family_id`` / ``lineage_group`` metadata, never task outcome.

The proxy is not paper OOD.  Models were not trained on the 20K tasks, so a
held-out family cannot measure generalization in the paper's sense; the
leave-one-out gaps mostly expose family-difficulty heterogeneity.  The report
says so explicitly and never promotes the proxy to an OOD number.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SCORED = "SCORED"


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """One task's existing scored outcome joined to its split metadata."""

    task_id: str
    lane_id: str
    family_id: str
    lineage_group: str
    status: str
    success: bool | None

    @property
    def is_scored(self) -> bool:
        return self.status == SCORED and self.success is not None


def _rate(success: int, scored: int) -> float | None:
    return (success / scored) if scored else None


def compute_split_drop(
    outcomes: Iterable[TaskOutcome],
    split_by_task: Mapping[str, str],
    *,
    iid_label: str = "iid",
    ood_labels: tuple[str, ...] = ("comp", "transfer"),
) -> dict[str, object]:
    """Compute per-split success and the OOD drop against ``iid_label``.

    Only scored outcomes with a split label enter a denominator.  Unlabelled
    tasks and non-scored tasks are counted separately.
    """
    buckets: dict[str, list[TaskOutcome]] = defaultdict(list)
    unlabelled = 0
    for outcome in outcomes:
        split = split_by_task.get(outcome.task_id)
        if split is None:
            unlabelled += 1
            continue
        buckets[split].append(outcome)

    per_split: dict[str, dict[str, object]] = {}
    for split, rows in sorted(buckets.items()):
        scored = [row for row in rows if row.is_scored]
        success = sum(1 for row in scored if row.success)
        per_split[split] = {
            "tasks": len(rows),
            "scored": len(scored),
            "success": success,
            "rate": _rate(success, len(scored)),
        }

    iid_rate = per_split.get(iid_label, {}).get("rate")
    drops = {}
    for label in ood_labels:
        rate = per_split.get(label, {}).get("rate")
        drops[label] = (
            iid_rate - rate if isinstance(iid_rate, float)
            and isinstance(rate, float) else None)
    return {
        "per_split": per_split,
        "iid_label": iid_label,
        "ood_drops": drops,
        "unlabelled_tasks": unlabelled,
    }


def _group_rows(
    outcomes: Iterable[TaskOutcome], unit: str,
) -> dict[str, dict[str, list[TaskOutcome]]]:
    grouped: dict[str, dict[str, list[TaskOutcome]]] = defaultdict(
        lambda: defaultdict(list))
    for outcome in outcomes:
        key = outcome.family_id if unit == "family_id" else outcome.lineage_group
        grouped[outcome.lane_id][key].append(outcome)
    return grouped


def _rate_rows(rows: Iterable[TaskOutcome]) -> tuple[int, int, float | None]:
    scored = [row for row in rows if row.is_scored]
    success = sum(1 for row in scored if row.success)
    return len(scored), success, _rate(success, len(scored))


def family_leave_one_out(
    outcomes: Iterable[TaskOutcome], *, unit: str = "family_id",
) -> dict[str, object]:
    """Leave-one-family-out proxy, stratified within lane.

    ``unit`` must be ``"family_id"`` or ``"lineage_group"``.  For each lane
    with at least two units, every unit is held out once and compared with the
    remaining units of the same lane.  ``drop = reference_rate - heldout_rate``
    (positive = held-out worse).  Lanes with a single unit cannot provide a
    same-lane reference and are reported as not evaluable.
    """
    if unit not in ("family_id", "lineage_group"):
        raise ValueError("unit must be family_id or lineage_group")
    grouped = _group_rows(outcomes, unit)
    records: list[dict[str, object]] = []
    per_lane: dict[str, dict[str, object]] = {}
    excluded_tasks = 0
    total_tasks = 0

    for lane in sorted(grouped):
        units = grouped[lane]
        all_rows = [row for rows in units.values() for row in rows]
        total_tasks += len(all_rows)
        excluded_tasks += sum(1 for row in all_rows if not row.is_scored)
        if len(units) < 2:
            per_lane[lane] = {
                "units": len(units), "tasks": len(all_rows),
                "evaluated": False, "unweighted_mean_drop": None,
                "task_weighted_mean_drop": None,
            }
            continue

        lane_records: list[dict[str, object]] = []
        for heldout in sorted(units):
            held_rows = units[heldout]
            ref_rows = [row for other, rows in units.items()
                        if other != heldout for row in rows]
            held_scored, held_success, held_rate = _rate_rows(held_rows)
            ref_scored, ref_success, ref_rate = _rate_rows(ref_rows)
            if held_scored == 0 or ref_scored == 0:
                continue
            if held_rate is None or ref_rate is None:
                continue
            drop = ref_rate - held_rate
            record = {
                "lane_id": lane,
                "unit": heldout,
                "heldout_tasks": len(held_rows),
                "heldout_scored": held_scored,
                "heldout_success": held_success,
                "heldout_rate": held_rate,
                "reference_scored": ref_scored,
                "reference_success": ref_success,
                "reference_rate": ref_rate,
                "drop": drop,
            }
            lane_records.append(record)
            records.append(record)

        if lane_records:
            lane_drops = [float(row["drop"]) for row in lane_records]
            lane_weight = sum(int(row["heldout_scored"]) for row in lane_records)
            per_lane[lane] = {
                "units": len(units),
                "tasks": len(all_rows),
                "evaluated": True,
                "evaluated_units": len(lane_records),
                "unweighted_mean_drop": sum(lane_drops) / len(lane_drops),
                "task_weighted_mean_drop": sum(
                    float(row["drop"]) * int(row["heldout_scored"])
                    for row in lane_records) / lane_weight,
            }
        else:
            per_lane[lane] = {
                "units": len(units), "tasks": len(all_rows),
                "evaluated": False, "unweighted_mean_drop": None,
                "task_weighted_mean_drop": None,
            }

    drops = [float(row["drop"]) for row in records]
    weights = [int(row["heldout_scored"]) for row in records]
    lane_means = [
        float(row["unweighted_mean_drop"]) for row in per_lane.values()
        if row["unweighted_mean_drop"] is not None]
    return {
        "unit": unit,
        "drop_definition": "reference_rate - heldout_rate (positive = heldout worse)",
        "units_total": sum(len(units) for units in grouped.values()),
        "units_evaluated": len(records),
        "tasks_total": total_tasks,
        "tasks_evaluated": sum(weights),
        "excluded_tasks": excluded_tasks,
        "tasks_not_evaluable": total_tasks - excluded_tasks - sum(weights),
        "unweighted_mean_drop": (sum(drops) / len(drops)) if drops else None,
        "task_weighted_mean_drop": (
            sum(drop * weight for drop, weight in zip(drops, weights))
            / sum(weights)) if weights else None,
        "median_drop": statistics.median(drops) if drops else None,
        "min_drop": min(drops) if drops else None,
        "max_drop": max(drops) if drops else None,
        "macro_lane_mean_drop": (
            sum(lane_means) / len(lane_means)) if lane_means else None,
        "per_lane": per_lane,
        "records": records,
    }