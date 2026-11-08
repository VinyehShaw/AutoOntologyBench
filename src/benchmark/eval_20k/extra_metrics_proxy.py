"""Offline proxy computations for the 20K FRR/MRR/OOD feasibility report."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping

from .extra_metrics_data import (
    REPO_ROOT,
    RUNS,
    cell_keys,
    prediction_files,
    read_jsonl,
)
from .metrics_decision import (
    compute_decision_metrics,
    decision_cells_from_maps,
    unwrap_decisions,
    unwrap_reference_decisions,
)
from .metrics_ood import TaskOutcome, family_leave_one_out
from .oracle import OracleStore, default_oracle_roots

_L02_COUNTERS = (
    "tasks_labeled", "tasks_oracle_missing", "tasks_reference_missing",
    "tasks_prediction_missing", "truth_cells", "predicted_cells",
    "spurious_cells", "malformed_predictions",
)
_L01_COUNTERS = (
    "tasks_labeled", "tasks_oracle_missing", "tasks_with_decisions",
    "truth_cells", "predicted_cells", "matched_key_cells", "exact_key_tasks",
)


def decision_proxy(tasks: dict[str, dict]) -> dict[str, object]:
    l02 = [row for row in tasks.values()
           if row.get("family_id") == "l02-f05-exact-decision-set"]
    l01 = [row for row in tasks.values()
           if row.get("family_id") == "l01-f20-exact-decision-set"]
    store = OracleStore(default_oracle_roots(include_repo_staging=True))
    payloads: dict[str, dict] = {}
    oracle_errors: dict[str, str] = {}
    for row in l02 + l01:
        try:
            payloads[row["task_id"]] = store.load(
                row["lane_id"], row["instance_id"], row["oracle_handle"])
        except Exception as exc:  # noqa: BLE001 - environment evidence
            oracle_errors[row["task_id"]] = "%s: %s" % (type(exc).__name__, exc)

    l02_preds = prediction_files("g4")
    l01_preds = prediction_files("g2")
    models: dict[str, object] = {}
    for model in sorted(set(l02_preds) | set(l01_preds)):
        cells = []
        stats: Counter[str] = Counter({key: 0 for key in _L02_COUNTERS})
        answers = l02_preds.get(model, (None, {}))[1]
        for row in l02:
            payload = payloads.get(row["task_id"])
            if payload is None:
                stats["tasks_oracle_missing"] += 1
                continue
            truth = unwrap_reference_decisions(payload.get("reference_answer"))
            if truth is None:
                stats["tasks_reference_missing"] += 1
                continue
            stats["tasks_labeled"] += 1
            if row["task_id"] not in answers:
                stats["tasks_prediction_missing"] += 1
            result = decision_cells_from_maps(
                row["task_id"], truth, unwrap_decisions(answers.get(row["task_id"])))
            cells.extend(result.cells)
            stats["truth_cells"] += len(result.cells)
            stats["predicted_cells"] += result.predicted_cell_count
            stats["spurious_cells"] += len(result.spurious_cells)
            stats["malformed_predictions"] += int(result.malformed_prediction)

        l01_stats: Counter[str] = Counter({key: 0 for key in _L01_COUNTERS})
        l01_answers = l01_preds.get(model, (None, {}))[1]
        for row in l01:
            payload = payloads.get(row["task_id"])
            if payload is None:
                l01_stats["tasks_oracle_missing"] += 1
                continue
            truth = unwrap_reference_decisions(payload.get("reference_answer"))
            predicted = unwrap_decisions(l01_answers.get(row["task_id"]))
            l01_stats["tasks_labeled"] += 1
            if predicted is not None:
                l01_stats["tasks_with_decisions"] += 1
            truth_keys = cell_keys(truth)
            predicted_keys = cell_keys(predicted)
            l01_stats["truth_cells"] += len(truth_keys)
            l01_stats["predicted_cells"] += len(predicted_keys)
            l01_stats["matched_key_cells"] += len(truth_keys & predicted_keys)
            l01_stats["exact_key_tasks"] += int(truth_keys == predicted_keys)

        models[model] = {
            "l02_exact_decision_set": {
                **dict(stats),
                **compute_decision_metrics(cells),
                "prediction_file": str(l02_preds.get(model, (None,))[0] or ""),
            },
            "l01_exact_decision_set_key_alignment": {
                **dict(l01_stats),
                "prediction_file": str(l01_preds.get(model, (None,))[0] or ""),
            },
        }
    return {
        "status": "PROXY",
        "scope": "L02 l02-f05-exact-decision-set only (50 tasks, 480 cells)",
        "not_paper_metric": (
            "These are task-answer ALLOW/DENY cells, not consumer/executor "
            "decision steps. L01 is excluded because its oracle policy ids are "
            "not exposed in the public task, so prediction cells cannot align."),
        "oracle_errors": oracle_errors,
        "models": models,
    }


def ood_proxy(tasks: dict[str, dict]) -> dict[str, object]:
    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    provenance: dict[str, list[str]] = defaultdict(list)
    for report_path in sorted(RUNS.glob("eval_20k_full2-g*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model = report.get("predictions", {}).get("model")
        rows_path = report_path.parent / "rows.jsonl"
        if not model or not rows_path.is_file():
            continue
        provenance[model].append(str(rows_path.relative_to(REPO_ROOT)))
        rows_by_model[model].extend(read_jsonl(rows_path))

    models: dict[str, object] = {}
    for model in sorted(rows_by_model):
        outcomes = []
        statuses: Counter[str] = Counter()
        missing_metadata = 0
        seen: set[str] = set()
        for row in rows_by_model[model]:
            task_id = row.get("task_id")
            if task_id in seen:
                continue
            seen.add(task_id)
            meta = tasks.get(task_id)
            if meta is None:
                missing_metadata += 1
                continue
            statuses[str(row.get("status"))] += 1
            outcomes.append(TaskOutcome(
                task_id=task_id, lane_id=meta["lane_id"],
                family_id=meta["family_id"], lineage_group=meta["lineage_group"],
                status=str(row.get("status")), success=row.get("task_success")))
        scored = [row for row in outcomes if row.is_scored]
        success = sum(1 for row in scored if row.success)
        models[model] = {
            "run_rows": provenance[model],
            "unique_tasks": len(seen),
            "joined_to_manifest": len(outcomes),
            "missing_metadata": missing_metadata,
            "statuses": dict(statuses),
            "scored": len(scored),
            "success": success,
            "tsr": (success / len(scored)) if scored else None,
            "family_leave_one_out": family_leave_one_out(outcomes, unit="family_id"),
            "lineage_group_leave_one_out": family_leave_one_out(
                outcomes, unit="lineage_group"),
        }
    return {
        "status": "PROXY",
        "split_definition": (
            "Leave-one-family-out within lane. The held-out unit is defined "
            "solely by lane_id + family_id (or lineage_group); no outcome "
            "field enters assignment, and only existing score rows are read."),
        "not_paper_metric": (
            "Leave-one-family-out, stratified within lane, over existing "
            "answer-only task success. Models were not trained on the 20K set; "
            "the gaps measure family-difficulty heterogeneity, not paper OOD."),
        "models": models,
    }