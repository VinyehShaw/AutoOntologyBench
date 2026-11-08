from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.eval_20k.metrics_regime import (
    COST_KEYS,
    REGIMES,
    RegimeJoinError,
    RegimeValidationError,
    aggregate_regimes,
    build_report,
    load_run_inputs,
)

DEFAULT_COSTS = {
    "api_calls": 1.0,
    "tokens_input": 100.0,
    "tokens_output": 50.0,
    "wall_seconds": 10.0,
}


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _report_overall(task_ids: list[str], scored_rows: list[dict],
                    prediction_rows: list[dict]) -> dict:
    counts = {
        "tasks": len(task_ids),
        "scored": 0,
        "success": 0,
        "verified_success": 0,
        "error": 0,
        "missing": 0,
        "not_supported": 0,
        "oracle_missing": 0,
    }
    status_field = {
        "MISSING": "missing",
        "NOT_SUPPORTED": "not_supported",
        "ORACLE_MISSING": "oracle_missing",
        "ERROR": "error",
    }
    for row in scored_rows:
        status = row["status"]
        if status == "SCORED":
            counts["scored"] += 1
            if row.get("task_success"):
                counts["success"] += 1
            if row.get("valid_task_success"):
                counts["verified_success"] += 1
        else:
            counts[status_field[status]] += 1
    costs = {key: {"total": 0.0, "rows": 0} for key in COST_KEYS}
    for row in prediction_rows:
        for key in COST_KEYS:
            if key in row:
                costs[key]["total"] += float(row[key])
                costs[key]["rows"] += 1
    return {**counts, "costs": costs}


def _make_run(tmp_path: Path, *, run_id: str, group: str, model: str,
              task_regimes: list[tuple[str, str]],
              outcomes: dict[str, dict],
              extra_predictions: list[dict] | None = None,
              extra_scored: list[dict] | None = None) -> Path:
    """Write one synthetic run and return its report path."""
    run_dir = tmp_path / "runs" / run_id
    tasks_path = tmp_path / "tasks" / ("%s.jsonl" % group)
    tasks_sha = _write_jsonl(
        tasks_path,
        [{"task_id": task_id, "task_regime": regime}
         for task_id, regime in task_regimes])

    prediction_rows: list[dict] = []
    scored_rows: list[dict] = []
    for task_id, _regime in task_regimes:
        outcome = outcomes[task_id]
        status = outcome.get("status", "SCORED")
        costs = dict(DEFAULT_COSTS)
        if "costs" in outcome:
            costs = dict(outcome["costs"])
        if status != "MISSING":
            prediction_rows.append({
                "task_id": task_id,
                "model": model,
                "answer": {},
                **costs,
            })
        scored_rows.append({
            "task_id": task_id,
            "status": status,
            "task_success": outcome.get("success", False),
            "valid_task_success": outcome.get("valid", False),
            "costs": costs if status != "MISSING" else {},
        })
    prediction_rows.extend(extra_predictions or [])
    scored_rows.extend(extra_scored or [])

    predictions_path = run_dir / "predictions.jsonl"
    predictions_sha = _write_jsonl(
        predictions_path,
        [{"schema": "aob-submission-20k/1", "model": model,
          "source_tasks": _rel(tasks_path, tmp_path)}] + prediction_rows)
    rows_path = run_dir / "rows.jsonl"
    _write_jsonl(rows_path, scored_rows)

    overall = _report_overall(
        [task_id for task_id, _regime in task_regimes],
        scored_rows, prediction_rows)
    report = {
        "schema": "aob-eval-20k/1",
        "tasks": {"path": _rel(tasks_path, tmp_path),
                  "sha256": tasks_sha, "count": len(task_regimes)},
        "predictions": {"path": _rel(predictions_path, tmp_path),
                        "sha256": predictions_sha,
                        "rows": len(prediction_rows), "model": model},
        "metrics": {"overall": overall},
    }
    report_path = run_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report_path


def _load(report_path: Path, tmp_path: Path):
    return load_run_inputs(report_path, tasks_dir=tmp_path / "tasks",
                           repo_root=tmp_path)


def test_synthetic_join_by_task_regime(tmp_path: Path):
    task_regimes = [
        ("t-app", "application"),
        ("t-rep", "repair"),
        ("t-org", "organization"),
        ("t-con", "construction"),
    ]
    outcomes = {
        "t-app": {"success": True, "valid": True,
                  "costs": {"api_calls": 1, "tokens_input": 10,
                            "tokens_output": 5, "wall_seconds": 1.0}},
        "t-rep": {"success": False, "valid": False,
                  "costs": {"api_calls": 2, "tokens_input": 20,
                            "tokens_output": 6, "wall_seconds": 2.0}},
        "t-org": {"success": True, "valid": False,
                  "costs": {"api_calls": 3, "tokens_input": 30,
                            "tokens_output": 7, "wall_seconds": 3.0}},
        "t-con": {"success": True, "valid": True,
                  "costs": {"api_calls": 4, "tokens_input": 40,
                            "tokens_output": 8, "wall_seconds": 4.0}},
    }
    report_path = _make_run(
        tmp_path, run_id="eval_20k_full2-g1-m1_aaaaaa", group="g1",
        model="m1", task_regimes=task_regimes, outcomes=outcomes)

    report = aggregate_regimes([_load(report_path, tmp_path)])

    model = report["models"]["m1"]
    assert model["overall"]["tasks"] == 4
    assert model["overall"]["scored"] == 4
    assert model["overall"]["success"] == 3
    assert model["overall"]["verified_success"] == 2
    assert model["overall"]["tsr"] == 3 / 4
    assert model["overall"]["tsr_denominator"] == 4
    assert model["overall"]["coverage"] == 1.0
    assert model["regimes"]["application"]["success"] == 1
    assert model["regimes"]["repair"]["success"] == 0
    assert model["regimes"]["organization"]["verified_success"] == 0
    assert model["regimes"]["construction"]["verified_success"] == 1
    assert model["overall"]["costs"]["api_calls"]["total"] == 10.0
    assert model["overall"]["costs"]["api_calls"]["rows"] == 4
    assert report["sources"][0]["tasks_sha256"]
    assert report["sources"][0]["predictions_sha256"]
    assert report["sources"][0]["rows_sha256"]


def test_timestamped_run_directory_defaults_to_local_group(tmp_path: Path):
    report_path = _make_run(
        tmp_path, run_id="eval_20k_20261105T084421Z_Z788jz", group="g1",
        model="m1",
        task_regimes=[("t-app", "application"), ("t-rep", "repair")],
        outcomes={
            "t-app": {"success": True, "valid": True},
            "t-rep": {"success": False, "valid": False},
        })

    run = _load(report_path, tmp_path)
    report = aggregate_regimes([run])

    assert run.group == "local"
    assert report["models"]["m1"]["groups"] == ["local"]
    assert report["reconciliation"]["all_models_have_four_groups"] is False
    assert report["models"]["m1"]["overall"]["tasks"] == 2


def test_regime_key_set_is_stable_even_when_unobserved(tmp_path: Path):
    report_path = _make_run(
        tmp_path, run_id="eval_20k_full2-g1-m1_bbbbbb", group="g1",
        model="m1", task_regimes=[("t-app", "application")],
        outcomes={"t-app": {"success": True, "valid": True}})

    report = aggregate_regimes([_load(report_path, tmp_path)])

    assert report["regimes"] == sorted(REGIMES)
    model = report["models"]["m1"]
    assert set(model["regimes"]) == set(REGIMES)
    assert model["observed_regimes"] == ["application"]
    assert set(model["missing_regimes"]) == set(REGIMES) - {"application"}
    assert model["regimes"]["repair"]["tasks"] == 0
    assert model["regimes"]["repair"]["tsr"] is None
    assert model["regimes"]["repair"]["coverage"] is None


def test_full_per_model_reconciliation_across_four_groups(tmp_path: Path):
    group_tasks = {
        "g1": [("a", "application"), ("b", "repair"), ("c", "organization")],
        "g2": [("d", "construction"), ("e", "application"), ("f", "repair")],
        "g3": [("g", "organization"), ("h", "construction"),
               ("i", "application")],
        "g4": [("j", "repair"), ("k", "organization"),
               ("l", "construction")],
    }
    for model in ("m1", "m2"):
        for group, task_regimes in group_tasks.items():
            outcomes = {}
            for index, (task_id, _regime) in enumerate(task_regimes):
                success = (index + len(model)) % 2 == 0
                outcomes[task_id] = {
                    "success": success,
                    "valid": success,
                    "costs": {"api_calls": index + 1,
                              "tokens_input": 10 * (index + 1),
                              "tokens_output": 5 * (index + 1),
                              "wall_seconds": float(index + 1)},
                }
            _make_run(
                tmp_path,
                run_id="eval_20k_full2-%s-%s_%06d" % (group, model, 1),
                group=group, model=model, task_regimes=task_regimes,
                outcomes=outcomes)

    report = build_report(
        str(tmp_path / "runs" / "eval_20k_full2-*" / "report.json"),
        tasks_dir=tmp_path / "tasks", repo_root=tmp_path)

    assert report["run_count"] == 8
    assert report["reconciliation"]["ok"] is True
    assert report["reconciliation"]["all_models_have_four_groups"] is True
    for model in ("m1", "m2"):
        expected = report["models"][model]["report_expected"]
        overall = report["models"][model]["overall"]
        assert overall["tasks"] == 12
        assert overall["tasks"] == expected["tasks"]
        assert overall["success"] == expected["success"]
        assert overall["verified_success"] == expected["verified_success"]
        regime_tasks = sum(cell["tasks"]
                           for cell in report["models"][model]["regimes"].values())
        assert regime_tasks == overall["tasks"]
        assert report["reconciliation"]["per_model"][model][
            "matches_report_overall"] is True


def test_missing_task_id_fails_loudly(tmp_path: Path):
    report_path = _make_run(
        tmp_path, run_id="eval_20k_full2-g1-m1_cccccc", group="g1",
        model="m1", task_regimes=[("t-app", "application")],
        outcomes={"t-app": {"success": True, "valid": True}},
        extra_predictions=[{
            "task_id": "ghost-task",
            "model": "m1",
            "answer": {},
            **DEFAULT_COSTS,
        }])

    with pytest.raises(RegimeJoinError, match="cannot be joined"):
        aggregate_regimes([_load(report_path, tmp_path)])


def test_reconciliation_mismatch_fails_loudly(tmp_path: Path):
    report_path = _make_run(
        tmp_path, run_id="eval_20k_full2-g1-m1_dddddd", group="g1",
        model="m1", task_regimes=[("t-app", "application")],
        outcomes={"t-app": {"success": True, "valid": True}})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["metrics"]["overall"]["tasks"] += 1
    report_path.write_text(json.dumps(data, sort_keys=True) + "\n",
                           encoding="utf-8")

    with pytest.raises(RegimeValidationError, match="does not reconcile"):
        aggregate_regimes([_load(report_path, tmp_path)])