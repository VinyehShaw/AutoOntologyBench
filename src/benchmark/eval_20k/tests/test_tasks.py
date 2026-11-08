"""Task loading, schema validation, leak guard and the build fallback."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.eval_20k.tasks import (
    ORACLE_ONLY_KEYS,
    TaskValidationError,
    build_fallback_tasks,
    load_tasks,
    validate_task_record,
)
from benchmark.eval_20k.tests.conftest import BUILD_DIR, TASKS_SMOKE


def _valid_row() -> dict:
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": "autoontobench-bank-20000-candidate-dev1",
        "lane_id": "L05",
        "family_id": "l05.fg_query",
        "base_goal_id": "l05b-abc",
        "instance_id": "l05i-abc",
        "task_id": "L05-l05i-abc",
        "task_regime": "analytical",
        "complexity": "low",
        "semantic_profile": "rdkit-smarts-native/1",
        "public": {"user_goal": "list molecules", "budget": {"actions": 1}},
        "oracle_handle": "orl05_abc",
        "status": "QUALIFIED",
        "qualification": {"native_verified": True},
        "source_snapshot_ids": ["snap"],
        "source_unit_ids": ["u1"],
        "difficulty": "structural-low",
        "split": "test",
    }


def test_validate_valid_row():
    record = validate_task_record(_valid_row(), line_no=1)
    assert record.task_id == "L05-l05i-abc"
    assert record.lane_id == "L05"


def test_reject_wrong_split():
    row = _valid_row()
    row["split"] = "dev"
    with pytest.raises(TaskValidationError, match="split"):
        validate_task_record(row, line_no=1)


def test_reject_missing_keys():
    row = _valid_row()
    del row["oracle_handle"]
    with pytest.raises(TaskValidationError, match="oracle_handle"):
        validate_task_record(row, line_no=1)


@pytest.mark.parametrize("key", sorted(ORACLE_ONLY_KEYS))
def test_reject_oracle_only_keys(key):
    row = _valid_row()
    row[key] = {"secret": True}
    with pytest.raises(TaskValidationError, match="private/oracle-only"):
        validate_task_record(row, line_no=1)


def test_reject_nested_private_key():
    row = _valid_row()
    row["public"]["private_gold"] = "x"
    with pytest.raises(TaskValidationError, match="private"):
        validate_task_record(row, line_no=1)


def test_load_smoke_fixture():
    task_set = load_tasks(TASKS_SMOKE, expected_count=3)
    assert len(task_set.records) == 3
    assert all(r.lane_id == "L05" for r in task_set.records)
    assert len(task_set.by_id) == 3
    assert task_set.sha256


def test_load_count_mismatch():
    with pytest.raises(TaskValidationError, match="expected 20000"):
        load_tasks(TASKS_SMOKE)


def test_load_duplicate_task_id(tmp_path: Path):
    row = json.dumps(_valid_row())
    path = tmp_path / "tasks.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(TaskValidationError, match="duplicate task_id"):
        load_tasks(path, expected_count=None)


def test_fallback_builds_public_test_file(tmp_path: Path):
    build = tmp_path / "build"
    build.mkdir()
    private_row = _valid_row()
    private_row.pop("split")
    private_row["native_evidence"] = {"prove_exists": {"answer_size": 3}}
    private_row["private_gold"] = "secret"
    other = dict(private_row)
    other["task_id"] = "L05-l05i-def"
    other["instance_id"] = "l05i-def"
    with (build / "qualified_tasks.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(private_row) + "\n")
        fh.write(json.dumps(other) + "\n")
    out = tmp_path / "derived" / "qualified_tasks_test.jsonl"
    task_set = build_fallback_tasks(build, out, expected_count=2)
    assert len(task_set.records) == 2
    assert task_set.derived_from == str(build)
    for record in task_set.records:
        assert record.raw["split"] == "test"
        assert "native_evidence" not in record.raw
        assert "private_gold" not in record.raw


def test_fallback_refuses_release_20k_tasks(tmp_path: Path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "qualified_tasks.jsonl").write_text("", encoding="utf-8")
    target = Path(__file__).resolve().parents[3] / "release_20k" / "tasks" / "x.jsonl"
    with pytest.raises(TaskValidationError, match="release_20k/tasks"):
        build_fallback_tasks(build, target, expected_count=None)


@pytest.mark.skipif(not (BUILD_DIR / "qualified_tasks.jsonl").is_file(),
                    reason="private 20K build not present")
def test_fallback_projects_real_build_rows(tmp_path: Path):
    from benchmark.eval_20k.tasks import iter_build_rows, project_public_row

    rows = []
    for i, row in enumerate(iter_build_rows(BUILD_DIR)):
        rows.append(row)
        if i >= 1:
            break
    assert len(rows) == 2
    for row in rows:
        projected = project_public_row(row)
        projected["split"] = "test"
        record = validate_task_record(projected, line_no=1)
        assert record.task_id == row["task_id"]
        assert "native_evidence" not in projected
