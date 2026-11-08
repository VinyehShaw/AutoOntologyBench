"""model_runner tests: prompt construction, per-lane sampling, checkpoint
resume, and per-row error recording — all against a fake in-memory client.
No real API calls are made anywhere in this file.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from autoontobench.models.dashscope import DashScopeClient
from autoontobench.models.dashscope_types import CallResult, Usage

from benchmark.eval_20k import model_runner
from benchmark.eval_20k.model_runner import parse_answer, select_tasks
from benchmark.eval_20k.submission import load_submission
from benchmark.eval_20k.tasks import load_tasks

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "eval.sh"


def make_task_row(lane: str, n: int) -> dict:
    iid = "%si-%016x" % (lane.lower(), n)
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": "fixture",
        "lane_id": lane,
        "family_id": "%s.family" % lane.lower(),
        "base_goal_id": "%sb-%012x" % (lane.lower(), n),
        "instance_id": iid,
        "task_id": "%s-%s" % (lane, iid),
        "task_regime": "application",
        "complexity": "low",
        "semantic_profile": "fixture/1",
        "public": {
            "user_goal": "Return the ids for case %d." % n,
            "allowed_tools": ["fixture.read"],
            "budget": {"actions": 2},
            "extra_material": {"case": n},
        },
        "oracle_handle": "or%s-%016x" % (lane.lower()[1:], n),
        "status": "QUALIFIED",
        "qualification": {"leakage_clean": True},
        "source_snapshot_ids": [],
        "source_unit_ids": [],
        "difficulty": "easy",
        "split": "test",
    }


def write_tasks(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class FakeClient:
    """In-memory client with the DashScopeClient.call contract."""

    def __init__(self, handler=None):
        self.handler = handler
        self.requests = []

    async def call(self, req):
        self.requests.append(req)
        cid = DashScopeClient.request_id(req)
        if self.handler is not None:
            return self.handler(req, cid)
        return CallResult(
            custom_id=cid, model_exact_id=req.model, prompt_sha=req.prompt_sha,
            content=json.dumps({"task_id": req.custom_id,
                                "answer": {"echo": req.custom_id}}),
            usage=Usage(input=11, output=7), latency_s=0.01, attempts=1)


def run_cli(tmp_path: Path, rows: list[dict], client, *extra: str) -> int:
    tasks_path = tmp_path / "tasks.jsonl"
    write_tasks(tasks_path, rows)
    return model_runner.main(
        ["--tasks", str(tasks_path), "--out", str(tmp_path / "preds.jsonl"),
         *extra], client=client)


# -- prompt -------------------------------------------------------------------

def test_prompt_carries_public_task_fields(tmp_path):
    row = make_task_row("L05", 1)
    client = FakeClient()
    rc = run_cli(tmp_path, [row], client, "--sample-per-lane", "1")
    assert rc == 0
    (req,) = client.requests
    assert req.params.json_mode is True
    assert req.model == model_runner.DEFAULT_MODEL
    user = req.messages[1]["content"]
    assert row["task_id"] in user
    assert "Return the ids for case 1." in user
    assert "fixture.read" in user
    assert '"actions": 2' in user
    assert "extra_material" in user
    assert '"answer"' in user  # strict JSON contract is stated


# -- token budget -----------------------------------------------------------------

def test_default_max_tokens_is_reasoning_headroom(tmp_path):
    args = model_runner.build_parser().parse_args(
        ["--out", str(tmp_path / "preds.jsonl")])
    assert args.max_tokens == 32768
    client = FakeClient()
    assert run_cli(tmp_path, [make_task_row("L09", 1)], client) == 0
    (req,) = client.requests
    assert req.params.max_tokens == 32768


def test_explicit_max_tokens_overrides_default(tmp_path):
    client = FakeClient()
    rc = run_cli(tmp_path, [make_task_row("L09", 1)], client,
                 "--max-tokens", "16384")
    assert rc == 0
    (req,) = client.requests
    assert req.params.max_tokens == 16384


# -- sampling -------------------------------------------------------------------

def test_sample_per_lane_caps_deterministic_and_zero_means_all(tmp_path):
    rows = [make_task_row("L05", i) for i in range(6)] + \
           [make_task_row("L07", i) for i in range(4)]
    file_order = [r["task_id"] for r in rows]
    tasks_path = tmp_path / "tasks.jsonl"
    write_tasks(tasks_path, rows)
    records = load_tasks(tasks_path, expected_count=None).records
    sel = select_tasks(records, None, 3)
    assert len(sel) == 6
    assert sum(1 for r in sel if r.lane_id == "L05") == 3
    assert len({r.task_id for r in sel}) == 6
    again = select_tasks(records, None, 3)
    assert [r.task_id for r in again] == [r.task_id for r in sel]
    positions = [file_order.index(r.task_id) for r in sel]
    assert positions == sorted(positions)  # selection keeps file order
    assert len(select_tasks(records, None, 0)) == 10


def test_sample_lane_filter(tmp_path):
    rows = [make_task_row("L05", i) for i in range(3)] + \
           [make_task_row("L07", i) for i in range(3)]
    tasks_path = tmp_path / "tasks.jsonl"
    write_tasks(tasks_path, rows)
    records = load_tasks(tasks_path, expected_count=None).records
    sel = select_tasks(records, {"L05"}, 0)
    assert {r.lane_id for r in sel} == {"L05"}
    assert len(sel) == 3


# -- end-to-end with fake client ----------------------------------------------

def test_predictions_file_is_scorable_and_complete(tmp_path):
    rows = [make_task_row("L05", i) for i in range(2)] + [make_task_row("L07", 7)]
    rc = run_cli(tmp_path, rows, FakeClient())
    assert rc == 0
    sub = load_submission(tmp_path / "preds.jsonl")  # validates the schema
    assert len(sub.predictions) == 3
    assert sub.model == model_runner.DEFAULT_MODEL
    for row in rows:
        pred = sub.by_task_id[row["task_id"]]
        assert pred.answer == {"echo": row["task_id"]}
        assert pred.costs["tokens_input"] == 11
        assert pred.costs["tokens_output"] == 7
        assert pred.costs["api_calls"] == 1
    sampled = load_tasks(tmp_path / "tasks_sampled.jsonl", expected_count=None)
    assert {r.task_id for r in sampled.records} == {r["task_id"] for r in rows}


def test_invalid_model_output_recorded_as_error_marker(tmp_path):
    def handler(req, cid):
        return CallResult(custom_id=cid, model_exact_id=req.model,
                          prompt_sha=req.prompt_sha, content="not json at all",
                          attempts=1)

    rc = run_cli(tmp_path, [make_task_row("L05", 1)], FakeClient(handler))
    assert rc == 0
    (pred,) = load_submission(tmp_path / "preds.jsonl").predictions
    assert pred.answer["_error"] == "invalid_json"
    assert pred.provenance["notes"] == "invalid_json"


def test_api_error_recorded_per_row_not_dropped(tmp_path):
    def handler(req, cid):
        return CallResult(custom_id=cid, model_exact_id=req.model,
                          prompt_sha=req.prompt_sha, content=None,
                          error="429 too many requests", attempts=5)

    rc = run_cli(tmp_path, [make_task_row("L05", i) for i in range(2)],
                 FakeClient(handler))
    assert rc == 0
    preds = load_submission(tmp_path / "preds.jsonl").predictions
    assert len(preds) == 2
    for pred in preds:
        assert pred.answer["_error"] == "api_error"
        assert "429" in pred.answer["detail"]
        assert pred.costs["api_calls"] == 5


def test_missing_answer_key_is_a_failure_marker(tmp_path):
    answer, kind = parse_answer(json.dumps({"answer_ids": [1, 2]}), "T1")
    assert kind == "missing_answer_key"
    assert answer["_error"] == "missing_answer_key"
    answer, kind = parse_answer('```json\n{"answer": {"x": 1}}\n```', "T1")
    assert kind is None
    assert answer == {"x": 1}
    answer, kind = parse_answer('{"task_id": "T2", "answer": {}}', "T1")
    assert kind == "task_id_mismatch"


# -- resume ----------------------------------------------------------------------

def test_resume_retries_only_failed_calls(tmp_path):
    rows = [make_task_row("L05", i) for i in range(4)]
    failing = rows[2]["task_id"]

    def handler1(req, cid):
        if req.custom_id == failing:
            return CallResult(custom_id=cid, model_exact_id=req.model,
                              prompt_sha=req.prompt_sha, content=None,
                              error="503 unavailable", attempts=5)
        return CallResult(custom_id=cid, model_exact_id=req.model,
                          prompt_sha=req.prompt_sha,
                          content=json.dumps({"answer": {"echo": req.custom_id}}),
                          usage=Usage(input=1, output=1), attempts=1)

    first = FakeClient(handler1)
    assert run_cli(tmp_path, rows, first) == 0
    assert len(first.requests) == 4

    second = FakeClient()
    tasks_path = tmp_path / "tasks.jsonl"
    rc = model_runner.main(
        ["--tasks", str(tasks_path), "--out", str(tmp_path / "preds.jsonl"),
         "--resume"], client=second)
    assert rc == 0
    assert [r.custom_id for r in second.requests] == [failing]
    preds = load_submission(tmp_path / "preds.jsonl").predictions
    assert len(preds) == 4
    assert all(isinstance(p.answer, dict) and "_error" not in p.answer
               for p in preds)


def test_refuses_to_overwrite_without_resume(tmp_path):
    rows = [make_task_row("L05", 1)]
    assert run_cli(tmp_path, rows, FakeClient()) == 0
    third = FakeClient()
    rc = run_cli(tmp_path, rows, third)
    assert rc == 2
    assert third.requests == []


def test_missing_api_key_exits_3_without_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    rows = [make_task_row("L05", 1)]
    tasks_path = tmp_path / "tasks.jsonl"
    write_tasks(tasks_path, rows)
    rc = model_runner.main(["--tasks", str(tasks_path),
                            "--out", str(tmp_path / "preds.jsonl")])
    assert rc == 3


# -- one-click script -------------------------------------------------------------

def test_one_click_script_syntax():
    assert SCRIPT.is_file()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_one_click_script_forwards_max_tokens_override(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "scripts").mkdir(parents=True)
    (repo / "runs").mkdir()
    script = repo / "src" / "scripts" / "eval.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task_row("L05", 1)])
    argv_log = tmp_path / "argv.log"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_ARGV_LOG\"\n"
        "exit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env = dict(os.environ, OPENAI_API_KEY="dummy", PYTHON_BIN=str(fake_python),
               FAKE_ARGV_LOG=str(argv_log), EVAL_MAX_TOKENS="12345")
    proc = subprocess.run(["bash", str(script), "--tasks", str(tasks)],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0  # the fake generator stops the script at step 1
    argv = argv_log.read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-m", "benchmark.eval_20k.model_runner"]
    assert argv[argv.index("--max-tokens") + 1] == "12345"


def test_one_click_script_requires_api_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    proc = subprocess.run(["bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "OPENAI_API_KEY" in proc.stderr


def test_one_click_script_never_echoes_key(tmp_path):
    env = dict(os.environ, OPENAI_API_KEY="sentinel-secret-value")
    env.pop("OPENAI_BASE_URL", None)
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--tasks", str(tmp_path / "missing.jsonl")],
        env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "sentinel-secret-value" not in (proc.stdout + proc.stderr)
