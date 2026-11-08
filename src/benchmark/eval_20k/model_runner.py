"""Prediction generator for the 20K harness (model side, never scoring).

Calls an OpenAI-compatible chat endpoint or the Anthropic Messages API over
the public task file and writes a submission JSONL that
``python -m benchmark.eval_20k.run`` scores unchanged. Honesty rules:
answers come only from the model's response to the public task record; a
response that is not a strict JSON object with an ``answer`` key, and every
API error, is recorded per row as an explicit ``{"_error": ...}`` answer
marker (scored as a failure) — never dropped, never guessed. The API key
comes from ``OPENAI_API_KEY`` and is never printed, logged or persisted (the
DashScope logger records prompt hashes, not content).

Resume: completed calls checkpoint to ``<out>.checkpoint.jsonl`` (keyed by
``task_id``); ``--resume`` with the same ``--out`` skips tasks whose recorded
call succeeded and retries errored calls, then rewrites the predictions file
in full. The per-lane sample uses a fixed seed: same tasks file + same
``--sample-per-lane`` always selects the same subset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from autoontobench.models.dashscope import DashScopeClient, run_batch
from autoontobench.models.dashscope_runtime import CheckpointStore
from autoontobench.models.dashscope_types import (
    CallParams,
    CallRequest,
    CallResult,
)

from . import SUBMISSION_SCHEMA
from .answer_contracts import contract_hint, normalize_answer
from .providers import AnthropicClient
from .tasks import TaskRecord, TaskValidationError, load_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_REL = Path("release_v3") / "tasks" / "qualified_tasks_test.jsonl"
DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_SAMPLE_PER_LANE = 50
# Reasoning models spend output tokens on reasoning_content before the visible
# answer, so a 4096 cap truncated them (finish_reason=length). 32768 is the
# verified endpoint cap documented in configs/model_registry.json.
DEFAULT_MAX_TOKENS = 32768
SAMPLE_SEED = 20261102
ERROR_MARKER_KEY = "_error"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m benchmark.eval_20k.model_runner",
        description="Generate 20K predictions from an OpenAI-compatible endpoint "
                    "(OpenAI, vLLM, DashScope, ...) or Anthropic Messages API. "
                    "Writes a submission JSONL for benchmark.eval_20k.run plus "
                    "the sampled tasks file.")
    ap.add_argument("--tasks", default=str(REPO_ROOT / DEFAULT_TASKS_REL),
                    help="public tasks JSONL (default: %(default)s)")
    ap.add_argument("--out", required=True,
                    help="predictions JSONL to write (plus "
                         "<out>.checkpoint.jsonl and sibling tasks_sampled.jsonl)")
    ap.add_argument("--provider", default=os.environ.get("EVAL_PROVIDER", "openai"),
                    choices=("openai", "openai-compatible", "vllm", "anthropic", "claude"),
                    help="API dialect: openai/openai-compatible (includes vLLM), "
                         "or anthropic/claude native Messages API")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None,
                    help="API base URL (defaults depend on --provider/env)")
    ap.add_argument("--api-key-env", default=None,
                    help="environment variable holding the API key "
                         "(defaults: OPENAI_API_KEY / VLLM_API_KEY / ANTHROPIC_API_KEY)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-per-lane", type=int, default=DEFAULT_SAMPLE_PER_LANE,
                    help="tasks sampled per lane; 0 = all (default %(default)s)")
    ap.add_argument("--lane-sample", action="append", default=[],
                    help="per-lane override, e.g. L11=10 or L11=10,L08=20 "
                         "(repeatable; overrides --sample-per-lane for those lanes)")
    ap.add_argument("--lanes", action="append", default=[],
                    help="comma-separated lane filter, e.g. L01,L05 (repeatable)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>.checkpoint.jsonl (retry errored calls)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="output-token budget; reasoning models need headroom "
                         "(default: %(default)s)")
    # Default 0.0: deterministic decoding, and some served models (e.g.
    # kimi-k3 on DashScope) reject any non-zero temperature with a 400.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true",
                    help="enable provider thinking/reasoning (default: disabled "
                         "for qwen models so a one-click smoke cannot stall)")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="do not send response_format=json_object (needed for "
                         "some vLLM/OpenAI-compatible servers)")
    return ap


def parse_lanes(raw: list[str]) -> set[str] | None:
    lanes = {part.strip().upper() for item in raw for part in item.split(",")}
    lanes.discard("")
    return lanes or None


def parse_lane_samples(raw: list[str]) -> dict[str, int]:
    """Parse comma-separated per-lane caps: ``L11=10,L08=20``."""
    out: dict[str, int] = {}
    for item in raw:
        for part in item.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError("invalid --lane-sample %r (expected LANE=N)" % part)
            lane, _, value = part.partition("=")
            lane = lane.strip().upper()
            n = int(value.strip())
            if not lane or n < 0:
                raise ValueError("invalid --lane-sample %r" % part)
            out[lane] = n
    return out


def select_tasks(records: list[TaskRecord] | tuple[TaskRecord, ...],
                 lanes: set[str] | None, per_lane: int,
                 seed: int = SAMPLE_SEED,
                 lane_samples: dict[str, int] | None = None) -> list[TaskRecord]:
    """Fixed-seed per-lane sample, returned in original file order."""
    if per_lane < 0:
        raise ValueError("per_lane must be >= 0 (0 = all)")
    by_lane: dict[str, list[tuple[int, TaskRecord]]] = {}
    for idx, record in enumerate(records):
        if lanes is not None and record.lane_id not in lanes:
            continue
        by_lane.setdefault(record.lane_id, []).append((idx, record))
    picked: list[tuple[int, TaskRecord]] = []
    for lane in sorted(by_lane):
        rows = list(by_lane[lane])
        random.Random("%d:%s" % (seed, lane)).shuffle(rows)
        cap = (lane_samples or {}).get(lane, per_lane)
        picked.extend(rows[:cap] if cap else rows)
    picked.sort(key=lambda pair: pair[0])
    return [record for _, record in picked]


def build_messages(task: TaskRecord) -> list[dict[str, str]]:
    """Prompt built only from the public task record."""
    public = task.public
    user_goal = public.get("user_goal", "(no user_goal field; use the materials)")
    materials = {k: v for k, v in public.items()
                 if k not in ("user_goal", "allowed_tools", "budget")}
    lane_contract = contract_hint(task) or (
        "The answer payload schema is lane-specific; derive it from the "
        "user goal and the public materials above.")
    user = (
        "Task-ID: %s\n"
        "Lane: %s | Family: %s | Regime: %s | Complexity: %s\n\n"
        "## User goal\n%s\n\n"
        "## Allowed tools\n%s\n\n"
        "## Budget\n%s\n\n"
        "## Public task materials\n%s\n\n"
        "## Output contract\n"
        "Reply with exactly one JSON object and nothing else:\n"
        '{"task_id": "%s", "answer": <answer payload>}\n'
        "## Answer payload contract for lane %s (exact — the scorer is a "
        "strict native verifier, not an LLM)\n%s\n"
        "Derive the answer only from the materials above; do not invent "
        "tool outputs."
    ) % (
        task.task_id, task.lane_id, task.family_id, task.task_regime,
        task.complexity, user_goal,
        json.dumps(public.get("allowed_tools"), ensure_ascii=False),
        json.dumps(public.get("budget"), ensure_ascii=False),
        json.dumps(materials, ensure_ascii=False, sort_keys=True),
        task.task_id,
        task.lane_id,
        lane_contract,
    )
    return [
        {"role": "system", "content":
         "You are an expert solver for the AutoOntologyBench 20K benchmark. "
         "You receive one public task and must reply with a single strict "
         "JSON object — no prose, no markdown fences."},
        {"role": "user", "content": user},
    ]


def _error_answer(kind: str, detail: str) -> dict[str, str]:
    return {ERROR_MARKER_KEY: kind, "detail": detail[:300]}


def parse_answer(content: str | None, task_id: str) -> tuple[Any, str | None]:
    """Extract the answer payload; on failure return (error marker, kind)."""
    if content is None or not content.strip():
        return _error_answer("empty_content", "model returned no content"), "empty_content"
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"\s*```$", "", re.sub(r"^```[a-zA-Z]*\s*", "", text))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return _error_answer("invalid_json", "output is not JSON: %s" % exc), "invalid_json"
    if not isinstance(obj, dict):
        return _error_answer("not_object", "top-level JSON is %s" % type(obj).__name__), "not_object"
    if "answer" not in obj:
        return _error_answer("missing_answer_key",
                             "JSON object has keys %s, no 'answer'" % sorted(obj)[:8]), "missing_answer_key"
    if "task_id" in obj and obj["task_id"] != task_id:
        return _error_answer("task_id_mismatch", "expected %s got %r" % (task_id, obj["task_id"])), "task_id_mismatch"
    return obj["answer"], None


def row_from_result(task: TaskRecord, res: CallResult, model: str) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task.task_id, "model": model}
    if res.error:
        row["answer"] = _error_answer("api_error", res.error)
        row["notes"] = "api_error"
    else:
        answer, kind = parse_answer(res.content, task.task_id)
        row["answer"] = answer
        if kind:
            row["notes"] = kind
        else:
            answer, note = normalize_answer(task, answer)
            if note is not None:
                row["answer"] = answer
                row["notes"] = "normalized:" + note
    if res.usage.input is not None:
        row["tokens_input"] = res.usage.input
    if res.usage.output is not None:
        row["tokens_output"] = res.usage.output
    row["api_calls"] = max(1, res.attempts)
    row["wall_seconds"] = round(res.latency_s, 3)
    return row


async def generate(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    tasks = load_tasks(args.tasks, expected_count=None)
    lanes = parse_lanes(args.lanes)
    lane_samples = parse_lane_samples(args.lane_sample)
    selection = select_tasks(tasks.records, lanes, args.sample_per_lane,
                             lane_samples=lane_samples)
    out_path = Path(args.out)
    ckpt_path = out_path.with_name(out_path.name + ".checkpoint.jsonl")
    # main() guarantees a fresh --out when --resume is off, so the store is
    # empty then. On resume: keep successful completions, retry errored calls.
    store = CheckpointStore(ckpt_path)
    if args.resume:
        store.completed = {k: v for k, v in store.completed.items()
                           if v.error is None}
    pending = [t for t in selection if t.task_id not in store.completed]
    requests = [
        CallRequest(
            model=args.model,
            messages=build_messages(task),
            params=CallParams(temperature=args.temperature,
                              max_tokens=args.max_tokens,
                              json_mode=not args.no_json_mode,
                              extra_body=(None if args.thinking or not args.model.lower().startswith("qwen")
                                          else {"enable_thinking": False})),
            custom_id=task.task_id,
        )
        for task in pending
    ]
    if requests:
        await run_batch(client, requests, workers=args.workers, checkpoint=store)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_path = out_path.parent / "tasks_sampled.jsonl"
    stats = {"rows": 0, "api_error": 0, "parse_error": 0, "per_lane": {}}
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": SUBMISSION_SCHEMA, "model": args.model,
                             "source_tasks": str(args.tasks)},
                            ensure_ascii=False) + "\n")
        for task in selection:
            res = store.completed[task.task_id]
            row = row_from_result(task, res, args.model)
            kind = row.get("notes")
            if kind == "api_error":
                stats["api_error"] += 1
            elif kind:
                stats["parse_error"] += 1
            lane = stats["per_lane"].setdefault(task.lane_id, 0)
            stats["per_lane"][task.lane_id] = lane + 1
            stats["rows"] += 1
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with sampled_path.open("w", encoding="utf-8") as fh:
        for task in selection:
            fh.write(json.dumps(task.raw, ensure_ascii=False, sort_keys=True) + "\n")
    stats["out"] = str(out_path)
    stats["tasks_sampled"] = str(sampled_path)
    stats["checkpoint"] = str(ckpt_path)
    return stats


def _provider_name(raw: str) -> str:
    provider = raw.strip().lower()
    if provider in ("openai-compatible", "vllm"):
        return "vllm" if provider == "vllm" else "openai"
    if provider == "claude":
        return "anthropic"
    return provider


def build_client(args: argparse.Namespace, out_path: Path) -> tuple[Any, str | None]:
    provider = _provider_name(args.provider)
    if provider == "anthropic":
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            return None, "API key not set in %s" % key_env
        base_url = (args.base_url or os.environ.get("ANTHROPIC_BASE_URL")
                    or "https://api.anthropic.com/v1")
        return AnthropicClient(
            api_key=key,
            base_url=base_url,
            version=os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        ), None
    if provider == "vllm":
        key_env = args.api_key_env or "VLLM_API_KEY"
        key = (os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
               or "EMPTY")
        base_url = (args.base_url or os.environ.get("VLLM_BASE_URL")
                    or "http://127.0.0.1:8000/v1")
    else:
        key_env = args.api_key_env or "OPENAI_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            return None, "API key not set in %s" % key_env
        base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1")
    return DashScopeClient(
        api_key=key,
        base_url=base_url,
        log_root=out_path.parent / "api_logs",
        request_timeout_s=float(os.environ.get("EVAL_REQUEST_TIMEOUT_S", "180")),
        max_attempts=int(os.environ.get("EVAL_MAX_ATTEMPTS", "5")),
    ), None


async def _generate_and_close(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    try:
        return await generate(args, client)
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)
    ckpt_path = out_path.with_name(out_path.name + ".checkpoint.jsonl")
    if not args.resume and (out_path.exists() or ckpt_path.exists()):
        print("error: %s (or its checkpoint) exists; pass --resume or choose "
              "a new --out" % out_path, file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        client, error = build_client(args, out_path)
        if error:
            print("error: %s (the key is never printed or persisted)" % error,
                  file=sys.stderr)
            return 3
    try:
        stats = asyncio.run(_generate_and_close(args, client))
    except TaskValidationError as exc:
        print("error: tasks: %s" % exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print("wrote %d prediction rows -> %s" % (stats["rows"], stats["out"]))
    print("sampled tasks -> %s" % stats["tasks_sampled"])
    print("per lane: %s" % ", ".join(
        "%s=%d" % (lane, n) for lane, n in sorted(stats["per_lane"].items())))
    if stats["api_error"] or stats["parse_error"]:
        print("WARNING: %d API-error rows, %d unparseable-output rows "
              "(recorded as explicit error-marker answers, scored as failures)"
              % (stats["api_error"], stats["parse_error"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
