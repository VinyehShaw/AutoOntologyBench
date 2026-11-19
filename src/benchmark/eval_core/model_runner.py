#!/usr/bin/env python3
"""Generate Core-track answers from an API or local OpenAI-compatible server.

This is the optional model front-end. It never scores; use
``benchmark.eval_core.run`` with a private oracle for that.

Providers:
  openai / openai-compatible  -- OpenAI-compatible /v1/chat/completions
  vllm                        -- local OpenAI-compatible vLLM server
  anthropic / claude          -- Anthropic Messages API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import SUBMISSION_SCHEMA
from .providers import AnthropicMessagesClient, ModelResult, OpenAICompatibleClient
from .tasks import (
    TaskRecord,
    TaskValidationError,
    default_tasks_path,
    load_tasks,
    parse_csv,
    select_tasks,
)

DEFAULT_TASKS = default_tasks_path()
DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_SAMPLE_PER_WORLD = 10
DEFAULT_MAX_TOKENS = 16384
ERROR_KEY = "_error"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.eval_core.model_runner",
        description="Generate Core-track answers from an API or vLLM endpoint.",
    )
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", required=True, help="predictions JSONL to write")
    parser.add_argument(
        "--provider",
        default=os.environ.get("EVAL_PROVIDER", "openai"),
        choices=("openai", "openai-compatible", "vllm", "anthropic", "claude"),
    )
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-per-world", type=int, default=DEFAULT_SAMPLE_PER_WORLD)
    parser.add_argument("--worlds", default="")
    parser.add_argument("--lanes", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    return parser


def build_messages(task: TaskRecord) -> list[dict[str, str]]:
    actions = json.dumps(task.actions, ensure_ascii=False, sort_keys=True)
    user = f"""Task ID: {task.release_task_id}
World: {task.world_id}
Lane: {task.lane_id}
Split: {task.split}
Spec: {task.raw.get('spec_version', '')}
Actions: {actions}
Instance IDs: {json.dumps(task.raw.get('instance_ids', []), ensure_ascii=False)}
Public features: {json.dumps(task.raw.get('features', []), ensure_ascii=False)}

## Task
{task.prompt}

## Required output
Reply with exactly one JSON object and no prose:
{{"task_id": "{task.release_task_id}", "answer": <answer>}}

Answer contract:
- Query: `<answer>` is the exact result payload (JSON list/dict/string/number).
- Validation/conformance: `<answer>` is "ok" or "reject:<reason>".
- Multi-step: `<answer>` is a JSON list with one entry per action; each entry is
  the action payload or {{"ok": true/false, "reason": "...", "payload": ...}}.
Do not invent tool output. Return JSON only.
"""
    return [
        {
            "role": "system",
            "content": (
                "You are solving the AutoOntologyBench Core-track benchmark. "
                "Return strict JSON only; the scorer is deterministic."
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_answer(content: str | None, task_id: str) -> tuple[Any, str | None]:
    if content is None or not content.strip():
        return {ERROR_KEY: "empty_content", "detail": "model returned no content"}, "empty_content"
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"\s*```$", "", re.sub(r"^```[a-zA-Z]*\s*", "", text))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return {ERROR_KEY: "invalid_json", "detail": str(exc)[:300]}, "invalid_json"
    if not isinstance(obj, dict):
        return {ERROR_KEY: "not_object", "detail": type(obj).__name__}, "not_object"
    if obj.get("task_id") not in (None, task_id):
        return {ERROR_KEY: "task_id_mismatch", "detail": str(obj.get("task_id"))}, "task_id_mismatch"
    if "answer" not in obj:
        return {ERROR_KEY: "missing_answer", "detail": str(sorted(obj)[:8])}, "missing_answer"
    return obj["answer"], None


def _provider_name(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("openai", "openai-compatible", "vllm"):
        return raw
    if raw in ("anthropic", "claude"):
        return "anthropic"
    raise ValueError(f"unsupported provider: {raw!r}")


def build_client(args: argparse.Namespace) -> tuple[Any, str]:
    provider = _provider_name(args.provider)
    if provider == "anthropic":
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set")
        base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
        return AnthropicMessagesClient(api_key=key, base_url=base_url), "anthropic"
    key_env = args.api_key_env
    if provider == "vllm":
        key_env = key_env or "VLLM_API_KEY"
        key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        base_url = args.base_url or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
    else:
        key_env = key_env or "OPENAI_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set")
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAICompatibleClient(base_url=base_url, api_key=key), provider


def _row_from_result(task: TaskRecord, result: ModelResult, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task.release_task_id, "model": args.model}
    if result.error:
        row["answer"] = {ERROR_KEY: "api_error", "detail": result.error[:300]}
        row["notes"] = "api_error"
    else:
        answer, kind = parse_answer(result.content, task.release_task_id)
        row["answer"] = answer
        if kind:
            row["notes"] = kind
    if result.tokens_input is not None:
        row["tokens_input"] = result.tokens_input
    if result.tokens_output is not None:
        row["tokens_output"] = result.tokens_output
    row["api_calls"] = max(1, result.attempts)
    row["wall_seconds"] = round(result.latency_s, 3)
    return row


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("task_id"), str):
                rows[row["task_id"]] = row
    return rows


async def generate(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(args.tasks, expected_count=None)
    selected = select_tasks(
        tasks.records,
        worlds=parse_csv(args.worlds),
        lanes=parse_csv(args.lanes),
        sample_per_world=args.sample_per_world,
        limit=args.limit,
    )
    if not selected:
        raise RuntimeError("no tasks selected")
    out_path = Path(args.out)
    checkpoint_path = out_path.with_name(out_path.name + ".checkpoint.jsonl")
    if not args.resume and (out_path.exists() or checkpoint_path.exists()):
        raise RuntimeError(f"{out_path} or its checkpoint exists; pass --resume")
    existing = _load_checkpoint(checkpoint_path) if args.resume else {}
    pending = [task for task in selected if task.release_task_id not in existing]
    client, provider = build_client(args)
    rows: dict[str, dict[str, Any]] = dict(existing)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    lock = asyncio.Lock()

    async def one(task: TaskRecord) -> None:
        async with semaphore:
            extra_body = None if (
                args.thinking or not args.model.lower().startswith("qwen")
            ) else {"enable_thinking": False}
            result = await client.call(
                model=args.model,
                messages=build_messages(task),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                json_mode=not args.no_json_mode,
                extra_body=extra_body,
            )
            row = _row_from_result(task, result, args)
            async with lock:
                rows[task.release_task_id] = row
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    try:
        await asyncio.gather(*(one(task) for task in pending))
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()

    ordered = select_tasks(
        tasks.records,
        worlds=parse_csv(args.worlds),
        lanes=parse_csv(args.lanes),
        sample_per_world=args.sample_per_world,
        limit=args.limit,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_path = out_path.parent / "tasks_sampled.jsonl"
    with sampled_path.open("w", encoding="utf-8") as handle:
        for task in ordered:
            handle.write(json.dumps(task.raw, ensure_ascii=False, sort_keys=True) + "\n")

    stats = {"rows": 0, "api_error": 0, "parse_error": 0, "per_world": {}}
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": SUBMISSION_SCHEMA,
                    "model": args.model,
                    "provider": provider,
                    "source_tasks": str(args.tasks),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for task in ordered:
            row = rows.get(task.release_task_id) or {
                "task_id": task.release_task_id,
                "answer": {ERROR_KEY: "missing_result"},
                "notes": "missing_result",
            }
            if row.get("notes") == "api_error":
                stats["api_error"] += 1
            elif row.get("notes"):
                stats["parse_error"] += 1
            stats["rows"] += 1
            stats["per_world"][task.world_id] = stats["per_world"].get(task.world_id, 0) + 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stats.update(
        {
            "out": str(out_path),
            "tasks_sampled": str(sampled_path),
            "checkpoint": str(checkpoint_path),
        }
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats = asyncio.run(generate(args))
    except (TaskValidationError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())