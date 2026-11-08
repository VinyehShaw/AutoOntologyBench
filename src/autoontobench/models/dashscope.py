"""DashScope async client for the Review-5 Task Factory (owner: W5).

OpenAI-compatible endpoint. Contract (W5 brief M1-1): bounded worker queue per
model (`run_batch(workers=N)`); dynamic rate limiting (AIMD spacing + 429/5xx
exponential backoff with jitter); idempotent request ids derived from
(model, prompt_sha, params); checkpoint/resume via append-only result store;
prompt_sha dedup; sanitized append-only JSONL logs under
`data/erule/review5/taskfactory/api_logs/<model>/<date>.jsonl`.
SECURITY: the API key comes from the environment (or the user env file) and is
NEVER written to any file or log line.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncOpenAI

from autoontobench.models.dashscope_runtime import (
    CallLogger, CheckpointStore, DynamicLimiter,
)
from autoontobench.models.dashscope_types import (
    CallParams, CallRequest, CallResult, Usage,
)

__all__ = [
    "CallParams", "CallRequest", "CallResult", "Usage",
    "CallLogger", "CheckpointStore", "DynamicLimiter",
    "DashScopeClient", "run_batch", "load_dashscope_env",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "data/erule/review5/taskfactory/api_logs"
DEFAULT_ENV_FILE = Path("/mnt/faster_nas/benyou.mys/.env.dashscope")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_dashscope_env(env_file: Path | None = None) -> None:
    """Load KEY=VALUE lines into os.environ (setdefault). Never prints values."""
    path = env_file or Path(os.environ.get("DASHSCOPE_ENV_FILE", DEFAULT_ENV_FILE))
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


RETRYABLE_TOKENS = ("429", "500", "502", "503", "504", "timeout", "connection", "rate")


class DashScopeClient:
    """One client per event loop: limiter/logger asyncio locks are loop-bound."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        log_root: Path = DEFAULT_LOG_ROOT,
        max_attempts: int = 5,
        request_timeout_s: float = 180.0,
    ) -> None:
        load_dashscope_env()
        key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY not set (source the env file first)")
        url = base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        self._client = AsyncOpenAI(api_key=key, base_url=url, timeout=request_timeout_s)
        self.logger = CallLogger(log_root)
        self.max_attempts = max_attempts
        self._limiters: dict[str, DynamicLimiter] = {}

    @property
    def sdk(self) -> AsyncOpenAI:
        """Raw SDK for non-chat endpoints (e.g. files/batches probe)."""
        return self._client

    def _limiter(self, model: str) -> DynamicLimiter:
        return self._limiters.setdefault(model, DynamicLimiter())

    @staticmethod
    def request_id(req: CallRequest) -> str:
        if req.custom_id:
            return req.custom_id
        blob = f"{req.model}:{req.prompt_sha}:{req.params.model_dump_json()}"
        return "req-" + hashlib.sha256(blob.encode()).hexdigest()[:24]

    async def call(self, req: CallRequest) -> CallResult:
        """Single call with retry/backoff. Always logged exactly once."""
        cid = self.request_id(req)
        limiter = self._limiter(req.model)
        res = CallResult(custom_id=cid, model_exact_id=req.model, prompt_sha=req.prompt_sha)
        for attempt in range(1, self.max_attempts + 1):
            res.attempts = attempt
            res.ts = datetime.now(timezone.utc).isoformat()
            await limiter.acquire()
            t0 = time.monotonic()
            try:
                kwargs: dict[str, Any] = {
                    "model": req.model,
                    "messages": req.messages,
                    "temperature": req.params.temperature,
                    "max_tokens": req.params.max_tokens,
                }
                if req.params.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if req.params.seed is not None:
                    kwargs["seed"] = req.params.seed
                if req.params.extra_body:
                    kwargs["extra_body"] = req.params.extra_body
                resp = await self._client.chat.completions.create(**kwargs)
                res.latency_s = time.monotonic() - t0
                choice = resp.choices[0] if resp.choices else None
                res.response_id = getattr(resp, "id", None)
                res.model_exact_id = getattr(resp, "model", req.model)
                res.content = choice.message.content if choice else None
                res.finish_reason = getattr(choice, "finish_reason", None) if choice else None
                u = getattr(resp, "usage", None)
                if u is not None:
                    res.usage = Usage(
                        input=getattr(u, "prompt_tokens", None),
                        output=getattr(u, "completion_tokens", None),
                        reasoning=getattr(getattr(u, "completion_tokens_details", None),
                                          "reasoning_tokens", None),
                    )
                res.error = None
                if res.finish_reason == "length":
                    res.error = "finish_reason=length (output budget exhausted by reasoning tokens)"
                elif res.content is None:
                    res.error = "empty content"
                limiter.on_success()
                break
            except Exception as exc:  # boundary: classify, backoff, retry or record
                res.latency_s = time.monotonic() - t0
                msg = f"{type(exc).__name__}: {exc}"[:500]
                retryable = any(tok in msg.lower() for tok in RETRYABLE_TOKENS)
                if retryable:
                    limiter.on_throttle()
                if retryable and attempt < self.max_attempts:
                    await asyncio.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
                    continue
                res.error = msg
                break
        await self.logger.log(req.params, res)
        return res


async def run_batch(
    client: DashScopeClient,
    requests: Iterable[CallRequest],
    workers: int = 8,
    checkpoint: CheckpointStore | None = None,
) -> list[CallResult]:
    """Bounded worker pool. Checkpoint resume + prompt_sha dedup. Order-preserving."""
    reqs = list(requests)
    results: list[CallResult | None] = [None] * len(reqs)

    pending: list[int] = []
    for i, req in enumerate(reqs):
        cid = DashScopeClient.request_id(req)
        if checkpoint and cid in checkpoint.completed:
            results[i] = checkpoint.completed[cid].model_copy(update={"dedup_hit": False})
        else:
            pending.append(i)

    executor_for: dict[str, int] = {}
    for i in pending:
        executor_for.setdefault(reqs[i].dedup_key(), i)

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in pending:
        if executor_for[reqs[i].dedup_key()] == i:
            queue.put_nowait(i)

    async def worker() -> None:
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res = await client.call(reqs[i])
            results[i] = res
            if checkpoint:
                checkpoint.record(res)
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(max(1, workers))))

    for i in pending:
        if results[i] is None:
            src = results[executor_for[reqs[i].dedup_key()]]
            assert src is not None
            results[i] = src.model_copy(update={
                "custom_id": DashScopeClient.request_id(reqs[i]), "dedup_hit": True})
            if checkpoint:
                checkpoint.record(results[i])  # type: ignore[arg-type]

    assert all(r is not None for r in results), "run_batch: unfulfilled request slot"
    return [r for r in results if r is not None]
