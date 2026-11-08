"""Runtime support for the DashScope client (owner: W5): sanitized call logging,
checkpoint/resume store, and the AIMD dynamic rate limiter. Split from
`dashscope.py` to keep both modules reviewable. Not part of the public surface.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoontobench.models.dashscope_types import CallParams, CallResult


class CallLogger:
    """Append-only sanitized JSONL logger. One file per model per UTC day."""

    def __init__(self, log_root: Path) -> None:
        self.log_root = log_root
        self._lock = asyncio.Lock()

    def _path(self, model: str) -> Path:
        safe = model.replace("/", "_")
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_root / safe / f"{day}.jsonl"

    async def log(self, req_params: Any, res: CallResult) -> None:
        row = {
            "ts": res.ts,
            "model_exact_id": res.model_exact_id,
            "prompt_sha": res.prompt_sha,
            "response_id": res.response_id,
            "usage": res.usage.model_dump(),
            "latency_s": round(res.latency_s, 3),
            "error": res.error,
            "custom_id": res.custom_id,
            "params": req_params.log_dict(),
            "finish_reason": res.finish_reason,
            "attempts": res.attempts,
            "dedup_hit": res.dedup_hit,
        }
        path = self._path(res.model_exact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class CheckpointStore:
    """Append-only JSONL of completed results; resume skips known custom_ids."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.completed: dict[str, CallResult] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    res = CallResult.model_validate_json(line)
                    self.completed[res.custom_id] = res

    def record(self, res: CallResult) -> None:
        self.completed[res.custom_id] = res
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(res.model_dump_json() + "\n")


class DynamicLimiter:
    """AIMD inter-request spacing per model. 429/5xx -> spacing x2; success -> x0.9."""

    def __init__(self, min_spacing_s: float = 0.05, max_spacing_s: float = 10.0) -> None:
        self._spacing = min_spacing_s
        self._min, self._max = min_spacing_s, max_spacing_s
        self._last_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            wait = self._spacing - (time.monotonic() - self._last_start)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_start = time.monotonic()

    def on_success(self) -> None:
        self._spacing = max(self._min, self._spacing * 0.9)

    def on_throttle(self) -> None:
        self._spacing = min(self._max, self._spacing * 2 + 0.1)

    @property
    def spacing(self) -> float:
        return self._spacing
