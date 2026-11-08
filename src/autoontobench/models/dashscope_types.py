"""Typed payloads for the DashScope client (owner: W5). Pure pydantic models;
no I/O, no SDK. Split out so runtime and client stay small and reviewable.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class CallParams(BaseModel):
    temperature: float = 0.3
    max_tokens: int = 2048  # reasoning models need generous budgets (see MODEL_PROBE.md)
    json_mode: bool = True
    seed: int | None = None
    extra_body: dict[str, Any] | None = None

    def log_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens,
                "json_mode": self.json_mode, "seed": self.seed}


class Usage(BaseModel):
    input: int | None = None
    output: int | None = None
    reasoning: int | None = None


class CallRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    params: CallParams = Field(default_factory=CallParams)
    custom_id: str | None = None  # default: derived idempotency key

    @property
    def prompt_sha(self) -> str:
        blob = json.dumps(self.messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def dedup_key(self) -> str:
        return f"{self.model}:{self.prompt_sha}:{self.params.model_dump_json()}"


class CallResult(BaseModel):
    custom_id: str
    model_exact_id: str
    prompt_sha: str
    response_id: str | None = None
    content: str | None = None
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    error: str | None = None
    attempts: int = 0
    dedup_hit: bool = False
    ts: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.content is not None
