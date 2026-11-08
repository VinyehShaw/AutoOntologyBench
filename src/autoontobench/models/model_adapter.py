"""Model adapter protocol + DashScope wiring (owner: W5).

Task-factory stages consume `JsonModel` (parse-at-the-boundary JSON completion),
never raw SDK objects. Every call goes through DashScopeClient so logging,
checkpointing and dedup are uniform.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from autoontobench.models.dashscope import CallParams, CallRequest, CallResult, DashScopeClient


class JsonCompletionError(Exception):
    """Raised when the model call succeeded but the body is not parseable JSON."""


class JsonModel(Protocol):
    async def complete_json(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        temperature: float = 0.3, seed: int | None = None, custom_id: str | None = None,
    ) -> tuple[dict[str, Any], CallResult]: ...


class DashScopeJsonModel:
    def __init__(self, client: DashScopeClient, model: str) -> None:
        self.client = client
        self.model = model

    async def complete_json(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        temperature: float = 0.3, seed: int | None = None, custom_id: str | None = None,
    ) -> tuple[dict[str, Any], CallResult]:
        req = CallRequest(model=self.model, messages=messages, custom_id=custom_id,
                          params=CallParams(temperature=temperature, max_tokens=max_tokens,
                                            json_mode=True, seed=seed))
        res = await self.client.call(req)
        if res.error or res.content is None:
            raise JsonCompletionError(f"{self.model} call failed: {res.error}")
        try:
            return json.loads(res.content), res
        except json.JSONDecodeError as exc:
            raise JsonCompletionError(
                f"{self.model} returned non-JSON despite json_object mode: {exc}") from exc
