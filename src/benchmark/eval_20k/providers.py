from __future__ import annotations

import time
from typing import Any

import httpx

from autoontobench.models.dashscope_types import CallRequest, CallResult, Usage


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        *,
        version: str = "2023-06-01",
        timeout_s: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._version = version
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(self, req: CallRequest) -> CallResult:
        result = CallResult(
            custom_id=req.custom_id or req.prompt_sha,
            model_exact_id=req.model,
            prompt_sha=req.prompt_sha,
            attempts=1,
        )
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for message in req.messages:
            if message.get("role") == "system":
                system_parts.append(str(message.get("content", "")))
            else:
                messages.append(
                    {"role": str(message.get("role", "user")),
                     "content": str(message.get("content", ""))}
                )
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.params.max_tokens,
            "temperature": req.params.temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
            "content-type": "application/json",
        }
        url = self._base_url
        if not url.endswith("/messages"):
            url += "/messages"
        started = time.monotonic()
        try:
            response = await self._http.post(url, json=payload, headers=headers)
            result.latency_s = time.monotonic() - started
            data = response.json()
            if response.status_code >= 400:
                error = data.get("error")
                if isinstance(error, dict):
                    error = error.get("message") or error
                result.error = ("HTTP %s: %s" % (response.status_code, error))[:500]
                return result
            result.response_id = data.get("id")
            result.model_exact_id = data.get("model") or req.model
            result.finish_reason = data.get("stop_reason")
            blocks = data.get("content") or []
            text = "".join(
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            result.content = text or None
            usage = data.get("usage") or {}
            result.usage = Usage(
                input=usage.get("input_tokens"),
                output=usage.get("output_tokens"),
            )
            if result.finish_reason == "max_tokens":
                result.error = "finish_reason=max_tokens (output budget exhausted)"
            elif result.content is None:
                result.error = "empty content"
            return result
        except Exception as exc:
            result.latency_s = time.monotonic() - started
            result.error = ("%s: %s" % (type(exc).__name__, exc))[:500]
            return result