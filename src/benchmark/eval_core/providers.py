"""Dependency-free HTTP model clients for the Core evaluator.

Uses only the Python standard library so the release can be evaluated in a
minimal environment. The two clients speak OpenAI-compatible chat completions
and Anthropic Messages.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResult:
    content: str | None = None
    error: str | None = None
    attempts: int = 1
    tokens_input: int | None = None
    tokens_output: int | None = None
    latency_s: float = 0.0


def _post_once(url: str, payload: dict[str, Any], headers: dict[str, str],
               timeout_s: float) -> tuple[int, bytes, str | None]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), None
    except Exception as exc:  # noqa: BLE001 - reported as a model error
        return 0, b"", f"{type(exc).__name__}: {exc}"[:500]


async def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout_s: float = 180.0,
    attempts: int = 3,
) -> tuple[int, bytes, str | None, int, float]:
    started = time.monotonic()
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        status, body, error = await asyncio.to_thread(
            _post_once, url, payload, headers, timeout_s
        )
        if status and status != 429 and status < 500:
            return status, body, None, attempt, time.monotonic() - started
        if error:
            last_error = error
        elif status:
            last_error = f"HTTP {status}"
        if attempt < attempts:
            await asyncio.sleep(min(8.0, 2 ** attempt))
    return status, body, (last_error or "request failed")[:500], attempts, time.monotonic() - started


def _decode(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


class OpenAICompatibleClient:
    """OpenAI-compatible ``/v1/chat/completions`` client (works with vLLM)."""

    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def aclose(self) -> None:
        return None

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if extra_body:
            payload.update(extra_body)
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        status, body, error, attempts, latency = await _post_with_retry(
            url,
            payload,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout_s=self.timeout_s,
        )
        if error:
            return ModelResult(error=error, attempts=attempts, latency_s=latency)
        data = _decode(body)
        if not isinstance(data, dict):
            return ModelResult(
                error=f"HTTP {status}: non-JSON response",
                attempts=attempts,
                latency_s=latency,
            )
        if status >= 400:
            return ModelResult(
                error=f"HTTP {status}: {data.get('error', data)}"[:500],
                attempts=attempts,
                latency_s=latency,
            )
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        usage = data.get("usage") or {}
        result = ModelResult(
            content=content if isinstance(content, str) else None,
            attempts=attempts,
            tokens_input=usage.get("prompt_tokens"),
            tokens_output=usage.get("completion_tokens"),
            latency_s=latency,
        )
        if not result.content:
            result.error = f"empty content (finish_reason={choice.get('finish_reason')})"
        return result


class AnthropicMessagesClient:
    """Anthropic Messages API client using stdlib HTTP."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        timeout_s: float = 180.0,
        version: str = "2023-06-01",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.version = version

    async def aclose(self) -> None:
        return None

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResult:
        del json_mode, extra_body
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(str(message.get("content", "")))
            else:
                chat.append(
                    {"role": str(message.get("role", "user")),
                     "content": str(message.get("content", ""))}
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        url = self.base_url
        if not url.endswith("/messages"):
            url += "/messages"
        status, body, error, attempts, latency = await _post_with_retry(
            url,
            payload,
            {
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
                "content-type": "application/json",
            },
            timeout_s=self.timeout_s,
        )
        if error:
            return ModelResult(error=error, attempts=attempts, latency_s=latency)
        data = _decode(body)
        if not isinstance(data, dict):
            return ModelResult(
                error=f"HTTP {status}: non-JSON response",
                attempts=attempts,
                latency_s=latency,
            )
        if status >= 400:
            return ModelResult(
                error=f"HTTP {status}: {data.get('error', data)}"[:500],
                attempts=attempts,
                latency_s=latency,
            )
        blocks = data.get("content") or []
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        result = ModelResult(
            content=text or None,
            attempts=attempts,
            tokens_input=usage.get("input_tokens"),
            tokens_output=usage.get("output_tokens"),
            latency_s=latency,
        )
        if not result.content:
            result.error = f"empty content (stop_reason={data.get('stop_reason')})"
        return result