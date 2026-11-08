from __future__ import annotations

import asyncio
import json

import httpx

from autoontobench.models.dashscope_types import CallParams, CallRequest
from benchmark.eval_20k import model_runner
from benchmark.eval_20k.providers import AnthropicClient


def test_anthropic_messages_request_and_response():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "model": "claude-test",
                "content": [{"type": "text", "text": '{"answer": 42}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = AnthropicClient(
        "sk-test",
        "https://api.example.test/v1",
        transport=httpx.MockTransport(handler),
    )
    request = CallRequest(
        model="claude-test",
        messages=[
            {"role": "system", "content": "follow the schema"},
            {"role": "user", "content": "solve this"},
        ],
        params=CallParams(temperature=0, max_tokens=128, json_mode=True),
        custom_id="task-1",
    )
    result = asyncio.run(client.call(request))
    asyncio.run(client.aclose())

    body = seen["body"]
    assert body["model"] == "claude-test"
    assert body["system"] == "follow the schema"
    assert body["messages"] == [{"role": "user", "content": "solve this"}]
    assert body["max_tokens"] == 128
    assert "response_format" not in body
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert result.content == '{"answer": 42}'
    assert result.usage.input == 10
    assert result.usage.output == 5
    assert result.error is None
    assert result.model_exact_id == "claude-test"


def test_build_vllm_client_does_not_require_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    args = model_runner.build_parser().parse_args([
        "--out", str(tmp_path / "preds.jsonl"),
        "--provider", "vllm",
        "--model", "local-model",
    ])
    client, error = model_runner.build_client(args, tmp_path / "preds.jsonl")
    assert error is None
    assert client is not None


def test_build_anthropic_client_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = model_runner.build_parser().parse_args([
        "--out", str(tmp_path / "preds.jsonl"),
        "--provider", "anthropic",
        "--model", "claude-test",
    ])
    client, error = model_runner.build_client(args, tmp_path / "preds.jsonl")
    assert client is None
    assert "ANTHROPIC_API_KEY" in error