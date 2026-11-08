"""Typed append-only tool-call log for the Review-5 tool API (owner: W6).

One :class:`ToolLogRow` per executed tool call, streamed to JSONL as it
happens (append-only within a run). Carries the Review-5B §5 required fields:
``tool_call_id``, ``action`` (tool args, sanitized), ``observation_hash``
(sha256 of the canonical observation payload returned to the agent),
``charge``, ``wall_time``, ``o_sha`` (ontology hash in play), and the
``channel`` native/public distinction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

CHANNELS = ("native", "ontology", "artifact", "public", "session")
_STRING_CAP = 500


def canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sanitize_args(args: Any) -> Any:
    """Cap long strings in tool args before they enter the log row."""
    return _sanitize(args, _STRING_CAP)


def _sanitize(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= cap else value[:cap] + "...<truncated>"
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v, cap) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, cap) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)[:cap]


@dataclass(frozen=True)
class ToolLogRow:
    tool_call_id: str
    seq: int
    tool: str
    action: dict[str, Any]
    ok: bool
    error: str | None
    charge: dict[str, int]
    observation_sha: str
    o_sha: str | None
    channel: str
    wall_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolLogWriter:
    """Append-only JSONL writer (one line per row, flushed per write)."""

    def __init__(self, path: Path | None):
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: ToolLogRow) -> None:
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
