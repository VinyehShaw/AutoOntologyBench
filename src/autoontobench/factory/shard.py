"""Shard framework: deterministic candidate sharding with immutable commits and
per-commit checkpoints (FINAL-BUILD-ORDER-v2 section 5.2).

A shard is the unit of resumable work: ~`candidate_size` candidates (default
250; adaptive down when the remaining quota for a family is smaller). Workers
write immutable shard files under ``bank/staging/<lane>/<shard_id>.jsonl``
(tmp + fsync + os.replace); the single writer then commits the manifest row.
``resume()`` lists already-committed shards so a restarted build never
recomputes them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SHARD_SIZE = 250


@dataclass(frozen=True)
class ShardSpec:
    lane: str
    family: str
    shard_id: str
    seed_start: int
    candidate_count: int


def plan_shards(lane: str, family: str, base_goal: str, total_candidates: int,
                *, size: int = DEFAULT_SHARD_SIZE, seed: int = 0) -> list[ShardSpec]:
    """Split a family's candidate budget into shards of <= `size`; the tail
    shard shrinks adaptively instead of padding with filler candidates."""
    if total_candidates <= 0:
        return []
    size = max(1, size)
    specs: list[ShardSpec] = []
    remaining = total_candidates
    index = 0
    while remaining > 0:
        n = min(size, remaining)
        shard_id = f"{lane}-{family}-{base_goal}-s{index:04d}"
        specs.append(ShardSpec(lane=lane, family=family, shard_id=shard_id,
                               seed_start=seed + index * size, candidate_count=n))
        remaining -= n
        index += 1
    return specs


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardStore:
    """Immutable shard files + sidecar checkpoints under a staging root."""

    def __init__(self, staging_root: Path) -> None:
        self.root = staging_root
        self.root.mkdir(parents=True, exist_ok=True)

    def shard_path(self, spec: ShardSpec) -> Path:
        return self.root / spec.lane / f"{spec.shard_id}.jsonl"

    def checkpoint_path(self, spec: ShardSpec) -> Path:
        return self.root / spec.lane / f"{spec.shard_id}.checkpoint.json"

    def is_committed(self, spec: ShardSpec) -> bool:
        cp = self.checkpoint_path(spec)
        sp = self.shard_path(spec)
        if not (cp.is_file() and sp.is_file()):
            return False
        meta = json.loads(cp.read_text(encoding="utf-8"))
        return bool(meta.get("sha256")) and meta["sha256"] == _sha256_file(sp)

    def commit(self, spec: ShardSpec, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Write shard immutably. Re-commit with identical content = no-op."""
        rows = list(records)
        target = self.shard_path(spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        digest = _sha256_file(tmp)
        checkpoint = self.checkpoint_path(spec)
        if target.exists():
            if _sha256_file(target) == digest and self.is_committed(spec):
                tmp.unlink(missing_ok=True)
                return json.loads(checkpoint.read_text(encoding="utf-8"))
            raise ValueError(f"immutable shard exists with different content: {target}")
        os.replace(tmp, target)
        meta: dict[str, Any] = {
            "shard_id": spec.shard_id, "lane": spec.lane, "family": spec.family,
            "candidate_count": len(rows), "sha256": digest, "path": str(target),
        }
        cp_tmp = checkpoint.with_suffix(".tmp")
        cp_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(cp_tmp, checkpoint)
        return meta

    def committed(self, lane: str | None = None) -> list[dict[str, Any]]:
        """All committed shard checkpoints (resume source of truth)."""
        lanes = [self.root / lane] if lane else sorted(p for p in self.root.iterdir() if p.is_dir())
        out: list[dict[str, Any]] = []
        for lane_dir in lanes:
            for cp in sorted(lane_dir.glob("*.checkpoint.json")):
                out.append(json.loads(cp.read_text(encoding="utf-8")))
        return out
