"""Persistent build queue (bank/queue.db), SQLite WAL, spec section 5.2.

Idempotency key = sha256(source_lock_hash|lane|family|base_goal|seed|
generator_version) as a UNIQUE column; re-enqueue returns the existing row.
Bounded exponential backoff + jitter, honors retry_not_before (Retry-After);
items past max_attempts become DEAD (dead-letter partition). Atomic claims via
UPDATE...RETURNING inside BEGIN IMMEDIATE. Workers commit immutable shard
files; this class is the single writer of queue state.
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

STAGES: tuple[str, ...] = (
    "source_snapshot_ready",
    "inventory_shards",
    "family_goal_shards",
    "candidate_ast",
    "statically_screened",
    "native_checked",
    "semantic_witnessed",
    "dedup_resolved",
    "language_finalized",
    "qualified",
    "release_bundle",
)

REJECTION_CATEGORIES: tuple[str, ...] = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD", "FETCHER_MISSING", "PLUGIN_MISSING",
    "SOURCE_UNPINNED", "FETCH_FAILED", "WORKER_ERROR")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT, idempotency_key TEXT NOT NULL UNIQUE,
  source_lock_hash TEXT NOT NULL, lane TEXT NOT NULL, family TEXT NOT NULL,
  base_goal TEXT NOT NULL, seed INTEGER NOT NULL, generator_version TEXT NOT NULL,
  stage_index INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'PENDING',
  attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 5,
  worker TEXT, enqueued_at REAL NOT NULL, started_at REAL, finished_at REAL,
  duration_s REAL, verdict TEXT, rejection_reason TEXT,
  cost_usd REAL NOT NULL DEFAULT 0.0, tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0, payload_hash TEXT, payload_ref TEXT,
  retry_not_before REAL, last_error TEXT, checkpoint_ref TEXT);
CREATE INDEX IF NOT EXISTS idx_wi_status ON work_items(status, stage_index);
CREATE INDEX IF NOT EXISTS idx_wi_lane ON work_items(lane, status);
"""


def idempotency_key(source_lock_hash: str, lane: str, family: str,
                    base_goal: str, seed: int, generator_version: str) -> str:
    blob = f"{source_lock_hash}|{lane}|{family}|{base_goal}|{seed}|{generator_version}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stage_name(index: int) -> str:
    return STAGES[index] if 0 <= index < len(STAGES) else "<terminal>"


@dataclass(frozen=True)
class WorkItem:
    id: int
    idempotency_key: str
    lane: str
    family: str
    base_goal: str
    seed: int
    stage_index: int
    status: str
    attempts: int
    payload_ref: str | None
    checkpoint_ref: str | None

    @property
    def stage(self) -> str:
        return stage_name(self.stage_index)


def _row(r: sqlite3.Row) -> WorkItem:
    return WorkItem(id=r["id"], idempotency_key=r["idempotency_key"], lane=r["lane"],
                    family=r["family"], base_goal=r["base_goal"], seed=r["seed"],
                    stage_index=r["stage_index"], status=r["status"], attempts=r["attempts"],
                    payload_ref=r["payload_ref"], checkpoint_ref=r["checkpoint_ref"])


class BuildQueue:
    def __init__(self, path: Path, *, max_attempts: int = 5,
                 backoff_base_s: float = 5.0, backoff_cap_s: float = 900.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._max_attempts = max_attempts
        self._base = backoff_base_s
        self._cap = backoff_cap_s

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "BuildQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- enqueue
    def enqueue(self, *, source_lock_hash: str, lane: str, family: str,
                base_goal: str, seed: int, generator_version: str,
                max_attempts: int | None = None) -> tuple[int, bool]:
        """INSERT OR IGNORE. Returns (row_id, created)."""
        key = idempotency_key(source_lock_hash, lane, family, base_goal, seed,
                              generator_version)
        now = time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO work_items (idempotency_key, source_lock_hash,"
                " lane, family, base_goal, seed, generator_version, max_attempts,"
                " enqueued_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (key, source_lock_hash, lane, family, base_goal, seed,
                 generator_version, max_attempts or self._max_attempts, now))
            if cur.rowcount:
                return int(cur.lastrowid), True
            row = self._db.execute(
                "SELECT id FROM work_items WHERE idempotency_key=?", (key,)).fetchone()
            return int(row["id"]), False

    # ------------------------------------------------------------------ claim
    def claim(self, worker: str, *, lane: str | None = None,
              now: float | None = None) -> WorkItem | None:
        """Atomically claim the oldest eligible PENDING item."""
        now = time.time() if now is None else now
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            cond = "status='PENDING' AND (retry_not_before IS NULL OR retry_not_before<=?)"
            params: list[object] = [now]
            if lane:
                cond += " AND lane=?"
                params.append(lane)
            row = self._db.execute(
                f"SELECT id FROM work_items WHERE {cond} ORDER BY enqueued_at LIMIT 1",
                params).fetchone()
            if row is None:
                return None
            cur = self._db.execute(
                "UPDATE work_items SET status='RUNNING', worker=?, started_at=?,"
                " attempts=attempts+1 WHERE id=? AND status='PENDING' RETURNING *",
                (worker, now, row["id"]))
            return _row(cur.fetchone())

    def claim_id(self, item_id: int, worker: str) -> WorkItem | None:
        """Claim one specific PENDING item (single-process pipelines that must
        advance a known row, never an unrelated pending one)."""
        now = time.time()
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            cur = self._db.execute(
                "UPDATE work_items SET status='RUNNING', worker=?, started_at=?,"
                " attempts=attempts+1 WHERE id=? AND status='PENDING' RETURNING *",
                (worker, now, item_id))
            row = cur.fetchone()
            return _row(row) if row else None

    # ---------------------------------------------------------------- outcome
    def advance(self, item_id: int, *, verdict: str, duration_s: float,
                payload_hash: str | None = None, payload_ref: str | None = None,
                checkpoint_ref: str | None = None, cost_usd: float = 0.0,
                tokens_in: int = 0, tokens_out: int = 0) -> WorkItem:
        """Mark current stage done; move to next stage or terminal DONE."""
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT stage_index FROM work_items WHERE id=?", (item_id,)).fetchone()
            next_index = int(row["stage_index"]) + 1
            terminal = next_index >= len(STAGES)
            cur = self._db.execute(
                "UPDATE work_items SET stage_index=?, status=?, finished_at=?,"
                " duration_s=?, verdict=?, payload_hash=COALESCE(?,payload_hash),"
                " payload_ref=COALESCE(?,payload_ref),"
                " checkpoint_ref=COALESCE(?,checkpoint_ref),"
                " cost_usd=cost_usd+?, tokens_in=tokens_in+?, tokens_out=tokens_out+?,"
                " started_at=NULL, worker=NULL, last_error=NULL WHERE id=? RETURNING *",
                (min(next_index, len(STAGES) - 1),
                 "DONE" if terminal else "PENDING",
                 now if terminal else None, duration_s, verdict, payload_hash,
                 payload_ref, checkpoint_ref, cost_usd, tokens_in, tokens_out, item_id))
            return _row(cur.fetchone())

    def fail(self, item_id: int, *, error: str, duration_s: float = 0.0,
             rejection_reason: str | None = None,
             retry_after_s: float | None = None) -> WorkItem:
        """Bounded exponential backoff; DEAD (dead-letter) after max_attempts."""
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT attempts, max_attempts FROM work_items WHERE id=?",
                (item_id,)).fetchone()
            attempts, max_attempts = int(row["attempts"]), int(row["max_attempts"])
            dead = attempts >= max_attempts or rejection_reason in (
                "LICENSE_HOLD", "PROFILE_UNSUPPORTED", "SOURCE_UNPINNED",
                "FETCHER_MISSING", "PLUGIN_MISSING")
            delay = min(self._cap, self._base * (2 ** max(0, attempts - 1)))
            delay += random.uniform(0, delay)
            if retry_after_s is not None:
                delay = max(delay, retry_after_s)
            cur = self._db.execute(
                "UPDATE work_items SET status=?, finished_at=?, duration_s=?,"
                " verdict='FAILED', rejection_reason=?, last_error=?,"
                " retry_not_before=?, started_at=NULL, worker=NULL"
                " WHERE id=? RETURNING *",
                ("DEAD" if dead else "PENDING", now, duration_s, rejection_reason,
                 error[:2000], None if dead else now + delay, item_id))
            return _row(cur.fetchone())

    def get(self, item_id: int) -> WorkItem | None:
        """Current state of one item (dispatch idempotency checks)."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM work_items WHERE id=?", (item_id,)).fetchone()
        return _row(row) if row else None

    def reset_item(self, item_id: int) -> WorkItem | None:
        """Explicit retry (``--resume``): clear a terminal/stale item back to
        PENDING at stage 0 with attempts/rejection state wiped. No-op on a
        PENDING item; returns the post-reset state."""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE work_items SET status='PENDING', stage_index=0, attempts=0,"
                " rejection_reason=NULL, retry_not_before=NULL, finished_at=NULL,"
                " duration_s=NULL, verdict=NULL, last_error=NULL, started_at=NULL,"
                " worker=NULL WHERE id=? AND status!='PENDING' RETURNING *", (item_id,))
            row = cur.fetchone()
            if row is None:
                row = self._db.execute(
                    "SELECT * FROM work_items WHERE id=?", (item_id,)).fetchone()
            return _row(row) if row else None

    def requeue_stale(self, *, running_older_than_s: float = 3600.0) -> int:
        """Crash recovery: RUNNING items whose claim is older than the cutoff go
        back to PENDING (worker died). Called once on queue open by the CLI."""
        cutoff = time.time() - running_older_than_s
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE work_items SET status='PENDING', started_at=NULL, worker=NULL"
                " WHERE status='RUNNING' AND started_at<?", (cutoff,))
            return cur.rowcount

    # ------------------------------------------------------------------ stats
    def counts(self) -> dict[str, int]:
        rows = self._db.execute(
            "SELECT status, COUNT(*) c FROM work_items GROUP BY status").fetchall()
        return {r["status"]: int(r["c"]) for r in rows}

    def by_lane(self) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT lane, status, COUNT(*) c FROM work_items GROUP BY lane, status"
            " ORDER BY lane").fetchall()
        out: dict[str, dict[str, object]] = {}
        for r in rows:
            out.setdefault(r["lane"], {"lane": r["lane"]})[r["status"].lower()] = int(r["c"])
        return list(out.values())

    def rejection_taxonomy(self) -> dict[str, int]:
        rows = self._db.execute(
            "SELECT rejection_reason, COUNT(*) c FROM work_items"
            " WHERE rejection_reason IS NOT NULL GROUP BY rejection_reason").fetchall()
        return {r["rejection_reason"]: int(r["c"]) for r in rows}

    def stage_utilization(self) -> dict[str, dict[str, int]]:
        rows = self._db.execute(
            "SELECT stage_index, status, COUNT(*) c FROM work_items"
            " GROUP BY stage_index, status").fetchall()
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(stage_name(int(r["stage_index"])), {})[r["status"].lower()] = int(r["c"])
        return out

    def queue_wait_percentiles(self) -> dict[str, float | None]:
        waits = [float(r["w"]) for r in self._db.execute(
            "SELECT (started_at - enqueued_at) w FROM work_items"
            " WHERE started_at IS NOT NULL AND enqueued_at IS NOT NULL").fetchall()]
        if not waits:
            return {"p50_s": None, "p95_s": None}
        waits.sort()
        p50 = waits[len(waits) // 2]
        p95 = waits[min(len(waits) - 1, int(len(waits) * 0.95))]
        return {"p50_s": round(p50, 3), "p95_s": round(p95, 3)}

    def terminal_rows(self, *, done_only: bool = False) -> list[sqlite3.Row]:
        cond = "status='DONE'" if done_only else "status IN ('DONE','DEAD')"
        return list(self._db.execute(
            f"SELECT * FROM work_items WHERE {cond} ORDER BY id").fetchall())

    def dead_letter(self) -> list[sqlite3.Row]:
        return list(self._db.execute(
            "SELECT * FROM work_items WHERE status='DEAD' ORDER BY id").fetchall())

    def distinct(self, column: str, *, status: str | None = None) -> int:
        if column not in ("family", "base_goal", "lane"):
            raise ValueError(f"unsupported distinct column {column!r}")
        sql = f"SELECT COUNT(DISTINCT {column}) c FROM work_items"
        params: tuple[object, ...] = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        return int(self._db.execute(sql, params).fetchone()["c"])
