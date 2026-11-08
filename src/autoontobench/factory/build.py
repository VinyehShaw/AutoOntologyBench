"""Build driver + telemetry for the Benchmark Factory (FINAL-BUILD-ORDER-v2
sections 5.2/5.5/5.6). Owns: config-driven lane orchestration over the
persistent queue, the five telemetry metric groups (single writer of
BUILD_THROUGHPUT.jsonl), REJECTION_REASONS.csv, and the per-candidate stage
executor shared by the official build and BYOK generation.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from autoontobench.factory.cli_support import (
    LoadedConfig, append_jsonl, utc_now_iso, write_json)
from autoontobench.factory.queue import BuildQueue, WorkItem


@dataclass(frozen=True)
class StageResult:
    verdict: str
    payload_hash: str | None = None
    payload_ref: str | None = None
    checkpoint_ref: str | None = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


#: A stage handler consumes a claimed work item and returns StageResult, or
#: raises StageFailure(reason_category, detail). Stage name -> handler.
StageHandler = Callable[[WorkItem], StageResult]


class StageFailure(Exception):
    def __init__(self, reason: str, detail: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(detail)
        self.reason = reason
        self.retry_after_s = retry_after_s


# ----------------------------------------------------------------- telemetry

def resource_probe() -> dict[str, Any]:
    out: dict[str, Any] = {"cpu_busy_pct": None, "ram_used_gb": None,
                           "disk_free_gb": None, "disk_free_inodes": None}
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            vals = [int(p) for p in fh.readline().split()[1:8]]
        idle, total = vals[3] + vals[4], sum(vals)
        prev = getattr(resource_probe, "_prev", None)
        resource_probe._prev = (idle, total)  # type: ignore[attr-defined]
        if prev:
            d_idle, d_total = idle - prev[0], total - prev[1]
            if d_total > 0:
                out["cpu_busy_pct"] = round(100.0 * (1 - d_idle / d_total), 1)
    except OSError:
        pass
    try:
        mem: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                mem[k] = int(rest.strip().split()[0])
        out["ram_used_gb"] = round((mem["MemTotal"] - mem["MemAvailable"]) / 2**20, 1)
    except (OSError, KeyError, ValueError):
        pass
    try:
        usage = shutil.disk_usage(Path.cwd())
        out["disk_free_gb"] = round(usage.free / 2**30, 1)
        out["disk_free_inodes"] = os.statvfs(Path.cwd()).f_favail
    except OSError:
        pass
    return out


def collect_metrics(queue: BuildQueue, *, build_id: str, window_s: float = 300.0,
                    quota: int = 8000) -> dict[str, Any]:
    """The five metric groups of spec section 5.5."""
    counts = queue.counts()
    rows = queue.terminal_rows()
    now = time.time()
    in_window = [r for r in rows if r["finished_at"] and r["finished_at"] >= now - 3600]
    done_window = [r for r in in_window if r["status"] == "DONE"]
    rejected = queue.rejection_taxonomy()
    total_terminal = len(rows) or 1
    done_total = counts.get("DONE", 0)
    eta: dict[str, Any] = {"estimated_remaining_qualified": max(0, quota - done_total),
                           "rolling_qualified_per_hour": round(len(done_window), 2)}
    rate = len(done_window)
    eta["eta_hours_internal_only"] = (
        round(eta["estimated_remaining_qualified"] / rate, 2) if rate else None)
    return {
        "ts": utc_now_iso(), "build_id": build_id,
        "rates": {
            "qualified_per_min": round(len(done_window) / 60.0, 3),
            "candidate_per_min": round(len(in_window) / 60.0, 3),
            "verified_per_min": round(len(done_window) / 60.0, 3),
            "active_mechanism_groups": queue.distinct("family"),
            "unique_base_goals": queue.distinct("base_goal"),
        },
        "yield": {
            "native_qualification_yield": round(done_total / total_terminal, 4),
            "rejections": rejected,
        },
        "stages": {
            "queue_wait": queue.queue_wait_percentiles(),
            "by_stage": queue.stage_utilization(),
            "counts": counts,
        },
        "resources": resource_probe(),
        "eta": eta,
    }


class TelemetryWriter:
    """Single writer for BUILD_THROUGHPUT.jsonl + REJECTION_REASONS.csv."""

    def __init__(self, telemetry_path: Path, failures_path: Path) -> None:
        self.telemetry_path = telemetry_path
        self.failures_path = failures_path

    def tick(self, queue: BuildQueue, *, build_id: str, quota: int) -> dict[str, Any]:
        snapshot = collect_metrics(queue, build_id=build_id, quota=quota)
        append_jsonl(self.telemetry_path, snapshot)
        self._write_rejections(queue)
        return snapshot

    def _write_rejections(self, queue: BuildQueue) -> None:
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["item_id", "lane", "family", "base_goal", "stage",
                    "rejection_reason", "detail", "finished_at"])
        for r in queue.terminal_rows():
            if r["rejection_reason"]:
                w.writerow([r["id"], r["lane"], r["family"], r["base_goal"],
                            f"stage_{r['stage_index']}", r["rejection_reason"],
                            (r["last_error"] or "")[:300], r["finished_at"]])
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.failures_path.with_suffix(".csv.tmp")
        tmp.write_text(buf.getvalue(), encoding="utf-8")
        os.replace(tmp, self.failures_path)

    def loop(self, queue: BuildQueue, *, build_id: str, quota: int,
             interval_s: float, stop: threading.Event) -> None:
        while not stop.wait(interval_s):
            self.tick(queue, build_id=build_id, quota=quota)


# --------------------------------------------------------------- run driver

def run_worker(queue: BuildQueue, *, worker_id: str, lane: str | None,
               handlers: dict[str, StageHandler], stop: threading.Event,
               idle_exit_s: float = 5.0) -> int:
    """Claim loop: dispatch each item's current stage to its handler. Returns
    the number of stage transitions executed."""
    done = 0
    idle_since = time.time()
    while not stop.is_set():
        item = queue.claim(worker_id, lane=lane)
        if item is None:
            if time.time() - idle_since >= idle_exit_s:
                break
            time.sleep(0.2)
            continue
        idle_since = time.time()
        handler = handlers.get(item.stage)
        started = time.time()
        if handler is None:
            queue.fail(item.id, error=f"no handler for stage {item.stage}",
                       rejection_reason="PLUGIN_MISSING")
            continue
        try:
            result = handler(item)
        except StageFailure as exc:
            queue.fail(item.id, error=str(exc), rejection_reason=exc.reason,
                       retry_after_s=exc.retry_after_s)
        except Exception as exc:  # worker boundary: unexpected = retryable error
            queue.fail(item.id, error=f"{type(exc).__name__}: {exc}",
                       rejection_reason="WORKER_ERROR")
        else:
            queue.advance(item.id, verdict=result.verdict, duration_s=time.time() - started,
                          payload_hash=result.payload_hash, payload_ref=result.payload_ref,
                          checkpoint_ref=result.checkpoint_ref, cost_usd=result.cost_usd,
                          tokens_in=result.tokens_in, tokens_out=result.tokens_out)
            done += 1
    return done
