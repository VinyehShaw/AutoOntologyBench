"""External-knowledge cache: content-addressed objects, pinned snapshots,
resume-safe downloads with per-host adaptive concurrency and SSRF guard.

Layout (FINAL-BUILD-ORDER-v2 section 2.3)::

    external_knowledge/
      objects/sha256/<digest[0:2]>/<digest>     # content-addressed blobs
      pinned/<source_id>/<release_id>/          # materialized snapshot + MANIFEST.json
      fetch_receipts/<source_id>/<ts>_<release>.json

Download policy: HTTPS Range resume, streaming sha256, content-length check,
size cap, per-host concurrency 4 -> 8 -> 16 with backpressure on 429/5xx
(halve) and slow ramp on sustained success. SSRF allowlist hook is imported
lazily from ``autoontobench.factory.gates`` when S11 has landed it.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from autoontobench.factory.cli_support import FetchError, utc_now_iso, write_json

_CHUNK = 1 << 16
_DEFAULT_MAX_BYTES = 2 * (1 << 30)  # 2 GiB hard cap per object unless overridden


class SsrfBlocked(FetchError):
    pass


def _default_ssrf_check(url: str) -> None:
    """Fallback guard: https only; no private/loopback literal IP hosts."""
    parts = urlsplit(url)
    if parts.scheme not in ("https",):
        raise SsrfBlocked(f"{url}: scheme {parts.scheme!r} not allowed (https only)")
    host = parts.hostname or ""
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise SsrfBlocked(f"{url}: private/loopback host blocked")
    except ValueError:
        if host in ("localhost",) or host.endswith(".internal"):
            raise SsrfBlocked(f"{url}: internal hostname blocked")


def _ssrf_check(url: str) -> None:
    """Lazy S11 hook: autoontobench.factory.gates.validate_url, if present."""
    try:
        from autoontobench.factory import gates  # type: ignore[import-not-found]

        validator = getattr(gates, "validate_url", None)
        if callable(validator):
            validator(url)
            return
    except ImportError:
        pass
    _default_ssrf_check(url)


class HostThrottle:
    """Per-host adaptive concurrency: start at `initial`, ramp through `ramp`
    on sustained success, halve on 429/5xx, floor at 1."""

    def __init__(self, initial: int = 4, ramp: tuple[int, ...] = (8, 16)) -> None:
        self._initial = initial
        self._ramp = ramp
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _entry(self, host: str) -> dict[str, Any]:
        return self._state.setdefault(
            host, {"limit": self._initial, "inflight": 0, "ok_streak": 0,
                   "cond": threading.Condition(self._lock)})

    def acquire(self, host: str) -> None:
        with self._lock:
            entry = self._entry(host)
            while entry["inflight"] >= entry["limit"]:
                entry["cond"].wait(timeout=5.0)
            entry["inflight"] += 1

    def release(self, host: str, *, ok: bool, limited: bool = False) -> None:
        with self._lock:
            entry = self._entry(host)
            entry["inflight"] = max(0, entry["inflight"] - 1)
            if limited:
                entry["limit"] = max(1, entry["limit"] // 2)
                entry["ok_streak"] = 0
            elif ok:
                entry["ok_streak"] += 1
                if entry["ok_streak"] >= 32:
                    ceiling = max((self._initial, *self._ramp))
                    nxt = [r for r in (self._initial, *self._ramp) if r > entry["limit"]]
                    entry["limit"] = min(nxt[0] if nxt else ceiling, ceiling)
                    entry["ok_streak"] = 0
            entry["cond"].notify_all()

    def limits(self) -> dict[str, int]:
        with self._lock:
            return {h: e["limit"] for h, e in self._state.items()}


@dataclass(frozen=True)
class DownloadResult:
    url: str
    digest: str
    bytes_written: int
    resumed: bool
    attempts: int


class ContentAddressedCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects" / "sha256"
        self.pinned = root / "pinned"
        self.receipts = root / "fetch_receipts"
        for d in (self.objects, self.pinned, self.receipts):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- object store
    def object_path(self, digest: str) -> Path:
        return self.objects / digest[:2] / digest

    def has(self, digest: str) -> bool:
        return self.object_path(digest).is_file()

    def put_bytes(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        dest = self.object_path(digest)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        return digest

    def read_bytes(self, digest: str) -> bytes:
        return self.object_path(digest).read_bytes()

    # -------------------------------------------------------------- download
    def download(self, url: str, *, expected_sha256: str | None = None,
                 max_bytes: int = _DEFAULT_MAX_BYTES, resume: bool = True,
                 throttle: HostThrottle | None = None,
                 max_attempts: int = 5, timeout_s: float = 120.0) -> DownloadResult:
        """Resume-safe fetch into the object store. Retries honor Retry-After
        with bounded exponential backoff + jitter. Raises FetchError."""
        _ssrf_check(url)
        host = urlsplit(url).hostname or "_"
        throttle = throttle or HostThrottle()
        tmp_dir = self.root / "objects" / ".partial"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        part = tmp_dir / (hashlib.sha256(url.encode()).hexdigest() + ".part")

        attempts = 0
        delay = 2.0
        while True:
            attempts += 1
            throttle.acquire(host)
            ok, limited = False, False
            try:
                result = self._attempt(url, part, expected_sha256=expected_sha256,
                                       max_bytes=max_bytes, resume=resume,
                                       timeout_s=timeout_s, attempts=attempts)
                ok = True
                return result
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                limited = code == 429 or 500 <= code < 600
                retry_after = exc.response.headers.get("retry-after")
                if attempts >= max_attempts or (not limited and code < 500):
                    raise FetchError(f"{url}: HTTP {code} after {attempts} attempt(s)") from exc
                wait = max(_retry_after_seconds(retry_after), delay)
            except (httpx.TransportError, httpx.ReadTimeout, OSError) as exc:
                if attempts >= max_attempts:
                    raise FetchError(f"{url}: transport failure after {attempts} attempts: {exc}") from exc
                wait = delay
            finally:
                throttle.release(host, ok=ok, limited=limited)
            time.sleep(wait + random.uniform(0, delay))
            delay = min(delay * 2, 120.0)

    def _attempt(self, url: str, part: Path, *, expected_sha256: str | None,
                 max_bytes: int, resume: bool, timeout_s: float,
                 attempts: int) -> DownloadResult:
        have = part.stat().st_size if resume and part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with httpx.stream("GET", url, headers=headers, timeout=timeout_s,
                          follow_redirects=True) as resp:
            if have and resp.status_code == 200:  # server ignored Range: restart
                have = 0
            elif resp.status_code not in (200, 206):
                raise httpx.HTTPStatusError("unexpected status", request=resp.request,
                                            response=resp)
            declared = resp.headers.get("content-length")
            total = (int(declared) + have) if declared is not None else None
            if total is not None and total > max_bytes:
                part.unlink(missing_ok=True)
                raise FetchError(f"{url}: {total} bytes exceeds cap {max_bytes}")
            sha = hashlib.sha256()
            written = 0
            if have:
                sha.update(part.read_bytes())  # re-hash resumed prefix
            with part.open("ab" if have else "wb") as fh:
                for chunk in resp.iter_bytes(_CHUNK):
                    written += len(chunk)
                    if written + have > max_bytes:
                        raise FetchError(f"{url}: exceeded cap {max_bytes} mid-stream")
                    fh.write(chunk)
                    sha.update(chunk)
        digest = sha.hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise FetchError(f"{url}: sha256 mismatch {digest} != {expected_sha256}")
        dest = self.object_path(digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, dest)
        return DownloadResult(url=url, digest=digest, bytes_written=written + have,
                              resumed=bool(have), attempts=attempts)

    # --------------------------------------------------------------- receipts
    def write_receipt(self, receipt: dict[str, Any]) -> Path:
        source_id = str(receipt["source_id"])
        release = str(receipt.get("release_id", "unknown"))
        stamp = utc_now_iso().replace(":", "").replace("+", "Z")
        out = self.receipts / source_id / f"{stamp}_{release}.json"
        write_json(out, receipt)
        return out

    def materialize(self, source_id: str, release_id: str,
                    files: dict[str, str]) -> Path:
        """Link/copy content-addressed objects into pinned/<src>/<rel>/ and
        write MANIFEST.json {relpath: sha256}. Existing manifest = resume."""
        dest = self.pinned / source_id / release_id
        dest.mkdir(parents=True, exist_ok=True)
        for relpath, digest in files.items():
            target = dest / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                continue
            data = self.read_bytes(digest)  # raises KeyError-ish if absent
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        write_json(dest / "MANIFEST.json", {"source_id": source_id,
                                            "release_id": release_id,
                                            "files": files,
                                            "written_at": utc_now_iso()})
        return dest


def _retry_after_seconds(header: str | None) -> float:
    if not header:
        return 0.0
    try:
        return max(0.0, float(header))
    except ValueError:
        return 0.0


def resolve_dns(host: str) -> list[str]:
    """Exported for gates/diagnostics."""
    try:
        return sorted({ai[4][0] for ai in socket.getaddrinfo(host, 443)})
    except OSError:
        return []
