"""SSRF-safe URL validation + archive expansion guards (owner: S11).

FINAL-BUILD-ORDER-v2 §6.3: user-supplied URLs/plugins must never reach private
network ranges; downloads get byte + decompression limits. This module is the
single funnel all factory fetch paths must pass through.

URL policy:
- scheme allowlist (default: https only)
- no userinfo; hostname required
- DNS resolution happens once inside `resolve_pinned`; EVERY resolved address
  must be globally routable unicast (`ip.is_global`). Private, loopback,
  link-local, multicast, reserved, unspecified -> rejected.
- DNS rebinding guard: `PinnedTarget` carries the validated IP set; the caller
  must dial one of those exact IPs and can re-check with `validate_against_pin`.

Archive policy (zip/tar):
- member count cap, total expanded byte cap, per-member cap, compression
  ratio cap (zip-bomb), path traversal (`..`, absolute, drive-letter) and
  link/device members rejected. Dispatch is on magic bytes, not extensions.
"""
from __future__ import annotations

import ipaddress
import socket
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"https"})
MAX_REDIRECTS = 5


@dataclass(frozen=True)
class UrlDecision:
    url: str
    allowed: bool
    reason: str
    resolved_ips: tuple[str, ...] = ()


@dataclass(frozen=True)
class PinnedTarget:
    """DNS rebinding guard: hostname + the exact IP set validated for it."""

    hostname: str
    port: int
    ips: tuple[str, ...]
    scheme: str

    def validate_against_pin(self, dialed_ips: Iterable[str]) -> bool:
        """The IP set actually dialed must be a subset of the pinned set."""
        return set(dialed_ips).issubset(set(self.ips))


def _resolve_host(
    hostname: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    out: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for _family, _type, _proto, _canon, sockaddr in infos:
        out.add(ipaddress.ip_address(sockaddr[0]))
    return tuple(sorted(out, key=str))


def validate_url(
    url: str,
    *,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
    resolve: bool = True,
) -> UrlDecision:
    """Validate one URL. The only network I/O performed is DNS resolution."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return UrlDecision(url, False, f"unparseable_url:{exc}")
    scheme = parts.scheme.lower()
    if scheme not in allowed_schemes:
        return UrlDecision(url, False, f"scheme_not_allowed:{scheme or '<none>'}")
    if parts.username is not None or parts.password is not None:
        return UrlDecision(url, False, "userinfo_rejected")
    host = parts.hostname
    if not host:
        return UrlDecision(url, False, "missing_hostname")
    # IP literal: check directly, no DNS needed.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if literal.is_global:
            return UrlDecision(url, True, "ok_ip_literal", (str(literal),))
        return UrlDecision(url, False, f"forbidden_ip:{literal}")
    if not resolve:
        return UrlDecision(url, True, "ok_unresolved", ())
    try:
        ips = _resolve_host(host)
    except socket.gaierror as exc:
        return UrlDecision(url, False, f"dns_failure:{exc}")
    if not ips:
        return UrlDecision(url, False, "dns_empty")
    str_ips = tuple(str(i) for i in ips)
    for ip in ips:
        if not ip.is_global:
            return UrlDecision(url, False, f"forbidden_resolved_ip:{ip}", str_ips)
    return UrlDecision(url, True, "ok", str_ips)


def resolve_pinned(
    url: str,
    *,
    port: int | None = None,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
) -> PinnedTarget:
    """Validate + pin DNS. Raises ValueError when the URL is not allowed."""
    decision = validate_url(url, allowed_schemes=allowed_schemes, resolve=True)
    if not decision.allowed:
        raise ValueError(f"url rejected: {decision.reason} ({url})")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    default_port = 443 if scheme == "https" else 80
    host = parts.hostname
    assert host is not None  # validate_url guarantees a hostname
    ips = decision.resolved_ips
    if not ips:  # defensive fallback; validate_url already resolved
        ips = tuple(str(i) for i in _resolve_host(host))
    return PinnedTarget(
        hostname=host, port=port or parts.port or default_port, ips=ips, scheme=scheme
    )


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 10_000
    max_total_expanded_bytes: int = 1 << 31  # 2 GiB
    max_member_bytes: int = 1 << 30  # 1 GiB
    max_compression_ratio: float = 100.0
    max_archive_bytes: int = 1 << 31


@dataclass(frozen=True)
class ArchiveReport:
    path: str
    ok: bool
    reason: str
    members: int = 0
    total_expanded_bytes: int = 0
    max_ratio: float = 0.0


def _member_name_safe(name: str) -> bool:
    if not name or name.startswith(("/", "\\")):
        return False
    if len(name) > 1 and name[1] == ":":  # windows drive letter
        return False
    return ".." not in name.replace("\\", "/").split("/")


def inspect_zip(path: Path, limits: ArchiveLimits = ArchiveLimits()) -> ArchiveReport:
    """Static zip inspection: no member is extracted."""
    try:
        if path.stat().st_size > limits.max_archive_bytes:
            return ArchiveReport(str(path), False, "archive_too_large")
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > limits.max_members:
                return ArchiveReport(str(path), False, "too_many_members", members=len(infos))
            total = 0
            worst_ratio = 0.0
            for info in infos:
                if not _member_name_safe(info.filename):
                    return ArchiveReport(str(path), False, f"unsafe_member_name:{info.filename}")
                total += info.file_size
                if info.file_size > limits.max_member_bytes:
                    return ArchiveReport(
                        str(path), False, f"member_too_large:{info.filename}",
                        members=len(infos), total_expanded_bytes=total,
                    )
                if info.compress_size > 0:
                    worst_ratio = max(worst_ratio, info.file_size / info.compress_size)
            if total > limits.max_total_expanded_bytes:
                return ArchiveReport(
                    str(path), False, "expanded_size_exceeded", members=len(infos),
                    total_expanded_bytes=total, max_ratio=worst_ratio,
                )
            if worst_ratio > limits.max_compression_ratio:
                return ArchiveReport(
                    str(path), False, "compression_ratio_exceeded", members=len(infos),
                    total_expanded_bytes=total, max_ratio=worst_ratio,
                )
            return ArchiveReport(str(path), True, "ok", members=len(infos),
                                 total_expanded_bytes=total, max_ratio=worst_ratio)
    except zipfile.BadZipFile:
        return ArchiveReport(str(path), False, "bad_zip")


_TAR_FORBIDDEN_TYPES = frozenset(
    t
    for t in (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
        getattr(tarfile, "DEVTYPE", None),
    )
    if t is not None
)


def inspect_tar(path: Path, limits: ArchiveLimits = ArchiveLimits()) -> ArchiveReport:
    """Static tar inspection (handles .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz)."""
    try:
        if path.stat().st_size > limits.max_archive_bytes:
            return ArchiveReport(str(path), False, "archive_too_large")
        with tarfile.open(path, mode="r:*") as tf:
            members = tf.getmembers()
            if len(members) > limits.max_members:
                return ArchiveReport(str(path), False, "too_many_members", members=len(members))
            total = 0
            for member in members:
                if not _member_name_safe(member.name):
                    return ArchiveReport(str(path), False, f"unsafe_member_name:{member.name}")
                if member.type in _TAR_FORBIDDEN_TYPES:
                    return ArchiveReport(
                        str(path), False, f"forbidden_member_type:{member.name}"
                    )
                if member.isfile():
                    total += member.size
                    if member.size > limits.max_member_bytes:
                        return ArchiveReport(
                            str(path), False, f"member_too_large:{member.name}",
                            members=len(members), total_expanded_bytes=total,
                        )
            if total > limits.max_total_expanded_bytes:
                return ArchiveReport(
                    str(path), False, "expanded_size_exceeded",
                    members=len(members), total_expanded_bytes=total,
                )
            return ArchiveReport(str(path), True, "ok", members=len(members),
                                 total_expanded_bytes=total)
    except (tarfile.TarError, OSError) as exc:
        return ArchiveReport(str(path), False, f"bad_tar:{exc}")


def inspect_archive(path: Path, limits: ArchiveLimits = ArchiveLimits()) -> ArchiveReport:
    """Dispatch on magic bytes (never on the filename extension)."""
    with path.open("rb") as fh:
        header = fh.read(6)
        if header[:4] == b"PK\x03\x04":
            return inspect_zip(path, limits)
        if header[:2] == b"\x1f\x8b" or header[:3] == b"BZh" or header[:6] == b"\xfd7zXZ\x00":
            return inspect_tar(path, limits)
        fh.seek(257)
        if fh.read(5) == b"ustar":
            return inspect_tar(path, limits)
    return ArchiveReport(str(path), False, "unknown_archive_format")
