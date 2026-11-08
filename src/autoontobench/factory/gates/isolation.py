"""Public/private physical isolation checks (owner: S11).

FINAL-BUILD-ORDER-v2 §2.3/§6.3 and Shared Contract §G0:
- the private root and the public root must be physically disjoint trees
  (neither nested under the other after symlink resolution)
- every private dir is 0700, every private file is owner-only (0600/0700)
- no symlink inside the public root may resolve into the private root
- no private *value* (file content hashes, registered private tokens) may
  appear under the public root

Reuses `autoontobench.safety.private_store.is_owner_only` for permission
verdicts. Complementary to `safety.leak_audit` (content-level scanning);
this module enforces the *physical* boundary.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from autoontobench.safety.private_store import is_owner_only


@dataclass(frozen=True)
class IsolationViolation:
    kind: str  # nested_root|symlink_escape|bad_private_mode|private_value_leak
    path: str
    detail: str


@dataclass(frozen=True)
class IsolationReport:
    public_root: str
    private_root: str
    ok: bool
    violations: tuple[IsolationViolation, ...] = ()

    def summary(self) -> str:
        if self.ok:
            return "isolation: OK (disjoint, owner-only private, no value leaks)"
        return f"isolation: {len(self.violations)} violation(s)"


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_disjoint(public_root: Path, private_root: Path) -> list[IsolationViolation]:
    """Resolved trees must not nest, in either direction."""
    violations: list[IsolationViolation] = []
    pub = public_root.resolve()
    priv = private_root.resolve()
    if _is_within(priv, pub):
        violations.append(
            IsolationViolation("nested_root", str(priv), "private root is inside public root")
        )
    if _is_within(pub, priv):
        violations.append(
            IsolationViolation("nested_root", str(pub), "public root is inside private root")
        )
    return violations


def check_private_permissions(private_root: Path) -> list[IsolationViolation]:
    """Every private dir/file must be owner-only (0700 dirs, no group/other bits)."""
    violations: list[IsolationViolation] = []
    if not private_root.exists():
        return violations
    for dirpath, dirnames, filenames in os.walk(private_root, followlinks=False):
        for name in dirnames:
            p = Path(dirpath) / name
            if not p.is_symlink() and not is_owner_only(p):
                violations.append(
                    IsolationViolation(
                        "bad_private_mode", str(p),
                        f"dir mode {oct(stat.S_IMODE(p.stat().st_mode))} grants group/other",
                    )
                )
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_symlink() and not is_owner_only(p):
                violations.append(
                    IsolationViolation(
                        "bad_private_mode", str(p),
                        f"file mode {oct(stat.S_IMODE(p.stat().st_mode))} grants group/other",
                    )
                )
    if private_root.exists() and not is_owner_only(private_root):
        violations.append(
            IsolationViolation(
                "bad_private_mode", str(private_root),
                f"root mode {oct(stat.S_IMODE(private_root.stat().st_mode))} grants group/other",
            )
        )
    return violations


def check_symlink_escape(public_root: Path, private_root: Path) -> list[IsolationViolation]:
    """No public symlink may resolve into the private tree."""
    violations: list[IsolationViolation] = []
    if not public_root.exists():
        return violations
    priv = private_root.resolve()
    for dirpath, dirnames, filenames in os.walk(public_root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                target = p.resolve()
                if _is_within(target, priv):
                    violations.append(
                        IsolationViolation(
                            "symlink_escape", str(p), f"resolves to {target} inside private root"
                        )
                    )
    return violations


def private_value_tokens(private_root: Path, *, sample_bytes: int = 1 << 20) -> set[str]:
    """Tokens that must never appear under the public root.

    Tokens = sha256 hexdigests of private file contents (first `sample_bytes`
    per file). Fast and catches verbatim gold/evidence copies.
    """
    tokens: set[str] = set()
    if not private_root.exists():
        return tokens
    for dirpath, _dirnames, filenames in os.walk(private_root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            try:
                blob = p.read_bytes()[:sample_bytes]
            except OSError:
                continue
            if blob:
                tokens.add(hashlib.sha256(blob).hexdigest())
    return tokens


def check_private_values(
    public_root: Path, tokens: set[str], *, max_file_bytes: int = 1 << 22
) -> list[IsolationViolation]:
    """Grep the public tree for registered private tokens (hex literals)."""
    violations: list[IsolationViolation] = []
    if not tokens or not public_root.exists():
        return violations
    for dirpath, _dirnames, filenames in os.walk(public_root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            try:
                if p.stat().st_size > max_file_bytes:
                    continue
                text = p.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            for token in tokens:
                if token in text:
                    violations.append(
                        IsolationViolation(
                            "private_value_leak", str(p),
                            f"contains private content hash {token[:16]}…",
                        )
                    )
    return violations


def run_isolation_checks(public_root: Path, private_root: Path) -> IsolationReport:
    """All four checks; ok == zero violations."""
    violations: list[IsolationViolation] = []
    violations += check_disjoint(public_root, private_root)
    violations += check_private_permissions(private_root)
    violations += check_symlink_escape(public_root, private_root)
    violations += check_private_values(public_root, private_value_tokens(private_root))
    return IsolationReport(
        public_root=str(public_root),
        private_root=str(private_root),
        ok=not violations,
        violations=tuple(violations),
    )
