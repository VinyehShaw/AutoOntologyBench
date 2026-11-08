"""Strict adversarial leak scanner for Review-5 public roots (owner: W1, G0 gate).

Scans a *public* root recursively and reports every forbidden signal named by
`verification/leak_fix_design.md` and Review-5B Implementation §5: forbidden
JSON keys, private/hidden/native path strings, absolute source paths, git
SHAs, rule-unit ids, gold Turtle bodies, ALLOW/DENY label maps, hidden
fingerprints (cross-checked against the private root when supplied), symlinks,
world-writable files and forbidden filename patterns. Rule tables live in
`leak_rules.py`. Permissions: with `--private-root`, each private world dir is
probed as uid/gid 65534 via `setpriv`; if unavailable the check is reported
`cannot_verify` (not counted as a leak).

Exit code: 1 when any finding has verdict `leaked`, 0 when clean, 2 on usage
error. JSON report written to `--json-out`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autoontobench.safety import leak_rules
from autoontobench.safety.leak_rules import (
    ARTIFACT_SHA256,
    FORBIDDEN_FILENAME,
    FORBIDDEN_JSON_KEYS,
    GOLD_TURTLE,
    HOST_ABS_PREFIXES,
    LABEL_MAP,
    PATH_PATTERNS,
)


@dataclass(frozen=True)
class Finding:
    world_id: str
    path: str
    item: str
    verdict: str  # "leaked" | "cannot_verify"
    evidence: dict[str, Any] = field(default_factory=dict)


def _world_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "<root>"


def _iter_paths(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(dirnames + filenames):
            yield Path(dirpath) / name


def scan_json_keys(obj: Any, path: str, hits: list[str]) -> None:
    """Append `path.key` for every forbidden key found recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_JSON_KEYS:
                hits.append(f"{path}.{k}")
            scan_json_keys(v, f"{path}.{k}", hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            scan_json_keys(v, f"{path}[{i}]", hits)


def _scan_text(root: Path, path: Path, text: str, findings: list[Finding]) -> None:
    world = _world_id(root, path)
    rel = str(path.relative_to(root))
    if path.suffix == ".ttl":
        m = GOLD_TURTLE.search(leak_rules.strip_turtle_comments(text))
        if m:
            findings.append(
                Finding(world, rel, "gold_rule_body", "leaked", {"token": m.group(0)})
            )
        sha = ARTIFACT_SHA256.search(leak_rules.strip_turtle_comments(text))
        if sha:
            findings.append(
                Finding(world, rel, "source_sha256_literal", "leaked",
                        {"match_prefix": sha.group(0)[:16]})
            )
    if path.suffix == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            findings.append(
                Finding(world, rel, "malformed_json", "leaked", {"error": str(exc)})
            )
        else:
            hits: list[str] = []
            scan_json_keys(obj, "$", hits)
            for h in hits:
                findings.append(Finding(world, rel, "forbidden_key", "leaked", {"key": h}))
    for name, pat in PATH_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(
                Finding(world, rel, f"forbidden_string:{name}", "leaked",
                        {"match": m.group(0)[:80]})
            )
    m = LABEL_MAP.search(text)
    if m:
        findings.append(Finding(world, rel, "label_map", "leaked",
                                {"match": m.group(0)[:80]}))
    for prefix in HOST_ABS_PREFIXES:
        if prefix in text:
            findings.append(Finding(world, rel, "host_absolute_path", "leaked",
                                    {"prefix": prefix}))
            break


def scan_public_root(public_root: Path, private_root: Path | None = None) -> list[Finding]:
    """Recursively scan `public_root`; return all findings."""
    public_root = public_root.resolve()
    findings: list[Finding] = []
    fingerprints: dict[str, dict[str, set[str]]] = {}
    if private_root is not None and private_root.exists():
        for world_dir in sorted(private_root.iterdir()):
            if world_dir.is_dir():
                fingerprints[world_dir.name] = leak_rules.hidden_fingerprints(world_dir)

    for path in _iter_paths(public_root):
        world = _world_id(public_root, path)
        rel = str(path.relative_to(public_root))
        if path.is_symlink():
            findings.append(
                Finding(world, rel, "symlink", "leaked", {"target": os.readlink(path)})
            )
            continue
        if path.is_dir():
            continue
        if FORBIDDEN_FILENAME.search(path.name):
            findings.append(
                Finding(world, rel, "forbidden_filename", "leaked", {"name": path.name})
            )
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o002:
            findings.append(Finding(world, rel, "world_writable", "leaked",
                                    {"mode": oct(mode)}))
        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError as exc:
            findings.append(Finding(world, rel, "unreadable", "cannot_verify",
                                    {"error": str(exc)}))
            continue
        _scan_text(public_root, path, text, findings)
        for kind, needles in fingerprints.get(world, {}).items():
            for n in sorted(needles):
                if n and n in text:
                    digest = hashlib.sha256(n.encode()).hexdigest()[:16]
                    findings.append(Finding(world, rel, f"hidden_fingerprint:{kind}",
                                            "leaked", {"needle_sha256": digest}))
    return findings


def check_private_permissions(private_root: Path) -> list[Finding]:
    """Defence-in-depth: private world dirs must not be readable by uid 65534."""
    probe = ["setpriv", "--reuid", "65534", "--regid", "65534", "--clear-groups"]
    try:
        subprocess.run(probe + ["true"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return [Finding("<private>", str(private_root), "perm_nobody_read",
                        "cannot_verify", {"reason": "setpriv unavailable or refused"})]
    findings: list[Finding] = []
    for world_dir in sorted(private_root.iterdir()):
        if not world_dir.is_dir():
            continue
        for sub in (world_dir / "hidden", world_dir / "native", world_dir):
            if not sub.exists():
                continue
            rc = subprocess.run(probe + ["test", "-r", str(sub)]).returncode
            if rc == 0:
                findings.append(Finding(world_dir.name, str(sub), "perm_nobody_read",
                                        "leaked",
                                        {"mode": oct(stat.S_IMODE(sub.stat().st_mode))}))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial leak scanner (Review-5 G0)")
    ap.add_argument("--public-root", required=True, type=Path)
    ap.add_argument("--private-root", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--skip-permissions", action="store_true")
    args = ap.parse_args(argv)

    if not args.public_root.is_dir():
        print(f"public root not found: {args.public_root}", file=sys.stderr)
        return 2

    findings = scan_public_root(args.public_root, args.private_root)
    if args.private_root is not None and not args.skip_permissions:
        findings += check_private_permissions(args.private_root)

    leaked = [f for f in findings if f.verdict == "leaked"]
    cannot = [f for f in findings if f.verdict == "cannot_verify"]
    report = {
        "public_root": str(args.public_root),
        "private_root": str(args.private_root) if args.private_root else None,
        "leaked": len(leaked),
        "cannot_verify": len(cannot),
        "findings": [
            {"world_id": f.world_id, "path": f.path, "item": f.item,
             "verdict": f.verdict, "evidence": f.evidence}
            for f in findings
        ],
    }
    blob = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(blob + "\n", encoding="utf-8")
    print(f"leak_audit: {len(leaked)} leaked, {len(cannot)} cannot_verify "
          f"(public_root={args.public_root})")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
