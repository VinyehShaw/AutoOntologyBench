"""Forbidden-signal tables + text helpers for the leak scanner (owner: W1).

Pure data/pattern module — separated from `leak_audit.py` to keep each file
under the 250-LOC review ceiling. No I/O except fingerprint extraction from a
private world dir.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FORBIDDEN_JSON_KEYS: frozenset[str] = frozenset(
    {
        # mandated by the W1 brief
        "source_sha",
        "src_rego_sha256",
        "counterfactual_variant_provenance",
        "semantic_witnesses",
        "identifying_probe_ids",
        "min_probe_cost",
        "identifiability_status",
        "provenance",
        # observed leak carriers (leak_fix_design.md §1/§2)
        "source_rule_unit_id",
        "source_commit",
        "constraint_path",
        "rule_unit_id",
        "src_rego_path",
        "counterfactual_rego_sha256",
        "hidden_rule_spec",
        "hidden_decision_cases",
        "hidden_stateful_trajectories",
        "probe_labels",
    }
)

PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_path", re.compile(r"(?:^|[\s\"'(=:])(?:\.\./)*private/[\w.\-/]+")),
    ("hidden_path", re.compile(r"(?:^|[\s\"'(=:])(?:\.\./)*hidden/[\w.\-]+")),
    ("native_path", re.compile(r"(?:^|[\s\"'(=:])(?:\.\./)*native/[\w.\-]+")),
    ("absolute_source_path", re.compile(r"references/sources/[\w.\-/]+")),
    ("git_commit_sha", re.compile(r"\b[0-9a-f]{40}\b")),
    ("rule_unit_id", re.compile(r"gatekeeper/general/[\w.\-/]+")),
)

GOLD_TURTLE = re.compile(r"sh\s*:\s*(sparql|select|message)\b")
# 64-hex literals are forbidden inside Turtle artifacts: commitment hashes live in
# JSON `hashes` blocks; a hex256 in an .ttl is a source/provenance fingerprint.
ARTIFACT_SHA256 = re.compile(r"\b[0-9a-f]{64}\b")
LABEL_MAP = re.compile(r"(?:decision|label)\"?\s*[:=]\s*\"?(ALLOW|DENY)\b")
FORBIDDEN_FILENAME = re.compile(
    r"(hidden|native|probe_labels|decision_cases|gold|expert_reference)", re.IGNORECASE
)
HOST_ABS_PREFIXES = ("/mnt/", "/home/", "/root/", "/tmp/")


def strip_turtle_comments(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        in_str = False
        kept: list[str] = []
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                in_str = not in_str
            if ch == "#" and not in_str:
                break
            kept.append(ch)
        out.append("".join(kept))
    return "\n".join(out)


def hidden_fingerprints(private_world: Path) -> dict[str, set[str]]:
    """Extract exact hidden strings to grep for in the public twin."""
    fp: dict[str, set[str]] = {"case_ids": set(), "input_ids": set(), "src_shas": set()}
    dc = private_world / "hidden" / "decision_cases.jsonl"
    if dc.exists():
        for line in dc.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key, bucket in (("case_id", "case_ids"), ("input_id", "input_ids")):
                v = row.get(key)
                if isinstance(v, str) and v:
                    fp[bucket].add(v)
    hs = private_world / "hidden" / "hidden_rule_spec.json"
    if hs.exists():
        spec = json.loads(hs.read_text(encoding="utf-8", errors="replace"))
        src = spec.get("source", {})
        for k in ("src_rego_sha256", "source_commit"):
            v = src.get(k)
            if isinstance(v, str) and len(v) >= 8:
                fp["src_shas"].add(v)
    dcontract = private_world / "hidden" / "decision_contract.json"
    if dcontract.exists():
        contract = json.loads(dcontract.read_text(encoding="utf-8", errors="replace"))
        text = contract.get("contract_text")
        if isinstance(text, str) and len(text) >= 32:
            fp.setdefault("contract_text", set()).add(text)
    return fp
