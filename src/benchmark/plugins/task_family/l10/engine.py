"""L10 engine core: independent BDDL parser + pinned-source knowledge tables.

This module is leg A of the lane's differential evidence: an independent,
dependency-free BDDL s-expression parser and normalizer. Leg B lives in
``upstream.py`` (the vendored ``bddl`` package from the pinned snapshot).
Both legs must agree on every parsed structure used by a task.

Knowledge tables (predicate arity, valid synsets, valid room constants) are
loaded from the pinned BEHAVIOR-1K snapshot only — never hardcoded here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from source_registry.fetchers import l10_behavior_bddl as fetcher

PINNED_ROOT = fetcher.PINNED_ROOT
BDDL_ROOT = PINNED_ROOT / "bddl3" / "bddl"
ACTIVITY_ROOT = BDDL_ROOT / "activity_definitions"
DOMAIN_FILE = ACTIVITY_ROOT / "domain_behavior-1k.bddl"
SYNSETS_CSV = BDDL_ROOT / "generated_data" / "synsets.csv"
ROOMS_CSV = BDDL_ROOT / "generated_data" / "allowed_room_types.csv"

SOURCE_SNAPSHOT_ID = "behavior-1k-bddl@v3.9.3-post2"

#: Logical operators of the BDDL goal grammar (mirrors upstream TOKEN_MAPPING;
#: asserted equal to the vendored mapping at table-load time).
OPERATORS = ("and", "or", "not", "imply", "forall", "exists", "forn",
             "forpairs", "fornpairs")
QUANTIFIERS = ("forall", "exists", "forn", "forpairs", "fornpairs")
CONNECTIVES = ("and", "or", "not", "imply")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class BddlParseError(ValueError):
    """Raised when a BDDL text cannot be tokenized/parsed."""


def scan_tokens(text: str) -> list[Any]:
    """Tokenize BDDL text into nested lists (comments stripped, lowercased)."""
    stripped = re.sub(r";.*$", "", text.lower(), flags=re.MULTILINE)
    stack: list[list[Any]] = []
    tokens: list[Any] = []
    for tok in re.findall(r"[()]|[^\s()]+", stripped):
        if tok == "(":
            stack.append(tokens)
            tokens = []
        elif tok == ")":
            if not stack:
                raise BddlParseError("unbalanced close parenthesis")
            inner = tokens
            tokens = stack.pop()
            tokens.append(inner)
        else:
            tokens.append(tok)
    if stack:
        raise BddlParseError("unbalanced open parenthesis")
    if len(tokens) != 1 or not isinstance(tokens[0], list):
        raise BddlParseError("malformed top-level expression")
    return tokens[0]


def canon(node: Any) -> Any:
    """Canonical hashable form of a parsed s-expression."""
    if isinstance(node, list):
        return tuple(canon(x) for x in node)
    return node


@dataclass(frozen=True)
class ParsedProblem:
    """Normalized BDDL problem (leg-A representation)."""

    name: str
    domain: str
    objects: tuple[tuple[str, tuple[str, ...]], ...]  # (category, instances) sorted by category
    init: tuple[Any, ...]  # canonical literals: ("atom"...) or ("not", ("atom"...))
    goal: tuple[Any, ...]  # canonical conjunct list (outer "and" stripped, upstream-style)
    goal_raw: Any  # canonical raw :goal expression (with outer "and" if present)

    @property
    def object_map(self) -> dict[str, list[str]]:
        return {cat: list(insts) for cat, insts in self.objects}

    @property
    def instances(self) -> frozenset[str]:
        return frozenset(i for _, insts in self.objects for i in insts)

    @property
    def instance_category(self) -> dict[str, str]:
        return {i: cat for cat, insts in self.objects for i in insts}


def parse_problem_text(text: str) -> ParsedProblem:
    """Parse a BDDL problem file (independent implementation, leg A)."""
    top = scan_tokens(text)
    if not top or top[0] != "define":
        raise BddlParseError("missing (define ...)")
    name = "unknown"
    domain = ""
    objects: dict[str, list[str]] = {}
    init: list[Any] = []
    goal_raw: Any = None
    for group in top[1:]:
        if not isinstance(group, list) or not group:
            continue
        head = group[0]
        if head == "problem":
            name = group[-1]
        elif head == ":domain":
            domain = group[-1]
        elif head == ":objects":
            # Tokens arrive flat: inst inst ... - cat inst ... - cat.
            # Untyped trailing instances land in category "object" (upstream rule).
            objects = {}
            cur: list[str] = []
            rest = group[1:]
            i = 0
            while i < len(rest):
                if rest[i] == "-":
                    cat = rest[i + 1]
                    objects.setdefault(cat, []).extend(cur)
                    cur = []
                    i += 2
                else:
                    cur.append(rest[i])
                    i += 1
            if cur:
                objects.setdefault("object", []).extend(cur)
        elif head == ":init":
            init = [canon(lit) for lit in group[1:]]
        elif head == ":goal":
            if len(group) < 2:
                raise BddlParseError("empty :goal")
            goal_raw = canon(group[1])
    if goal_raw is None:
        raise BddlParseError("missing :goal")
    if goal_raw and goal_raw[0] == "and":
        goal = tuple(goal_raw[1:])
    else:
        goal = (goal_raw,)
    return ParsedProblem(
        name=name,
        domain=domain,
        objects=tuple(sorted((c, tuple(insts)) for c, insts in objects.items())),
        init=tuple(init),
        goal=goal,
        goal_raw=goal_raw,
    )


# ---------------------------------------------------------------------------
# Pinned-source knowledge tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeTables:
    """Domain knowledge loaded from the pinned snapshot."""

    predicates: tuple[tuple[str, int], ...]  # predicate -> arity
    synsets: frozenset[str]
    synset_object_type: tuple[tuple[str, str], ...]  # synset -> objectType column
    rooms: frozenset[str]
    domain_file_sha256: str
    synsets_sha256: str
    rooms_sha256: str

    @property
    def predicate_map(self) -> dict[str, int]:
        return dict(self.predicates)

    @property
    def object_type_map(self) -> dict[str, str]:
        return dict(self.synset_object_type)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_tables() -> KnowledgeTables:
    """Load predicate/synset/room tables from the pinned snapshot (cached)."""
    domain = scan_tokens(DOMAIN_FILE.read_text(encoding="utf-8"))
    predicates: dict[str, int] = {}
    for group in domain[1:]:
        if isinstance(group, list) and group and group[0] == ":predicates":
            for decl in group[1:]:
                if isinstance(decl, list) and decl:
                    predicates[decl[0]] = len(decl) - 1
    if not predicates:
        raise BddlParseError("domain file has no :predicates")
    synsets: dict[str, str] = {}
    with SYNSETS_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            syn = (row.get("synset") or "").strip()
            if syn:
                synsets[syn] = (row.get("objectType") or "").strip()
    rooms: set[str] = set()
    with ROOMS_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    for row in rows[1:]:  # header: "Room Name"
        if row and row[0].strip():
            rooms.add(row[0].strip())
    return KnowledgeTables(
        predicates=tuple(sorted(predicates.items())),
        synsets=frozenset(synsets),
        synset_object_type=tuple(sorted(synsets.items())),
        rooms=frozenset(rooms),
        domain_file_sha256=_sha256_file(DOMAIN_FILE),
        synsets_sha256=_sha256_file(SYNSETS_CSV),
        rooms_sha256=_sha256_file(ROOMS_CSV),
    )


def table_shas(tables: KnowledgeTables) -> dict[str, str]:
    return {
        "domain_file_sha256": tables.domain_file_sha256,
        "synsets_sha256": tables.synsets_sha256,
        "rooms_sha256": tables.rooms_sha256,
    }


def iter_activity_paths() -> Iterator[Path]:
    yield from sorted(ACTIVITY_ROOT.glob("*/problem*.bddl"))


def load_activity(path: Path) -> ParsedProblem:
    return parse_problem_text(path.read_text(encoding="utf-8"))


def activity_rel_path(path: Path) -> str:
    return str(path.relative_to(PINNED_ROOT))
