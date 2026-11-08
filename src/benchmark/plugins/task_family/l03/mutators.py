"""L03 mutators: seeded defects injected into canonical table payloads.

Mutators are proposals only (build order §4.1); the native runtime decides
whether a mutation is behaviorally real (golden vs broken differ on the case
universe). Every mutator returns the mutated payload plus a machine-readable
mutation descriptor for the oracle.
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any

_CMP = re.compile(r"^(<=?|>=?|!=?)\s*(.+)$")
_INTERVAL = re.compile(
    r"^([\[\(\]])\s*(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)\s*([\]\)\[])$"
)
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_STRING = re.compile(r'^"(?:[^"\\]|\\.)*"$')

_FLIP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "!=", "!=": "="}

_HIT_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "UNIQUE": ("FIRST", "COLLECT"),
    "FIRST": ("COLLECT", "RULE ORDER"),
    "RULE ORDER": ("COLLECT", "FIRST"),
    "OUTPUT ORDER": ("RULE ORDER", "COLLECT"),
    "PRIORITY": ("FIRST", "RULE ORDER"),
    "ANY": ("FIRST", "COLLECT"),
    "COLLECT": ("FIRST", "UNIQUE"),
}


def mutate_rule_entry(table: dict[str, Any], rng: random.Random
                      ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Single-cell input-entry defect. Returns (broken_table, mutation)."""
    eligible: list[tuple[int, int]] = []
    for ri, rule in enumerate(table["rules"]):
        for ci, entry in enumerate(rule["input_entries"]):
            if _mutation_for(entry, table, ci, rng) is not None:
                eligible.append((ri, ci))
    if not eligible:
        return None
    ri, ci = rng.choice(sorted(eligible))
    original = table["rules"][ri]["input_entries"][ci]
    mutated = _mutation_for(original, table, ci, rng)
    assert mutated is not None
    broken = copy.deepcopy(table)
    broken["rules"][ri]["input_entries"][ci] = mutated
    return broken, {
        "slot": "input_entry",
        "rule_index": ri,
        "rule_id": table["rules"][ri]["id"],
        "column": ci,
        "variable": table["inputs"][ci]["expression"],
        "from": original,
        "to": mutated,
    }


def _mutation_for(entry: str, table: dict[str, Any], col: int,
                  rng: random.Random) -> str | None:
    text = entry.strip()
    if not text:
        return None
    choices: list[str] = []
    cmp_m = _CMP.match(text)
    if cmp_m and cmp_m.group(1) in _FLIP:
        choices.append(_FLIP[cmp_m.group(1)] + " " + cmp_m.group(2))
        num = cmp_m.group(2).strip()
        if _NUMBER.match(num):
            v = int(num) if re.fullmatch(r"-?\d+", num) else float(num)
            choices.append(cmp_m.group(1) + " " + repr(v + 1))
    if _NUMBER.match(text):
        v: Any = int(text) if re.fullmatch(r"-?\d+", text) else float(text)
        choices.append(repr(v + 1))
    iv = _INTERVAL.match(text)
    if iv:
        lo, hi = iv.group(2), iv.group(3)
        hi_v = int(hi) if re.fullmatch(r"-?\d+", hi) else float(hi)
        choices.append("%s%s..%s%s" % (iv.group(1), lo, repr(hi_v + 1), iv.group(4)))
    if _STRING.match(text):
        pool = _column_string_pool(table, col) - {text[1:-1]}
        if pool:
            choices.append('"%s"' % rng.choice(sorted(pool)))
        else:
            choices.append('"zzz-distractor"')
    if text in ("true", "false"):
        choices.append("false" if text == "true" else "true")
    choices.append("-")  # dropped condition: rule becomes broader
    return rng.choice(choices) if choices else None


def _column_string_pool(table: dict[str, Any], col: int) -> set[str]:
    pool: set[str] = set()
    for rule in table["rules"]:
        for atom in rule["input_entries"][col].split(","):
            atom = atom.strip()
            if _STRING.match(atom):
                pool.add(atom[1:-1])
    return pool


def mutate_hit_policy(table: dict[str, Any], rng: random.Random
                      ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    alternatives = _HIT_ALTERNATIVES.get(table.get("hit_policy") or "UNIQUE")
    if not alternatives:
        return None
    broken = copy.deepcopy(table)
    new_policy = rng.choice(alternatives)
    broken["hit_policy"] = new_policy
    if new_policy != "COLLECT":
        broken["aggregation"] = ""
    return broken, {
        "slot": "hit_policy",
        "from": table.get("hit_policy") or "UNIQUE",
        "to": new_policy,
        "aggregation_from": table.get("aggregation") or "",
        "aggregation_to": broken.get("aggregation") or "",
    }


def mutate_binding(chain_tables: dict[str, dict[str, Any]],
                   edges: dict[str, list[str]],
                   rng: random.Random) -> tuple[dict[str, dict[str, Any]],
                                                dict[str, Any]] | None:
    """Rename a consuming decision's input expression that is bound to a
    required decision's output name (cross-table contract defect)."""
    produced: dict[str, str] = {}  # output name -> producing decision id
    for dec_id, payload in chain_tables.items():
        outputs = payload.get("outputs") or []
        if outputs:
            produced[outputs[0]["name"]] = dec_id
    candidates: list[tuple[str, str]] = []  # (consuming decision, expression)
    for dec_id, deps in edges.items():
        if not deps:
            continue
        for clause in chain_tables[dec_id].get("inputs", []):
            expr = clause["expression"]
            if expr in produced and produced[expr] in deps:
                candidates.append((dec_id, expr))
    if not candidates:
        return None
    dec_id, expr = rng.choice(sorted(candidates))
    broken = copy.deepcopy(chain_tables)
    new_expr = expr + "__unbound"
    for clause in broken[dec_id]["inputs"]:
        if clause["expression"] == expr:
            clause["expression"] = new_expr
    return broken, {
        "slot": "binding",
        "decision_id": dec_id,
        "from": expr,
        "to": new_expr,
        "producer_decision_id": produced[expr],
    }
