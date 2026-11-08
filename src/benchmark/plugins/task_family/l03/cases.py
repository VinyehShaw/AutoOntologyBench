"""L03 case machinery: typed input-domain derivation + case-universe enumeration.

Every family that needs input cases shares this: from a parsed table we derive
per-column candidate values (literals from rule entries, comparison thresholds
with +/-1 probes, interval bounds and midpoints, booleans), enumerate the
bounded cross product, and evaluate the pinned ``aob-dmn-static`` runtime on
each case. The result is the deterministic case universe that generation and
verification both draw from - one code path, no divergent semantics.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

from benchmark.plugins.verifier.l03.dmn_model import DecisionTable, DmnUnsupportedError
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_table
from benchmark.plugins.verifier.l03.feel import FeelUnsupportedError, split_alternatives

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CMP = re.compile(r"^(<=?|>=?|!=?)\s*(.+)$")
_INTERVAL = re.compile(
    r"^([\[\(\]])\s*(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)\s*([\]\)\[])$"
)
_STRING = re.compile(r'^"(?:[^"\\]|\\.)*"$')
_LEGACY_EQ = re.compile(r"^cellInput\s*==\s*(.+)$")

MAX_UNIVERSE = 512


def column_domain(table: DecisionTable, col: int, *, cap: int = 12) -> list[Any]:
    """Deterministic candidate values for input column ``col``."""
    clause = table.inputs[col]
    values: list[Any] = []
    seen: set[Any] = set()

    def add(v: Any) -> None:
        key = v if isinstance(v, (str, int, float, bool)) else repr(v)
        if key not in seen and len(values) < cap:
            seen.add(key)
            values.append(v)

    for rule in table.rules:
        entry = rule.input_entries[col].strip()
        if entry.startswith("not(") and entry.endswith(")"):
            entry = entry[4:-1]
        for atom in split_alternatives(entry):
            atom = atom.strip()
            legacy = _LEGACY_EQ.match(atom)
            if legacy:
                atom = legacy.group(1).strip()
            if _STRING.match(atom):
                add(atom[1:-1])
            elif _NUMBER.fullmatch(atom):
                num: Any = int(atom) if re.fullmatch(r"-?\d+", atom) else float(atom)
                add(num)
                if clause.type_ref in ("integer", "long"):
                    add(num + 1)
                    add(num - 1)
            elif atom in ("true", "false"):
                add(atom == "true")
            else:
                cmp_m = _CMP.match(atom)
                if cmp_m and _NUMBER.fullmatch(cmp_m.group(2).strip()):
                    thr = _parse_num(cmp_m.group(2).strip())
                    add(thr)
                    add(thr + 1)
                    add(thr - 1)
                    add(thr + 10)
                    add(thr - 10)
                    continue
                iv = _INTERVAL.match(atom)
                if iv:
                    lo, hi = _parse_num(iv.group(2)), _parse_num(iv.group(3))
                    add(lo)
                    add(hi)
                    add(lo - 1)
                    add(hi + 1)
                    if isinstance(lo, int) and isinstance(hi, int) and lo <= hi:
                        add((lo + hi) // 2)
                    continue
                if atom and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_ \-.]*", atom):
                    add(atom)
    if not values:
        if clause.type_ref in ("integer", "long"):
            values = [0, 1, -1, 5, 10]
        elif clause.type_ref == "double":
            values = [0.0, 1.0, -1.0, 0.5]
        elif clause.type_ref == "boolean":
            values = [True, False]
        else:
            values = ["__unseen__"]
    elif clause.type_ref == "boolean":
        for b in (True, False):
            add(b)
    return values


def _parse_num(text: str) -> int | float:
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def case_universe(table: DecisionTable, *, cap: int = MAX_UNIVERSE,
                  column_cap: int = 12) -> list[dict[str, Any]]:
    """Bounded cross product of per-column domains, as variable -> value dicts.

    ``column_cap`` widens the per-column domain (default 12 keeps the universe
    small; larger values surface more overlap cases for COLLECT-style tables
    with many rules). Deterministic for fixed (cap, column_cap).
    """
    domains = [column_domain(table, i, cap=column_cap) for i in range(len(table.inputs))]
    total = 1
    for d in domains:
        total *= len(d)
    if total > cap:
        per_col = max(2, round(cap ** (1 / max(len(domains), 1))))
        domains = [d[:per_col] for d in domains]
    names = [c.expression for c in table.inputs]
    seen_cases: set[str] = set()
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*domains):
        case = dict(zip(names, combo))
        sig = repr(sorted(case.items()))
        if sig not in seen_cases:
            seen_cases.add(sig)
            out.append(case)
        if len(out) >= cap:
            break
    return out


def eval_universe(table: DecisionTable,
                  cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Evaluate every case; rows carry input/output/matched-rule-id evidence."""
    cases = cases if cases is not None else case_universe(table)
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            res = evaluate_table(table, case)
        except DmnUnsupportedError:
            continue
        rows.append({
            "input": case,
            "output": res.value,
            "matched_rule_ids": list(res.matched_rule_ids),
            "violations": list(res.violations),
        })
    return rows


def distinct_output_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One representative case per distinct (output, matched-set) signature."""
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = repr((row["output"], sorted(row["matched_rule_ids"])))
        seen.setdefault(key, row)
    return [seen[k] for k in sorted(seen)]
