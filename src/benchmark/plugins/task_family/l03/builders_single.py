"""L03 candidate builders: single-table application + construction families.

f01 exact_output / f02 rule_audit / f07 collect_result / f08 input_synthesis.

Every builder receives the index row (canonical table payload), evaluates the
pinned runtime over the derived case universe, and yields AST dicts whose
reference answers are already the native verdicts. The verifier later
recomputes every one of them independently from the pinned XML; generation-time
answers are proposals, not truth.

Yielded dicts carry ``reference_*`` fields (the native verdicts observed at
generation time) plus the deterministic ``universe`` parameters so the
verifier can reproduce the exact case universe.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l03 import cases as case_mod
from benchmark.plugins.task_family.l03.families import complexity_of_table
from benchmark.plugins.verifier.l03.dmn_model import table_from_payload

#: Default deterministic case-universe parameters (see ``cases.py``).
UNIVERSE_DEFAULT = {"cap": case_mod.MAX_UNIVERSE, "column_cap": 12}
#: Wide universe for COLLECT tables: many-rule tables need wider per-column
#: domains before rule overlaps (multi-match cases) become visible.
UNIVERSE_WIDE = {"cap": 2048, "column_cap": 64}


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_policy": row["hit_policy"],
        "aggregation": row["aggregation"],
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "rules": row["rules"],
    }


def _universe(row: dict[str, Any], params: dict[str, int] | None = None
              ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _payload(row)
    table = table_from_payload(payload)
    params = params or UNIVERSE_DEFAULT
    cases = case_mod.case_universe(table, **params)
    return payload, case_mod.eval_universe(table, cases)


def _pick_cases(rows: list[dict[str, Any]], n: int, rng: random.Random,
                predicate) -> list[dict[str, Any]]:
    pool = [r for r in rows if predicate(r)]
    pool = sorted(pool, key=lambda r: repr(sorted(r["input"].items())))
    rng.shuffle(pool)
    return pool[:n]


def build_exact_output(row: dict[str, Any], n_instances: int,
                       rng: random.Random) -> Iterator[dict[str, Any]]:
    payload, rows = _universe(row)
    outputs = {repr(r["output"]) for r in rows if r["matched_rule_ids"]}
    if len(rows) < n_instances or len(outputs) < 2:
        return
    picked = _pick_cases(
        rows, n_instances, rng,
        lambda r: r["matched_rule_ids"] and r["output"] is not None and not r["violations"],
    )
    for case_row in picked:
        yield {
            "kind": "exact_output",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "input_case": case_row["input"],
            "reference_output": case_row["output"],
            "reference_matched_rule_ids": list(case_row["matched_rule_ids"]),
            "universe": dict(UNIVERSE_DEFAULT),
            "complexity": complexity_of_table(
                len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"])),
        }


def build_rule_audit(row: dict[str, Any], n_instances: int,
                     rng: random.Random) -> Iterator[dict[str, Any]]:
    payload, rows = _universe(row)
    n_rules = len(payload["rules"])
    if n_rules < 3 or len(rows) < n_instances:
        return
    picked = _pick_cases(
        rows, n_instances, rng,
        lambda r: 0 < len(r["matched_rule_ids"]) < n_rules,
    )
    for case_row in picked:
        yield {
            "kind": "rule_audit",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "input_case": case_row["input"],
            "reference_matched_rule_ids": list(case_row["matched_rule_ids"]),
            "universe": dict(UNIVERSE_DEFAULT),
            "complexity": complexity_of_table(
                len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"])),
        }


def build_collect_result(row: dict[str, Any], n_instances: int,
                         rng: random.Random) -> Iterator[dict[str, Any]]:
    if row["hit_policy"] != "COLLECT":
        return
    payload, rows = _universe(row, UNIVERSE_WIDE)
    multi = [r for r in rows if len(r["matched_rule_ids"]) >= 2]
    if len(multi) < 1:
        return
    picked = _pick_cases(multi, n_instances, rng, lambda r: True)
    for case_row in picked:
        yield {
            "kind": "collect_result",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "input_case": case_row["input"],
            "reference_output": case_row["output"],
            "reference_matched_rule_ids": list(case_row["matched_rule_ids"]),
            "universe": dict(UNIVERSE_WIDE),
            "complexity": complexity_of_table(
                len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"])),
        }


def build_input_synthesis(row: dict[str, Any], n_instances: int,
                          rng: random.Random) -> Iterator[dict[str, Any]]:
    payload, rows = _universe(row)
    if len(payload["rules"]) < 3:
        return
    hits = [r for r in rows if r["output"] is not None and r["matched_rule_ids"]]
    by_output: dict[str, list[dict[str, Any]]] = {}
    for r in hits:
        by_output.setdefault(repr(r["output"]), []).append(r)
    targets = [k for k, v in sorted(by_output.items())
               if 0 < len(v) <= max(2, len(hits) // 2)]
    if not targets:
        return
    emitted = 0
    for target in targets:
        if emitted >= n_instances:
            break
        rep = by_output[target][0]
        yield {
            "kind": "input_synthesis",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "target_output": rep["output"],
            "reference_input": rep["input"],
            "universe": dict(UNIVERSE_DEFAULT),
            "complexity": complexity_of_table(
                len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"])),
        }
        emitted += 1
