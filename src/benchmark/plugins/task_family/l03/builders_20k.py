"""L03 20K-expansion candidate builders (families f09..f16).

Every builder receives index rows (Camunda combined index and/or the DMN TCK
index), evaluates the pinned ``aob-dmn-static`` runtime over the derived case
universe, and yields AST dicts whose reference answers are the native verdicts
observed at generation time. The verifier later recomputes every one of them
independently from the pinned XML (and, for TCK families, re-parses the pinned
TCK test file); generation-time answers are proposals, not truth.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l03 import cases as case_mod
from benchmark.plugins.task_family.l03.builders_repair import _suite
from benchmark.plugins.task_family.l03.builders_single import (
    UNIVERSE_DEFAULT,
    UNIVERSE_WIDE,
    _payload,
    _universe,
)
from benchmark.plugins.task_family.l03.families import (
    complexity_of_chain,
    complexity_of_table,
)
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_chain, evaluate_table
from benchmark.plugins.verifier.l03.dmn_model import table_from_payload
from benchmark.plugins.verifier.l03.plugin import chain_root_universe

ROOT_CAP = 64
SUITE_MIN = 3
SUITE_MAX = 6
AGGREGATION_ALTERNATIVES = ("SUM", "MIN", "MAX", "COUNT", "")
MAX_MUTATION_TRIES = 8


def _behavior(broken_payload: dict[str, Any],
              golden_rows: list[dict[str, Any]]) -> list[Any] | None:
    try:
        broken = table_from_payload(broken_payload)
    except Exception:
        return None
    out: list[Any] = []
    for row in golden_rows:
        try:
            res = evaluate_table(broken, row["input"])
        except Exception:
            return None
        out.append(res.value)
    return out


def _golden_rows(payload: dict[str, Any], universe: dict[str, int]) -> list[dict[str, Any]]:
    table = table_from_payload(payload)
    cases = case_mod.case_universe(table, **universe)
    return case_mod.eval_universe(table, cases)


# -- f09: TCK scenario output (dual oracle) --------------------------------------

def build_tck_output(row: dict[str, Any], scenario: dict[str, Any],
                     n_instances: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    yield {
        "kind": "tck_output",
        "unit_id": row["unit_id"],
        "decision_id": row["decision_id"],
        "decision_name": row["decision_name"],
        "table": payload,
        "input_case": dict(scenario["inputs"]),
        "reference_output": scenario["engine_value"],
        "tck_test_unit_id": scenario["unit_id"],
        "tck_test_path": "external_knowledge/pinned/" + scenario["unit_id"],
        "test_case_id": scenario["test_case_id"],
        "tck_expected": scenario["expected"],
        "universe": dict(UNIVERSE_DEFAULT),
        "complexity": complexity_of_table(
            len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"])),
    }


# -- f10: suite conformance audit -------------------------------------------------

def build_suite_audit(row: dict[str, Any], n_instances: int,
                      rng: random.Random,
                      tck_scenarios: list[dict[str, Any]] | None = None
                      ) -> Iterator[dict[str, Any]]:
    """One visible suite case has an expected value inconsistent with the table.

    The tampered value is always a true expectation of *another* case in the
    suite, so value-frequency shortcuts cannot identify the tampered case.
    """
    payload = _payload(row)
    if tck_scenarios:
        suite_cases = [
            {"case_id": "tck-%s" % s["test_case_id"], "input": dict(s["inputs"]),
             "expected": s["engine_value"]}
            for s in tck_scenarios
        ]
        tck_unit = tck_scenarios[0]["unit_id"]
    else:
        _payload_unused, rows = _universe(row)
        hits = [r for r in rows if r["matched_rule_ids"] and r["output"] is not None]
        if len(hits) < SUITE_MIN:
            return
        suite_cases = [
            {"case_id": "case-%02d" % i, "input": dict(r["input"]),
             "expected": r["output"]}
            for i, r in enumerate(hits)
        ]
        tck_unit = None
    if len(suite_cases) < SUITE_MIN:
        return
    true_values = [c["expected"] for c in suite_cases]
    if len({repr(v) for v in true_values}) < 2:
        return
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen: set[str] = set()
    emitted = 0
    order = list(range(len(suite_cases)))
    for _ in range(4 * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        rng.shuffle(order)
        take = min(len(suite_cases), rng.randint(SUITE_MIN, min(SUITE_MAX, len(suite_cases))))
        idx = sorted(order[:take])
        sub = [suite_cases[i] for i in idx]
        sub_true = [c["expected"] for c in sub]
        eligible = [i for i in range(len(sub))
                    if len({repr(v) for j, v in enumerate(sub_true) if j != i}) >= 2]
        if not eligible:
            continue
        tamper_at = eligible[rng.randrange(len(eligible))]
        true_v = sub[tamper_at]["expected"]
        candidates = [v for v in sub_true if not _values_eq(v, true_v)]
        if not candidates:
            continue
        shown = copy.deepcopy(candidates[rng.randrange(len(candidates))])
        suite = copy.deepcopy(sub)
        suite[tamper_at]["expected"] = shown
        sig = repr([(c["case_id"], repr(c["input"]), repr(c["expected"])) for c in suite])
        if sig in seen:
            continue
        seen.add(sig)
        yield {
            "kind": "suite_audit",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "suite": suite,
            "tamper": {
                "case_id": sub[tamper_at]["case_id"],
                "true_value": true_v,
                "shown_value": shown,
            },
            "tck_test_unit_id": tck_unit,
            "universe": dict(UNIVERSE_DEFAULT),
            "complexity": complexity,
        }
        emitted += 1


def _values_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_eq(a[k], b[k]) for k in a)
    return a == b


# -- f11: output-entry repair -------------------------------------------------------

def _output_entry_candidates(payload: dict[str, Any], rule_index: int,
                             output_index: int) -> list[str]:
    original = payload["rules"][rule_index]["output_entries"][output_index]
    candidates: list[str] = []
    for r in payload["rules"]:
        entry = r["output_entries"][output_index]
        if entry != original and entry not in candidates:
            candidates.append(entry)
    entry = original.strip()
    if entry.lstrip("-").replace(".", "", 1).isdigit():
        num = float(entry)
        for delta in (1, -1, 2):
            cand = num + delta
            text = ("%d" % cand) if cand == int(cand) else repr(cand)
            if text not in candidates:
                candidates.append(text)
    elif entry in ("true", "false"):
        other = "false" if entry == "true" else "true"
        if other not in candidates:
            candidates.append(other)
    elif entry.startswith('"') and entry.endswith('"'):
        inner = entry[1:-1]
        for suffix in ("X", "Y"):
            cand = '"%s%s"' % (inner, suffix)
            if cand not in candidates:
                candidates.append(cand)
    return candidates


def build_output_repair(row: dict[str, Any], n_instances: int,
                        rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_DEFAULT)
    golden_rows = _golden_rows(payload, universe)
    if len(golden_rows) < SUITE_MIN or len(payload["rules"]) < 2:
        return
    golden_out = [r["output"] for r in golden_rows]
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    n_outputs = len(payload["outputs"])
    seen: set[tuple] = set()
    emitted = 0
    sites = [(ri, oi) for ri in range(len(payload["rules"]))
             for oi in range(n_outputs)]
    for _ in range(MAX_MUTATION_TRIES * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        if not sites:
            return
        ri, oi = sites[rng.randrange(len(sites))]
        candidates = _output_entry_candidates(payload, ri, oi)
        if not candidates:
            continue
        to = candidates[rng.randrange(len(candidates))]
        frm = payload["rules"][ri]["output_entries"][oi]
        sig = (ri, oi, frm, to)
        if sig in seen:
            continue
        broken = copy.deepcopy(payload)
        broken["rules"][ri]["output_entries"][oi] = to
        broken_out = _behavior(broken, golden_rows)
        if broken_out is None:
            continue
        failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out))
                   if not _values_eq(g, b)]
        if not failing:
            continue
        suite = _suite(golden_rows, failing, rng)
        if suite is None:
            continue
        seen.add(sig)
        yield {
            "kind": "output_repair",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "broken_table": broken,
            "mutation": {"slot": "output_entry", "rule_index": ri,
                         "output_index": oi, "from": frm, "to": to},
            "suite": suite,
            "universe": universe,
            "complexity": complexity,
        }
        emitted += 1


# -- f12: aggregation repair ---------------------------------------------------------

def aggregation_variants(payload: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    original = payload.get("aggregation") or ""
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for alt in AGGREGATION_ALTERNATIVES:
        if alt == original:
            continue
        broken = copy.deepcopy(payload)
        broken["aggregation"] = alt
        out.append((broken, {"slot": "aggregation", "from": original, "to": alt}))
    return out


def build_aggregation_repair(row: dict[str, Any], broken: dict[str, Any],
                             mutation: dict[str, Any], n_instances: int,
                             rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_WIDE)
    golden_rows = _golden_cases(payload, universe)
    if len(golden_rows) < SUITE_MIN:
        return
    golden_out = [r["output"] for r in golden_rows]
    broken_out = _behavior(broken, golden_rows)
    if broken_out is None:
        return
    failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out))
               if not _values_eq(g, b)]
    if not failing:
        return
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen: set[str] = set()
    emitted = 0
    for _ in range(6 * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        suite = _suite(golden_rows, failing, rng)
        if suite is None:
            continue
        sig = repr(sorted((repr(s["input"]) for s in suite)))
        if sig in seen:
            continue
        seen.add(sig)
        yield {
            "kind": "aggregation_repair",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "broken_table": broken,
            "mutation": mutation,
            "suite": suite,
            "universe": universe,
            "complexity": complexity,
        }
        emitted += 1


def _golden_cases(payload: dict[str, Any], universe: dict[str, int]) -> list[dict[str, Any]]:
    table = table_from_payload(payload)
    cases = case_mod.case_universe(table, **universe)
    return case_mod.eval_universe(table, cases)


# -- f13: rule-deletion repair ---------------------------------------------------------

def build_rule_deletion_repair(row: dict[str, Any], n_instances: int,
                               rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_DEFAULT)
    golden_rows = _golden_cases(payload, universe)
    if len(golden_rows) < SUITE_MIN or len(payload["rules"]) < 3:
        return
    golden_out = [r["output"] for r in golden_rows]
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen: set[int] = set()
    emitted = 0
    for _ in range(MAX_MUTATION_TRIES * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        ri = rng.randrange(len(payload["rules"]))
        if ri in seen:
            continue
        broken = copy.deepcopy(payload)
        removed = broken["rules"].pop(ri)
        broken_out = _behavior(broken, golden_rows)
        if broken_out is None:
            continue
        failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out))
                   if not _values_eq(g, b)]
        if not failing:
            continue
        suite = _suite(golden_rows, failing, rng)
        if suite is None:
            continue
        seen.add(ri)
        yield {
            "kind": "rule_deletion_repair",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "broken_table": broken,
            "mutation": {"slot": "rule_deletion", "rule_index": ri,
                         "rule_id": removed["id"]},
            "suite": suite,
            "universe": universe,
            "complexity": complexity,
        }
        emitted += 1


# -- f14 / f15: partial-input audit + chain synthesis -----------------------------------

def _chain_tables(material: dict[str, Any]) -> dict[str, Any]:
    return {d: table_from_payload(p) for d, p in material["tables"].items()}


def classify_partial(table: Any, partial: dict[str, Any]
                     ) -> tuple[list[int], list[int]]:
    """Three-valued rule classification under a partial input assignment.

    Returns (necessarily_matching, possibly_matching) rule indexes. A clause
    whose variable is unbound is unknown unless its test is empty (always
    true); a rule necessarily matches when no clause is false and none is
    unknown; it possibly matches when no clause is false.
    """
    from benchmark.plugins.verifier.l03.feel import match_unary

    nec: list[int] = []
    poss: list[int] = []
    for idx, rule in enumerate(table.rules):
        has_false = False
        has_unknown = False
        for clause, test in zip(table.inputs, rule.input_entries):
            if clause.expression in partial:
                if not match_unary(test, partial[clause.expression], clause.type_ref):
                    has_false = True
                    break
            elif test.strip() not in ("", "-"):
                has_unknown = True
        if not has_false:
            poss.append(idx)
            if not has_unknown:
                nec.append(idx)
    return nec, poss


def build_partial_input_audit(row: dict[str, Any], n_instances: int,
                              rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_DEFAULT)
    table = table_from_payload(payload)
    rows = _golden_cases(payload, universe)
    hits = [r for r in rows if r["matched_rule_ids"] and r["output"] is not None]
    if len(hits) < SUITE_MIN or len(payload["inputs"]) < 2:
        return
    rule_ids = [r["id"] for r in payload["rules"]]
    if any(not rid for rid in rule_ids) or len(set(rule_ids)) != len(rule_ids):
        return
    dom = {c.expression: case_mod.column_domain(table, i)
           for i, c in enumerate(table.inputs)}
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen: set[tuple] = set()
    emitted = 0
    pool = sorted(hits, key=lambda r: repr(sorted(r["input"].items())))
    rng.shuffle(pool)
    for case_row in pool:
        if emitted >= n_instances:
            return
        for mv in [c.expression for c in table.inputs]:
            if emitted >= n_instances:
                return
            partial = {k: v for k, v in case_row["input"].items() if k != mv}
            if not partial:
                continue
            nec, poss = classify_partial(table, partial)
            amb = [i for i in poss if i not in nec]
            if not poss or not amb:
                continue
            outs = []
            for v in dom[mv]:
                full = dict(partial)
                full[mv] = v
                try:
                    res = evaluate_table(table, full)
                    outs.append(res.value)
                except Exception:
                    continue
            if len({repr(o) for o in outs}) < 2:
                continue
            sig = (repr(sorted(partial.items())), mv)
            if sig in seen:
                continue
            seen.add(sig)
            yield {
                "kind": "partial_input_audit",
                "unit_id": row["unit_id"],
                "decision_id": row["decision_id"],
                "table": payload,
                "partial_input": partial,
                "missing_var": mv,
                "reference": {
                    "necessarily_matching": [rule_ids[i] for i in nec],
                    "possibly_matching": [rule_ids[i] for i in poss],
                },
                "universe": universe,
                "complexity": complexity,
            }
            emitted += 1


def build_chain_synthesis(chain: dict[str, Any],
                          rows_by_dec: dict[str, dict[str, Any]],
                          terminal: str, n_instances: int,
                          rng: random.Random) -> Iterator[dict[str, Any]]:
    from benchmark.plugins.task_family.l03.builders_chain import chain_material

    material = chain_material(chain, rows_by_dec)
    if material is None:
        return
    material["terminal_decision_id"] = terminal
    tables = _chain_tables(material)
    root_cases = chain_root_universe(tables, material["edges"], root_cap=ROOT_CAP)
    outputs: list[Any] = []
    for case in root_cases:
        try:
            res, _memo = evaluate_chain(tables, material["edges"], terminal, case)
            outputs.append(res.value)
        except Exception:
            outputs.append(None)
    ok_idx = [i for i, o in enumerate(outputs) if o is not None]
    if len(ok_idx) < SUITE_MIN:
        return
    by_out: dict[str, list[int]] = {}
    for i in ok_idx:
        by_out.setdefault(repr(outputs[i]), []).append(i)
    targets = [k for k, v in sorted(by_out.items())
               if 0 < len(v) <= max(1, len(ok_idx) // 2)]
    if not targets:
        return
    complexity = complexity_of_chain(len(material["decision_ids"]))
    emitted = 0
    for target in targets:
        if emitted >= n_instances:
            return
        producers = by_out[target]
        rng.shuffle(producers)
        i = producers[0]
        yield {
            "kind": "chain_synthesis",
            "unit_id": chain["unit_id"],
            "decision_id": terminal,
            "chain": material,
            "target_output": outputs[i],
            "reference_input": root_cases[i],
            "universe": {"root_cap": ROOT_CAP},
            "complexity": complexity,
        }
        emitted += 1


# -- f16: ordered-policy output -----------------------------------------------------------

def build_ordered_output(row: dict[str, Any], n_instances: int,
                         rng: random.Random) -> Iterator[dict[str, Any]]:
    if row["hit_policy"] not in ("RULE ORDER", "OUTPUT ORDER"):
        return
    payload, rows = _universe(row)
    multi = [r for r in rows
             if len(r["matched_rule_ids"]) >= 2 and r["output"] is not None]
    if len(multi) < n_instances:
        return
    outputs = {repr(r["output"]) for r in multi}
    if len(outputs) < 2:
        return
    pool = sorted(multi, key=lambda r: repr(sorted(r["input"].items())))
    rng.shuffle(pool)
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    for case_row in pool[:n_instances]:
        yield {
            "kind": "ordered_output",
            "unit_id": row["unit_id"],
            "decision_id": row["decision_id"],
            "table": payload,
            "input_case": case_row["input"],
            "reference_output": case_row["output"],
            "reference_matched_rule_ids": list(case_row["matched_rule_ids"]),
            "universe": dict(UNIVERSE_DEFAULT),
            "complexity": complexity,
        }
