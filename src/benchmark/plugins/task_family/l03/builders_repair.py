"""L03 repair-family builders: f03 rule-entry repair, f04 hit-policy repair.

Both mutate the canonical table payload into a *broken* variant and emit the
golden payload, the mutation descriptor and a visible case suite whose expected
values are the golden (pinned) behavior. Mutations are proposals only: a
mutation becomes an instance solely when the pinned runtime shows a behavioral
differential (golden vs broken differ on the golden case universe); otherwise
the candidate is skipped here and would be rejected by the verifier anyway.

f03 base goal = one decision table; instances = distinct seeded single-cell
mutations. f04 base goal = (table, broken hit-policy variant) because a table
admits only 2-3 policy alternatives; instances = distinct seeded case suites
(the shown failing/passing evidence subset varies, the repair target is fixed).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l03 import cases as case_mod, mutators
from benchmark.plugins.task_family.l03.builders_single import (
    UNIVERSE_DEFAULT, _payload, _universe,
)
from benchmark.plugins.task_family.l03.families import complexity_of_table
from benchmark.plugins.verifier.l03.dmn_model import table_from_payload

SUITE_MIN = 3
SUITE_MAX = 6
MAX_MUTATION_TRIES = 8


def _golden_cases(payload: dict[str, Any],
                  universe: dict[str, int]) -> list[dict[str, Any]]:
    table = table_from_payload(payload)
    cases = case_mod.case_universe(table, **universe)
    return case_mod.eval_universe(table, cases)


def _behavior(broken_payload: dict[str, Any],
              golden_rows: list[dict[str, Any]]) -> list[Any] | None:
    """Broken outputs over the golden case list; None when evaluation fails."""
    try:
        broken = table_from_payload(broken_payload)
    except Exception:
        return None
    out: list[Any] = []
    for row in golden_rows:
        try:
            res = case_mod.eval_universe(broken, [row["input"]])
        except Exception:
            return None
        if not res:
            return None
        out.append(res[0]["output"])
    return out


def _suite(golden_rows: list[dict[str, Any]], failing: list[int],
           rng: random.Random) -> list[dict[str, Any]] | None:
    """Seeded visible suite: 2..4 failing cases plus passing fill, <=6 cases.

    Passing fill uses only cases where the golden table actually produces an
    output (a no-match case carries no policy signal). All-failing variants
    (e.g. COLLECT collapsed to a scalar policy) yield failing-only suites.
    """
    if not failing or len(golden_rows) < 2:
        return None
    hi = min(4, len(failing))
    take_failing = rng.randint(min(2, hi), hi)
    failing_pick = rng.sample(failing, take_failing)
    passing = [i for i, r in enumerate(golden_rows)
               if i not in set(failing) and r["output"] is not None]
    room = SUITE_MAX - len(failing_pick)
    passing_pick = rng.sample(passing, min(room, len(passing))) if passing and room > 0 else []
    idx = sorted(set(failing_pick) | set(passing_pick))
    if len(idx) < 2:
        return None
    return [
        {"input": golden_rows[i]["input"], "expected": golden_rows[i]["output"]}
        for i in idx
    ]


def build_rule_entry_repair(row: dict[str, Any], n_instances: int,
                            rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_DEFAULT)
    golden_rows = _golden_cases(payload, universe)
    if len(golden_rows) < SUITE_MIN:
        return
    golden_out = [r["output"] for r in golden_rows]
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen_mutations: set[tuple] = set()
    emitted = 0
    for _ in range(MAX_MUTATION_TRIES * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        mutated = mutators.mutate_rule_entry(payload, rng)
        if mutated is None:
            continue
        broken, mutation = mutated
        sig = (mutation["rule_index"], mutation["column"],
               mutation["from"], mutation["to"])
        if sig in seen_mutations:
            continue
        broken_out = _behavior(broken, golden_rows)
        if broken_out is None:
            continue
        failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out)) if g != b]
        if not failing:
            continue
        suite = _suite(golden_rows, failing, rng)
        if suite is None:
            continue
        seen_mutations.add(sig)
        yield {
            "kind": "rule_repair",
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


def hit_policy_variants(payload: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """All (broken_table, mutation) variants for the table's hit policy."""
    original = payload.get("hit_policy") or "UNIQUE"
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for alt in mutators._HIT_ALTERNATIVES.get(original, ()):
        broken = copy.deepcopy(payload)
        broken["hit_policy"] = alt
        if alt != "COLLECT":
            broken["aggregation"] = ""
        out.append((broken, {
            "slot": "hit_policy",
            "from": original,
            "to": alt,
            "aggregation_from": payload.get("aggregation") or "",
            "aggregation_to": broken.get("aggregation") or "",
        }))
    return out


def build_hit_policy_repair(row: dict[str, Any], broken: dict[str, Any],
                            mutation: dict[str, Any], n_instances: int,
                            rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = _payload(row)
    universe = dict(UNIVERSE_DEFAULT)
    golden_rows = _golden_cases(payload, universe)
    if len(golden_rows) < SUITE_MIN:
        return
    golden_out = [r["output"] for r in golden_rows]
    broken_out = _behavior(broken, golden_rows)
    if broken_out is None:
        return
    failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out)) if g != b]
    if not failing:
        return
    complexity = complexity_of_table(
        len(payload["rules"]), len(payload["inputs"]), len(payload["outputs"]))
    seen_suites: set[str] = set()
    emitted = 0
    for _ in range(6 * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        suite = _suite(golden_rows, failing, rng)
        if suite is None:
            continue
        sig = repr(sorted((repr(s["input"]) for s in suite)))
        if sig in seen_suites:
            continue
        seen_suites.add(sig)
        yield {
            "kind": "hit_policy_repair",
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
