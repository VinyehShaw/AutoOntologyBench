"""L03 chain-family builders: f05 cross-table chain eval, f06 binding repair.

A chain is one pinned DMN file's usable decisions linked by
informationRequirements (from ``chains.jsonl``; acyclic, >=2 usable decisions).
f05 base goal = (file, terminal decision); instances = distinct root input
cases. f06 base goal = (file, terminal decision, binding site); instances =
distinct seeded root-case suites. The root-input universe is derived by the
verifier-owned :func:`chain_root_universe` so generation and verification
share one deterministic code path.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l03.families import complexity_of_chain
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_chain
from benchmark.plugins.verifier.l03.dmn_model import table_from_payload
from benchmark.plugins.verifier.l03.plugin import chain_root_universe

ROOT_CAP = 64
SUITE_MIN = 2
SUITE_MAX = 4


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_policy": row["hit_policy"],
        "aggregation": row["aggregation"],
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "rules": row["rules"],
    }


def chain_material(chain: dict[str, Any],
                   rows_by_dec: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows_by_dec) < 2:
        return None
    tables: dict[str, Any] = {}
    digests: dict[str, str] = {}
    names: dict[str, str] = {}
    for dec_id in chain["decision_ids"]:
        row = rows_by_dec.get(dec_id)
        if row is None:
            return None
        try:
            table_from_payload(_payload(row))
        except Exception:
            return None
        tables[dec_id] = _payload(row)
        digests[dec_id] = row["table_digest"]
        names[dec_id] = row["decision_name"]
    return {
        "decision_ids": list(chain["decision_ids"]),
        "terminal_decision_id": None,
        "edges": {d: list(deps) for d, deps in chain["edges"].items()},
        "tables": tables,
        "table_digests": digests,
        "decision_names": names,
    }


def _decision_tables(material: dict[str, Any]) -> dict[str, Any]:
    return {d: table_from_payload(p) for d, p in material["tables"].items()}


def _root_outputs(material: dict[str, Any], terminal: str,
                  tables: dict[str, Any] | None = None) -> tuple[list[Any], list[dict[str, Any]]] | None:
    tables = tables or _decision_tables(material)
    root_cases = chain_root_universe(tables, material["edges"], root_cap=ROOT_CAP)
    outputs: list[Any] = []
    for case in root_cases:
        try:
            res, _memo = evaluate_chain(
                tables, material["edges"], terminal, case,
                decision_names=material["decision_names"])
        except Exception:
            return None
        outputs.append(res.value)
    return outputs, root_cases


def binding_sites(material: dict[str, Any]) -> list[tuple[str, str]]:
    """(consuming decision, input expression) pairs bound to a required decision."""
    produced: dict[str, str] = {}
    for dec_id, payload in material["tables"].items():
        outputs = payload.get("outputs") or []
        if outputs:
            produced[outputs[0]["name"]] = dec_id
    sites: list[tuple[str, str]] = []
    for dec_id, deps in material["edges"].items():
        if not deps:
            continue
        for clause in material["tables"][dec_id].get("inputs", []):
            expr = clause["expression"]
            if expr in produced and produced[expr] in deps:
                sites.append((dec_id, expr))
    return sorted(set(sites))


def mutate_binding_site(material: dict[str, Any], dec_id: str, expr: str
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
    broken = copy.deepcopy(material["tables"])
    new_expr = expr + "__unbound"
    for clause in broken[dec_id]["inputs"]:
        if clause["expression"] == expr:
            clause["expression"] = new_expr
    producer = None
    for d, payload in material["tables"].items():
        outputs = payload.get("outputs") or []
        if outputs and outputs[0]["name"] == expr:
            producer = d
    return broken, {
        "slot": "binding",
        "decision_id": dec_id,
        "from": expr,
        "to": new_expr,
        "producer_decision_id": producer,
    }


def build_chain_eval(chain: dict[str, Any], rows_by_dec: dict[str, dict[str, Any]],
                     terminal: str, n_instances: int,
                     rng: random.Random) -> Iterator[dict[str, Any]]:
    material = chain_material(chain, rows_by_dec)
    if material is None:
        return
    material["terminal_decision_id"] = terminal
    result = _root_outputs(material, terminal)
    if result is None:
        return
    outputs, root_cases = result
    ok = [(case, out) for case, out in zip(root_cases, outputs)
          if out is not None]
    distinct = {repr(out) for _, out in ok}
    if len(ok) < n_instances or len(distinct) < 2:
        return
    pool = sorted(ok, key=lambda p: repr(sorted(p[0].items())))
    rng.shuffle(pool)
    complexity = complexity_of_chain(len(material["decision_ids"]))
    for case, out in pool[:n_instances]:
        yield {
            "kind": "chain_eval",
            "unit_id": chain["unit_id"],
            "decision_id": terminal,
            "chain": material,
            "input_case": case,
            "reference_output": out,
            "universe": {"root_cap": ROOT_CAP},
            "complexity": complexity,
        }


def build_binding_repair(chain: dict[str, Any], rows_by_dec: dict[str, dict[str, Any]],
                         terminal: str, site: tuple[str, str], n_instances: int,
                         rng: random.Random) -> Iterator[dict[str, Any]]:
    material = chain_material(chain, rows_by_dec)
    if material is None:
        return
    material["terminal_decision_id"] = terminal
    result = _root_outputs(material, terminal)
    if result is None:
        return
    golden_outputs, root_cases = result
    broken_tables, mutation = mutate_binding_site(material, site[0], site[1])
    broken = {d: table_from_payload(p) for d, p in broken_tables.items()}
    broken_outputs: list[Any] = []
    for case in root_cases:
        try:
            res, _memo = evaluate_chain(
                broken, material["edges"], terminal, case,
                decision_names=material["decision_names"])
            broken_outputs.append(res.value)
        except Exception:
            broken_outputs.append(None)
    failing = [i for i, (g, b) in enumerate(zip(golden_outputs, broken_outputs))
               if g != b]
    if not failing:
        return
    ok_idx = [i for i, o in enumerate(golden_outputs) if o is not None]
    if len(ok_idx) < SUITE_MIN:
        return
    complexity = complexity_of_chain(len(material["decision_ids"]))
    seen: set[str] = set()
    emitted = 0
    for _ in range(6 * max(n_instances, 1)):
        if emitted >= n_instances:
            return
        take_fail = min(len(failing), rng.randint(1, 2))
        fail_pick = rng.sample(failing, take_fail)
        room = SUITE_MAX - len(fail_pick)
        pass_pool = [i for i in ok_idx if i not in set(fail_pick)]
        pass_pick = rng.sample(pass_pool, min(room, len(pass_pool))) if room > 0 else []
        idx = sorted(set(fail_pick) | set(pass_pick))
        if len(idx) < SUITE_MIN:
            continue
        suite = [{"input": root_cases[i], "expected": golden_outputs[i]} for i in idx]
        sig = repr(sorted((repr(s["input"]) for s in suite)))
        if sig in seen:
            continue
        seen.add(sig)
        yield {
            "kind": "binding_repair",
            "unit_id": chain["unit_id"],
            "decision_id": terminal,
            "chain": material,
            "broken_chain": broken_tables,
            "mutation": mutation,
            "suite": suite,
            "universe": {"root_cap": ROOT_CAP},
            "complexity": complexity,
        }
        emitted += 1
