"""L03 task-family plugin (§3.2 facade): supported_profiles / propose / ground.

Drives the eight structural families of ``families.py`` over the pinned
combined index. Goal selection is capacity-aware: candidates are ranked by
how many instances the family can honestly yield from that unit (probed with
the same pinned runtime the verifier uses), the strongest goals are taken,
and per-goal instance counts are capped by probed capacity so the family
quota is met without inflating weak units. The seed drives instance-level
material (case picks, mutations, suites) via sha256-derived per-goal RNGs.

Base-goal coordinates are family-specific sites, not renames: single-table
families key on the decision, f04 adds the broken hit-policy variant, f05
keys on the chain terminal, f06 on the chain terminal plus binding site.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.task_family.l03 import (
    builders_chain,
    builders_repair,
    builders_single,
)
from benchmark.plugins.task_family.l03.builders_single import (
    UNIVERSE_DEFAULT,
    UNIVERSE_WIDE,
)
from benchmark.plugins.task_family.l03.families import (
    FAMILY_BY_ID,
    GOAL_BUDGETS,
    SEMANTIC_PROFILE,
)
from benchmark.plugins.task_family.l03 import cases as case_mod
from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.verifier.l03.dmn_model import sha256_json, table_from_payload

LANE = "l03"
INDEX_DIR = Path("external_knowledge/indexes/camunda-dmn-examples/combined")
DATASET_ID = "autoontobench-bank-8000-rc1"
DEFAULT_SEED = 20261027

#: family_id -> (base goals, instances). f07 is capped at 9 table-level goals
#: by the COLLECT corpus (8 tables with >=5 multi-match cases + one with 4);
#: the missing goal moved to f01 (richest corpus). Sums to exactly 400/80/8.
FAMILY_QUOTA: dict[str, tuple[int, int]] = {
    "l03-f01-exact-decision-output": (11, 56),
    "l03-f02-rule-match-audit": (10, 50),
    "l03-f03-rule-entry-repair": (10, 50),
    "l03-f04-hit-policy-repair": (10, 50),
    "l03-f05-cross-table-chain-eval": (10, 50),
    "l03-f06-cross-table-binding-repair": (10, 50),
    "l03-f07-collect-aggregation-result": (9, 44),
    "l03-f08-hidden-rule-input-synthesis": (10, 50),
}

ALLOWED_TOOLS = {
    "application": ["dmn_table_inspect", "dmn_table_evaluate"],
    "repair": ["dmn_table_inspect", "dmn_table_evaluate", "dmn_table_edit"],
    "construction": ["dmn_table_inspect", "dmn_table_probe"],
}

_GOAL_TEXT = {
    "l03-f01-exact-decision-output": (
        "Evaluate the DMN decision table '{decision}' ({rules} rules, hit policy "
        "{policy}) for the given typed input case and produce the exact decision "
        "output value. Submit {{\"output\": <value>}}."
    ),
    "l03-f02-rule-match-audit": (
        "For the DMN decision table '{decision}' ({rules} rules, hit policy "
        "{policy}) and the given typed input case, produce the exact ordered list "
        "of rule ids that match under the table's hit policy (in rule order). "
        "Submit {{\"matched_rule_ids\": [...]}}."
    ),
    "l03-f03-rule-entry-repair": (
        "The decision table '{decision}' below returns wrong outputs for some "
        "cases: exactly one input-entry cell was corrupted. Repair that single "
        "cell so the table produces the intended outputs for every case in the "
        "visible suite. Submit the full repaired table payload as "
        "{{\"table\": {{...}}}}."
    ),
    "l03-f04-hit-policy-repair": (
        "The decision table '{decision}' below had its hit policy changed and now "
        "returns wrong-shaped results. Restore the hit policy (and aggregation, "
        "if any) so every case in the suite matches the intended result. Submit "
        "the full repaired table payload as {{\"table\": {{...}}}}."
    ),
    "l03-f05-cross-table-chain-eval": (
        "Evaluate the terminal decision '{decision}' of the multi-table DMN "
        "requirements graph below ({decisions} decisions), propagating "
        "intermediate decision outputs from the given root input case, and "
        "produce its exact output value. Submit {{\"output\": <value>}}."
    ),
    "l03-f06-cross-table-binding-repair": (
        "In the decision chain below (terminal decision '{decision}'), one "
        "decision no longer receives its required decision's output because a "
        "variable binding was renamed. Restore the cross-table binding so the "
        "chain evaluates to the intended terminal results for every case in the "
        "suite. Submit the full repaired chain tables as "
        "{{\"tables\": {{decision_id: table}}}}."
    ),
    "l03-f07-collect-aggregation-result": (
        "Compute the exact COLLECT result (list or aggregate) of the decision "
        "table '{decision}' (hit policy {policy}) for the given input case, which "
        "matches multiple rules. Submit {{\"output\": <value>}}."
    ),
    "l03-f08-hidden-rule-input-synthesis": (
        "Only the schema, hit policy and rule outputs of the decision table "
        "'{decision}' are visible (input entries are masked). Construct a typed "
        "input assignment for the input variables that makes the decision return "
        "the target output. Submit {{\"input\": {{variable: value}}}}."
    ),
}


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_policy": row["hit_policy"],
        "aggregation": row["aggregation"],
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "rules": row["rules"],
    }


def _goal_rng(seed: int, family_id: str, *key: Any) -> random.Random:
    return random.Random(int(sha256_json([seed, family_id, *key])[:16], 16))


def _adaptive_counts(instances_target: int, caps: list[int]) -> list[int]:
    """Per-goal instance counts that sum to the target when capacity allows.

    Starts every goal at the base count capped by its probed capacity, then
    distributes the remainder to the highest-capacity goals first (wrapping
    around), so a family whose weakest goals are capacity-bound still reaches
    its quota through its strong goals.
    """
    n = len(caps)
    if n == 0:
        return []
    base = instances_target // n
    counts = [min(base, max(c, 1)) for c in caps]
    remaining = instances_target - sum(counts)
    order = sorted(range(n), key=lambda i: -caps[i])
    while remaining > 0:
        progressed = False
        for i in order:
            if counts[i] < caps[i]:
                counts[i] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


class L03TaskFamilies:
    """TaskFamilyPlugin for lane L03 (Camunda DMN static tasks)."""

    plugin_id = "l03.families"

    def __init__(self) -> None:
        self._index: dict[str, Any] | None = None
        self._rows_cache: dict[tuple, list[dict[str, Any]]] = {}

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    # -- index ------------------------------------------------------------------
    def load_index(self, index_dir: Path | None = None) -> dict[str, Any]:
        if self._index is not None and index_dir is None:
            return self._index
        base = Path(index_dir) if index_dir else INDEX_DIR
        with (base / "tables.jsonl").open(encoding="utf-8") as fh:
            tables = [json.loads(line) for line in fh if line.strip()]
        with (base / "chains.jsonl").open(encoding="utf-8") as fh:
            chains = [json.loads(line) for line in fh if line.strip()]
        index = {"tables": tables, "chains": chains}
        if index_dir is None:
            self._index = index
        return index

    def _universe_rows(self, row: dict[str, Any],
                       params: dict[str, int]) -> list[dict[str, Any]]:
        key = (row["unit_id"], row["decision_id"], tuple(sorted(params.items())))
        if key not in self._rows_cache:
            table = table_from_payload(_payload(row))
            cases = case_mod.case_universe(table, **params)
            self._rows_cache[key] = case_mod.eval_universe(table, cases)
        return self._rows_cache[key]

    def _rows_by_dec(self) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for row in self.load_index()["tables"]:
            out.setdefault(row["unit_id"], {})[row["decision_id"]] = row
        return out

    # -- propose ------------------------------------------------------------------
    def propose(self, snapshot: SourceSnapshot | None = None, *, family: str,
                seed: int = DEFAULT_SEED,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = FAMILY_BY_ID[family]
        goals_target, instances_target = FAMILY_QUOTA[family]
        constraints = constraints or {}
        goals_target = int(constraints.get("goals", goals_target))
        instances_target = int(constraints.get("instances", instances_target))
        kind = spec.goal_kind
        if kind in ("exact_output", "rule_audit", "collect_result",
                    "input_synthesis"):
            yield from self._propose_table_family(family, seed, goals_target,
                                                  instances_target)
        elif kind == "rule_repair":
            yield from self._propose_rule_repair(family, seed, goals_target,
                                                 instances_target)
        elif kind == "hit_policy_repair":
            yield from self._propose_hit_policy_repair(family, seed, goals_target,
                                                       instances_target)
        elif kind == "chain_eval":
            yield from self._propose_chain_eval(family, seed, goals_target,
                                                instances_target)
        elif kind == "binding_repair":
            yield from self._propose_binding_repair(family, seed, goals_target,
                                                    instances_target)
        else:  # pragma: no cover - families.py is exhaustive
            raise ValueError("unknown goal_kind %r" % kind)

    def _propose_table_family(self, family_id: str, seed: int, goals_target: int,
                              instances_target: int) -> Iterator[CandidateTaskAST]:
        kind = FAMILY_BY_ID[family_id].goal_kind
        builder = {
            "exact_output": builders_single.build_exact_output,
            "rule_audit": builders_single.build_rule_audit,
            "collect_result": builders_single.build_collect_result,
            "input_synthesis": builders_single.build_input_synthesis,
        }[kind]
        params = UNIVERSE_WIDE if kind == "collect_result" else UNIVERSE_DEFAULT
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self.load_index()["tables"]:
            rows = self._universe_rows(row, params)
            cap = _table_capacity(kind, row, rows)
            if cap > 0:
                capacities.append((cap, row))
        rng = random.Random(seed)
        rng.shuffle(capacities)
        capacities.sort(key=lambda p: -p[0])
        selected = capacities[:goals_target]
        counts = _adaptive_counts(instances_target, [cap for cap, _row in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builder(row, min(n, cap), goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _propose_rule_repair(self, family_id: str, seed: int, goals_target: int,
                             instances_target: int) -> Iterator[CandidateTaskAST]:
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self.load_index()["tables"]:
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            if len(rows) < builders_repair.SUITE_MIN:
                continue
            if self._has_differential_mutation(row, rows, seed):
                capacities.append((len(rows), row))
        rng = random.Random(seed)
        rng.shuffle(capacities)
        capacities.sort(key=lambda p: -p[0])
        selected = capacities[:goals_target]
        counts = _adaptive_counts(instances_target, [cap for cap, _row in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_repair.build_rule_entry_repair(
                    row, min(n, cap), goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _has_differential_mutation(self, row: dict[str, Any],
                                   rows: list[dict[str, Any]], seed: int) -> bool:
        payload = _payload(row)
        probe_rng = _goal_rng(seed, "probe", row["unit_id"], row["decision_id"])
        golden_out = [r["output"] for r in rows]
        for _ in range(2 * builders_repair.MAX_MUTATION_TRIES):
            mutated = builders_repair.mutators.mutate_rule_entry(payload, probe_rng)
            if mutated is None:
                continue
            broken, _mutation = mutated
            broken_out = builders_repair._behavior(broken, rows)
            if broken_out is None:
                continue
            if any(g != b for g, b in zip(golden_out, broken_out)):
                return True
        return False

    def _propose_hit_policy_repair(self, family_id: str, seed: int,
                                   goals_target: int,
                                   instances_target: int) -> Iterator[CandidateTaskAST]:
        goals: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for row in self.load_index()["tables"]:
            payload = _payload(row)
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            if len(rows) < builders_repair.SUITE_MIN:
                continue
            golden_out = [r["output"] for r in rows]
            for broken, mutation in builders_repair.hit_policy_variants(payload):
                broken_out = builders_repair._behavior(broken, rows)
                if broken_out is None:
                    continue
                failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out))
                           if g != b]
                if not failing:
                    continue
                cap = self._suite_yield(rows, failing, seed, row, mutation)
                if cap > 0:
                    goals.append((cap, row, broken, mutation))
        rng = random.Random(seed)
        rng.shuffle(goals)
        goals.sort(key=lambda p: -p[0])
        selected = goals[:goals_target]
        counts = _adaptive_counts(instances_target, [cap for cap, _r, _b, _m in selected])
        for (cap, row, broken, mutation), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"],
                                 row["decision_id"], mutation["to"])
            for yielded in builders_repair.build_hit_policy_repair(
                    row, broken, mutation, min(n, cap), goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _suite_yield(self, rows: list[dict[str, Any]], failing: list[int],
                     seed: int, row: dict[str, Any],
                     mutation: dict[str, Any]) -> int:
        probe_rng = _goal_rng(seed, "probe", row["unit_id"], mutation["to"])
        seen: set[str] = set()
        for _ in range(30):
            suite = builders_repair._suite(rows, failing, probe_rng)
            if suite is None:
                continue
            seen.add(repr(sorted((repr(s["input"]) for s in suite))))
        return len(seen)

    def _propose_chain_eval(self, family_id: str, seed: int, goals_target: int,
                            instances_target: int) -> Iterator[CandidateTaskAST]:
        goals = self._chain_goals()
        rng = random.Random(seed)
        rng.shuffle(goals)
        goals.sort(key=lambda p: -p[0])
        selected = goals[:goals_target]
        counts = _adaptive_counts(instances_target, [cap for cap, _c, _t in selected])
        rows_by_unit = self._rows_by_dec()
        for (cap, chain, terminal), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, chain["unit_id"], terminal)
            for yielded in builders_chain.build_chain_eval(
                    chain, rows_by_unit[chain["unit_id"]], terminal,
                    min(n, cap), goal_rng):
                yield self._wrap(family_id, chain, yielded)

    def _chain_goals(self) -> list[tuple[int, dict[str, Any], str]]:
        goals: list[tuple[int, dict[str, Any], str]] = []
        rows_by_unit = self._rows_by_dec()
        for chain in self.load_index()["chains"]:
            rows_by_dec = rows_by_unit.get(chain["unit_id"], {})
            material = builders_chain.chain_material(chain, rows_by_dec)
            if material is None:
                continue
            for terminal in chain["terminal_decision_ids"]:
                result = builders_chain._root_outputs(material, terminal)
                if result is None:
                    continue
                outputs, _root_cases = result
                ok = [o for o in outputs if o is not None]
                if len(ok) >= 5 and len({repr(o) for o in ok}) >= 2:
                    goals.append((len(ok), chain, terminal))
        return goals

    def _propose_binding_repair(self, family_id: str, seed: int, goals_target: int,
                                instances_target: int) -> Iterator[CandidateTaskAST]:
        goals = self._binding_goals()
        rng = random.Random(seed)
        rng.shuffle(goals)
        goals.sort(key=lambda p: -p[0])
        selected = goals[:goals_target]
        counts = _adaptive_counts(instances_target,
                                  [cap for cap, _c, _t, _s in selected])
        rows_by_unit = self._rows_by_dec()
        for (cap, chain, terminal, site), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, chain["unit_id"], terminal,
                                 site[0], site[1])
            for yielded in builders_chain.build_binding_repair(
                    chain, rows_by_unit[chain["unit_id"]], terminal, site,
                    min(n, cap), goal_rng):
                yield self._wrap(family_id, chain, yielded)

    def _binding_goals(self) -> list[tuple[int, dict[str, Any], str, tuple[str, str]]]:
        goals: list[tuple[int, dict[str, Any], str, tuple[str, str]]] = []
        rows_by_unit = self._rows_by_dec()
        for chain in self.load_index()["chains"]:
            rows_by_dec = rows_by_unit.get(chain["unit_id"], {})
            material = builders_chain.chain_material(chain, rows_by_dec)
            if material is None:
                continue
            sites = builders_chain.binding_sites(material)
            if not sites:
                continue
            for terminal in chain["terminal_decision_ids"]:
                result = builders_chain._root_outputs(material, terminal)
                if result is None:
                    continue
                golden_outputs, root_cases = result
                ok_idx = [i for i, o in enumerate(golden_outputs) if o is not None]
                if len(ok_idx) < builders_chain.SUITE_MIN:
                    continue
                for site in sites:
                    broken_tables, _mutation = builders_chain.mutate_binding_site(
                        material, site[0], site[1])
                    broken = {d: table_from_payload(p)
                              for d, p in broken_tables.items()}
                    failing = 0
                    for i in ok_idx:
                        try:
                            res, _memo = builders_chain.evaluate_chain(
                                broken, material["edges"], terminal, root_cases[i],
                                decision_names=material["decision_names"])
                        except Exception:
                            failing += 1
                            continue
                        if res.value != golden_outputs[i]:
                            failing += 1
                    if failing > 0:
                        goals.append((failing, chain, terminal, site))
        return goals

    # -- wrap / ground -----------------------------------------------------------------
    def _wrap(self, family_id: str, row: dict[str, Any],
              yielded: dict[str, Any]) -> CandidateTaskAST:
        spec = FAMILY_BY_ID[family_id]
        kind = yielded["kind"]
        coordinate = _coordinate(kind, yielded)
        base_goal_id = "bg-" + sha256_json(
            [family_id, yielded["unit_id"], coordinate])[:20]
        instance_material = _instance_material(kind, yielded)
        instance_id = "in-" + sha256_json(
            [family_id, base_goal_id, instance_material])[:20]
        ast = {
            **yielded,
            "family_id": family_id,
            "regime": spec.regime,
            "goal_kind": kind,
            "coordinate": coordinate,
            "source_path": "external_knowledge/pinned/" + yielded["unit_id"],
            "release_id": row.get("release_id") or yielded.get("release_id"),
            "table_digest": row.get("table_digest") if kind not in (
                "chain_eval", "binding_repair") else None,
            "reference": _reference(kind, yielded),
        }
        return CandidateTaskAST(
            lane_id="L03", family_id=family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime,
            semantic_profile=SEMANTIC_PROFILE, ast=ast,
            source_unit_ids=(yielded["unit_id"],),
        )

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        oracle_ref = "l03-oracle-%s" % sha256_json(
            [candidate.instance_id, ast["unit_id"], ast["goal_kind"]])[:16]
        public = {
            "user_goal": render_user_goal(ast),
            "allowed_tools": list(ALLOWED_TOOLS[ast["regime"]]),
            "budget": dict(GOAL_BUDGETS[ast["complexity"]]),
            "observation_refs": observation_refs(ast),
        }
        public.update(public_payload(ast))
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={})


def _table_capacity(kind: str, row: dict[str, Any],
                    rows: list[dict[str, Any]]) -> int:
    if kind == "exact_output":
        clean = [r for r in rows if r["matched_rule_ids"]
                 and r["output"] is not None and not r["violations"]]
        outputs = {repr(r["output"]) for r in rows if r["matched_rule_ids"]}
        return len(clean) if len(outputs) >= 2 else 0
    if kind == "rule_audit":
        n_rules = len(row["rules"])
        ok = [r for r in rows if 0 < len(r["matched_rule_ids"]) < n_rules]
        return len(ok) if n_rules >= 3 else 0
    if kind == "collect_result":
        if row["hit_policy"] != "COLLECT":
            return 0
        return len([r for r in rows if len(r["matched_rule_ids"]) >= 2])
    if kind == "input_synthesis":
        if len(row["rules"]) < 3:
            return 0
        hits = [r for r in rows if r["output"] is not None and r["matched_rule_ids"]]
        by_out: dict[str, list] = {}
        for r in hits:
            by_out.setdefault(repr(r["output"]), []).append(r)
        return sum(1 for k, v in sorted(by_out.items())
                   if 0 < len(v) <= max(2, len(hits) // 2))
    return 0


def _coordinate(kind: str, yielded: dict[str, Any]) -> dict[str, Any]:
    if kind in ("exact_output", "rule_audit", "collect_result",
                "input_synthesis", "rule_repair"):
        return {"decision_id": yielded["decision_id"]}
    if kind == "hit_policy_repair":
        return {"decision_id": yielded["decision_id"],
                "broken_hit_policy": yielded["mutation"]["to"],
                "broken_aggregation": yielded["mutation"]["aggregation_to"]}
    if kind == "chain_eval":
        return {"terminal_decision_id": yielded["decision_id"]}
    if kind == "binding_repair":
        return {"terminal_decision_id": yielded["decision_id"],
                "binding_decision_id": yielded["mutation"]["decision_id"],
                "binding_expression": yielded["mutation"]["from"]}
    return {}


def _instance_material(kind: str, yielded: dict[str, Any]) -> Any:
    if kind in ("exact_output", "rule_audit", "collect_result", "chain_eval"):
        return yielded["input_case"]
    if kind == "rule_repair":
        return yielded["mutation"]
    if kind in ("hit_policy_repair", "binding_repair"):
        return yielded["suite"]
    if kind == "input_synthesis":
        return yielded["target_output"]
    return None


def _reference(kind: str, yielded: dict[str, Any]) -> Any:
    if kind in ("exact_output", "chain_eval", "collect_result"):
        return yielded["reference_output"]
    if kind == "rule_audit":
        return yielded["reference_matched_rule_ids"]
    if kind in ("rule_repair", "hit_policy_repair"):
        return yielded["table"]
    if kind == "binding_repair":
        return yielded["chain"]["tables"]
    if kind == "input_synthesis":
        return yielded["reference_input"]
    return None


def observation_refs(ast: dict[str, Any]) -> list[str]:
    """Pinned-XML pointers for families whose gold is NOT the pinned table
    itself; repair/synthesis families keep the task self-contained so the
    pinned original cannot be read off the referenced file."""
    if ast["goal_kind"] in ("exact_output", "rule_audit", "collect_result",
                            "chain_eval"):
        return [ast["source_path"]]
    return []


def masked_table(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_policy": table["hit_policy"],
        "aggregation": table.get("aggregation") or "",
        "inputs": [dict(c) for c in table["inputs"]],
        "outputs": [dict(o) for o in table["outputs"]],
        "rules": [
            {
                "id": r["id"],
                "input_entries": ["?"] * len(table["inputs"]),
                "output_entries": list(r["output_entries"]),
            }
            for r in table["rules"]
        ],
    }


def public_chain(chain: dict[str, Any], *, broken: bool = False,
                 broken_chain: dict[str, Any] | None = None) -> dict[str, Any]:
    tables = broken_chain if broken else chain["tables"]
    return {
        "decision_ids": list(chain["decision_ids"]),
        "terminal_decision_id": chain["terminal_decision_id"],
        "edges": {d: list(deps) for d, deps in chain["edges"].items()},
        "tables": tables,
        "decision_names": dict(chain["decision_names"]),
        "table_digests": dict(chain["table_digests"]),
    }


def public_payload(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["goal_kind"]
    if kind in ("exact_output", "rule_audit", "collect_result"):
        return {"table": ast["table"], "input_case": ast["input_case"]}
    if kind in ("rule_repair", "hit_policy_repair"):
        return {"broken_table": ast["broken_table"], "suite": ast["suite"]}
    if kind == "chain_eval":
        return {"chain": public_chain(ast["chain"]),
                "input_case": ast["input_case"]}
    if kind == "binding_repair":
        return {"chain": public_chain(ast["chain"], broken=True,
                                      broken_chain=ast["broken_chain"]),
                "suite": ast["suite"]}
    if kind == "input_synthesis":
        return {"masked_table": masked_table(ast["table"]),
                "target_output": ast["target_output"]}
    return {}


def render_user_goal(ast: dict[str, Any]) -> str:
    text = _GOAL_TEXT[ast["family_id"]]
    table = ast.get("table") or {}
    return text.format(
        decision=ast.get("decision_id") or "?",
        rules=len(table.get("rules") or []),
        policy=table.get("hit_policy") or "?",
        decisions=len((ast.get("chain") or {}).get("decision_ids") or []),
    )


PLUGIN = L03TaskFamilies()
