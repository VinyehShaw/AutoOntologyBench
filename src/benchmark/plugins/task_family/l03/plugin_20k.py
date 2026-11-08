"""L03 20K-expansion task-family plugin: propose / ground for families f09..f16.

Pool policy (genuine-newness by construction):
- table families draw from DMN TCK tables (new pinned source) plus Camunda
  combined-index tables whose source units were NOT used by the 8K build
  (115 unused units / 169 tables);
- chain families (f14/f15) use a genuinely new mechanism (intermediate
  evaluation / root synthesis) over all 29 indexed chains - task signatures
  differ from every 8K family by goal_kind, and the pipeline verifies
  task_id + signature disjointness against the 8K qualified set;
- f09/f10 TCK instances come from pinned TCK test scenarios (dual oracle).

Goal selection is capacity-aware like the 8K plugin: candidates are ranked by
probed instance yield on the same pinned runtime, strongest goals first, and
per-goal counts are capped by probed capacity.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.task_family.l03 import builders_20k
from benchmark.plugins.task_family.l03.builders_single import (
    UNIVERSE_DEFAULT,
    _payload,
)
from benchmark.plugins.task_family.l03.families_20k import (
    FAMILY_BY_ID,
    FAMILY_QUOTA,
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
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
DEFAULT_SEED = 20261101
CAM_INDEX_DIR = Path("external_knowledge/indexes/camunda-dmn-examples/combined")
TCK_INDEX_DIR = Path("external_knowledge/indexes/dmn-tck/combined")
LEGACY_STAGING = Path("bank/staging/l03")

ALLOWED_TOOLS = {
    "application": ["dmn_table_inspect", "dmn_table_evaluate"],
    "repair": ["dmn_table_inspect", "dmn_table_evaluate", "dmn_table_edit"],
    "construction": ["dmn_table_inspect", "dmn_table_probe"],
}

_GOAL_TEXT = {
    "l03-f09-tck-scenario-output": (
        "The decision table '{decision}' ({rules} rules, hit policy {policy}) "
        "below is a DMN TCK conformance table. Evaluate it for the given typed "
        "input case (a TCK test scenario) and produce the exact decision output "
        "value. Submit {{\"output\": <value>}}."
    ),
    "l03-f10-suite-conformance-audit": (
        "The decision table '{decision}' ({rules} rules, hit policy {policy}) "
        "is claimed to satisfy every test case in the visible suite, but "
        "exactly one suite case has an expected value inconsistent with the "
        "table. Identify that case. Submit "
        "{{\"inconsistent_case_id\": \"<case_id>\"}}."
    ),
    "l03-f11-output-entry-repair": (
        "The decision table '{decision}' below returns wrong values for some "
        "cases: exactly one output-entry cell was corrupted, so matched rules "
        "return the wrong result. Repair that single cell so the table "
        "produces the intended outputs for every case in the visible suite. "
        "Submit the full repaired table payload as {{\"table\": {{...}}}}."
    ),
    "l03-f12-aggregation-repair": (
        "The COLLECT decision table '{decision}' below had its aggregation "
        "changed and now returns wrong aggregate results. Restore the "
        "aggregation (SUM/MIN/MAX/COUNT, or none) so every case in the suite "
        "matches the intended result. Submit the full repaired table payload "
        "as {{\"table\": {{...}}}}."
    ),
    "l03-f13-rule-deletion-repair": (
        "One whole rule was deleted from the decision table '{decision}' "
        "below, so it returns wrong results for some cases. Restore the "
        "table's full intended behavior for every case in the visible suite. "
        "Submit the full repaired table payload as {{\"table\": {{...}}}}."
    ),
    "l03-f14-partial-input-rule-audit": (
        "The decision table '{decision}' ({rules} rules, hit policy {policy}) "
        "is evaluated with a PARTIAL input assignment: the variables "
        "{given_vars} are known, the variable '{missing_var}' is unknown. "
        "Classify the table's rules: (a) rules that match under EVERY "
        "completion of '{missing_var}', (b) rules that match under SOME "
        "completion. Submit {{\"necessarily_matching\": [...], "
        "\"possibly_matching\": [...]}} with rule ids."
    ),
    "l03-f15-chain-root-synthesis": (
        "Only the decision chain below (terminal decision '{decision}') and a "
        "probe budget are visible. Construct a root input assignment for the "
        "input variables that makes the terminal decision return the target "
        "output. Submit {{\"input\": {{variable: value}}}}."
    ),
    "l03-f16-ordered-policy-output": (
        "Compute the exact ordered output list of the decision table "
        "'{decision}' ({rules} rules, hit policy {policy}) for the given input "
        "case, which matches multiple rules. Submit {{\"output\": <value>}}."
    ),
}


def _goal_rng(seed: int, family_id: str, *key: Any) -> random.Random:
    return random.Random(int(sha256_json([seed, family_id, *key])[:16], 16))


def _adaptive_counts(instances_target: int, caps: list[int]) -> list[int]:
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


def legacy_used_units() -> set[str]:
    used: set[str] = set()
    for shard in sorted(LEGACY_STAGING.glob("qualified/shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                used.update(json.loads(line)["source_unit_ids"])
    return used


class L03TaskFamilies20K:
    """TaskFamilyPlugin for the L03 20K expansion (families f09..f16)."""

    plugin_id = "l03.families-20k"

    def __init__(self) -> None:
        self._index: dict[str, Any] | None = None
        self._rows_cache: dict[tuple, list[dict[str, Any]]] = {}

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def load_index(self) -> dict[str, Any]:
        if self._index is not None:
            return self._index
        with (CAM_INDEX_DIR / "tables.jsonl").open(encoding="utf-8") as fh:
            cam_tables = [json.loads(line) for line in fh if line.strip()]
        with (CAM_INDEX_DIR / "chains.jsonl").open(encoding="utf-8") as fh:
            cam_chains = [json.loads(line) for line in fh if line.strip()]
        with (TCK_INDEX_DIR / "tables.jsonl").open(encoding="utf-8") as fh:
            tck_tables = [json.loads(line) for line in fh if line.strip()]
        with (TCK_INDEX_DIR / "tck_tests.jsonl").open(encoding="utf-8") as fh:
            tck_tests = [json.loads(line) for line in fh if line.strip()]
        used = legacy_used_units()
        self._index = {
            "cam_tables": cam_tables,
            "cam_chains": cam_chains,
            "tck_tables": tck_tables,
            "tck_tests": tck_tests,
            "legacy_used_units": used,
            "cam_unused_tables": [t for t in cam_tables if t["unit_id"] not in used],
        }
        return self._index

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
        for row in self.load_index()["cam_tables"]:
            out.setdefault(row["unit_id"], {})[row["decision_id"]] = row
        return out

    def _table_pool(self) -> list[dict[str, Any]]:
        return self.load_index()["cam_unused_tables"] + self.load_index()["tck_tables"]

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
        if kind == "tck_output":
            yield from self._propose_tck_output(family, seed, goals_target,
                                                instances_target)
        elif kind == "suite_audit":
            yield from self._propose_suite_audit(family, seed, goals_target,
                                                 instances_target)
        elif kind == "output_repair":
            yield from self._propose_output_repair(family, seed, goals_target,
                                                   instances_target)
        elif kind == "aggregation_repair":
            yield from self._propose_aggregation_repair(family, seed, goals_target,
                                                        instances_target)
        elif kind == "rule_deletion_repair":
            yield from self._propose_rule_deletion(family, seed, goals_target,
                                                   instances_target)
        elif kind == "partial_input_audit":
            yield from self._propose_partial_input_audit(family, seed, goals_target,
                                                         instances_target)
        elif kind == "chain_synthesis":
            yield from self._propose_chain_synthesis(family, seed, goals_target,
                                                     instances_target)
        elif kind == "ordered_output":
            yield from self._propose_ordered_output(family, seed, goals_target,
                                                    instances_target)
        else:  # pragma: no cover - families_20k is exhaustive
            raise ValueError("unknown goal_kind %r" % kind)

    def _select(self, seed: int, capacities: list[tuple[int, Any]], goals_target: int
                ) -> list[tuple[int, Any]]:
        rng = random.Random(seed)
        shuffled = list(capacities)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda p: -p[0])
        return shuffled[:goals_target]

    def _propose_tck_output(self, family_id: str, seed: int, goals_target: int,
                            instances_target: int) -> Iterator[CandidateTaskAST]:
        index = self.load_index()
        row_by_path = {(r["path"], r["decision_id"]): r for r in index["tck_tables"]}
        scenarios = [t for t in index["tck_tests"] if t["conformant"]]
        goals: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for s in scenarios:
            row = row_by_path.get((s["model_path"], s["decision_id"]))
            if row is None:
                continue
            goals.setdefault((row["unit_id"], row["decision_id"]), []).append(s)
        capacities = [(len(v), k) for k, v in goals.items()]
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _k in selected])
        for (cap, key), n in zip(selected, counts):
            unit_id, decision_id = key
            row = next(r for r in index["tck_tables"]
                       if r["unit_id"] == unit_id and r["decision_id"] == decision_id)
            goal_rng = _goal_rng(seed, family_id, unit_id, decision_id)
            for scenario in goals[key][:n]:
                for yielded in builders_20k.build_tck_output(row, scenario, 1, goal_rng):
                    yield self._wrap(family_id, row, yielded)

    def _propose_suite_audit(self, family_id: str, seed: int, goals_target: int,
                             instances_target: int) -> Iterator[CandidateTaskAST]:
        index = self.load_index()
        tck_by_decision: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for s in index["tck_tests"]:
            if not s["conformant"]:
                continue
            tck_by_decision.setdefault((s["model_path"], s["decision_id"]), []).append(s)
        capacities: list[tuple[int, dict[str, Any], list[dict[str, Any]] | None]] = []
        for row in index["tck_tables"]:
            scenarios = tck_by_decision.get((row["path"], row["decision_id"]))
            if scenarios and len(scenarios) >= builders_20k.SUITE_MIN:
                capacities.append((len(scenarios) * 3, row, scenarios))
        for row in self._table_pool():
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            hits = [r for r in rows if r["matched_rule_ids"] and r["output"] is not None]
            if len(hits) >= 4 and len({repr(r["output"]) for r in hits}) >= 2:
                capacities.append((min(len(hits), 24), row, None))
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r, _s in selected])
        for (cap, row, scenarios), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_20k.build_suite_audit(row, n, goal_rng, scenarios):
                yield self._wrap(family_id, row, yielded)

    def _propose_output_repair(self, family_id: str, seed: int, goals_target: int,
                               instances_target: int) -> Iterator[CandidateTaskAST]:
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self._table_pool():
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            if len(rows) < builders_20k.SUITE_MIN or len(row["rules"]) < 2:
                continue
            outs = {repr(r["output"]) for r in rows if r["matched_rule_ids"]}
            if len(outs) >= 2:
                capacities.append((len(rows), row))
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_20k.build_output_repair(row, n, goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _propose_aggregation_repair(self, family_id: str, seed: int,
                                    goals_target: int,
                                    instances_target: int) -> Iterator[CandidateTaskAST]:
        goals: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for row in self._table_pool():
            if row["hit_policy"] != "COLLECT":
                continue
            payload = _payload(row)
            rows = self._universe_rows(row, {"cap": 2048, "column_cap": 64})
            if len(rows) < builders_20k.SUITE_MIN:
                continue
            golden_out = [r["output"] for r in rows]
            for broken, mutation in builders_20k.aggregation_variants(payload):
                broken_out = builders_20k._behavior(broken, rows)
                if broken_out is None:
                    continue
                failing = [i for i, (g, b) in enumerate(zip(golden_out, broken_out))
                           if not builders_20k._values_eq(g, b)]
                if failing:
                    goals.append((len(failing), row, broken, mutation))
        selected = self._select(seed, goals, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r, _b, _m in selected])
        for (cap, row, broken, mutation), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"],
                                 row["decision_id"], mutation["to"])
            for yielded in builders_20k.build_aggregation_repair(
                    row, broken, mutation, n, goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _propose_rule_deletion(self, family_id: str, seed: int, goals_target: int,
                               instances_target: int) -> Iterator[CandidateTaskAST]:
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self._table_pool():
            if len(row["rules"]) < 3:
                continue
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            if len(rows) >= builders_20k.SUITE_MIN:
                capacities.append((min(len(rows), len(row["rules"]) * 2), row))
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_20k.build_rule_deletion_repair(row, n, goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _propose_partial_input_audit(self, family_id: str, seed: int,
                                     goals_target: int,
                                     instances_target: int) -> Iterator[CandidateTaskAST]:
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self._table_pool():
            if len(row["inputs"]) < 2:
                continue
            rule_ids = [r["id"] for r in row["rules"]]
            if any(not rid for rid in rule_ids) or len(set(rule_ids)) != len(rule_ids):
                continue
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            hits = [r for r in rows if r["matched_rule_ids"] and r["output"] is not None]
            if len(hits) >= 3:
                capacities.append((min(len(hits), 24), row))
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_20k.build_partial_input_audit(row, n, goal_rng):
                yield self._wrap(family_id, row, yielded)

    def _propose_chain_synthesis(self, family_id: str, seed: int, goals_target: int,
                                 instances_target: int) -> Iterator[CandidateTaskAST]:
        goals = self._synthesis_goals()
        selected = self._select(seed, goals, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _c, _t in selected])
        rows_by_unit = self._rows_by_dec()
        for (cap, chain, terminal), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, chain["unit_id"], terminal)
            for yielded in builders_20k.build_chain_synthesis(
                    chain, rows_by_unit[chain["unit_id"]], terminal, n, goal_rng):
                yield self._wrap(family_id, chain, yielded)

    def _synthesis_goals(self) -> list[tuple[int, dict[str, Any], str]]:
        from benchmark.plugins.task_family.l03.builders_chain import chain_material
        from benchmark.plugins.verifier.l03.dmn_eval import evaluate_chain
        from benchmark.plugins.verifier.l03.plugin import chain_root_universe

        goals: list[tuple[int, dict[str, Any], str]] = []
        rows_by_unit = self._rows_by_dec()
        for chain in self.load_index()["cam_chains"]:
            rows_by_dec = rows_by_unit.get(chain["unit_id"], {})
            material = chain_material(chain, rows_by_dec)
            if material is None:
                continue
            tables = {d: table_from_payload(p) for d, p in material["tables"].items()}
            for terminal in chain["terminal_decision_ids"]:
                material["terminal_decision_id"] = terminal
                root_cases = chain_root_universe(tables, material["edges"],
                                                 root_cap=builders_20k.ROOT_CAP)
                outputs: list[Any] = []
                for case in root_cases:
                    try:
                        res, _memo = evaluate_chain(
                            tables, material["edges"], terminal, case,
                            decision_names=material["decision_names"])
                        outputs.append(res.value)
                    except Exception:
                        outputs.append(None)
                ok_idx = [i for i, o in enumerate(outputs) if o is not None]
                if len(ok_idx) < builders_20k.SUITE_MIN:
                    continue
                by_out: dict[str, list[int]] = {}
                for i in ok_idx:
                    by_out.setdefault(repr(outputs[i]), []).append(i)
                targets = [k for k, v in sorted(by_out.items())
                           if 0 < len(v) <= max(1, len(ok_idx) // 2)]
                if targets:
                    goals.append((len(targets) * 2, chain, terminal))
        return goals

    def _propose_ordered_output(self, family_id: str, seed: int, goals_target: int,
                                instances_target: int) -> Iterator[CandidateTaskAST]:
        capacities: list[tuple[int, dict[str, Any]]] = []
        for row in self._table_pool():
            if row["hit_policy"] not in ("RULE ORDER", "OUTPUT ORDER"):
                continue
            # Honest exclusion: engine ordering disagrees with the TCK reference on these.
            if row["unit_id"].startswith("dmn-tck/") and row["hit_policy"] == "OUTPUT ORDER":
                continue
            rows = self._universe_rows(row, UNIVERSE_DEFAULT)
            multi = [r for r in rows
                     if len(r["matched_rule_ids"]) >= 2 and r["output"] is not None]
            if len(multi) >= 2 and len({repr(r["output"]) for r in multi}) >= 2:
                capacities.append((len(multi), row))
        selected = self._select(seed, capacities, goals_target)
        counts = _adaptive_counts(instances_target, [cap for cap, _r in selected])
        for (cap, row), n in zip(selected, counts):
            goal_rng = _goal_rng(seed, family_id, row["unit_id"], row["decision_id"])
            for yielded in builders_20k.build_ordered_output(row, n, goal_rng):
                yield self._wrap(family_id, row, yielded)

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
                "chain_intermediate", "chain_synthesis") else None,
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


def _coordinate(kind: str, yielded: dict[str, Any]) -> dict[str, Any]:
    if kind in ("tck_output", "ordered_output", "suite_audit", "output_repair",
                "partial_input_audit", "rule_deletion_repair"):
        return {"decision_id": yielded["decision_id"]}
    if kind == "aggregation_repair":
        return {"decision_id": yielded["decision_id"],
                "broken_aggregation": yielded["mutation"]["to"]}
    if kind == "chain_synthesis":
        return {"terminal_decision_id": yielded["decision_id"]}
    return {}


def _instance_material(kind: str, yielded: dict[str, Any]) -> Any:
    if kind in ("tck_output", "ordered_output"):
        return yielded["input_case"]
    if kind == "partial_input_audit":
        return [yielded["partial_input"], yielded["missing_var"]]
    if kind == "suite_audit":
        return yielded["suite"]
    if kind == "output_repair":
        return yielded["mutation"]
    if kind == "aggregation_repair":
        return yielded["suite"]
    if kind == "rule_deletion_repair":
        return yielded["mutation"]
    if kind == "chain_synthesis":
        return [yielded["target_output"], yielded["reference_input"]]
    return None


def _reference(kind: str, yielded: dict[str, Any]) -> Any:
    if kind in ("tck_output", "ordered_output"):
        return yielded["reference_output"]
    if kind == "partial_input_audit":
        return yielded["reference"]
    if kind == "suite_audit":
        return yielded["tamper"]["case_id"]
    if kind in ("output_repair", "aggregation_repair", "rule_deletion_repair"):
        return yielded["table"]
    if kind == "chain_synthesis":
        return yielded["reference_input"]
    return None


def observation_refs(ast: dict[str, Any]) -> list[str]:
    if ast["goal_kind"] in ("tck_output", "suite_audit", "ordered_output",
                            "partial_input_audit"):
        return [ast["source_path"]]
    return []


def public_payload(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["goal_kind"]
    if kind in ("tck_output", "ordered_output"):
        return {"table": ast["table"], "input_case": ast["input_case"]}
    if kind == "suite_audit":
        return {"table": ast["table"], "suite": ast["suite"]}
    if kind in ("output_repair", "aggregation_repair", "rule_deletion_repair"):
        return {"broken_table": ast["broken_table"], "suite": ast["suite"]}
    if kind == "partial_input_audit":
        return {"table": ast["table"], "partial_input": ast["partial_input"],
                "missing_var": ast["missing_var"]}
    if kind == "chain_synthesis":
        return {"chain": public_chain(ast["chain"]),
                "target_output": ast["target_output"]}
    return {}


def public_chain(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_ids": list(chain["decision_ids"]),
        "terminal_decision_id": chain["terminal_decision_id"],
        "edges": {d: list(deps) for d, deps in chain["edges"].items()},
        "tables": chain["tables"],
        "decision_names": dict(chain["decision_names"]),
        "table_digests": dict(chain["table_digests"]),
    }


def render_user_goal(ast: dict[str, Any]) -> str:
    text = _GOAL_TEXT[ast["family_id"]]
    table = ast.get("table") or {}
    return text.format(
        decision=ast.get("decision_id") or "?",
        rules=len(table.get("rules") or []),
        policy=table.get("hit_policy") or "?",
        decisions=len((ast.get("chain") or {}).get("decision_ids") or []),
        given_vars=", ".join(sorted(ast.get("partial_input") or {})) or "?",
        missing_var=ast.get("missing_var") or "?",
    )
