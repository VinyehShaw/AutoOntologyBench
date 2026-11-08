"""L03 native verifier: deterministic DMN truth recomputed from pinned XML.

The verifier is the private oracle. Nothing stored in a task record is
trusted: for every verdict the golden decision table(s) are re-parsed from
the pinned Camunda XML (per-decision ``table_digest`` re-checked), the case
universe is re-derived deterministically, and the family-specific truth is
recomputed with the ``aob-dmn-static`` engine. Repair submissions are judged
by full-universe behavioral equality with the XML-derived golden table, not
by string similarity to any stored answer. No LLM output is gold; no Java
runtime is assumed or required.

Plugin surface (governing doc §3.2): ``prove_exists`` / ``verify_terminal`` /
``certify_semantic_relevance``. ``PLUGIN`` is the class itself — the
constructor takes the private oracle directory, so the factory binds lane
args at dispatch time (S10 pattern for constructor-arg plugins).
"""
from __future__ import annotations

import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l03 import cases as case_mod
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_chain, evaluate_table
from benchmark.plugins.verifier.l03.dmn_model import (
    ENGINE_ID,
    ENGINE_VERSION,
    DecisionTable,
    DmnUnsupportedError,
    parse_definitions,
    sha256_json,
    table_digest,
    table_from_payload,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from source_registry.fetchers import l03_camunda_dmn as fetcher

ENGINE = "%s/%s" % (ENGINE_ID, ENGINE_VERSION)
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_SNAPSHOT_IDS = [
    "camunda-dmn-examples@%s" % c["release_id"] for c in fetcher.CONSTITUENTS
]
_COMMIT_BY_RELEASE = {c["release_id"]: c["commit"] for c in fetcher.CONSTITUENTS}
ROOT_CAP = 64


class VerifierError(DmnUnsupportedError):
    """Recomputation failed (drift / profile / inconsistency)."""


def chain_root_universe(tables: dict[str, DecisionTable],
                        edges: dict[str, list[str]],
                        *, root_cap: int = ROOT_CAP) -> list[dict[str, Any]]:
    """Deterministic root-input case list for a DRG chain.

    Root variables are every input expression of every decision that is not
    bound by some decision's output name; domains come from the first
    consuming table's column domain. The bounded cross product (capped) is
    the case universe generation and verification share.
    """
    produced: set[str] = set()
    for dec_id in sorted(tables):
        for out in tables[dec_id].outputs:
            produced.add(out.name)
    domains: dict[str, list[Any]] = {}
    for dec_id in sorted(tables):
        for i, clause in enumerate(tables[dec_id].inputs):
            if clause.expression in produced or clause.expression in domains:
                continue
            domains[clause.expression] = case_mod.column_domain(tables[dec_id], i)
    if not domains:
        return []
    root_vars = list(domains)
    combos = itertools.islice(
        itertools.product(*[domains[v] for v in root_vars]), root_cap)
    return [dict(zip(root_vars, combo)) for combo in combos]


def values_equal(a: Any, b: Any) -> bool:
    """Type-strict equality (bool is not int); recursive over lists/dicts."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(values_equal(a[k], b[k]) for k in a)
    return a == b


class L03NativeVerifier:
    """NativeVerifierPlugin for lane L03 (static Camunda DMN)."""

    plugin_id = "l03.dmn-static-verifier"

    def __init__(self, oracle_dir: str | Path) -> None:
        self.oracle_dir = Path(oracle_dir)
        self.oracle_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.oracle_dir, 0o700)

    # -- pinned-XML re-derivation ------------------------------------------------
    def _xml_path(self, ast: dict[str, Any]) -> Path:
        path = REPO_ROOT / ast["source_path"]
        if not path.is_file():
            raise VerifierError("pinned XML missing: %s" % ast["source_path"])
        return path

    def _golden_table(self, ast: dict[str, Any]) -> DecisionTable:
        defs = parse_definitions(self._xml_path(ast))
        dec = defs.decisions.get(ast["decision_id"])
        if dec is None or dec.table is None:
            raise VerifierError("decision %r absent or tableless in pinned XML"
                                % ast["decision_id"])
        if table_digest(dec.table) != ast["table_digest"]:
            raise VerifierError("table_digest drift vs pinned XML")
        return dec.table

    def _golden_chain(self, ast: dict[str, Any]) -> dict[str, DecisionTable]:
        chain = ast["chain"]
        defs = parse_definitions(self._xml_path(ast))
        out: dict[str, DecisionTable] = {}
        for dec_id in chain["decision_ids"]:
            dec = defs.decisions.get(dec_id)
            if dec is None or dec.table is None:
                raise VerifierError("chain decision %r absent in pinned XML" % dec_id)
            if table_digest(dec.table) != chain["table_digests"][dec_id]:
                raise VerifierError("table_digest drift for %r vs pinned XML" % dec_id)
            out[dec_id] = dec.table
        return out

    def _universe_rows(self, table: DecisionTable,
                       params: dict[str, int]) -> list[dict[str, Any]]:
        cases = case_mod.case_universe(table, **params)
        return case_mod.eval_universe(table, cases)

    def _table_behavior(self, payload: dict[str, Any],
                        rows: list[dict[str, Any]]) -> list[Any] | None:
        try:
            table = table_from_payload(payload)
        except Exception:
            return None
        out: list[Any] = []
        for row in rows:
            try:
                res = evaluate_table(table, row["input"])
            except DmnUnsupportedError:
                return None
            out.append(res.value)
        return out

    def _chain_terminal_outputs(self, tables: dict[str, DecisionTable],
                                edges: dict[str, list[str]], terminal: str,
                                root_cases: list[dict[str, Any]],
                                decision_names: dict[str, str] | None = None,
                                ) -> list[Any] | None:
        """Terminal output per root case; a case whose evaluation raises
        (missing binding, unsupported policy on that path) contributes None -
        that is exactly the broken-binding differential, not an abort."""
        out: list[Any] = []
        for case in root_cases:
            try:
                res, _memo = evaluate_chain(tables, edges, terminal, case,
                                            decision_names=decision_names)
                out.append(res.value)
            except DmnUnsupportedError:
                out.append(None)
        return out

    # -- family recomputation ------------------------------------------------------
    def _recompute(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["goal_kind"]
        out: dict[str, Any] = {"goal_kind": kind}
        if kind in ("exact_output", "rule_audit", "collect_result"):
            table = self._golden_table(ast)
            res = evaluate_table(table, ast["input_case"])
            out["value"] = res.value
            out["matched_rule_ids"] = list(res.matched_rule_ids)
            out["rows"] = self._universe_rows(table, ast["universe"])
        elif kind == "input_synthesis":
            table = self._golden_table(ast)
            out["target_value"] = evaluate_table(table, ast["reference"]).value
            out["rows"] = self._universe_rows(table, ast["universe"])
        elif kind in ("rule_repair", "hit_policy_repair"):
            table = self._golden_table(ast)
            rows = self._universe_rows(table, ast["universe"])
            out["rows"] = rows
            out["golden_behavior"] = [r["output"] for r in rows]
            out["broken_behavior"] = self._table_behavior(ast["broken_table"], rows)
            out["golden_payload_digest"] = table_digest(
                table_from_payload(ast["table"]))
        elif kind == "chain_eval":
            tables = self._golden_chain(ast)
            chain = ast["chain"]
            res, memo = evaluate_chain(
                tables, chain["edges"], chain["terminal_decision_id"],
                ast["input_case"], decision_names=chain["decision_names"])
            out["value"] = res.value
            out["chain_depth"] = len(chain["decision_ids"])
            out["root_cases"] = chain_root_universe(tables, chain["edges"])
            out["root_outputs"] = self._chain_terminal_outputs(
                tables, chain["edges"], chain["terminal_decision_id"],
                out["root_cases"], chain["decision_names"])
        elif kind == "binding_repair":
            tables = self._golden_chain(ast)
            chain = ast["chain"]
            root_cases = chain_root_universe(tables, chain["edges"])
            out["root_cases"] = root_cases
            out["golden_root_outputs"] = self._chain_terminal_outputs(
                tables, chain["edges"], chain["terminal_decision_id"],
                root_cases, chain["decision_names"])
            broken_tables = {
                dec: table_from_payload(payload)
                for dec, payload in ast["broken_chain"].items()
            }
            out["broken_root_outputs"] = self._chain_terminal_outputs(
                broken_tables, chain["edges"], chain["terminal_decision_id"],
                root_cases, chain["decision_names"])
        else:
            raise VerifierError("unknown goal_kind %r" % kind)
        return out

    # -- prove_exists ----------------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        ev: dict[str, Any] = {
            "engine": ENGINE,
            "source_commits": sorted(_COMMIT_BY_RELEASE.values()),
            "goal_kind": ast["goal_kind"],
            "unit_id": ast["unit_id"],
            "decision_id": ast.get("decision_id"),
        }
        try:
            re = self._recompute(ast)
        except VerifierError as exc:
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:%s" % str(exc)[:200])
        except DmnUnsupportedError as exc:
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:%s" % str(exc)[:200])
        gate = _GATES[ast["goal_kind"]](ast, re, ev)
        if gate is not None:
            return ProofOrCounterexample(False, ev, gate)
        return ProofOrCounterexample(True, ev)

    # -- oracle store ------------------------------------------------------------------
    def write_oracle(self, grounded_task: GroundedTask,
                     evidence: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "lane_id": "L03",
            "family_id": grounded_task.ast.family_id,
            "goal_kind": ast["goal_kind"],
            "engine": ENGINE,
            "source_commits": sorted(_COMMIT_BY_RELEASE.values()),
            "unit_id": ast["unit_id"],
            "decision_id": ast.get("decision_id"),
            "table_digest": ast.get("table_digest"),
            "source_path": ast["source_path"],
            "coordinate": ast["coordinate"],
            "universe": ast["universe"],
            "table": ast.get("table"),
            "broken_table": ast.get("broken_table"),
            "chain": ast.get("chain"),
            "broken_chain": ast.get("broken_chain"),
            "mutation": ast.get("mutation"),
            "suite": ast.get("suite"),
            "input_case": ast.get("input_case"),
            "target_output": ast.get("target_output"),
            "reference": ast["reference"],
            "prove_evidence": evidence,
        }
        payload["oracle_sha256"] = _sha256_json(payload)
        path = self.oracle_dir / ("%s.json" % grounded_task.oracle_ref)
        fd, tmp = tempfile.mkstemp(dir=str(self.oracle_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path

    def load_oracle(self, oracle_ref: str) -> dict[str, Any]:
        return json.loads(
            (self.oracle_dir / ("%s.json" % oracle_ref)).read_text(encoding="utf-8"))

    # -- verify_terminal -----------------------------------------------------------------
    def verify_terminal(self, task: GroundedTask,
                        submitted_state_or_answer: Any) -> NativeVerdict:
        oracle = self.load_oracle(task.oracle_ref)
        ast = _ast_from_oracle(oracle)
        submitted = submitted_state_or_answer or {}
        checks: list[dict[str, Any]] = []
        try:
            re = self._recompute(ast)
        except (VerifierError, DmnUnsupportedError) as exc:
            return NativeVerdict(False, {"error": str(exc)[:200], "engine": ENGINE})
        _TERMINAL[ast["goal_kind"]](ast, re, submitted, checks)
        valid = all(c["pass"] for c in checks)
        return NativeVerdict(valid, {"checks": checks, "engine": ENGINE})

    # -- certify_semantic_relevance ----------------------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        oracle = self.load_oracle(task.oracle_ref)
        ast = _ast_from_oracle(oracle)
        try:
            re = self._recompute(ast)
            payload = _WITNESS[ast["goal_kind"]](ast, re)
        except (VerifierError, DmnUnsupportedError) as exc:
            return SemanticWitness(False, {"reason": str(exc)[:200]})
        return SemanticWitness(bool(payload.pop("ok")), payload)


def _sha256_json(payload: object) -> str:
    return sha256_json(payload)


def _ast_from_oracle(oracle: dict[str, Any]) -> dict[str, Any]:
    ast = {
        "goal_kind": oracle["goal_kind"],
        "unit_id": oracle["unit_id"],
        "decision_id": oracle.get("decision_id"),
        "table_digest": oracle.get("table_digest"),
        "source_path": oracle["source_path"],
        "coordinate": oracle.get("coordinate"),
        "universe": oracle.get("universe") or {"cap": case_mod.MAX_UNIVERSE, "column_cap": 12},
        "table": oracle.get("table"),
        "broken_table": oracle.get("broken_table"),
        "chain": oracle.get("chain"),
        "broken_chain": oracle.get("broken_chain"),
        "mutation": oracle.get("mutation"),
        "suite": oracle.get("suite"),
        "input_case": oracle.get("input_case"),
        "target_output": oracle.get("target_output"),
        "reference": oracle.get("reference"),
    }
    return ast


# -- prove gates -------------------------------------------------------------------
def _gate_exact_output(ast, re, ev):
    if not values_equal(ast["reference"], re["value"]):
        return "UNSOLVABLE:reference_mismatch"
    if re["value"] is None or not re["matched_rule_ids"]:
        return "UNSOLVABLE:no_output"
    ev["value"] = re["value"]
    ev["matched_rule_ids"] = re["matched_rule_ids"]
    distinct = {repr(r["output"]) for r in re["rows"] if r["matched_rule_ids"]}
    if len(distinct) < 2:
        return "TRIVIAL:constant_output"
    return None


def _gate_rule_audit(ast, re, ev):
    if list(ast["reference"]) != list(re["matched_rule_ids"]):
        return "UNSOLVABLE:reference_mismatch"
    n_rules = len(ast["table"]["rules"])
    if not re["matched_rule_ids"] or len(re["matched_rule_ids"]) >= n_rules:
        return "TRIVIAL:degenerate_match_set"
    ev["matched_rule_ids"] = re["matched_rule_ids"]
    return None


def _gate_rule_repair(ast, re, ev):
    if re["broken_behavior"] is None:
        return "PROFILE_UNSUPPORTED:broken_table_unevaluable"
    if re["golden_payload_digest"] != ast["table_digest"]:
        return "PROFILE_UNSUPPORTED:golden_payload_drift"
    failing = [i for i, (g, b) in enumerate(zip(re["golden_behavior"], re["broken_behavior"]))
               if not values_equal(g, b)]
    if not failing:
        return "TRIVIAL:no_behavioral_difference"
    if not _single_cell_mutation(ast["table"], ast["broken_table"], ast["mutation"]):
        return "UNSOLVABLE:mutation_not_single_cell"
    if not _suite_consistent(ast["suite"], re["rows"]):
        return "UNSOLVABLE:suite_inconsistent"
    if not _suite_shows_failure(ast["suite"], ast["broken_table"], re["rows"]):
        return "UNSOLVABLE:suite_not_showing_failure"
    ev["failing_cases"] = len(failing)
    ev["universe_cases"] = len(re["rows"])
    return None


def _gate_hit_policy_repair(ast, re, ev):
    if re["broken_behavior"] is None:
        return "PROFILE_UNSUPPORTED:broken_table_unevaluable"
    if re["golden_payload_digest"] != ast["table_digest"]:
        return "PROFILE_UNSUPPORTED:golden_payload_drift"
    if not _policy_only_mutation(ast["table"], ast["broken_table"]):
        return "UNSOLVABLE:mutation_not_policy_only"
    failing = [i for i, (g, b) in enumerate(zip(re["golden_behavior"], re["broken_behavior"]))
               if not values_equal(g, b)]
    if not failing:
        return "TRIVIAL:no_behavioral_difference"
    if not _suite_consistent(ast["suite"], re["rows"]):
        return "UNSOLVABLE:suite_inconsistent"
    if not _suite_shows_failure(ast["suite"], ast["broken_table"], re["rows"]):
        return "UNSOLVABLE:suite_not_showing_failure"
    ev["failing_cases"] = len(failing)
    ev["universe_cases"] = len(re["rows"])
    return None


def _gate_chain_eval(ast, re, ev):
    if not values_equal(ast["reference"], re["value"]):
        return "UNSOLVABLE:reference_mismatch"
    if re["value"] is None:
        return "UNSOLVABLE:no_output"
    if re["chain_depth"] < 2:
        return "TRIVIAL:single_decision"
    ev["value"] = re["value"]
    ev["chain_depth"] = re["chain_depth"]
    return None


def _gate_binding_repair(ast, re, ev):
    if re["broken_root_outputs"] is None or re["golden_root_outputs"] is None:
        return "PROFILE_UNSUPPORTED:chain_unevaluable"
    failing = [i for i, (g, b) in enumerate(zip(re["golden_root_outputs"], re["broken_root_outputs"]))
               if not values_equal(g, b)]
    if not failing:
        return "TRIVIAL:no_behavioral_difference"
    if not _binding_only_mutation(ast["chain"]["tables"], ast["broken_chain"], ast["mutation"]):
        return "UNSOLVABLE:mutation_not_binding_only"
    if not _chain_suite_consistent(ast, re):
        return "UNSOLVABLE:suite_inconsistent"
    if not _chain_suite_shows_failure(ast, re):
        return "UNSOLVABLE:suite_not_showing_failure"
    ev["failing_cases"] = len(failing)
    ev["root_cases"] = len(re["root_cases"])
    return None


def _gate_collect_result(ast, re, ev):
    if not values_equal(ast["reference"], re["value"]):
        return "UNSOLVABLE:reference_mismatch"
    if re["value"] is None:
        return "UNSOLVABLE:no_output"
    if len(re["matched_rule_ids"]) < 2:
        return "TRIVIAL:single_match_collect"
    ev["value"] = re["value"]
    ev["matched_rule_ids"] = re["matched_rule_ids"]
    return None


def _gate_input_synthesis(ast, re, ev):
    if not values_equal(re["target_value"], ast["target_output"]):
        return "UNSOLVABLE:reference_fails"
    hits = [r for r in re["rows"] if r["matched_rule_ids"]]
    producing = [r for r in hits if values_equal(r["output"], ast["target_output"])]
    if not producing:
        return "UNSOLVABLE:target_unreachable"
    if len(producing) > max(1, len(hits) // 2):
        return "TRIVIAL:target_too_common"
    ev["target_fraction"] = round(len(producing) / max(len(hits), 1), 3)
    ev["universe_cases"] = len(re["rows"])
    return None


_GATES = {
    "exact_output": _gate_exact_output,
    "rule_audit": _gate_rule_audit,
    "rule_repair": _gate_rule_repair,
    "hit_policy_repair": _gate_hit_policy_repair,
    "chain_eval": _gate_chain_eval,
    "binding_repair": _gate_binding_repair,
    "collect_result": _gate_collect_result,
    "input_synthesis": _gate_input_synthesis,
}


# -- mutation shape checks ------------------------------------------------------------
def _single_cell_mutation(golden: dict[str, Any], broken: dict[str, Any],
                          mutation: dict[str, Any]) -> bool:
    if golden["hit_policy"] != broken["hit_policy"]:
        return False
    if golden.get("aggregation") != broken.get("aggregation"):
        return False
    if golden["inputs"] != broken["inputs"] or golden["outputs"] != broken["outputs"]:
        return False
    if len(golden["rules"]) != len(broken["rules"]):
        return False
    ri = mutation["rule_index"]
    for i, (g, b) in enumerate(zip(golden["rules"], broken["rules"])):
        if g["id"] != b["id"] or g["output_entries"] != b["output_entries"]:
            return False
        if i == ri:
            if len(g["input_entries"]) != len(b["input_entries"]):
                return False
            for ci, (ge, be) in enumerate(zip(g["input_entries"], b["input_entries"])):
                if ci != mutation["column"] and ge != be:
                    return False
            if g["input_entries"][mutation["column"]] != mutation["from"]:
                return False
            if b["input_entries"][mutation["column"]] != mutation["to"]:
                return False
        elif g["input_entries"] != b["input_entries"]:
            return False
    return True


def _policy_only_mutation(golden: dict[str, Any], broken: dict[str, Any]) -> bool:
    if golden["inputs"] != broken["inputs"] or golden["outputs"] != broken["outputs"]:
        return False
    if golden["rules"] != broken["rules"]:
        return False
    if golden["hit_policy"] == broken["hit_policy"] and \
            golden.get("aggregation") == broken.get("aggregation"):
        return False
    return True


def _binding_only_mutation(golden_tables: dict[str, Any],
                           broken_tables: dict[str, Any],
                           mutation: dict[str, Any]) -> bool:
    if set(golden_tables) != set(broken_tables):
        return False
    dec_id = mutation["decision_id"]
    for d, g in golden_tables.items():
        b = broken_tables[d]
        if d == dec_id:
            if g["rules"] != b["rules"] or g["outputs"] != b["outputs"] \
                    or g["hit_policy"] != b["hit_policy"]:
                return False
            if len(g["inputs"]) != len(b["inputs"]):
                return False
            for gc, bc in zip(g["inputs"], b["inputs"]):
                if gc == bc:
                    continue
                if gc["expression"] != mutation["from"] or bc["expression"] != mutation["to"]:
                    return False
                if gc["type_ref"] != bc["type_ref"]:
                    return False
        elif g != b:
            return False
    return True


def _case_key(case: dict[str, Any]) -> str:
    """Canonical case signature: dict repr is insertion-order dependent, and
    the oracle JSON round-trip (sort_keys) reorders keys."""
    return json.dumps(case, sort_keys=True)


def _suite_consistent(suite: list[dict[str, Any]],
                      rows: list[dict[str, Any]]) -> bool:
    by_input = {_case_key(r["input"]): r["output"] for r in rows}
    for case in suite:
        key = _case_key(case["input"])
        if key not in by_input:
            return False
        if not values_equal(by_input[key], case["expected"]):
            return False
    return True


def _suite_shows_failure(suite: list[dict[str, Any]], broken_table: dict[str, Any],
                         rows: list[dict[str, Any]]) -> bool:
    try:
        broken = table_from_payload(broken_table)
    except Exception:
        return False
    for case in suite:
        try:
            res = evaluate_table(broken, case["input"])
        except DmnUnsupportedError:
            return False
        if not values_equal(res.value, case["expected"]):
            return True
    return False


def _chain_suite_consistent(ast: dict[str, Any], re: dict[str, Any]) -> bool:
    return _chain_suite_check(ast, re, golden=True)


def _chain_suite_check(ast: dict[str, Any], re: dict[str, Any], *, golden: bool) -> bool:
    outputs = re["golden_root_outputs"] if golden else re["broken_root_outputs"]
    if outputs is None:
        return False
    root_cases = re["root_cases"]
    by_input = {_case_key(c): o for c, o in zip(root_cases, outputs)}
    for case in ast["suite"]:
        key = _case_key(case["input"])
        if key not in by_input:
            return False
        if not values_equal(by_input[key], case["expected"]):
            return False
    return True


def _chain_suite_shows_failure(ast: dict[str, Any], re: dict[str, Any]) -> bool:
    return not _chain_suite_check(ast, re, golden=False)


# -- terminal checks ---------------------------------------------------------------------
def _term_value(ast, re, submitted, checks, key="output"):
    answer = submitted.get(key)
    checks.append({
        "check": "exact_value",
        "pass": answer is not None and values_equal(answer, re["value"]),
    })


def _term_rule_audit(ast, re, submitted, checks):
    answer = submitted.get("matched_rule_ids")
    ok = isinstance(answer, list) and [str(x) for x in answer] == list(re["matched_rule_ids"])
    checks.append({"check": "exact_matched_rule_set", "pass": bool(ok)})


def _term_table_repair(ast, re, submitted, checks):
    answer = submitted.get("table")
    if not isinstance(answer, dict):
        checks.append({"check": "repaired_table", "pass": False,
                       "error": "submission missing 'table' payload"})
        return
    behavior = None
    try:
        behavior = _verifier_table_behavior(answer, re["rows"])
    except Exception:
        behavior = None
    if behavior is None:
        checks.append({"check": "repaired_table", "pass": False,
                       "error": "submitted table not evaluable on the case universe"})
        return
    matches = all(values_equal(g, b) for g, b in zip(re["golden_behavior"], behavior)) \
        and len(behavior) == len(re["golden_behavior"])
    checks.append({"check": "full_universe_behavioral_equality", "pass": bool(matches)})
    differs = json.dumps(answer, sort_keys=True) != json.dumps(ast["broken_table"], sort_keys=True)
    checks.append({"check": "differs_from_broken", "pass": bool(differs)})


def _verifier_table_behavior(payload: dict[str, Any],
                             rows: list[dict[str, Any]]) -> list[Any]:
    table = table_from_payload(payload)
    out: list[Any] = []
    for row in rows:
        res = evaluate_table(table, row["input"])
        out.append(res.value)
    return out


def _term_binding_repair(ast, re, submitted, checks):
    answer = submitted.get("tables")
    if not isinstance(answer, dict) or set(answer) != set(ast["chain"]["tables"]):
        checks.append({"check": "repaired_chain", "pass": False,
                       "error": "submission missing 'tables' map for every chain decision"})
        return
    try:
        tables = {d: table_from_payload(p) for d, p in answer.items()}
    except Exception:
        checks.append({"check": "repaired_chain", "pass": False,
                       "error": "submitted chain tables not parseable"})
        return
    chain = ast["chain"]
    behavior = []
    for case in re["root_cases"]:
        try:
            res, _ = evaluate_chain(tables, chain["edges"],
                                    chain["terminal_decision_id"], case,
                                    decision_names=chain["decision_names"])
        except DmnUnsupportedError:
            checks.append({"check": "full_root_case_behavioral_equality", "pass": False,
                           "error": "submitted chain not evaluable"})
            return
        behavior.append(res.value)
    matches = all(values_equal(g, b) for g, b in zip(re["golden_root_outputs"], behavior))
    checks.append({"check": "full_root_case_behavioral_equality", "pass": bool(matches)})
    differs = json.dumps(answer, sort_keys=True) != json.dumps(ast["broken_chain"], sort_keys=True)
    checks.append({"check": "differs_from_broken", "pass": bool(differs)})


def _term_input_synthesis(ast, re, submitted, checks):
    answer = submitted.get("input")
    if not isinstance(answer, dict):
        checks.append({"check": "input_assignment", "pass": False,
                       "error": "submission missing 'input' assignment"})
        return
    try:
        table = table_from_payload(ast["table"])
        res = evaluate_table(table, answer)
    except DmnUnsupportedError as exc:
        checks.append({"check": "input_assignment", "pass": False,
                       "error": "evaluation failed: %s" % str(exc)[:120]})
        return
    checks.append({
        "check": "produces_target_output",
        "pass": values_equal(res.value, ast["target_output"]),
    })


_TERMINAL = {
    "exact_output": _term_value,
    "rule_audit": _term_rule_audit,
    "rule_repair": _term_table_repair,
    "hit_policy_repair": _term_table_repair,
    "chain_eval": _term_value,
    "binding_repair": _term_binding_repair,
    "collect_result": _term_value,
    "input_synthesis": _term_input_synthesis,
}


# -- semantic witnesses ---------------------------------------------------------------------
def _witness_output_discrimination(ast, re):
    others = [r for r in re["rows"]
              if r["matched_rule_ids"] and not values_equal(r["output"], re["value"])]
    return {"ok": bool(others), "mode": "output_discrimination",
            "distinct_other_outputs": len({repr(r["output"]) for r in others})}


def _witness_match_set(ast, re):
    others = [r for r in re["rows"]
              if r["matched_rule_ids"] and list(r["matched_rule_ids"]) != list(re["matched_rule_ids"])]
    return {"ok": bool(others), "mode": "match_set_discrimination",
            "distinct_other_match_sets": len(
                {repr(sorted(r["matched_rule_ids"])) for r in others})}


def _witness_mutation_differential(ast, re, behavior_key, golden_key):
    behavior = re[behavior_key]
    if behavior is None:
        return {"ok": False, "mode": "mutation_differential", "reason": "unevaluable"}
    golden = re[golden_key]
    failing = sum(1 for g, b in zip(golden, behavior) if not values_equal(g, b))
    return {"ok": failing > 0, "mode": "mutation_differential",
            "golden_pass_cases": len(behavior) - failing, "broken_fail_cases": failing}


def _witness_chain(ast, re):
    others = [o for o in (re.get("root_outputs") or [])
              if o is not None and not values_equal(o, re["value"])]
    return {"ok": bool(others) and re["chain_depth"] >= 2,
            "mode": "terminal_discrimination", "chain_depth": re["chain_depth"],
            "distinct_other_outputs": len({repr(o) for o in others})}


def _witness_collect(ast, re):
    table = table_from_payload(ast["table"])
    unique = DecisionTable(
        hit_policy="UNIQUE", aggregation="", inputs=table.inputs,
        outputs=table.outputs, rules=table.rules)
    res = evaluate_table(unique, ast["input_case"])
    return {"ok": bool(res.violations), "mode": "collect_necessity",
            "unique_policy_violations": list(res.violations),
            "matched_rules": len(re["matched_rule_ids"])}


def _witness_input_synthesis(ast, re):
    hits = [r for r in re["rows"] if r["matched_rule_ids"]]
    producing = [r for r in hits if values_equal(r["output"], ast["target_output"])]
    others = [r for r in hits if not values_equal(r["output"], ast["target_output"])]
    return {"ok": bool(others) and 0 < len(producing) <= max(1, len(hits) // 2),
            "mode": "target_discrimination",
            "producing_fraction": round(len(producing) / max(len(hits), 1), 3),
            "other_outputs": len({repr(r["output"]) for r in others})}


_WITNESS = {
    "exact_output": _witness_output_discrimination,
    "rule_audit": _witness_match_set,
    "rule_repair": lambda ast, re: _witness_mutation_differential(
        ast, re, "broken_behavior", "golden_behavior"),
    "hit_policy_repair": lambda ast, re: _witness_mutation_differential(
        ast, re, "broken_behavior", "golden_behavior"),
    "chain_eval": _witness_chain,
    "binding_repair": lambda ast, re: _witness_mutation_differential(
        ast, re, "broken_root_outputs", "golden_root_outputs"),
    "collect_result": _witness_collect,
    "input_synthesis": _witness_input_synthesis,
}


PLUGIN = L03NativeVerifier
