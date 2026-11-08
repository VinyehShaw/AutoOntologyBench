"""L03 20K-expansion native verifier: extends ``L03NativeVerifier`` (f09..f16).

Same discipline as the 8K verifier: nothing stored in a task record is
trusted. For every verdict the golden decision table(s) are re-parsed from
the pinned XML (per-decision ``table_digest`` re-checked), the case universe
is re-derived deterministically, and the family-specific truth is recomputed
with ``aob-dmn-static``. TCK families additionally re-parse the pinned TCK
test file and re-derive the TCK expectation, so the dual-oracle conformance
claim is itself recomputed from pinned sources.

New goal kinds (legacy kinds are delegated to the 8K verifier unchanged):
``tck_output``, ``suite_audit``, ``output_repair``, ``aggregation_repair``,
``rule_deletion_repair``, ``chain_intermediate``, ``chain_synthesis``,
``ordered_output``.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.plugins.source.l03.tck_index import _parse_expected
from benchmark.plugins.task_family.l03 import cases as case_mod
from benchmark.plugins.task_family.l03.builders_20k import classify_partial
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_chain, evaluate_table
from benchmark.plugins.verifier.l03.dmn_model import (
    DecisionTable,
    DmnUnsupportedError,
    table_digest,
    table_from_payload,
)
from benchmark.plugins.verifier.l03.plugin import (
    ENGINE,
    L03NativeVerifier,
    VerifierError,
    _case_key,
    _sha256_json,
    chain_root_universe,
    values_equal,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[4]


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname


def _children(el: etree._Element, name: str) -> list[etree._Element]:
    return [c for c in el if isinstance(c.tag, str) and _local(c) == name]


def load_tck_expectation(test_path: Path, decision_name: str,
                         test_case_id: str) -> Any:
    """Re-derive the TCK expected value for (decision, testCase) from XML."""
    tree = etree.parse(str(test_path))
    for tc in tree.iter():
        if not (isinstance(tc.tag, str) and _local(tc) == "testCase"):
            continue
        if (tc.get("id") or "") != test_case_id:
            continue
        for node in _children(tc, "resultNode"):
            if (node.get("name") or "") != decision_name:
                continue
            expected = _children(node, "expected")
            if len(expected) != 1:
                raise VerifierError("TCK expected block shape changed for %s#%s"
                                    % (decision_name, test_case_id))
            ok, value, reason = _parse_expected(expected[0])
            if not ok:
                raise VerifierError("TCK expected unparsable: %s" % reason)
            return value
    raise VerifierError("TCK testCase %r / resultNode %r not found in %s"
                        % (test_case_id, decision_name, test_path))


class L03NativeVerifier20K(L03NativeVerifier):
    """NativeVerifierPlugin for the L03 20K expansion (static Camunda DMN + TCK)."""

    plugin_id = "l03.dmn-static-verifier-20k"

    # -- recompute ---------------------------------------------------------------
    def _recompute(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["goal_kind"]
        out: dict[str, Any] = {"goal_kind": kind}
        if kind == "tck_output":
            table = self._golden_table(ast)
            res = evaluate_table(table, ast["input_case"])
            out["value"] = res.value
            out["matched_rule_ids"] = list(res.matched_rule_ids)
            out["rows"] = self._universe_rows(table, ast["universe"])
            out["tck_expected"] = load_tck_expectation(
                REPO_ROOT / ast["tck_test_path"], ast["decision_name"],
                ast["test_case_id"])
        elif kind == "suite_audit":
            table = self._golden_table(ast)
            out["rows"] = self._universe_rows(table, ast["universe"])
            out["suite_values"] = [
                evaluate_table(table, case["input"]).value for case in ast["suite"]]
        elif kind in ("output_repair", "aggregation_repair", "rule_deletion_repair"):
            table = self._golden_table(ast)
            rows = self._universe_rows(table, ast["universe"])
            out["rows"] = rows
            out["golden_behavior"] = [r["output"] for r in rows]
            out["broken_behavior"] = self._table_behavior(ast["broken_table"], rows)
            out["golden_payload_digest"] = table_digest(
                table_from_payload(ast["table"]))
        elif kind == "partial_input_audit":
            table = self._golden_table(ast)
            nec, poss = classify_partial(table, ast["partial_input"])
            out["necessarily_matching"] = list(nec)
            out["possibly_matching"] = list(poss)
            out["rule_ids"] = [r.id for r in table.rules]
            dom = [case_mod.column_domain(table, i)
                   for i, c in enumerate(table.inputs)
                   if c.expression == ast["missing_var"]]
            completions = []
            if dom:
                for v in dom[0]:
                    full = dict(ast["partial_input"])
                    full[ast["missing_var"]] = v
                    try:
                        res = evaluate_table(table, full)
                        completions.append(res.value)
                    except DmnUnsupportedError:
                        continue
            out["completion_outputs"] = completions
        elif kind == "chain_synthesis":
            tables = self._golden_chain(ast)
            chain = ast["chain"]
            root_cases = chain_root_universe(tables, chain["edges"])
            out["root_cases"] = root_cases
            out["root_outputs"] = self._chain_terminal_outputs(
                tables, chain["edges"], chain["terminal_decision_id"],
                root_cases, chain["decision_names"])
            res, _memo = evaluate_chain(
                tables, chain["edges"], chain["terminal_decision_id"],
                ast["reference"], decision_names=chain["decision_names"])
            out["reference_value"] = res.value
        elif kind == "ordered_output":
            table = self._golden_table(ast)
            res = evaluate_table(table, ast["input_case"])
            out["value"] = res.value
            out["matched_rule_ids"] = list(res.matched_rule_ids)
            out["rows"] = self._universe_rows(table, ast["universe"])
        else:
            return super()._recompute(ast)
        return out

    # -- prove_exists ----------------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        ev: dict[str, Any] = {
            "engine": ENGINE,
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
        gate = _GATES_20K[ast["goal_kind"]](ast, re, ev)
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
            "unit_id": ast["unit_id"],
            "decision_id": ast.get("decision_id"),
            "decision_name": ast.get("decision_name"),
            "table_digest": ast.get("table_digest"),
            "source_path": ast["source_path"],
            "coordinate": ast.get("coordinate"),
            "universe": ast.get("universe"),
            "table": ast.get("table"),
            "broken_table": ast.get("broken_table"),
            "chain": ast.get("chain"),
            "mutation": ast.get("mutation"),
            "suite": ast.get("suite"),
            "input_case": ast.get("input_case"),
            "partial_input": ast.get("partial_input"),
            "missing_var": ast.get("missing_var"),
            "target_output": ast.get("target_output"),
            "reference": ast["reference"],
            "tck_test_path": ast.get("tck_test_path"),
            "test_case_id": ast.get("test_case_id"),
            "tck_expected": ast.get("tck_expected"),
            "tamper": ast.get("tamper"),
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

    # -- verify_terminal -----------------------------------------------------------------
    def verify_terminal(self, task: GroundedTask,
                        submitted_state_or_answer: Any) -> NativeVerdict:
        oracle = self.load_oracle(task.oracle_ref)
        ast = _ast_from_oracle_20k(oracle)
        submitted = submitted_state_or_answer or {}
        checks: list[dict[str, Any]] = []
        try:
            re = self._recompute(ast)
        except (VerifierError, DmnUnsupportedError) as exc:
            return NativeVerdict(False, {"error": str(exc)[:200], "engine": ENGINE})
        _TERMINAL_20K[ast["goal_kind"]](ast, re, submitted, checks)
        valid = all(c["pass"] for c in checks)
        return NativeVerdict(valid, {"checks": checks, "engine": ENGINE})

    # -- certify_semantic_relevance ----------------------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        oracle = self.load_oracle(task.oracle_ref)
        ast = _ast_from_oracle_20k(oracle)
        try:
            re = self._recompute(ast)
            payload = _WITNESS_20K[ast["goal_kind"]](ast, re)
        except (VerifierError, DmnUnsupportedError) as exc:
            return SemanticWitness(False, {"reason": str(exc)[:200]})
        return SemanticWitness(bool(payload.pop("ok")), payload)


def _ast_from_oracle_20k(oracle: dict[str, Any]) -> dict[str, Any]:
    ast = {
        "goal_kind": oracle["goal_kind"],
        "unit_id": oracle["unit_id"],
        "decision_id": oracle.get("decision_id"),
        "decision_name": oracle.get("decision_name"),
        "table_digest": oracle.get("table_digest"),
        "source_path": oracle["source_path"],
        "coordinate": oracle.get("coordinate"),
        "universe": oracle.get("universe") or {"cap": case_mod.MAX_UNIVERSE,
                                               "column_cap": 12},
        "table": oracle.get("table"),
        "broken_table": oracle.get("broken_table"),
        "chain": oracle.get("chain"),
        "mutation": oracle.get("mutation"),
        "suite": oracle.get("suite"),
        "input_case": oracle.get("input_case"),
        "partial_input": oracle.get("partial_input"),
        "missing_var": oracle.get("missing_var"),
        "target_output": oracle.get("target_output"),
        "reference": oracle.get("reference"),
        "tck_test_path": oracle.get("tck_test_path"),
        "test_case_id": oracle.get("test_case_id"),
        "tck_expected": oracle.get("tck_expected"),
        "tamper": oracle.get("tamper"),
    }
    return ast


# -- prove gates -------------------------------------------------------------------
def _gate_tck_output(ast, re, ev):
    if not values_equal(ast["reference"], re["value"]):
        return "UNSOLVABLE:reference_mismatch"
    if not values_equal(re["value"], re["tck_expected"]):
        return "PROFILE_UNSUPPORTED:tck_engine_disagreement"
    if re["value"] is None or not re["matched_rule_ids"]:
        return "UNSOLVABLE:no_output"
    ev["value"] = re["value"]
    ev["tck_expected"] = re["tck_expected"]
    ev["matched_rule_ids"] = re["matched_rule_ids"]
    distinct = {repr(r["output"]) for r in re["rows"] if r["matched_rule_ids"]}
    if len(distinct) < 2:
        return "TRIVIAL:constant_output"
    return None


def _gate_suite_audit(ast, re, ev):
    suite = ast["suite"]
    values = re["suite_values"]
    if len(suite) != len(values):
        return "UNSOLVABLE:suite_shape"
    inconsistent = [i for i, (case, v) in enumerate(zip(suite, values))
                    if not values_equal(case["expected"], v)]
    if len(inconsistent) != 1:
        return "UNSOLVABLE:suite_not_exactly_one_inconsistent"
    idx = inconsistent[0]
    if suite[idx]["case_id"] != ast["tamper"]["case_id"]:
        return "UNSOLVABLE:tamper_case_mismatch"
    true_values = [v for i, v in enumerate(values) if i != idx]
    if len({repr(v) for v in true_values}) < 2:
        return "TRIVIAL:suite_constant_true_values"
    shown = [case["expected"] for case in suite]
    dup = any(i != idx and values_equal(s, shown[idx]) for i, s in enumerate(shown))
    if not dup:
        return "TRIVIAL:tampered_value_unduplicated"
    ev["inconsistent_case_id"] = suite[idx]["case_id"]
    ev["suite_size"] = len(suite)
    ev["distinct_true_values"] = len({repr(v) for v in true_values})
    return None


def _gate_output_repair(ast, re, ev):
    if re["broken_behavior"] is None:
        return "PROFILE_UNSUPPORTED:broken_table_unevaluable"
    if re["golden_payload_digest"] != ast["table_digest"]:
        return "PROFILE_UNSUPPORTED:golden_payload_drift"
    failing = [i for i, (g, b) in enumerate(zip(re["golden_behavior"], re["broken_behavior"]))
               if not values_equal(g, b)]
    if not failing:
        return "TRIVIAL:no_behavioral_difference"
    if not _single_output_cell_mutation(ast["table"], ast["broken_table"], ast["mutation"]):
        return "UNSOLVABLE:mutation_not_single_output_cell"
    if not _suite_consistent(ast["suite"], re["rows"]):
        return "UNSOLVABLE:suite_inconsistent"
    if not _suite_shows_failure(ast["suite"], ast["broken_table"], re["rows"]):
        return "UNSOLVABLE:suite_not_showing_failure"
    ev["failing_cases"] = len(failing)
    ev["universe_cases"] = len(re["rows"])
    return None


def _gate_aggregation_repair(ast, re, ev):
    if re["broken_behavior"] is None:
        return "PROFILE_UNSUPPORTED:broken_table_unevaluable"
    if re["golden_payload_digest"] != ast["table_digest"]:
        return "PROFILE_UNSUPPORTED:golden_payload_drift"
    if not _aggregation_only_mutation(ast["table"], ast["broken_table"]):
        return "UNSOLVABLE:mutation_not_aggregation_only"
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


def _gate_rule_deletion_repair(ast, re, ev):
    if re["broken_behavior"] is None:
        return "PROFILE_UNSUPPORTED:broken_table_unevaluable"
    if re["golden_payload_digest"] != ast["table_digest"]:
        return "PROFILE_UNSUPPORTED:golden_payload_drift"
    if not _rule_deletion_only_mutation(ast["table"], ast["broken_table"], ast["mutation"]):
        return "UNSOLVABLE:mutation_not_rule_deletion"
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


def _gate_partial_input_audit(ast, re, ev):
    rule_ids = re["rule_ids"]
    ref = ast["reference"]
    nec_ids = [rule_ids[i] for i in re["necessarily_matching"]]
    poss_ids = [rule_ids[i] for i in re["possibly_matching"]]
    if list(ref["necessarily_matching"]) != nec_ids or \
            list(ref["possibly_matching"]) != poss_ids:
        return "UNSOLVABLE:reference_mismatch"
    amb = [i for i in re["possibly_matching"] if i not in re["necessarily_matching"]]
    if not re["possibly_matching"] or not amb:
        return "TRIVIAL:no_ambiguous_rules"
    if len({repr(o) for o in re["completion_outputs"]}) < 2:
        return "TRIVIAL:completion_outputs_constant"
    ev["necessarily_matching"] = nec_ids
    ev["possibly_matching"] = poss_ids
    ev["ambiguous_rules"] = len(amb)
    ev["distinct_completion_outputs"] = len(
        {repr(o) for o in re["completion_outputs"]})
    return None


def _gate_chain_synthesis(ast, re, ev):
    if not values_equal(re["reference_value"], ast["target_output"]):
        return "UNSOLVABLE:reference_fails"
    ok_idx = [i for i, o in enumerate(re["root_outputs"]) if o is not None]
    producing = [i for i in ok_idx
                 if values_equal(re["root_outputs"][i], ast["target_output"])]
    if not producing:
        return "UNSOLVABLE:target_unreachable"
    if len(producing) > max(1, len(ok_idx) // 2):
        return "TRIVIAL:target_too_common"
    ev["target_fraction"] = round(len(producing) / max(len(ok_idx), 1), 3)
    ev["root_cases"] = len(re["root_cases"])
    return None


def _gate_ordered_output(ast, re, ev):
    if not values_equal(ast["reference"], re["value"]):
        return "UNSOLVABLE:reference_mismatch"
    if re["value"] is None:
        return "UNSOLVABLE:no_output"
    if len(re["matched_rule_ids"]) < 2:
        return "TRIVIAL:single_match_ordered"
    ev["value"] = re["value"]
    ev["matched_rule_ids"] = re["matched_rule_ids"]
    return None


_GATES_20K = {
    "tck_output": _gate_tck_output,
    "suite_audit": _gate_suite_audit,
    "output_repair": _gate_output_repair,
    "aggregation_repair": _gate_aggregation_repair,
    "rule_deletion_repair": _gate_rule_deletion_repair,
    "partial_input_audit": _gate_partial_input_audit,
    "chain_synthesis": _gate_chain_synthesis,
    "ordered_output": _gate_ordered_output,
}


# -- mutation shape checks ------------------------------------------------------------
def _single_output_cell_mutation(golden: dict[str, Any], broken: dict[str, Any],
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
    oi = mutation["output_index"]
    for i, (g, b) in enumerate(zip(golden["rules"], broken["rules"])):
        if g["id"] != b["id"] or g["input_entries"] != b["input_entries"]:
            return False
        if len(g["output_entries"]) != len(b["output_entries"]):
            return False
        for ci, (ge, be) in enumerate(zip(g["output_entries"], b["output_entries"])):
            if i == ri and ci == oi:
                if ge != mutation["from"] or be != mutation["to"]:
                    return False
            elif ge != be:
                return False
    return True


def _aggregation_only_mutation(golden: dict[str, Any], broken: dict[str, Any]) -> bool:
    if golden["inputs"] != broken["inputs"] or golden["outputs"] != broken["outputs"]:
        return False
    if golden["rules"] != broken["rules"]:
        return False
    if golden["hit_policy"] != broken["hit_policy"]:
        return False
    if golden.get("aggregation") == broken.get("aggregation"):
        return False
    return True


def _rule_deletion_only_mutation(golden: dict[str, Any], broken: dict[str, Any],
                                 mutation: dict[str, Any]) -> bool:
    if golden["hit_policy"] != broken["hit_policy"]:
        return False
    if golden.get("aggregation") != broken.get("aggregation"):
        return False
    if golden["inputs"] != broken["inputs"] or golden["outputs"] != broken["outputs"]:
        return False
    if len(broken["rules"]) != len(golden["rules"]) - 1:
        return False
    ri = mutation["rule_index"]
    if not 0 <= ri < len(golden["rules"]):
        return False
    kept_golden = [g for i, g in enumerate(golden["rules"]) if i != ri]
    return kept_golden == broken["rules"]


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


# -- terminal checks ---------------------------------------------------------------------
def _term_value(ast, re, submitted, checks, key="output"):
    answer = submitted.get(key)
    checks.append({
        "check": "exact_value",
        "pass": answer is not None and values_equal(answer, re["value"]),
    })


def _term_suite_audit(ast, re, submitted, checks):
    answer = submitted.get("inconsistent_case_id")
    ok = isinstance(answer, str) and answer == ast["tamper"]["case_id"]
    checks.append({"check": "exact_inconsistent_case_id", "pass": bool(ok)})


def _term_table_repair(ast, re, submitted, checks):
    answer = submitted.get("table")
    if not isinstance(answer, dict):
        checks.append({"check": "repaired_table", "pass": False,
                       "error": "submission missing 'table' payload"})
        return
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


def _term_chain_synthesis(ast, re, submitted, checks):
    answer = submitted.get("input")
    if not isinstance(answer, dict):
        checks.append({"check": "input_assignment", "pass": False,
                       "error": "submission missing 'input' assignment"})
        return
    chain = ast["chain"]
    try:
        tables = {d: table_from_payload(p) for d, p in chain["tables"].items()}
        res, _ = evaluate_chain(tables, chain["edges"],
                                chain["terminal_decision_id"], answer,
                                decision_names=chain["decision_names"])
    except DmnUnsupportedError as exc:
        checks.append({"check": "produces_target_output", "pass": False,
                       "error": "evaluation failed: %s" % str(exc)[:120]})
        return
    checks.append({
        "check": "produces_target_output",
        "pass": values_equal(res.value, ast["target_output"]),
    })


def _term_partial_input_audit(ast, re, submitted, checks):
    answer = submitted.get("necessarily_matching")
    answer_poss = submitted.get("possibly_matching")
    rule_ids = re["rule_ids"]
    nec_ids = [rule_ids[i] for i in re["necessarily_matching"]]
    poss_ids = [rule_ids[i] for i in re["possibly_matching"]]
    ok_nec = isinstance(answer, list) and [str(x) for x in answer] == nec_ids
    ok_poss = isinstance(answer_poss, list) and [str(x) for x in answer_poss] == poss_ids
    checks.append({"check": "exact_necessarily_matching", "pass": bool(ok_nec)})
    checks.append({"check": "exact_possibly_matching", "pass": bool(ok_poss)})


_TERMINAL_20K = {
    "tck_output": _term_value,
    "suite_audit": _term_suite_audit,
    "output_repair": _term_table_repair,
    "aggregation_repair": _term_table_repair,
    "rule_deletion_repair": _term_table_repair,
    "partial_input_audit": _term_partial_input_audit,
    "chain_synthesis": _term_chain_synthesis,
    "ordered_output": _term_value,
}


# -- semantic witnesses ---------------------------------------------------------------------
def _witness_tck_output(ast, re):
    others = [r for r in re["rows"]
              if r["matched_rule_ids"] and not values_equal(r["output"], re["value"])]
    return {"ok": bool(others), "mode": "tck_output_discrimination",
            "distinct_other_outputs": len({repr(r["output"]) for r in others}),
            "dual_oracle": "aob-dmn-static == pinned TCK expectation"}


def _witness_suite_audit(ast, re):
    values = re["suite_values"]
    idx = next(i for i, case in enumerate(ast["suite"])
               if case["case_id"] == ast["tamper"]["case_id"])
    true_others = [v for i, v in enumerate(values) if i != idx]
    distinct_true = {repr(v) for v in true_others}
    shown = [case["expected"] for case in ast["suite"]]
    duplicated = any(i != idx and values_equal(s, shown[idx])
                     for i, s in enumerate(shown))
    return {"ok": bool(distinct_true) and duplicated,
            "mode": "tamper_requires_table_evaluation",
            "distinct_true_values": len(distinct_true),
            "tampered_value_duplicated_in_suite": bool(duplicated)}


def _witness_mutation_differential(ast, re):
    behavior = re["broken_behavior"]
    if behavior is None:
        return {"ok": False, "mode": "mutation_differential", "reason": "unevaluable"}
    golden = re["golden_behavior"]
    failing = sum(1 for g, b in zip(golden, behavior) if not values_equal(g, b))
    return {"ok": failing > 0, "mode": "mutation_differential",
            "golden_pass_cases": len(behavior) - failing, "broken_fail_cases": failing}


def _witness_partial_input_audit(ast, re):
    amb = [i for i in re["possibly_matching"] if i not in re["necessarily_matching"]]
    distinct = {repr(o) for o in re["completion_outputs"]}
    return {"ok": bool(amb) and len(distinct) >= 2,
            "mode": "partial_information_necessity",
            "ambiguous_rules": len(amb),
            "distinct_completion_outputs": len(distinct)}


def _witness_chain_synthesis(ast, re):
    ok_idx = [i for i, o in enumerate(re["root_outputs"]) if o is not None]
    producing = [i for i in ok_idx
                 if values_equal(re["root_outputs"][i], ast["target_output"])]
    others = [o for i, o in enumerate(re["root_outputs"])
              if o is not None and not values_equal(o, ast["target_output"])]
    return {"ok": bool(others) and 0 < len(producing) <= max(1, len(ok_idx) // 2),
            "mode": "target_discrimination",
            "producing_fraction": round(len(producing) / max(len(ok_idx), 1), 3),
            "other_outputs": len({repr(o) for o in others})}


def _witness_ordered_output(ast, re):
    others = [r for r in re["rows"]
              if len(r["matched_rule_ids"]) >= 2
              and not values_equal(r["output"], re["value"])]
    unique = DecisionTable(
        hit_policy="UNIQUE", aggregation="", inputs=table_from_payload(ast["table"]).inputs,
        outputs=table_from_payload(ast["table"]).outputs,
        rules=table_from_payload(ast["table"]).rules)
    res = evaluate_table(unique, ast["input_case"])
    return {"ok": bool(others) and bool(res.violations),
            "mode": "ordering_necessity",
            "distinct_other_outputs": len({repr(r["output"]) for r in others}),
            "unique_policy_violations": list(res.violations),
            "matched_rules": len(re["matched_rule_ids"])}


_WITNESS_20K = {
    "tck_output": _witness_tck_output,
    "suite_audit": _witness_suite_audit,
    "output_repair": _witness_mutation_differential,
    "aggregation_repair": _witness_mutation_differential,
    "rule_deletion_repair": _witness_mutation_differential,
    "partial_input_audit": _witness_partial_input_audit,
    "chain_synthesis": _witness_chain_synthesis,
    "ordered_output": _witness_ordered_output,
}
