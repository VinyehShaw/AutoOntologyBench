"""L10 NativeVerifierPlugin: deterministic BDDL symbolic verification.

Two independent legs must agree on every verdict:
  leg A — ``benchmark.plugins.task_family.l10.engine``/``checks``/``evalgoal``
          (independent s-expression parser + closed-world evaluator)
  leg B — the vendored ``bddl`` package from the pinned BEHAVIOR-1K snapshot
          (upstream parser + condition_evaluation compiler)

All gold is recomputed from the pinned snapshot bytes (digest-checked); stored
``native_evidence`` is never trusted. Every task is symbolic: static
predicate/arity/binding checks and closed-world goal evaluation. No simulator
is involved and no embodied success is claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from benchmark.plugins.task_family.l10 import engine, evalgoal, textgen, upstream
from benchmark.plugins.task_family.l10.checks import check_problem
from benchmark.plugins.task_family.l10.engine import ParsedProblem

REPO_ROOT = Path(__file__).resolve().parents[4]
ORACLE_DIR = REPO_ROOT / "bank" / "staging" / "l10" / "oracle"
ENGINE = "bddl-symbolic-static/1 (leg A: independent parser; leg B: vendored bddl 3.7.0)"
SOURCE_SNAPSHOT_ID = "behavior-1k-bddl@v3.9.3-post2"

_PROBLEM_CACHE: dict[str, tuple[str, ParsedProblem]] = {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_pinned(problem_rel: str, expected_digest: str) -> tuple[str, ParsedProblem]:
    key = problem_rel
    if key not in _PROBLEM_CACHE:
        path = engine.PINNED_ROOT / problem_rel
        text = path.read_text(encoding="utf-8")
        _PROBLEM_CACHE[key] = (text, engine.parse_problem_text(text))
    text, problem = _PROBLEM_CACHE[key]
    digest = _sha256_file(engine.PINNED_ROOT / problem_rel)
    if digest != expected_digest:
        raise RuntimeError(
            "pinned problem digest drift: %s (%s != %s)"
            % (problem_rel, digest[:12], expected_digest[:12]))
    return text, problem


def _parsed(problem: ParsedProblem, goal: tuple, init: tuple | None = None) -> ParsedProblem:
    return ParsedProblem(
        name=problem.name, domain=problem.domain, objects=problem.objects,
        init=problem.init if init is None else tuple(init),
        goal=tuple(goal), goal_raw=tuple(goal))


def _subgoal(problem: ParsedProblem, indices: list[int]) -> tuple:
    return tuple(problem.goal[i] for i in indices)


def _parse_atom_str(s: str) -> tuple:
    node = engine.scan_tokens(s)
    if not isinstance(node, list) or not node:
        raise ValueError("not an atom: %r" % s)
    return tuple(node)


def _upstream_eval_states(problem_text: str, truth_sets: list[frozenset]) -> list[bool]:
    """Leg-B evaluation of one problem text over several closed-world states."""
    mod = upstream._load_upstream()
    parsing, ce = mod["parsing"], mod["condition_evaluation"]
    cls_to_token = mod["cls_to_token"]
    _name, objects, initial_state, goal_state = parsing.parse_problem(
        "", 0, "behavior-1k", predefined_problem=problem_text)
    scope = ce.create_scope(objects)
    out: list[bool] = []
    for truth in truth_sets:
        def evaluate_fn(predicate_cls, *entities, _truth=tuple(truth)):
            token = cls_to_token.get(predicate_cls)
            return token is not None and (token,) + tuple(entities) in _truth
        compiled = ce.compile_state(goal_state, scope=scope, object_map=objects,
                                    generate_ground_options=False)
        satisfied, _results = ce.evaluate_state(compiled, evaluate_fn)
        out.append(bool(satisfied))
    return out


def _upstream_parse_ok(text: str) -> tuple[bool, str]:
    try:
        upstream.parse_upstream(text)
        return True, ""
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:120])


class BddlNativeVerifier:
    """NativeVerifierPlugin for lane L10 (symbolic BDDL checks only)."""

    plugin_id = "l10.bddl-verifier"

    def __init__(self, oracle_dir: str | Path | None = None) -> None:
        self.oracle_dir = Path(oracle_dir) if oracle_dir else ORACLE_DIR

    # ------------------------------------------------------------- material

    def _material(self, ast: dict[str, Any]) -> dict[str, Any]:
        text, problem = _load_pinned(ast["problem_rel"], ast["problem_digest"])
        kind = ast["goal_kind"]
        params = ast["params"]
        obj_map, inits, goal = textgen.problem_parts(problem)
        tables = engine.load_tables()
        out: dict[str, Any] = {
            "problem": problem, "text": text, "obj_map": obj_map,
            "inits": inits, "goal": goal, "tables": tables, "kind": kind,
            "params": params,
        }
        if kind in ("repair_predicate", "repair_arity", "repair_binding",
                    "repair_init_predicate", "repair_quantifier_category",
                    "defect_audit"):
            c_obj, c_init, c_goal = textgen.corrupt_parts(problem, kind, params)
            out["corrupted"] = (c_obj, c_init, c_goal)
            out["corrupted_text"] = textgen.serialize_problem(
                ast["activity"] + "-%d" % ast["variant"], c_obj, c_init, c_goal)
        if kind in ("obligation_query", "goal_construction", "minimal_state"):
            sub = _subgoal(problem, params["conjuncts"])
            out["subgoal"] = sub
            out["obligations"] = evalgoal.obligation_atoms(_parsed(problem, sub))
        if kind == "conjunct_truth_audit":
            add = params["add_atoms"]
            perturbed = list(inits)
            if add and add[0].startswith("@remove:"):
                pos = textgen.positive_init_positions(problem)
                perturbed.pop(pos[int(add[0].split(":", 1)[1])][0])
            else:
                for atom_str in add:
                    perturbed.append(_parse_atom_str(atom_str))
            out["perturbed_init"] = tuple(perturbed)
        if kind == "init_grouping":
            pos = textgen.positive_init_positions(problem)
            keep = set(range(len(pos))) - set(params["remove_init"])
            out["perturbed_init"] = tuple(
                atom for k, (_i, atom) in enumerate(pos) if k in keep)
        if kind == "objects_audit":
            pairs = [(inst, cat) for cat, insts in obj_map.items()
                     for inst in insts]
            for i, j in params["swaps"]:
                pairs[i], pairs[j] = (
                    (pairs[i][0], pairs[j][1]), (pairs[j][0], pairs[i][1]))
            out["pairs"] = pairs
        if kind == "category_join":
            partner_rel = "bddl3/bddl/activity_definitions/%s/problem0.bddl" % (
                params["partner_activity"])
            p_text, p_problem = _load_pinned(partner_rel, _sha256_file(
                engine.PINNED_ROOT / partner_rel))
            out["partner"] = p_problem
            out["partner_text"] = p_text
        return out

    # ------------------------------------------------------ reference answer

    def reference_submission(self, ast: dict[str, Any],
                             mat: dict[str, Any] | None = None) -> dict[str, Any]:
        if mat is None:
            mat = self._material(ast)
        kind = mat["kind"]
        if kind == "repair_predicate":
            leaves = textgen.goal_leaf_positions(mat["goal"])
            path, leaf = leaves[mat["params"]["leaf_index"] % len(leaves)]
            return {"fragment": str(leaf[0])}
        if kind == "repair_arity":
            leaves = textgen.goal_leaf_positions(mat["goal"])
            path, leaf = leaves[mat["params"]["leaf_index"] % len(leaves)]
            return {"fragment": str(leaf[-1])}
        if kind == "repair_binding":
            leaves = textgen.goal_leaf_positions(mat["goal"])
            path, leaf = leaves[mat["params"]["leaf_index"] % len(leaves)]
            return {"fragment": str(leaf[mat["params"]["arg_index"]])}
        if kind == "repair_init_predicate":
            pos = textgen.positive_init_positions(mat["problem"])
            _i, atom = pos[mat["params"]["init_index"] % len(pos)]
            return {"fragment": str(atom[0])}
        if kind == "repair_quantifier_category":
            quants = textgen.quantifier_positions(mat["goal"])
            _path, node = quants[mat["params"]["quant_index"] % len(quants)]
            head = node[0]
            iter_idx = 2 if head in ("forn", "fornpairs") else 1
            return {"fragment": str(node[iter_idx][2])}
        if kind == "obligation_query":
            return {"atoms": [textgen.atom_str(a) for a in mat["obligations"]]}
        if kind == "conjunct_truth_audit":
            truth = evalgoal.init_truth_set(
                _parsed(mat["problem"], mat["goal"], mat["perturbed_init"]))
            true_idx = sorted(
                i for i, c in enumerate(mat["goal"])
                if evalgoal.eval_node(c, {}, mat["problem"], truth))
            return {"true_conjuncts": true_idx}
        if kind == "init_grouping":
            groups: dict[str, list[str]] = {}
            for atom in mat["perturbed_init"]:
                groups.setdefault(str(atom[0]), []).append(textgen.atom_str(atom))
            return {"groups": {k: sorted(v) for k, v in sorted(groups.items())}}
        if kind == "objects_audit":
            mismatched = sorted(
                inst for inst, cat in mat["pairs"]
                if not inst.startswith(cat + "_"))
            return {"mismatched": mismatched}
        if kind == "goal_construction":
            return {"goal_text": textgen.sexpr(("and",) + tuple(mat["subgoal"]))}
        if kind == "minimal_state":
            return {"atoms": [textgen.atom_str(a) for a in mat["obligations"]]}
        if kind == "category_join":
            inst_cat = mat["problem"].instance_category
            cats: list[str] = []
            for _path, leaf in textgen.goal_leaf_positions(mat["goal"]):
                for arg in leaf[1:]:
                    if isinstance(arg, str) and arg.startswith("?"):
                        cat = inst_cat.get(arg[1:])
                        if cat is not None and cat not in cats:
                            cats.append(cat)
            mapping = {cat: sorted(mat["partner"].object_map.get(cat, []))
                       for cat in cats}
            return {"mapping": mapping}
        if kind == "defect_audit":
            c_obj, c_init, c_goal = mat["corrupted"]
            findings = check_problem(
                _parsed(mat["problem"], c_goal, c_init), mat["tables"])
            return {"findings": sorted(
                [f.section, f.code] for f in findings)}
        raise ValueError("unknown goal_kind: %s" % kind)

    # -------------------------------------------------------- verify_terminal

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        try:
            mat = self._material(ast)
        except Exception as exc:
            return NativeVerdict(False, {"error": "material: %s" % exc})
        kind = mat["kind"]
        try:
            return self._verify(mat, submitted)
        except Exception as exc:
            return NativeVerdict(False, {"error": "%s: %s" % (
                type(exc).__name__, str(exc)[:200])})

    def _verify(self, mat: dict[str, Any], submitted: Any) -> NativeVerdict:
        kind = mat["kind"]
        problem = mat["problem"]
        tables = mat["tables"]
        if not isinstance(submitted, dict):
            return NativeVerdict(False, {"error": "submission must be an object"})

        if kind in ("repair_predicate", "repair_arity", "repair_binding",
                    "repair_init_predicate", "repair_quantifier_category"):
            return self._verify_repair(mat, submitted)

        if kind == "obligation_query":
            atoms = submitted.get("atoms")
            if (not isinstance(atoms, list) or not atoms
                    or any(not isinstance(a, str) for a in atoms)
                    or sorted(set(atoms)) != sorted(atoms)):
                return NativeVerdict(False, {"error": "atoms must be a sorted "
                                                "list of unique strings"})
            want = [textgen.atom_str(a) for a in mat["obligations"]]
            if atoms != want:
                return NativeVerdict(False, {
                    "error": "obligation set mismatch",
                    "expected": len(want), "received": len(atoms),
                    "missing": sorted(set(want) - set(atoms))[:10],
                    "extra": sorted(set(atoms) - set(want))[:10]})
            truth = frozenset(tuple(_parse_atom_str(a)) for a in atoms)
            sub_text = textgen.serialize_problem(
                "verify-obligation", mat["obj_map"], mat["inits"], mat["subgoal"])
            leg_b = _upstream_eval_states(sub_text, [truth])
            detail = {"leg_a": "exact obligation expansion",
                      "leg_b_satisfied": leg_b[0],
                      "atoms": len(atoms)}
            if not leg_b[0]:
                return NativeVerdict(False, dict(
                    detail, error="leg B (vendored bddl) does not evaluate the "
                    "goal as satisfied over the submitted atoms"))
            return NativeVerdict(True, detail)

        if kind == "conjunct_truth_audit":
            idxs = submitted.get("true_conjuncts")
            n = len(mat["goal"])
            if (not isinstance(idxs, list)
                    or any(not isinstance(i, int) or i < 0 or i >= n for i in idxs)
                    or sorted(set(idxs)) != sorted(idxs)):
                return NativeVerdict(False, {"error": "true_conjuncts must be a "
                                                "sorted list of indices"})
            truth = evalgoal.init_truth_set(
                _parsed(problem, mat["goal"], mat["perturbed_init"]))
            want = sorted(
                i for i, c in enumerate(mat["goal"])
                if evalgoal.eval_node(c, {}, problem, truth))
            if idxs != want:
                return NativeVerdict(False, {
                    "error": "conjunct truth set mismatch",
                    "expected": want, "received": idxs})
            leg_b = []
            for i, conj in enumerate(mat["goal"]):
                text = textgen.serialize_problem(
                    "verify-conj-%d" % i, mat["obj_map"],
                    mat["perturbed_init"], (conj,))
                leg_b.append(_upstream_eval_states(text, [truth])[0])
            got_b = sorted(i for i, ok in enumerate(leg_b) if ok)
            if got_b != want:
                return NativeVerdict(False, {
                    "error": "leg B (vendored bddl) conjunct truth disagrees",
                    "leg_a": want, "leg_b": got_b})
            return NativeVerdict(True, {
                "conjuncts": n, "true_conjuncts": want,
                "leg_b_agrees": True})

        if kind == "init_grouping":
            groups = submitted.get("groups")
            if not isinstance(groups, dict) or not groups:
                return NativeVerdict(False, {"error": "groups must be a "
                                                "non-empty mapping"})
            want: dict[str, list[str]] = {}
            for atom in mat["perturbed_init"]:
                want.setdefault(str(atom[0]), []).append(textgen.atom_str(atom))
            want = {k: sorted(v) for k, v in want.items()}
            norm = {k: sorted(v) for k, v in groups.items()
                    if isinstance(v, list)}
            if norm != want:
                return NativeVerdict(False, {
                    "error": "grouping mismatch",
                    "expected_predicates": sorted(want),
                    "received_predicates": sorted(norm)})
            text = textgen.serialize_problem(
                "verify-grouping", mat["obj_map"], mat["perturbed_init"],
                tuple(mat["goal"]))
            theirs = upstream.parse_upstream(text)
            b_groups: dict[str, list[str]] = {}
            for lit in theirs.init:
                if isinstance(lit, tuple) and lit and lit[0] != "not":
                    b_groups.setdefault(str(lit[0]), []).append(
                        textgen.atom_str(lit))
            b_groups = {k: sorted(v) for k, v in b_groups.items()}
            if b_groups != want:
                return NativeVerdict(False, {
                    "error": "leg B grouping disagrees",
                    "leg_a": sorted(want), "leg_b": sorted(b_groups)})
            return NativeVerdict(True, {
                "predicates": len(want), "atoms": len(mat["perturbed_init"]),
                "leg_b_agrees": True})

        if kind == "objects_audit":
            mismatched = submitted.get("mismatched")
            if (not isinstance(mismatched, list)
                    or any(not isinstance(x, str) for x in mismatched)
                    or sorted(set(mismatched)) != sorted(mismatched)):
                return NativeVerdict(False, {"error": "mismatched must be a "
                                                "sorted list of instances"})
            want = sorted(
                inst for inst, cat in mat["pairs"]
                if not inst.startswith(cat + "_"))
            if mismatched != want:
                return NativeVerdict(False, {
                    "error": "mismatch set mismatch",
                    "expected": want, "received": mismatched})
            swapped_obj_map: dict[str, list[str]] = {}
            for inst, cat in mat["pairs"]:
                swapped_obj_map.setdefault(cat, []).append(inst)
            text = textgen.serialize_problem(
                "verify-objects", swapped_obj_map, mat["inits"],
                tuple(mat["goal"]))
            theirs = upstream.parse_upstream(text)
            b_pairs = [(inst, cat) for cat, insts in theirs.object_map.items()
                       for inst in insts]
            b_want = sorted(
                inst for inst, cat in b_pairs
                if not inst.startswith(cat + "_"))
            if b_want != want:
                return NativeVerdict(False, {
                    "error": "leg B mismatch set disagrees",
                    "leg_a": want, "leg_b": b_want})
            return NativeVerdict(True, {
                "objects": len(mat["pairs"]), "mismatched": want,
                "leg_b_agrees": True})

        if kind == "goal_construction":
            goal_text = submitted.get("goal_text")
            if not isinstance(goal_text, str) or not goal_text.strip():
                return NativeVerdict(False, {"error": "goal_text must be a "
                                                "non-empty string"})
            try:
                expr = engine.scan_tokens(goal_text)
            except engine.BddlParseError as exc:
                return NativeVerdict(False, {"error": "unparsable goal: %s" % exc})
            expr_t = tuple(engine.canon(expr))
            s1 = list(textgen.positive_init_atoms(
                mat["problem"])) + list(mat["obligations"])
            s2 = list(textgen.positive_init_atoms(mat["problem"]))
            checked = _parsed(problem, (expr_t,), s1)
            findings = check_problem(checked, tables)
            if findings:
                return NativeVerdict(False, {
                    "error": "submitted goal fails static checks",
                    "findings": [[f.section, f.code, f.detail[:80]]
                                 for f in findings[:5]]})
            truth1 = evalgoal.init_truth_set(_parsed(problem, (expr_t,), s1))
            truth2 = evalgoal.init_truth_set(_parsed(problem, (expr_t,), s2))
            a1 = evalgoal.eval_node(expr_t, {}, problem, truth1)
            a2 = evalgoal.eval_node(expr_t, {}, problem, truth2)
            text1 = textgen.serialize_problem(
                "verify-gc-1", mat["obj_map"], s1, (expr_t,))
            text2 = textgen.serialize_problem(
                "verify-gc-2", mat["obj_map"], s2, (expr_t,))
            b1 = _upstream_eval_states(text1, [truth1])[0]
            b2 = _upstream_eval_states(text2, [truth2])[0]
            detail = {"leg_a": [bool(a1), bool(a2)], "leg_b": [bool(b1), bool(b2)]}
            if not (a1 and not a2):
                return NativeVerdict(False, dict(
                    detail, error="goal must be satisfied in S1 and not in S2"))
            if [bool(a1), bool(a2)] != [bool(b1), bool(b2)]:
                return NativeVerdict(False, dict(
                    detail, error="leg B (vendored bddl) disagrees with leg A"))
            return NativeVerdict(True, detail)

        if kind == "minimal_state":
            atoms = submitted.get("atoms")
            if (not isinstance(atoms, list) or not atoms
                    or any(not isinstance(a, str) for a in atoms)
                    or sorted(set(atoms)) != sorted(atoms)):
                return NativeVerdict(False, {"error": "atoms must be a sorted "
                                                "list of unique strings"})
            try:
                parsed_atoms = [tuple(_parse_atom_str(a)) for a in atoms]
            except (ValueError, engine.BddlParseError) as exc:
                return NativeVerdict(False, {"error": "bad atom: %s" % exc})
            if any(any(str(x).startswith("?") for x in a[1:]) for a in parsed_atoms):
                return NativeVerdict(False, {"error": "atoms must be ground"})
            checked = _parsed(problem, mat["subgoal"], parsed_atoms)
            findings = check_problem(checked, tables)
            if findings:
                return NativeVerdict(False, {
                    "error": "submitted atoms fail static checks",
                    "findings": [[f.section, f.code, f.detail[:80]]
                                 for f in findings[:5]]})
            truth = frozenset(parsed_atoms)
            satisfied = all(
                evalgoal.eval_node(c, {}, problem, truth) for c in mat["subgoal"])
            if not satisfied:
                return NativeVerdict(False, {
                    "error": "goal not satisfied over submitted state",
                    "atoms": len(atoms)})
            minimal = True
            for i in range(len(parsed_atoms)):
                smaller = frozenset(parsed_atoms[:i] + parsed_atoms[i + 1:])
                if all(eval_node_ok(mat, c, smaller) for c in mat["subgoal"]):
                    minimal = False
                    break
            if not minimal:
                return NativeVerdict(False, {
                    "error": "state is not minimal: some atom can be removed",
                    "atoms": len(atoms)})
            text = textgen.serialize_problem(
                "verify-minstate", mat["obj_map"], parsed_atoms, mat["subgoal"])
            leg_b = _upstream_eval_states(text, [truth])[0]
            if not leg_b:
                return NativeVerdict(False, {
                    "error": "leg B (vendored bddl) does not evaluate the goal "
                    "as satisfied over the submitted state"})
            return NativeVerdict(True, {
                "atoms": len(atoms), "minimal": True, "leg_b_satisfied": True})

        if kind == "category_join":
            mapping = submitted.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                return NativeVerdict(False, {"error": "mapping must be a "
                                                "non-empty object"})
            want = self.reference_submission(
                {"goal_kind": "category_join", "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""},
                mat)["mapping"]
            norm = {k: sorted(v) for k, v in mapping.items()
                    if isinstance(v, list)}
            if norm != want:
                return NativeVerdict(False, {
                    "error": "category join mapping mismatch",
                    "expected_categories": sorted(want),
                    "received_categories": sorted(norm)})
            theirs_a = upstream.parse_upstream(mat["text"])
            theirs_b = upstream.parse_upstream(mat["partner_text"])
            inst_cat_a = {i: c for c, ii in theirs_a.object_map.items()
                          for i in ii}
            cats_b: dict[str, list[str]] = {}
            for _path, leaf in textgen.goal_leaf_positions(
                    tuple(theirs_a.goal)):
                for arg in leaf[1:]:
                    if isinstance(arg, str) and arg.startswith("?"):
                        cat = inst_cat_a.get(arg[1:])
                        if cat is not None:
                            cats_b.setdefault(
                                cat, sorted(theirs_b.object_map.get(cat, [])))
            if cats_b != want:
                return NativeVerdict(False, {
                    "error": "leg B (vendored bddl) join mapping disagrees",
                    "leg_a": {k: len(v) for k, v in want.items()},
                    "leg_b": {k: len(v) for k, v in cats_b.items()}})
            return NativeVerdict(True, {
                "categories": len(want),
                "partner_instances": sum(len(v) for v in want.values()),
                "leg_b_agrees": True})

        if kind == "defect_audit":
            findings = submitted.get("findings")
            if (not isinstance(findings, list) or not findings
                    or any(not isinstance(p, list) or len(p) != 2
                           or not all(isinstance(x, str) for x in p)
                           for p in findings)):
                return NativeVerdict(False, {"error": "findings must be a list "
                                                "of [section, code] pairs"})
            c_obj, c_init, c_goal = mat["corrupted"]
            want = sorted(
                [f.section, f.code] for f in check_problem(
                    _parsed(problem, c_goal, c_init), tables))
            norm = sorted([list(p) for p in findings])
            if norm != want:
                return NativeVerdict(False, {
                    "error": "finding multiset mismatch",
                    "expected": want, "received": norm})
            ok_b, err_b = _upstream_parse_ok(mat["corrupted_text"])
            detail = {"findings": len(want),
                      "distinct_codes": len({p[1] for p in want}),
                      "upstream_parses_corrupted": ok_b}
            if not ok_b:
                detail["upstream_error"] = err_b
                return NativeVerdict(True, detail)
            mine = engine.parse_problem_text(mat["corrupted_text"])
            if not upstream.parse_agrees(mat["corrupted_text"], mine):
                return NativeVerdict(False, {
                    "error": "leg B parses the corrupted text but disagrees "
                    "structurally with leg A"})
            detail["leg_b_agrees"] = True
            return NativeVerdict(True, detail)

        return NativeVerdict(False, {"error": "unknown kind %r" % kind})

    def _verify_repair(self, mat: dict[str, Any],
                       submitted: dict[str, Any]) -> NativeVerdict:
        fragment = submitted.get("fragment")
        if not isinstance(fragment, str) or not fragment.strip() \
                or any(ch in fragment for ch in " ()\n\t"):
            return NativeVerdict(False, {"error": "fragment must be a single "
                                            "BDDL token (no whitespace or "
                                            "parentheses)"})
        kind = mat["kind"]
        problem = mat["problem"]
        tables = mat["tables"]
        obj_map, inits, goal = mat["obj_map"], mat["inits"], mat["goal"]
        leaves = textgen.goal_leaf_positions(goal)
        params = mat["params"]

        if kind == "repair_predicate":
            i = params["leaf_index"] % len(leaves)
            path, leaf = leaves[i]
            repaired_goal = textgen.replace_goal_leaf(
                goal, path, (fragment,) + tuple(leaf[1:]))
            repaired_init = inits
        elif kind == "repair_arity":
            i = params["leaf_index"] % len(leaves)
            path, leaf = leaves[i]
            repaired_goal = textgen.replace_goal_leaf(
                goal, path, tuple(leaf[:-1]) + (fragment,))
            repaired_init = inits
        elif kind == "repair_binding":
            i = params["leaf_index"] % len(leaves)
            path, leaf = leaves[i]
            new_args = list(leaf)
            new_args[params["arg_index"]] = fragment
            repaired_goal = textgen.replace_goal_leaf(
                goal, path, tuple(new_args))
            repaired_init = inits
        elif kind == "repair_init_predicate":
            pos = textgen.positive_init_positions(problem)
            full_i, atom = pos[params["init_index"] % len(pos)]
            repaired_init = list(inits)
            repaired_init[full_i] = (fragment,) + tuple(atom[1:])
            repaired_goal = goal
        elif kind == "repair_quantifier_category":
            quants = textgen.quantifier_positions(goal)
            q = params["quant_index"] % len(quants)
            qpath, node = quants[q]
            head = node[0]
            iter_idx = 2 if head in ("forn", "fornpairs") else 1
            new_iter = (node[iter_idx][0], "-", fragment)
            new_node = list(node)
            new_node[iter_idx] = new_iter
            repaired_goal = textgen.replace_goal_quantifier(
                goal, qpath, tuple(new_node))
            repaired_init = inits
        else:
            return NativeVerdict(False, {"error": "not a repair kind"})

        repaired_text = textgen.serialize_problem(
            "repaired", obj_map, repaired_init, repaired_goal)
        try:
            mine = engine.parse_problem_text(repaired_text)
        except engine.BddlParseError as exc:
            return NativeVerdict(False, {"error": "repaired text unparsable: "
                                            "%s" % exc})
        findings = check_problem(mine, tables)
        if findings:
            return NativeVerdict(False, {
                "error": "repaired problem still fails static checks",
                "findings": [[f.section, f.code, f.detail[:80]]
                             for f in findings[:5]]})
        if (mine.objects != problem.objects
                or sorted(map(repr, mine.init)) != sorted(map(repr, problem.init))
                or sorted(map(repr, mine.goal)) != sorted(map(repr, problem.goal))):
            return NativeVerdict(False, {
                "error": "repaired problem does not canonically match the "
                "pinned original (wrong fragment)"})
        if not upstream.parse_agrees(repaired_text, mine):
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) disagrees with leg A on the "
                "repaired problem"})
        return NativeVerdict(True, {
            "fragment": fragment,
            "checks": "zero findings + canonical equality + leg A/B agreement"})

    # ---------------------------------------------------------- prove_exists

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        try:
            mat = self._material(ast)
        except Exception as exc:
            return ProofOrCounterexample(
                False, {"error": str(exc)[:200]},
                "PROFILE_UNSUPPORTED:material:%s" % type(exc).__name__)
        kind = mat["kind"]
        ev: dict[str, Any] = {
            "engine": ENGINE,
            "source_snapshot_id": SOURCE_SNAPSHOT_ID,
            "table_shas": engine.table_shas(mat["tables"]),
        }
        try:
            reference = self.reference_submission(ast)
        except Exception as exc:
            return ProofOrCounterexample(
                False, dict(ev, error=str(exc)[:200]),
                "PROFILE_UNSUPPORTED:reference:%s" % type(exc).__name__)
        task = GroundedTask(ast=grounded_task.ast, public={},
                            oracle_ref=grounded_task.oracle_ref,
                            native_evidence={})
        term = self.verify_terminal(task, reference)
        ev["reference_terminal"] = term.valid
        if not term.valid:
            return ProofOrCounterexample(
                False, ev, "UNSOLVABLE:reference_fails:%s"
                % json.dumps(term.detail)[:160])

        if kind in ("repair_predicate", "repair_arity", "repair_binding",
                    "repair_init_predicate", "repair_quantifier_category",
                    "defect_audit"):
            c_obj, c_init, c_goal = mat["corrupted"]
            findings = check_problem(
                _parsed(mat["problem"], c_goal, c_init), mat["tables"])
            ev["corrupted_findings"] = sorted(
                [f.section, f.code] for f in findings)
            if not findings:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:corruption_produces_no_findings")
            ok_b, err_b = _upstream_parse_ok(mat["corrupted_text"])
            ev["upstream_parses_corrupted"] = ok_b
            if not ok_b:
                ev["upstream_error"] = err_b
            if kind == "defect_audit" and len({p[1] for p in ev["corrupted_findings"]}) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:defect_audit_needs_2_distinct_codes")
            return ProofOrCounterexample(True, ev)

        if kind == "obligation_query":
            if not mat["obligations"]:
                return ProofOrCounterexample(False, ev, "TRIVIAL:empty_obligations")
            truth_init = evalgoal.init_truth_set(mat["problem"])
            if all(evalgoal.eval_node(c, {}, mat["problem"], truth_init)
                   for c in mat["subgoal"]):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:subgoal_already_satisfied")
            sub_text = textgen.serialize_problem(
                "prove-obligation", mat["obj_map"], mat["inits"], mat["subgoal"])
            truth_ref = frozenset(mat["obligations"])
            leg_b = _upstream_eval_states(sub_text, [truth_init, truth_ref])
            ev.update(obligations=len(mat["obligations"]),
                      leg_b_satisfied_over_reference=leg_b[1],
                      leg_b_satisfied_over_init=leg_b[0])
            if leg_b[1] is not True or leg_b[0] is not False:
                return ProofOrCounterexample(
                    False, ev, "PROFILE_UNSUPPORTED:legB_differential_mismatch")
            return ProofOrCounterexample(True, ev)

        if kind == "conjunct_truth_audit":
            if len(mat["goal"]) < 2:
                return ProofOrCounterexample(False, ev, "TRIVIAL:one_conjunct")
            return ProofOrCounterexample(True, ev)

        if kind == "init_grouping":
            preds = {str(a[0]) for a in mat["perturbed_init"]}
            ev.update(predicates=len(preds),
                      atoms=len(mat["perturbed_init"]))
            if len(preds) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_predicates")
            return ProofOrCounterexample(True, ev)

        if kind == "objects_audit":
            mismatched = [inst for inst, cat in mat["pairs"]
                          if not inst.startswith(cat + "_")]
            ev.update(objects=len(mat["pairs"]), mismatched=len(mismatched))
            if not mismatched:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:no_mismatch_after_swaps")
            if len(mismatched) == len(mat["pairs"]):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:all_objects_mismatch")
            return ProofOrCounterexample(True, ev)

        if kind == "goal_construction":
            if not mat["obligations"]:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:empty_obligations")
            pos_inits = textgen.positive_init_atoms(mat["problem"])
            if set(pos_inits) == set(list(pos_inits) + list(mat["obligations"])):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:s1_equals_s2")
            return ProofOrCounterexample(True, ev)

        if kind == "minimal_state":
            if not mat["obligations"]:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:empty_obligations")
            truth_init = evalgoal.init_truth_set(mat["problem"])
            if all(evalgoal.eval_node(c, {}, mat["problem"], truth_init)
                   for c in mat["subgoal"]):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:subgoal_already_satisfied")
            return ProofOrCounterexample(True, ev)

        if kind == "category_join":
            ref = self.reference_submission(ast)
            ev.update(categories=len(ref["mapping"]),
                      partner_instances=sum(
                          len(v) for v in ref["mapping"].values()))
            if len(ref["mapping"]) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_categories")
            return ProofOrCounterexample(True, ev)

        return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:kind=%s" % kind)

    # ------------------------------------------------- certify_semantic_relev

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        mat = self._material(ast)
        kind = mat["kind"]
        wrong = self._wrong_submission(mat)
        probe = GroundedTask(ast=task.ast, public={}, oracle_ref=task.oracle_ref,
                             native_evidence={})
        verdict = self.verify_terminal(probe, wrong)
        witness = {
            "engine": ENGINE,
            "counterfactual_kind": kind,
            "wrong_submission": _trim(wrong),
            "wrong_verdict_valid": verdict.valid,
            "wrong_verdict_detail": verdict.detail,
        }
        if verdict.valid:
            return SemanticWitness(False, witness)
        return SemanticWitness(True, witness)

    def _wrong_submission(self, mat: dict[str, Any]) -> dict[str, Any]:
        kind = mat["kind"]
        tables = mat["tables"]
        if kind == "repair_predicate":
            leaves = textgen.goal_leaf_positions(mat["goal"])
            _path, leaf = leaves[mat["params"]["leaf_index"] % len(leaves)]
            arity = tables.predicate_map.get(str(leaf[0]), len(leaf) - 1)
            same_arity = sorted(
                p for p, a in tables.predicate_map.items()
                if a == arity and p != str(leaf[0]))
            alt = same_arity[0] if same_arity else "ontop"
            return {"fragment": alt}
        if kind == "repair_arity":
            insts = sorted(mat["problem"].instances)
            orig = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            alt = next((i for i in insts if i != orig["fragment"]), insts[0])
            return {"fragment": alt}
        if kind == "repair_binding":
            insts = sorted(mat["problem"].instances)
            orig = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            alt = next((i for i in insts if "?" + i != orig["fragment"]),
                       insts[0])
            return {"fragment": "?" + alt}
        if kind == "repair_init_predicate":
            preds = sorted(tables.predicate_map)
            pos = textgen.positive_init_positions(mat["problem"])
            _i, atom = pos[mat["params"]["init_index"] % len(pos)]
            alt = next((p for p in preds if p != str(atom[0])), preds[0])
            return {"fragment": alt}
        if kind == "repair_quantifier_category":
            cats = sorted({c for c, _ in mat["problem"].objects})
            quants = textgen.quantifier_positions(mat["goal"])
            _p, node = quants[mat["params"]["quant_index"] % len(quants)]
            head = node[0]
            iter_idx = 2 if head in ("forn", "fornpairs") else 1
            orig = str(node[iter_idx][2])
            alt = next((c for c in cats if c != orig), cats[0])
            return {"fragment": alt}
        if kind == "obligation_query":
            atoms = [textgen.atom_str(a) for a in mat["obligations"]]
            if len(atoms) > 1:
                return {"atoms": atoms[:-1]}
            return {"atoms": atoms + ["(ontop agent.n.01_1 floor.n.01_1)"]}
        if kind == "conjunct_truth_audit":
            ref = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            idxs = ref["true_conjuncts"]
            n = len(mat["goal"])
            if idxs:
                return {"true_conjuncts": idxs[:-1]}
            return {"true_conjuncts": [0]}
        if kind == "init_grouping":
            ref = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            groups = {k: list(v) for k, v in ref["groups"].items()}
            keys = sorted(groups)
            if len(keys) >= 2:
                groups[keys[0]], groups[keys[1]] = (
                    groups[keys[1]], groups[keys[0]])
                return {"groups": groups}
            atom = textgen.atom_str(mat["perturbed_init"][0])
            return {"groups": {"wrongpred": [atom]}}
        if kind == "objects_audit":
            ref = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            matched = sorted(
                inst for inst, cat in mat["pairs"]
                if inst.startswith(cat + "_"))
            if matched:
                return {"mismatched": sorted(set(ref["mismatched"]) | {matched[0]})}
            return {"mismatched": []}
        if kind == "goal_construction":
            atom = textgen.atom_str(
                textgen.positive_init_atoms(mat["problem"])[0])
            return {"goal_text": atom}
        if kind == "minimal_state":
            atoms = [textgen.atom_str(a) for a in mat["obligations"]]
            insts = sorted(mat["problem"].instances)
            extra = "(ontop %s %s)" % (insts[0], insts[-1])
            return {"atoms": sorted(set(atoms) | {extra})}
        if kind == "category_join":
            ref = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            mapping = {k: list(v) for k, v in ref["mapping"].items()}
            key = sorted(mapping)[0]
            if mapping[key]:
                mapping[key] = mapping[key][:-1]
            else:
                mapping[key] = ["nonexistent_instance"]
            return {"mapping": mapping}
        if kind == "defect_audit":
            ref = self.reference_submission(
                {"goal_kind": kind, "params": mat["params"],
                 "problem_rel": "", "problem_digest": ""}, mat)
            findings = [list(p) for p in ref["findings"]]
            if len(findings) > 1:
                return {"findings": findings[:-1]}
            return {"findings": [["goal", "PRED_UNKNOWN"]]}
        raise ValueError("unknown kind %s" % kind)

    # ----------------------------------------------------------------- oracle

    def write_oracle(self, grounded_task: GroundedTask, evidence: dict[str, Any]) -> None:
        self.oracle_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.oracle_dir, 0o700)
        payload = {
            "candidate": {
                "lane_id": grounded_task.ast.lane_id,
                "family_id": grounded_task.ast.family_id,
                "base_goal_id": grounded_task.ast.base_goal_id,
                "instance_id": grounded_task.ast.instance_id,
                "regime": grounded_task.ast.regime,
                "semantic_profile": grounded_task.ast.semantic_profile,
                "source_unit_ids": list(grounded_task.ast.source_unit_ids),
            },
            "ast": grounded_task.ast.ast,
            "evidence": evidence,
        }
        _write_json_atomic(self.oracle_dir / ("%s.json" % grounded_task.oracle_ref),
                           payload)

    def load_oracle(self, oracle_ref: str) -> dict[str, Any]:
        path = self.oracle_dir / ("%s.json" % oracle_ref)
        return json.loads(path.read_text(encoding="utf-8"))


def eval_node_ok(mat: dict[str, Any], conj: Any, truth: frozenset) -> bool:
    return evalgoal.eval_node(conj, {}, mat["problem"], truth)


def _trim(obj: Any) -> Any:
    blob = json.dumps(obj, sort_keys=True, default=str)
    if len(blob) > 600:
        return blob[:600] + "...(trimmed)"
    return obj


PLUGIN = BddlNativeVerifier()

__all__ = ["BddlNativeVerifier", "PLUGIN", "ENGINE", "SOURCE_SNAPSHOT_ID"]
