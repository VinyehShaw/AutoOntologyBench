"""L10 20K-expansion NativeVerifierPlugin: deterministic BDDL symbolic checks.

Subclass of the legacy :class:`BddlNativeVerifier` covering the 12 NEW
family mechanisms (goal kinds ``quant_inventory`` .. ``predicate_usage``).
The two independent legs are unchanged:

  leg A — the lane's own parser/evaluator (``engine``/``checks``/``evalgoal``)
  leg B — the vendored upstream ``bddl`` 3.7.0 package from the pinned
          BEHAVIOR-1K snapshot (structural walks + closed-world evaluation)

Every verdict is recomputed from the pinned snapshot bytes (digest-checked);
stored ``native_evidence`` is never trusted. Symbolic only: no simulator, no
physics, no embodied execution, no OmniGibson claim anywhere.
"""
from __future__ import annotations

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
from benchmark.plugins.task_family.l10.propose_20k import (
    _closure_refs,
    _outermost_quants,
    _polarity_flags,
    _shape_tag,
)
from benchmark.plugins.verifier.l10.bddl_verifier import (
    ENGINE,
    SOURCE_SNAPSHOT_ID,
    BddlNativeVerifier,
    _load_pinned,
    _parse_atom_str,
    _upstream_eval_states,
)

NEW_KINDS = (
    "quant_inventory", "shape_census", "reference_closure", "polarity_audit",
    "discriminating_state", "weakening_goal", "contradiction_audit",
    "duplicate_audit", "predicate_join", "room_grouping",
    "provenance_partition", "predicate_usage",
)


def _parsed(problem: ParsedProblem, goal: tuple,
            init: tuple | None = None) -> ParsedProblem:
    return ParsedProblem(
        name=problem.name, domain=problem.domain, objects=problem.objects,
        init=problem.init if init is None else tuple(init),
        goal=tuple(goal), goal_raw=tuple(goal))


def _obligations(problem: ParsedProblem, goal: tuple) -> list[tuple]:
    return evalgoal.obligation_atoms(_parsed(problem, goal))


def _legb(text: str) -> ParsedProblem:
    return upstream.parse_upstream(text)


def _quant_triple(q: tuple) -> list[str]:
    """[head, var, category] for a single-iterable quantifier node.

    forall/exists keep the iterable at index 1; forn keeps its (N) count at
    index 1 and the iterable at index 2. forpairs/fornpairs have two
    iterables and are excluded from the inventory by design.
    """
    head = q[0]
    if head == "forn":
        return [str(head), str(q[2][0][1:]), str(q[2][2])]
    return [str(head), str(q[1][0][1:]), str(q[1][2])]


def _inventory_quants(goal: tuple) -> list[tuple]:
    """Outermost forall/exists/forn quantifiers (single iterable), in order."""
    return [q for q in _outermost_quants(goal)
            if q[0] in ("forall", "exists", "forn")]


class BddlVerifier20K(BddlNativeVerifier):
    """NativeVerifierPlugin for the L10 20K-expansion families."""

    plugin_id = "l10.bddl-verifier-20k"

    # ------------------------------------------------------------- material

    def _material(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["goal_kind"]
        if kind not in NEW_KINDS:
            raise ValueError("not a 20k goal_kind: %s" % kind)
        text, problem = _load_pinned(ast["problem_rel"], ast["problem_digest"])
        obj_map, inits, goal = textgen.problem_parts(problem)
        tables = engine.load_tables()
        out: dict[str, Any] = {
            "problem": problem, "text": text, "obj_map": obj_map,
            "inits": inits, "goal": goal, "tables": tables, "kind": kind,
            "params": ast["params"], "ast": ast,
        }
        params = ast["params"]
        if kind in ("quant_inventory", "shape_census", "reference_closure",
                    "polarity_audit", "provenance_partition",
                    "predicate_usage", "weakening_goal"):
            out["subgoal"] = tuple(problem.goal[i] for i in params["conjuncts"])
        if kind in ("weakening_goal", "provenance_partition"):
            out["obligations"] = _obligations(problem, out["subgoal"])
        if kind == "weakening_goal":
            out["s2"] = textgen.positive_init_atoms(problem)
            out["s1"] = list(out["s2"]) + list(out["obligations"])
        if kind == "discriminating_state":
            a_idx = params["a_conjuncts"]
            out["subgoal_a"] = tuple(problem.goal[j] for j in a_idx)
            out["subgoal_b"] = (problem.goal[params["b_index"]],)
            out["obligations_a"] = _obligations(problem, out["subgoal_a"])
            out["obligations_b"] = _obligations(problem, out["subgoal_b"])
        if kind == "contradiction_audit":
            pos = textgen.positive_init_atoms(problem)
            atom = pos[params["inject_index"] % len(pos)]
            insert_at = (params["inject_index"] * 3 + 1) % (len(inits) + 1)
            perturbed = list(inits)
            perturbed.insert(insert_at, ("not", atom))
            out["perturbed_init"] = tuple(perturbed)
        if kind == "duplicate_audit":
            perturbed = list(inits)
            for k, idx in enumerate(params["inject_indices"]):
                pos = textgen.positive_init_atoms(problem)
                atom = pos[idx % len(pos)]
                insert_at = (idx * 3 + 2 + k) % (len(perturbed) + 1)
                perturbed.insert(insert_at, atom)
            out["perturbed_init"] = tuple(perturbed)
        if kind == "predicate_join":
            partner_rel = ("bddl3/bddl/activity_definitions/%s/problem0.bddl"
                           % params["partner_activity"])
            p_text, p_problem = _load_pinned(
                partner_rel, params["partner_digest"])
            out["partner"] = p_problem
            out["partner_text"] = p_text
        if kind == "room_grouping":
            inrooms = [lit for lit in problem.init
                       if isinstance(lit, tuple) and lit and lit[0] == "inroom"]
            rem = [a for j, a in enumerate(inrooms)
                   if j != params["drop_index"] % len(inrooms)]
            out["shown_atoms"] = rem
        return out

    def _public_text(self, mat: dict[str, Any]) -> str:
        """The public problem text for structural leg-B walks."""
        ast = mat["ast"]
        name = ast["activity"] + "-%d" % ast["variant"]
        if mat["kind"] in ("contradiction_audit", "duplicate_audit"):
            return textgen.serialize_problem(
                name, mat["obj_map"], mat["perturbed_init"],
                tuple(mat["goal"]))
        return textgen.serialize_problem(
            name, mat["obj_map"], mat["inits"], tuple(mat["goal"]))

    # ------------------------------------------------------ reference answer

    def reference_submission(self, ast: dict[str, Any],
                             mat: dict[str, Any] | None = None) -> dict[str, Any]:
        if mat is None:
            mat = self._material(ast)
        kind = mat["kind"]
        problem = mat["problem"]
        if kind == "quant_inventory":
            quants = _inventory_quants(mat["subgoal"])
            return {"quantifiers": [_quant_triple(q) for q in quants]}
        if kind == "shape_census":
            return {"shapes": {
                str(p): _shape_tag(c)
                for p, c in enumerate(mat["subgoal"])}}
        if kind == "reference_closure":
            return {"closure": {
                str(p): sorted(_closure_refs(problem, c))
                for p, c in enumerate(mat["subgoal"])}}
        if kind == "polarity_audit":
            flags = _polarity_flags(mat["subgoal"])
            return {"negative_leaves": sorted(
                i for i, f in enumerate(flags) if f)}
        if kind == "discriminating_state":
            return {"atoms": sorted(
                textgen.atom_str(a) for a in mat["obligations_a"])}
        if kind == "weakening_goal":
            pos = textgen.positive_init_atoms(problem)
            first, second = pos[0], pos[1]
            return {"goal_text": textgen.sexpr(
                ("and", first, second))}
        if kind == "contradiction_audit":
            pos = set(textgen.positive_init_atoms(
                _parsed(problem, mat["goal"], mat["perturbed_init"])))
            neg = set()
            for lit in mat["perturbed_init"]:
                if isinstance(lit, tuple) and lit and lit[0] == "not":
                    neg.add(lit[1])
            return {"contradicted": sorted(
                textgen.atom_str(a) for a in pos & neg)}
        if kind == "duplicate_audit":
            from collections import Counter
            counts = Counter(
                textgen.atom_str(a) for a in textgen.positive_init_atoms(
                    _parsed(problem, mat["goal"], mat["perturbed_init"])))
            return {"duplicated": sorted(
                s for s, n in counts.items() if n > 1)}
        if kind == "predicate_join":
            goal_preds = sorted({
                str(leaf[0]) for _p, leaf in
                textgen.goal_leaf_positions(tuple(mat["goal"]))})
            mapping = {}
            for pred in goal_preds:
                mapping[pred] = sorted(
                    textgen.atom_str(a) for a in
                    textgen.positive_init_atoms(mat["partner"])
                    if str(a[0]) == pred)
            return {"mapping": mapping}
        if kind == "room_grouping":
            rooms: dict[str, list[str]] = {}
            for atom in mat["shown_atoms"]:
                rooms.setdefault(str(atom[1]), []).append(str(atom[0]))
            return {"rooms": {k: sorted(v) for k, v in sorted(rooms.items())}}
        if kind == "provenance_partition":
            prov: dict[str, list[int]] = {}
            for ci, c in enumerate(mat["subgoal"]):
                for a in _obligations(problem, (c,)):
                    prov.setdefault(textgen.atom_str(a), []).append(ci)
            return {"provenance": {
                k: sorted(v) for k, v in sorted(prov.items())}}
        if kind == "predicate_usage":
            init_preds = {str(a[0]) for a in
                          textgen.positive_init_atoms(problem)}
            goal_preds = {str(leaf[0]) for _p, leaf in
                          textgen.goal_leaf_positions(mat["subgoal"])}
            usage = {}
            for pred in sorted(init_preds | goal_preds):
                sections = []
                if pred in init_preds:
                    sections.append("init")
                if pred in goal_preds:
                    sections.append("goal")
                usage[pred] = sorted(sections)
            return {"usage": usage}
        raise ValueError("unknown 20k goal_kind: %s" % kind)

    # -------------------------------------------------------- verify_terminal

    def _verify(self, mat: dict[str, Any], submitted: Any) -> NativeVerdict:
        if not isinstance(submitted, dict):
            return NativeVerdict(False, {"error": "submission must be an "
                                            "object"})
        kind = mat["kind"]
        handler = getattr(self, "_verify_" + kind)
        return handler(mat, submitted)

    def _exact_or_fail(self, mat, submitted, key, want, leg_b_detail):
        got = submitted.get(key)
        if got != want:
            return NativeVerdict(False, {
                "error": "%s mismatch" % key, "expected": want,
                "received": got})
        return NativeVerdict(True, leg_b_detail)

    def _verify_quant_inventory(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["quantifiers"]
        got = submitted.get("quantifiers")
        if (not isinstance(got, list)
                or any(not isinstance(t, list) or len(t) != 3
                       or not all(isinstance(x, str) for x in t)
                       for t in got)):
            return NativeVerdict(False, {"error": "quantifiers must be a "
                                            "list of [head, var, category] "
                                            "string triples"})
        if got != want:
            return NativeVerdict(False, {
                "error": "quantifier inventory mismatch",
                "expected": want, "received": got})
        theirs = _legb(self._public_text(mat))
        b_sub = tuple(theirs.goal[i] for i in mat["params"]["conjuncts"])
        b_want = [_quant_triple(q) for q in _inventory_quants(b_sub)]
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) inventory disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "quantifiers": len(want), "leg_b_agrees": True})

    def _verify_shape_census(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["shapes"]
        got = submitted.get("shapes")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "shapes must be a "
                                            "non-empty mapping"})
        norm = {str(k): str(v) for k, v in got.items()}
        if norm != want:
            return NativeVerdict(False, {
                "error": "shape census mismatch",
                "expected": want, "received": norm})
        theirs = _legb(self._public_text(mat))
        b_sub = tuple(theirs.goal[i] for i in mat["params"]["conjuncts"])
        b_want = {str(p): _shape_tag(c) for p, c in enumerate(b_sub)}
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) shape census disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "conjuncts": len(want), "leg_b_agrees": True})

    def _verify_reference_closure(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["closure"]
        got = submitted.get("closure")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "closure must be a "
                                            "non-empty mapping"})
        norm = {str(k): sorted(v) for k, v in got.items()
                if isinstance(v, list)}
        if norm != want:
            return NativeVerdict(False, {
                "error": "reference closure mismatch",
                "expected": want, "received": norm})
        theirs = _legb(self._public_text(mat))
        b_sub = tuple(theirs.goal[i] for i in mat["params"]["conjuncts"])
        b_want = {str(p): sorted(_closure_refs(theirs, c))
                  for p, c in enumerate(b_sub)}
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) closure disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "conjuncts": len(want), "leg_b_agrees": True})

    def _verify_polarity_audit(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["negative_leaves"]
        got = submitted.get("negative_leaves")
        n = len(_polarity_flags(mat["subgoal"]))
        if (not isinstance(got, list)
                or any(not isinstance(i, int) or i < 0 or i >= n for i in got)
                or sorted(set(got)) != sorted(got)):
            return NativeVerdict(False, {"error": "negative_leaves must be "
                                            "a sorted list of indices"})
        if got != want:
            return NativeVerdict(False, {
                "error": "polarity set mismatch",
                "expected": want, "received": got})
        theirs = _legb(self._public_text(mat))
        b_sub = tuple(theirs.goal[i] for i in mat["params"]["conjuncts"])
        b_flags = _polarity_flags(b_sub)
        b_want = sorted(i for i, f in enumerate(b_flags) if f)
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) polarity set disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "leaves": n, "negative": len(want), "leg_b_agrees": True})

    def _verify_discriminating_state(self, mat, submitted) -> NativeVerdict:
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
        problem, tables = mat["problem"], mat["tables"]
        checked = _parsed(problem, mat["subgoal_a"], parsed_atoms)
        findings = check_problem(checked, tables)
        if findings:
            return NativeVerdict(False, {
                "error": "submitted atoms fail static checks",
                "findings": [[f.section, f.code, f.detail[:80]]
                             for f in findings[:5]]})
        truth = frozenset(parsed_atoms)
        a_ok = all(evalgoal.eval_node(c, {}, problem, truth)
                   for c in mat["subgoal_a"])
        b_false = not all(evalgoal.eval_node(c, {}, problem, truth)
                          for c in mat["subgoal_b"])
        if not (a_ok and b_false):
            return NativeVerdict(False, {
                "error": "state must satisfy sub-goal A and falsify "
                         "conjunct B",
                "a_satisfied": a_ok, "b_falsified": b_false})
        text_a = textgen.serialize_problem(
            "verify-ds-a", mat["obj_map"], parsed_atoms, mat["subgoal_a"])
        text_b = textgen.serialize_problem(
            "verify-ds-b", mat["obj_map"], parsed_atoms, mat["subgoal_b"])
        b_a = _upstream_eval_states(text_a, [truth])[0]
        b_b = _upstream_eval_states(text_b, [truth])[0]
        detail = {"atoms": len(atoms), "leg_b": [bool(b_a), bool(b_b)]}
        if not b_a or b_b:
            return NativeVerdict(False, dict(
                detail, error="leg B (vendored bddl) disagrees with leg A "
                "on the discriminating state"))
        return NativeVerdict(True, detail)

    def _verify_weakening_goal(self, mat, submitted) -> NativeVerdict:
        goal_text = submitted.get("goal_text")
        if not isinstance(goal_text, str) or not goal_text.strip():
            return NativeVerdict(False, {"error": "goal_text must be a "
                                            "non-empty string"})
        try:
            expr = engine.scan_tokens(goal_text)
        except engine.BddlParseError as exc:
            return NativeVerdict(False, {"error": "unparsable goal: %s" % exc})
        expr_t = tuple(engine.canon(expr))
        problem, tables = mat["problem"], mat["tables"]
        checked = _parsed(problem, (expr_t,), mat["s1"])
        findings = check_problem(checked, tables)
        if findings:
            return NativeVerdict(False, {
                "error": "submitted goal fails static checks",
                "findings": [[f.section, f.code, f.detail[:80]]
                             for f in findings[:5]]})
        truth1 = frozenset(mat["s1"])
        truth2 = frozenset(mat["s2"])
        empty = frozenset()
        a1 = evalgoal.eval_node(expr_t, {}, problem, truth1)
        a2 = evalgoal.eval_node(expr_t, {}, problem, truth2)
        a0 = evalgoal.eval_node(expr_t, {}, problem, empty)
        text = textgen.serialize_problem(
            "verify-wg", mat["obj_map"], mat["s1"], (expr_t,))
        b1, b2, b0 = _upstream_eval_states(text, [truth1, truth2, empty])
        detail = {"leg_a": [bool(a1), bool(a2), bool(a0)],
                  "leg_b": [bool(b1), bool(b2), bool(b0)]}
        if not (a1 and a2 and not a0):
            return NativeVerdict(False, dict(
                detail, error="goal must be satisfied under S1 AND S2 and "
                "falsified under the empty state"))
        if [bool(a1), bool(a2), bool(a0)] != [bool(b1), bool(b2), bool(b0)]:
            return NativeVerdict(False, dict(
                detail, error="leg B (vendored bddl) disagrees with leg A"))
        return NativeVerdict(True, detail)

    def _verify_contradiction_audit(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["contradicted"]
        got = submitted.get("contradicted")
        if (not isinstance(got, list)
                or any(not isinstance(x, str) for x in got)
                or sorted(set(got)) != sorted(got)):
            return NativeVerdict(False, {"error": "contradicted must be a "
                                            "sorted list of atom strings"})
        if got != want:
            return NativeVerdict(False, {
                "error": "contradiction set mismatch",
                "expected": want, "received": got})
        text = self._public_text(mat)
        theirs = _legb(text)
        b_pos = set()
        for lit in theirs.init:
            if isinstance(lit, tuple) and lit and lit[0] != "not":
                b_pos.add(lit)
        b_neg = set()
        for lit in theirs.init:
            if isinstance(lit, tuple) and lit and lit[0] == "not":
                b_neg.add(lit[1])
        b_want = sorted(textgen.atom_str(a) for a in b_pos & b_neg)
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) contradiction set disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "contradicted": len(want), "leg_b_agrees": True})

    def _verify_duplicate_audit(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["duplicated"]
        got = submitted.get("duplicated")
        if (not isinstance(got, list)
                or any(not isinstance(x, str) for x in got)
                or sorted(set(got)) != sorted(got)):
            return NativeVerdict(False, {"error": "duplicated must be a "
                                            "sorted list of atom strings"})
        if got != want:
            return NativeVerdict(False, {
                "error": "duplicate set mismatch",
                "expected": want, "received": got})
        theirs = _legb(self._public_text(mat))
        from collections import Counter
        b_counts = Counter(
            textgen.atom_str(a) for a in theirs.init
            if isinstance(a, tuple) and a and a[0] != "not")
        b_want = sorted(s for s, n in b_counts.items() if n > 1)
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) duplicate set disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "duplicated": len(want), "leg_b_agrees": True})

    def _verify_predicate_join(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["mapping"]
        got = submitted.get("mapping")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "mapping must be a "
                                            "non-empty object"})
        norm = {str(k): sorted(v) for k, v in got.items()
                if isinstance(v, list)}
        if norm != want:
            return NativeVerdict(False, {
                "error": "predicate join mapping mismatch",
                "expected_predicates": sorted(want),
                "received_predicates": sorted(norm)})
        # leg B: recompute from the public texts of both activities
        from benchmark.plugins.task_family.l10.propose_20k import (
            build_public_20k,
        )
        public = build_public_20k(mat["ast"])
        pa = _legb(public["task_material"]["problem_a_text"])
        pb = _legb(public["task_material"]["problem_b_text"])
        goal_preds = sorted({
            str(leaf[0]) for _p, leaf in
            textgen.goal_leaf_positions(tuple(pa.goal))})
        b_map = {}
        for pred in goal_preds:
            b_map[pred] = sorted(
                textgen.atom_str(a) for a in pb.init
                if isinstance(a, tuple) and a and a[0] != "not"
                and str(a[0]) == pred)
        if b_map != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) join mapping disagrees",
                "leg_a": {k: len(v) for k, v in want.items()},
                "leg_b": {k: len(v) for k, v in b_map.items()}})
        return NativeVerdict(True, {
            "predicates": len(want),
            "partner_atoms": sum(len(v) for v in want.values()),
            "leg_b_agrees": True})

    def _verify_room_grouping(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["rooms"]
        got = submitted.get("rooms")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "rooms must be a "
                                            "non-empty mapping"})
        norm = {str(k): sorted(v) for k, v in got.items()
                if isinstance(v, list)}
        if norm != want:
            return NativeVerdict(False, {
                "error": "room grouping mismatch",
                "expected_rooms": sorted(want),
                "received_rooms": sorted(norm)})
        text = textgen.serialize_problem(
            "verify-rooms", mat["obj_map"], mat["shown_atoms"],
            tuple(mat["goal"]))
        theirs = _legb(text)
        b_rooms: dict[str, list[str]] = {}
        for lit in theirs.init:
            if isinstance(lit, tuple) and lit and lit[0] == "inroom":
                b_rooms.setdefault(str(lit[1]), []).append(str(lit[0]))
        b_want = {k: sorted(v) for k, v in sorted(b_rooms.items())}
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) room grouping disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "rooms": len(want),
            "instances": sum(len(v) for v in want.values()),
            "leg_b_agrees": True})

    def _verify_provenance_partition(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["provenance"]
        got = submitted.get("provenance")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "provenance must be a "
                                            "non-empty mapping"})
        norm = {str(k): sorted(v) for k, v in got.items()
                if isinstance(v, list)}
        if norm != want:
            return NativeVerdict(False, {
                "error": "provenance partition mismatch",
                "expected_atoms": len(want), "received_atoms": len(norm)})
        # leg B: per-conjunct satisfaction + per-atom necessity (upstream)
        problem = mat["problem"]
        n_conj = len(mat["subgoal"])
        for ci in range(n_conj):
            atoms_ci = [tuple(_parse_atom_str(k)) for k, v in want.items()
                        if ci in v]
            text = textgen.serialize_problem(
                "verify-prov-%d" % ci, mat["obj_map"], atoms_ci,
                (mat["subgoal"][ci],))
            truth_sets = [frozenset(atoms_ci)] + [
                frozenset(atoms_ci[:i] + atoms_ci[i + 1:])
                for i in range(len(atoms_ci))]
            results = _upstream_eval_states(text, truth_sets)
            if not results[0]:
                return NativeVerdict(False, {
                    "error": "leg B: conjunct %d not satisfied over its "
                             "attributed atoms" % ci})
            if any(results[1:]):
                return NativeVerdict(False, {
                    "error": "leg B: some atom attributed to conjunct %d is "
                             "not necessary" % ci})
        text_all = textgen.serialize_problem(
            "verify-prov-all", mat["obj_map"],
            [tuple(_parse_atom_str(k)) for k in want],
            mat["subgoal"])
        ok_all = _upstream_eval_states(text_all, [
            frozenset(tuple(_parse_atom_str(k)) for k in want)])[0]
        if not ok_all:
            return NativeVerdict(False, {
                "error": "leg B: full partition does not satisfy the "
                         "sub-goal"})
        return NativeVerdict(True, {
            "atoms": len(want), "conjuncts": n_conj,
            "leg_b_agrees": True})

    def _verify_predicate_usage(self, mat, submitted) -> NativeVerdict:
        want = self.reference_submission(mat["ast"], mat)["usage"]
        got = submitted.get("usage")
        if not isinstance(got, dict) or not got:
            return NativeVerdict(False, {"error": "usage must be a "
                                            "non-empty mapping"})
        norm = {str(k): sorted(v) for k, v in got.items()
                if isinstance(v, list)}
        if norm != want:
            return NativeVerdict(False, {
                "error": "predicate usage matrix mismatch",
                "expected_predicates": sorted(want),
                "received_predicates": sorted(norm)})
        theirs = _legb(self._public_text(mat))
        b_init = {str(a[0]) for a in theirs.init
                  if isinstance(a, tuple) and a and a[0] != "not"}
        b_sub = tuple(theirs.goal[i] for i in mat["params"]["conjuncts"])
        b_goal = {str(leaf[0]) for _p, leaf in
                  textgen.goal_leaf_positions(b_sub)}
        b_want = {}
        for pred in sorted(b_init | b_goal):
            sections = []
            if pred in b_init:
                sections.append("init")
            if pred in b_goal:
                sections.append("goal")
            b_want[pred] = sorted(sections)
        if b_want != want:
            return NativeVerdict(False, {
                "error": "leg B (vendored bddl) usage matrix disagrees",
                "leg_a": want, "leg_b": b_want})
        return NativeVerdict(True, {
            "predicates": len(want), "leg_b_agrees": True})

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
                % _json(term.detail)[:160])

        problem = mat["problem"]
        if kind == "quant_inventory":
            quants = reference["quantifiers"]
            ev.update(quantifiers=len(quants))
            if len(quants) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_quantifiers")
            return ProofOrCounterexample(True, ev)
        if kind == "shape_census":
            tags = set(reference["shapes"].values())
            ev.update(conjuncts=len(reference["shapes"]),
                      distinct_shapes=len(tags))
            if len(tags) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_shapes")
            return ProofOrCounterexample(True, ev)
        if kind == "reference_closure":
            closures = [tuple(v) for v in reference["closure"].values()]
            ev.update(conjuncts=len(closures),
                      distinct_closures=len(set(closures)))
            if len(closures) < 2 or len(set(closures)) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:uniform_reference_closure")
            return ProofOrCounterexample(True, ev)
        if kind == "polarity_audit":
            flags = _polarity_flags(mat["subgoal"])
            ev.update(leaves=len(flags), negative=sum(flags))
            if not flags or not any(flags) or not any(
                    not f for f in flags):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:uniform_polarity")
            return ProofOrCounterexample(True, ev)
        if kind == "discriminating_state":
            if not mat["obligations_a"]:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:empty_obligations_a")
            truth_obl = frozenset(mat["obligations_a"])
            b_false = not all(evalgoal.eval_node(
                c, {}, problem, truth_obl) for c in mat["subgoal_b"])
            truth_init = evalgoal.init_truth_set(problem)
            a_init = all(evalgoal.eval_node(
                c, {}, problem, truth_init) for c in mat["subgoal_a"])
            ev.update(obligations_a=len(mat["obligations_a"]),
                      b_falsified_over_reference=b_false,
                      a_satisfied_over_init=a_init)
            if not b_false:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:b_satisfied_over_reference")
            if a_init:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:a_already_satisfied_over_init")
            return ProofOrCounterexample(True, ev)
        if kind == "weakening_goal":
            truth1 = frozenset(mat["s1"])
            truth2 = frozenset(mat["s2"])
            g1 = all(evalgoal.eval_node(c, {}, problem, truth1)
                     for c in mat["subgoal"])
            g2 = all(evalgoal.eval_node(c, {}, problem, truth2)
                     for c in mat["subgoal"])
            ev.update(reference_true_s1=g1, reference_true_s2=g2)
            if not (g1 and not g2):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:reference_goal_does_not_separate")
            return ProofOrCounterexample(True, ev)
        if kind == "contradiction_audit":
            contradicted = reference["contradicted"]
            orig = mat["problem"]
            pos = set(textgen.positive_init_atoms(orig))
            neg = set()
            for lit in orig.init:
                if isinstance(lit, tuple) and lit and lit[0] == "not":
                    neg.add(lit[1])
            ev.update(contradicted=len(contradicted),
                      original_contradictions=len(pos & neg))
            if len(contradicted) != 1:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:contradiction_count_%d"
                    % len(contradicted))
            return ProofOrCounterexample(True, ev)
        if kind == "duplicate_audit":
            duplicated = reference["duplicated"]
            n_pos = len(textgen.positive_init_atoms(
                _parsed(problem, mat["goal"], mat["perturbed_init"])))
            ev.update(duplicated=len(duplicated), positive_atoms=n_pos)
            if not duplicated:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:no_duplicates")
            if len(duplicated) >= n_pos:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:all_atoms_duplicated")
            return ProofOrCounterexample(True, ev)
        if kind == "predicate_join":
            mapping = reference["mapping"]
            non_empty = [k for k, v in mapping.items() if v]
            ev.update(predicates=len(mapping),
                      non_empty_predicates=len(non_empty))
            if len(mapping) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_predicates")
            if not non_empty:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:all_join_lists_empty")
            return ProofOrCounterexample(True, ev)
        if kind == "room_grouping":
            rooms = reference["rooms"]
            ev.update(rooms=len(rooms),
                      instances=sum(len(v) for v in rooms.values()))
            if len(rooms) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_rooms")
            if sum(len(v) for v in rooms.values()) < 3:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_3_instances")
            return ProofOrCounterexample(True, ev)
        if kind == "provenance_partition":
            prov = reference["provenance"]
            conjuncts = {ci for v in prov.values() for ci in v}
            ev.update(atoms=len(prov), conjuncts=len(conjuncts))
            if len(prov) < 3:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_3_atoms")
            if len(conjuncts) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_2_generating_conjuncts")
            return ProofOrCounterexample(True, ev)
        if kind == "predicate_usage":
            usage = reference["usage"]
            tags = {tuple(v) for v in usage.values()}
            ev.update(predicates=len(usage), distinct_tags=len(tags))
            if len(usage) < 3:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:fewer_than_3_predicates")
            if len(tags) < 2:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:uniform_usage")
            return ProofOrCounterexample(True, ev)
        return ProofOrCounterexample(
            False, ev, "PROFILE_UNSUPPORTED:kind=%s" % kind)

    # ------------------------------------------------- certify_semantic_relev

    def _wrong_submission(self, mat: dict[str, Any]) -> dict[str, Any]:
        kind = mat["kind"]
        ref = self.reference_submission(mat["ast"], mat)
        if kind == "quant_inventory":
            wrong = [list(t) for t in ref["quantifiers"]]
            wrong[0][1] = wrong[0][1] + "x"
            return {"quantifiers": wrong}
        if kind == "shape_census":
            wrong = dict(ref["shapes"])
            k = sorted(wrong)[0]
            wrong[k] = "atom" if wrong[k] != "atom" else "and"
            return {"shapes": wrong}
        if kind == "reference_closure":
            wrong = {k: list(v) for k, v in ref["closure"].items()}
            k = sorted(wrong)[0]
            if wrong[k]:
                wrong[k] = wrong[k][:-1]
            else:
                wrong[k] = ["nonexistent.n.01_9"]
            return {"closure": wrong}
        if kind == "polarity_audit":
            idxs = ref["negative_leaves"]
            n = len(_polarity_flags(mat["subgoal"]))
            if idxs:
                return {"negative_leaves": idxs[:-1]}
            return {"negative_leaves": [0]}
        if kind == "discriminating_state":
            exclusive = sorted(
                set(mat["obligations_b"]) - set(mat["obligations_a"]))
            atoms = list(ref["atoms"])
            if exclusive:
                atoms = sorted(set(atoms) | {
                    textgen.atom_str(exclusive[0])})
            elif atoms:
                atoms = atoms[:-1]
            return {"atoms": atoms}
        if kind == "weakening_goal":
            return {"goal_text": textgen.sexpr(
                ("and",) + tuple(mat["subgoal"]))}
        if kind == "contradiction_audit":
            pos = [textgen.atom_str(a) for a in
                   textgen.positive_init_atoms(mat["problem"])]
            alt = next((s for s in pos if s not in ref["contradicted"]),
                       pos[0] if pos else "(ontop a b)")
            return {"contradicted": [alt]}
        if kind == "duplicate_audit":
            pos = [textgen.atom_str(a) for a in
                   textgen.positive_init_atoms(mat["problem"])]
            alt = next((s for s in pos if s not in ref["duplicated"]),
                       pos[0] if pos else "(ontop a b)")
            return {"duplicated": sorted(set(ref["duplicated"]) | {alt})}
        if kind == "predicate_join":
            wrong = {k: list(v) for k, v in ref["mapping"].items()}
            for k in sorted(wrong):
                if wrong[k]:
                    wrong[k] = wrong[k][:-1]
                    break
            else:
                wrong[sorted(wrong)[0]] = ["(ontop a b)"]
            return {"mapping": wrong}
        if kind == "room_grouping":
            wrong = {k: list(v) for k, v in ref["rooms"].items()}
            keys = sorted(wrong)
            if len(keys) >= 2 and wrong[keys[0]]:
                inst = wrong[keys[0]][0]
                wrong[keys[0]] = [x for x in wrong[keys[0]] if x != inst]
                wrong[keys[1]] = sorted(wrong[keys[1]] + [inst])
            elif wrong.get(keys[0]):
                wrong[keys[0]] = wrong[keys[0]][:-1]
            return {"rooms": wrong}
        if kind == "provenance_partition":
            wrong = {k: list(v) for k, v in ref["provenance"].items()}
            k = sorted(wrong)[0]
            if len(wrong[k]) > 1:
                wrong[k] = wrong[k][:-1]
            else:
                wrong[k] = wrong[k] + [wrong[k][0] + 1]
            return {"provenance": wrong}
        if kind == "predicate_usage":
            wrong = {k: list(v) for k, v in ref["usage"].items()}
            k = sorted(wrong)[0]
            wrong[k] = sorted(set(wrong[k]) | {"init"}) \
                if "init" not in wrong[k] else [x for x in wrong[k]
                                                if x != "init"]
            return {"usage": wrong}
        raise ValueError("unknown 20k kind %s" % kind)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, default=str)


PLUGIN = BddlVerifier20K()

__all__ = ["BddlVerifier20K", "PLUGIN", "ENGINE", "SOURCE_SNAPSHOT_ID",
           "NEW_KINDS"]
