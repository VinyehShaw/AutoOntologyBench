"""L10 public payload rendering (task inputs only; never gold)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l10 import engine, evalgoal, textgen
from benchmark.plugins.task_family.l10.engine import ParsedProblem, sha256_json
from benchmark.plugins.task_family.l10.families import (
    ALLOWED_TOOLS,
    BUDGETS,
    FAMILY_BY_ID,
    SYMBOLIC_ONLY_NOTE,
)

_PROBLEM_CACHE: dict[str, tuple[ParsedProblem, str]] = {}


def load_problem(problem_rel: str) -> tuple[ParsedProblem, str]:
    if problem_rel not in _PROBLEM_CACHE:
        path = engine.PINNED_ROOT / problem_rel
        text = path.read_text(encoding="utf-8")
        _PROBLEM_CACHE[problem_rel] = (engine.parse_problem_text(text), text)
    return _PROBLEM_CACHE[problem_rel]


def _domain_predicates() -> dict[str, int]:
    return dict(engine.load_tables().predicate_map)


def _observation_refs(ast: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "path": "external_knowledge/pinned/behavior-1k-bddl/v3.9.3-post2/%s"
                    % ast["problem_rel"],
            "sha256": ast["problem_digest"],
        },
        {
            "path": "external_knowledge/pinned/behavior-1k-bddl/v3.9.3-post2/"
                    "bddl3/bddl/activity_definitions/domain_behavior-1k.bddl",
            "sha256": engine.load_tables().domain_file_sha256,
        },
    ]


def _base(ast: dict[str, Any], user_goal: str) -> dict[str, Any]:
    spec = FAMILY_BY_ID[ast["family_id"]]
    return {
        "user_goal": user_goal,
        "allowed_tools": list(ALLOWED_TOOLS[spec.regime]),
        "budget": dict(BUDGETS[ast["complexity"]]),
        "observation_refs": _observation_refs(ast),
        "semantic_scope": SYMBOLIC_ONLY_NOTE,
    }


def _subgoal(problem: ParsedProblem, indices: list[int]) -> tuple:
    return tuple(problem.goal[i] for i in indices)


def render_public(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["goal_kind"]
    params = ast["params"]
    problem, _text = load_problem(ast["problem_rel"])
    obj_map, inits, goal = textgen.problem_parts(problem)
    name = ast["activity"] + "-%d" % ast["variant"]

    if kind in ("repair_predicate", "repair_arity", "repair_binding",
                "repair_init_predicate", "repair_quantifier_category",
                "defect_audit"):
        c_obj, c_init, c_goal = textgen.corrupt_parts(problem, kind, params)
        corrupted = textgen.serialize_problem(name, c_obj, c_init, c_goal)
        if kind == "defect_audit":
            user_goal = (
                "The BDDL problem below has multiple injected structural defects. "
                "Run the domain's static checks (predicate declarations, arity, "
                "argument binding, init groundness, quantifier shape) and return "
                "the exact multiset of defects as (section, code) pairs. %s"
                % SYMBOLIC_ONLY_NOTE)
            defect_note = {
                "defect_count": len(params["defects"]),
                "defect_kinds": sorted({d["kind"] for d in params["defects"]}),
            }
            public = _base(ast, user_goal)
            public["task_material"] = {
                "problem_text": corrupted,
                "defects": defect_note,
                "domain_predicates": _domain_predicates(),
            }
            return public
        section = {"repair_predicate": "goal", "repair_arity": "goal",
                   "repair_binding": "goal", "repair_init_predicate": "init",
                   "repair_quantifier_category": "goal"}[kind]
        defect = dict(params)
        defect["section"] = section
        defect.pop("conjuncts", None)
        user_goal = (
            "The BDDL problem below has exactly one injected structural defect in "
            "its %s section. Identify the corrupted token and submit the correct "
            "replacement fragment (a single token string) so the problem passes "
            "every domain check and matches the activity's intended semantics. %s"
            % (section, SYMBOLIC_ONLY_NOTE))
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": corrupted,
            "defect": defect,
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "obligation_query":
        sub = _subgoal(problem, params["conjuncts"])
        user_goal = (
            "For the BDDL problem below, consider only the goal conjuncts at "
            "indices %s. Compute the exact set of ground atoms those conjuncts "
            "obligate: every atom that must hold (closed-world) for the sub-goal "
            "to be satisfied. Submit the sorted atom list. %s"
            % (params["conjuncts"], SYMBOLIC_ONLY_NOTE))
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": textgen.serialize_problem(
                name, obj_map, inits, tuple(problem.goal)),
            "subgoal_conjuncts": list(params["conjuncts"]),
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "conjunct_truth_audit":
        add = params["add_atoms"]
        perturbed = list(inits)
        removed_note = None
        if add and add[0].startswith("@remove:"):
            j = int(add[0].split(":", 1)[1])
            pos = textgen.positive_init_positions(problem)
            perturbed.pop(pos[j][0])
            removed_note = True
        else:
            for atom_str in add:
                perturbed.append(tuple(engine.scan_tokens(atom_str)))
        text = textgen.serialize_problem(
            name, obj_map, perturbed, tuple(problem.goal))
        user_goal = (
            "For the BDDL problem below, audit every goal conjunct under the "
            "closed-world initial state shown. Submit the exact sorted list of "
            "conjunct indices (0-based, in :goal order) that are already "
            "satisfied. %s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        material = {
            "problem_text": text,
            "conjunct_count": len(problem.goal),
        }
        if removed_note is not None:
            material["perturbation"] = "one init atom removed"
        else:
            material["perturbation"] = "%d atoms added to init" % len(add)
        public["task_material"] = material
        return public

    if kind == "init_grouping":
        pos = textgen.positive_init_positions(problem)
        keep = set(range(len(pos))) - set(params["remove_init"])
        perturbed = [atom for k, (_i, atom) in enumerate(pos) if k in keep]
        rng = random.Random(sha256_json([ast["variant_key"], "shuffle"])[:16])
        shuffled = list(perturbed)
        rng.shuffle(shuffled)
        user_goal = (
            "The list below is the shuffled initial-state atom set of a BDDL "
            "problem (one atom removed). Organize it by relation type: submit "
            "the exact mapping predicate -> sorted list of atom strings. %s"
            % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "init_atoms": [textgen.atom_str(a) for a in shuffled],
            "objects": {c: list(ii) for c, ii in obj_map.items()},
        }
        return public

    if kind == "objects_audit":
        pairs = [(inst, cat) for cat, insts in obj_map.items() for inst in insts]
        for i, j in params["swaps"]:
            pairs[i], pairs[j] = (
                (pairs[i][0], pairs[j][1]), (pairs[j][0], pairs[i][1]))
        rng = random.Random(sha256_json([ast["variant_key"], "objshuffle"])[:16])
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        user_goal = (
            "The shuffled (instance, category) declarations below come from a "
            "BDDL problem, but some categories were cross-swapped between "
            "instances. BDDL convention: a declared instance is named "
            "<category>_<index>. Submit the exact sorted list of instances whose "
            "declared category violates this convention. %s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "object_declarations": [
                {"instance": inst, "category": cat} for inst, cat in shuffled],
        }
        return public

    if kind == "goal_construction":
        sub = _subgoal(problem, params["conjuncts"])
        try:
            obligations = evalgoal.obligation_atoms(
                _problem_with_goal(problem, sub))
        except evalgoal.EvalError:
            obligations = []
        s2 = textgen.positive_init_atoms(problem)
        s1 = s2 + list(obligations)
        user_goal = (
            "Using only the domain predicates and the declared objects below, "
            "construct a BDDL goal expression that is satisfied under the "
            "closed-world state S1 but NOT under the state S2. Submit the goal "
            "expression text (e.g. \"(and (predicate obj1 obj2))\"). %s"
            % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "objects": {c: list(ii) for c, ii in obj_map.items()},
            "state_s1": [textgen.atom_str(a) for a in s1],
            "state_s2": [textgen.atom_str(a) for a in s2],
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "minimal_state":
        sub = _subgoal(problem, params["conjuncts"])
        goal_text = textgen.serialize_problem(
            name, obj_map, inits, sub)
        user_goal = (
            "Construct a minimal set of ground atoms over the declared objects "
            "that satisfies the goal in the problem below (closed-world). "
            "Minimality: removing any single atom must break satisfaction. "
            "Submit the sorted atom list. %s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": goal_text,
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "category_join":
        partner_rel = "bddl3/bddl/activity_definitions/%s/problem0.bddl" % (
            params["partner_activity"])
        partner, _ptext = load_problem(partner_rel)
        user_goal = (
            "The goal below references object instances of activity A. Join "
            "each distinct referenced category against activity B's object "
            "declarations: submit the exact mapping category -> sorted list of "
            "B's instances of that category (empty list if B declares none). %s"
            % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "goal_text": textgen.serialize_problem(
                name, obj_map, inits, tuple(problem.goal)),
            "objects_a": {c: list(ii) for c, ii in obj_map.items()},
            "objects_b": {c: list(ii) for c, ii in partner.object_map.items()},
        }
        return public

    raise ValueError("unknown goal_kind: %s" % kind)


def _problem_with_goal(problem: ParsedProblem, goal: tuple) -> ParsedProblem:
    from benchmark.plugins.task_family.l10.engine import ParsedProblem as PP
    return PP(name=problem.name, domain=problem.domain, objects=problem.objects,
              init=problem.init, goal=goal, goal_raw=goal)
