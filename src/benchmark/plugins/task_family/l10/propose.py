"""L10 candidate generation (propose) + grounding for the 13 BDDL families.

Generation is solver-grounded: every corruption/perturbation parameter is
chosen so that the pinned original problem provably supports the task (enough
goal leaves / refs / quantifiers / conjuncts / objects), and the independent
verifier re-proves everything from the pinned snapshot bytes. The generator
never writes gold into public payloads; ``ground`` emits only task inputs.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l10 import engine, evalgoal, textgen
from benchmark.plugins.task_family.l10.checks import check_problem, complexity_of
from benchmark.plugins.task_family.l10.engine import ParsedProblem, sha256_json
from benchmark.plugins.task_family.l10.families import (
    ALLOWED_TOOLS,
    BUDGETS,
    FAMILIES,
    FAMILY_BY_ID,
    INSTANCES_PER_GOAL,
    LANE_ID,
    SEMANTIC_PROFILE,
    SYMBOLIC_ONLY_NOTE,
    FamilySpec,
)

SEED_SALT = 20261027

_DOMAIN_UNIT = (
    "behavior-1k-bddl/bddl3/bddl/activity_definitions/domain_behavior-1k.bddl"
)


def _ids(family_id: str, base_key: Any, inst_key: Any) -> tuple[str, str]:
    base_goal_id = "bg-" + sha256_json({"f": family_id, "b": base_key})[:20]
    instance_id = "in-" + sha256_json({"b": base_goal_id, "i": inst_key})[:20]
    return base_goal_id, instance_id


def _rng(*parts: Any) -> random.Random:
    return random.Random(sha256_json(list(parts))[:16])


# ------------------------------------------------------------- anchor loading

@lru_cache(maxsize=1)
def _all_problems() -> tuple[tuple[Path, ParsedProblem], ...]:
    out = []
    for path in engine.iter_activity_paths():
        out.append((path, engine.load_activity(path)))
    return tuple(out)


@lru_cache(maxsize=1)
def _clean_problems() -> tuple[tuple[Path, ParsedProblem], ...]:
    tables = engine.load_tables()
    return tuple(
        (p, prob) for p, prob in _all_problems()
        if not check_problem(prob, tables)
    )


def _restricted_goal(problem: ParsedProblem) -> bool:
    """Goal built only from and/forall over leaf atoms (unique obligations)."""
    tables = engine.load_tables()
    pm = tables.predicate_map

    def rest(node: Any) -> bool:
        if not isinstance(node, tuple) or not node:
            return False
        h = node[0]
        if h == "and":
            return all(rest(x) for x in node[1:])
        if h == "forall":
            return len(node) == 3 and rest(node[2])
        return h in pm

    return all(rest(c) for c in problem.goal)


def _goal_refs(problem: ParsedProblem) -> list[str]:
    return [leaf[ai]
            for _path, leaf in textgen.goal_leaf_positions(tuple(problem.goal))
            for ai in range(1, len(leaf))
            if isinstance(leaf[ai], str) and leaf[ai].startswith("?")]


def _goal_categories(problem: ParsedProblem) -> list[str]:
    inst_cat = problem.instance_category
    cats: list[str] = []
    for ref in _goal_refs(problem):
        cat = inst_cat.get(ref[1:])
        if cat is not None and cat not in cats:
            cats.append(cat)
    return cats


def _rotate(items: list, n: int) -> list:
    """Deterministic rotation so families do not share anchor prefixes."""
    return items[n % len(items):] + items[: n % len(items)] if items else items


# ------------------------------------------------------- variant construction

def _subgoal_specs(problem: ParsedProblem, count: int) -> list[list[int]]:
    """Distinct conjunct-index lists: full goal + drop-one variants."""
    n = len(problem.goal)
    if n < 2:
        return []
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for idxs in ([list(range(n))] + [list(range(n))[:i] + list(range(n))[i + 1:]
                                     for i in range(n)]):
        key = tuple(idxs)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(idxs))
        if len(out) >= count:
            break
    return out


def _unique_leaf_positions(goal: tuple) -> list[int]:
    """Indices of goal leaves whose serialized form is unique in the goal.

    Corrupting a leaf with an identical twin would leave the gold expression
    verbatim in the public payload (leakage); those positions are unsafe.
    """
    leaves = textgen.goal_leaf_positions(tuple(goal))
    counts: dict[str, int] = {}
    for _p, leaf in leaves:
        s = textgen.sexpr(leaf)
        counts[s] = counts.get(s, 0) + 1
    return [i for i, (_p, leaf) in enumerate(leaves)
            if counts[textgen.sexpr(leaf)] == 1]


def _unique_leaf_paths(goal: tuple) -> set[tuple]:
    leaves = textgen.goal_leaf_positions(tuple(goal))
    counts: dict[str, int] = {}
    for _p, leaf in leaves:
        s = textgen.sexpr(leaf)
        counts[s] = counts.get(s, 0) + 1
    return {p for p, leaf in leaves if counts[textgen.sexpr(leaf)] == 1}


def _unique_init_positions(problem: ParsedProblem) -> list[int]:
    """Indices (into positive init atoms) unique in :init and absent from :goal."""
    pos = textgen.positive_init_positions(problem)
    goal_blob = " ".join(
        textgen.sexpr(leaf) for _p, leaf in
        textgen.goal_leaf_positions(tuple(problem.goal)))
    counts: dict[str, int] = {}
    for _i, atom in pos:
        s = textgen.sexpr(atom)
        counts[s] = counts.get(s, 0) + 1
    return [k for k, (_i, atom) in enumerate(pos)
            if counts[textgen.sexpr(atom)] == 1
            and textgen.sexpr(atom) not in goal_blob]


def _variant_params(spec: FamilySpec, path: Path, problem: ParsedProblem,
                    variant: int) -> dict[str, Any] | None:
    """Family-specific instance parameters, or None if unsupported."""
    kind = spec.goal_kind
    leaves = textgen.goal_leaf_positions(tuple(problem.goal))
    inits = textgen.positive_init_atoms(problem)
    if kind == "repair_predicate":
        safe = _unique_leaf_positions(tuple(problem.goal))
        if len(safe) < INSTANCES_PER_GOAL:
            return None
        i = safe[variant % len(safe)]
        return {"leaf_index": i, "corrupt_token": "predx%d" % variant}
    if kind == "repair_arity":
        safe = _unique_leaf_positions(tuple(problem.goal))
        if len(safe) < INSTANCES_PER_GOAL:
            return None
        i = safe[variant % len(safe)]
        return {"leaf_index": i}
    if kind == "repair_binding":
        refs = textgen.ref_positions(tuple(problem.goal))
        unique_paths = _unique_leaf_paths(tuple(problem.goal))
        safe_refs = [(lp, ai) for lp, ai in refs if lp in unique_paths]
        if len(safe_refs) < INSTANCES_PER_GOAL:
            return None
        leaf_path, arg_index = safe_refs[variant % len(safe_refs)]
        leaf_i = next(
            k for k, (p, _l) in enumerate(leaves) if p == leaf_path)
        return {"leaf_index": leaf_i, "arg_index": arg_index,
                "suffix": "x%d" % variant}
    if kind == "repair_init_predicate":
        safe = _unique_init_positions(problem)
        if len(safe) < INSTANCES_PER_GOAL:
            return None
        return {"init_index": safe[variant % len(safe)],
                "corrupt_token": "initx%d" % variant}
    if kind == "repair_quantifier_category":
        quants = textgen.quantifier_positions(tuple(problem.goal))
        if not quants:
            return None
        q = variant % len(quants)
        return {"quant_index": q, "corrupt_token": "qx%d" % variant}
    if kind in ("obligation_query", "goal_construction", "minimal_state"):
        if not _restricted_goal(problem):
            return None
        if evalgoal.goal_satisfied(problem):
            return None
        specs = _subgoal_specs(problem, INSTANCES_PER_GOAL)
        if len(specs) < INSTANCES_PER_GOAL:
            return None
        return {"conjuncts": specs[variant]}
    if kind == "conjunct_truth_audit":
        if not _restricted_goal(problem) or len(problem.goal) < 2:
            return None
        if evalgoal.goal_satisfied(problem):
            return None
        try:
            obligations = [textgen.atom_str(a)
                           for a in evalgoal.obligation_atoms(problem)]
        except evalgoal.EvalError:
            return None
        if not obligations:
            return None
        # cumulative additions of obligation atoms, then init removals
        cands: list[list[str]] = []
        for j in range(1, min(5, len(obligations)) + 1):
            cands.append(obligations[:j])
        for j in range(min(5, len(inits))):
            cands.append(["@remove:%d" % j])
        seen: set[tuple[str, ...]] = set()
        out: list[list[str]] = []
        for cand in cands:
            key = tuple(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) >= INSTANCES_PER_GOAL:
                break
        if len(out) < INSTANCES_PER_GOAL:
            return None
        return {"add_atoms": out[variant]}
    if kind == "init_grouping":
        if len(inits) < INSTANCES_PER_GOAL:
            return None
        return {"remove_init": [variant % len(inits)]}
    if kind == "objects_audit":
        insts = [i for _c, ii in problem.objects for i in ii]
        if len(insts) < 6:
            return None
        cat_of = problem.instance_category
        partners = [j for j in range(1, len(insts))
                    if cat_of.get(insts[j]) != cat_of.get(insts[0])]
        if len(partners) < INSTANCES_PER_GOAL:
            return None
        return {"swaps": [[0, partners[variant % len(partners)]]]}
    if kind == "category_join":
        cats = _goal_categories(problem)
        if len(cats) < 2:
            return None
        partners = _partners_for(path.parent.name, cats)
        if len(partners) < INSTANCES_PER_GOAL:
            return None
        partner = partners[variant % len(partners)]
        return {"partner_activity": partner}
    if kind == "defect_audit":
        refs = textgen.ref_positions(tuple(problem.goal))
        if len(leaves) < 3 or len(refs) < 2 or len(inits) < 2:
            return None
        combos = [
            [("predicate", "goal", 0), ("arity", "goal", 1)],
            [("binding", "goal", 0), ("init_predicate", "init", 0)],
            [("predicate", "goal", 0), ("arity", "goal", 1), ("binding", "goal", 0)],
            [("init_predicate", "init", 0), ("predicate", "goal", 0),
             ("arity", "goal", 2 % len(leaves))],
            [("predicate", "goal", 1), ("binding", "goal", 1 % len(refs)),
             ("init_predicate", "init", 1), ("arity", "goal", 0)],
        ]
        combo = combos[variant % len(combos)]
        return {"defects": [
            {"kind": k, "section": s, "index": i} for k, s, i in combo]}
    return None


@lru_cache(maxsize=1)
def _partner_index() -> dict[str, list[str]]:
    """activity -> sorted list of other activities sharing >=1 object category."""
    out: dict[str, list[str]] = {}
    acts = [(p.parent.name, set(prob.object_map)) for p, prob in _clean_problems()]
    for name, cats in acts:
        partners = sorted(
            oname for oname, ocats in acts
            if oname != name and cats & ocats)
        out[name] = partners
    return out


def _partners_for(activity: str, cats: list[str]) -> list[str]:
    idx = _partner_index()
    return [p for p in idx.get(activity, []) if _shares(p, set(cats))]


def _shares(activity: str, cats: set[str]) -> bool:
    for p, prob in _clean_problems():
        if p.parent.name == activity:
            return bool(cats & set(prob.object_map))
    return False


# ------------------------------------------------------------------ proposal

def _anchor_pool(spec: FamilySpec) -> list[tuple[Path, ParsedProblem]]:
    kind = spec.goal_kind
    pool = list(_clean_problems())
    if kind in ("repair_predicate", "repair_arity"):
        pool = [(p, prob) for p, prob in pool
                if len(_unique_leaf_positions(tuple(prob.goal)))
                >= INSTANCES_PER_GOAL]
    elif kind == "repair_binding":
        pool = [(p, prob) for p, prob in pool
                if len([1 for lp, _ai in textgen.ref_positions(
                    tuple(prob.goal))
                    if lp in _unique_leaf_paths(tuple(prob.goal))])
                >= INSTANCES_PER_GOAL]
    elif kind == "repair_init_predicate":
        pool = [(p, prob) for p, prob in pool
                if len(_unique_init_positions(prob)) >= INSTANCES_PER_GOAL]
    elif kind == "repair_quantifier_category":
        pool = [(p, prob) for p, prob in pool
                if textgen.quantifier_positions(tuple(prob.goal))]
        pool.sort(key=lambda ap: -len(
            textgen.quantifier_positions(tuple(ap[1].goal))))
    elif kind in ("obligation_query", "goal_construction", "minimal_state"):
        pool = [(p, prob) for p, prob in pool
                if _restricted_goal(prob) and len(prob.goal) >= 4
                and not evalgoal.goal_satisfied(prob)]
    elif kind == "conjunct_truth_audit":
        pool = [(p, prob) for p, prob in pool
                if _restricted_goal(prob) and len(prob.goal) >= 2
                and not evalgoal.goal_satisfied(prob)]
    elif kind == "init_grouping":
        pool = [(p, prob) for p, prob in pool
                if len(textgen.positive_init_atoms(prob)) >= INSTANCES_PER_GOAL]
    elif kind == "objects_audit":
        pool = [(p, prob) for p, prob in pool
                if len([i for _c, ii in prob.objects for i in ii]) >= 6]
    elif kind == "category_join":
        pool = [(p, prob) for p, prob in pool
                if len(_goal_categories(prob)) >= 2]
    elif kind == "defect_audit":
        pool = [(p, prob) for p, prob in pool
                if len(textgen.goal_leaf_positions(tuple(prob.goal))) >= 3
                and len(textgen.ref_positions(tuple(prob.goal))) >= 2
                and len(textgen.positive_init_atoms(prob)) >= 2]
    return _rotate(pool, sum(ord(c) for c in spec.family_id))


class BehaviorBddlTaskFamilies:
    """TaskFamilyPlugin for lane L10 (BEHAVIOR-1K BDDL symbolic)."""

    plugin_id = "l10.behavior-bddl-families"

    def __init__(self) -> None:
        self._tables = None

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        spec = FAMILY_BY_ID[family]
        pool = _anchor_pool(spec)
        base_quota = int(constraints.get(
            "base_goals", spec.base_goals))
        per_goal = int(constraints.get(
            "instances_per_goal", INSTANCES_PER_GOAL))
        made_bases = 0
        for path, problem in pool:
            if made_bases >= base_quota:
                break
            # all variants of one anchor share the base goal
            variants: list[dict[str, Any]] = []
            for v in range(per_goal):
                params = _variant_params(spec, path, problem, v)
                if params is None:
                    variants = []
                    break
                variants.append(params)
            if len(variants) < per_goal:
                continue
            base_key = str(path.relative_to(engine.PINNED_ROOT))
            base_goal_id, _ = _ids(family, base_key, 0)
            ok = True
            cands: list[CandidateTaskAST] = []
            for v, params in enumerate(variants):
                inst_key = {"v": v, "params": params}
                _bg, instance_id = _ids(family, base_key, inst_key)
                if instance_id in {c.instance_id for c in cands}:
                    ok = False
                    break
                digest = engine_sha256(path)
                ast: dict[str, Any] = {
                    "family_id": family,
                    "goal_kind": spec.goal_kind,
                    "activity": path.parent.name,
                    "problem_rel": base_key,
                    "problem_digest": digest,
                    "variant": v,
                    "variant_key": sha256_json(inst_key)[:12],
                    "complexity": complexity_of(problem),
                    "params": params,
                }
                cands.append(CandidateTaskAST(
                    lane_id=LANE_ID, family_id=family,
                    base_goal_id=base_goal_id, instance_id=instance_id,
                    regime=spec.regime, semantic_profile=SEMANTIC_PROFILE,
                    ast=ast,
                    source_unit_ids=_source_units(spec, path),
                ))
            if not ok or len(cands) != per_goal:
                continue
            made_bases += 1
            yield from cands

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        oracle_ref = "l10-oracle-%s" % sha256_json(
            [candidate.instance_id, ast["variant_key"]])[:16]
        public = build_public(ast)
        return GroundedTask(ast=candidate, public=public,
                            oracle_ref=oracle_ref, native_evidence={})


def engine_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _source_units(spec: FamilySpec, path: Path) -> tuple[str, ...]:
    units = ["behavior-1k-bddl/%s" % str(path.relative_to(engine.PINNED_ROOT)),
             _DOMAIN_UNIT]
    return tuple(units)


# -------------------------------------------------------------- public build

def build_public(ast: dict[str, Any]) -> dict[str, Any]:
    """Public task payload from ast params (recomputed; never contains gold)."""
    from benchmark.plugins.task_family.l10.public_render import render_public
    return render_public(ast)


PLUGIN = BehaviorBddlTaskFamilies()

__all__ = ["BehaviorBddlTaskFamilies", "PLUGIN", "FAMILIES", "FAMILY_BY_ID"]
