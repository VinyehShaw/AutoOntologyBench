"""L10 20K-expansion candidate generation (propose) + public grounding.

New mechanisms for build ``aob-20000-candidate-dev1`` (seed base 20261101).
Generation is solver-grounded exactly like the legacy lane: every anchor is
pre-filtered so the pinned problem provably supports the family mechanism,
and the independent verifier re-proves everything from the pinned snapshot
bytes. The generator never writes gold into public payloads.

Legacy modules (``propose.py`` / ``families.py`` / ``public_render.py``) are
untouched; only their shared engine/textgen/evalgoal layers are reused.
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
    INSTANCES_PER_GOAL,
    LANE_ID,
    SEMANTIC_PROFILE,
    SYMBOLIC_ONLY_NOTE,
)
from benchmark.plugins.task_family.l10.families_20k import (
    NEW_FAMILIES,
    NEW_FAMILY_BY_ID,
    FamilySpec20K,
)

SEED_SALT = 20261101

_DOMAIN_UNIT = (
    "behavior-1k-bddl/bddl3/bddl/activity_definitions/domain_behavior-1k.bddl"
)

_SUBGOAL_KINDS = (
    "quant_inventory", "shape_census", "reference_closure", "polarity_audit",
    "provenance_partition", "predicate_usage", "weakening_goal",
)


def _ids(family_id: str, base_key: Any, inst_key: Any) -> tuple[str, str]:
    base_goal_id = "bg-" + sha256_json(
        {"f": family_id, "b": base_key, "s": SEED_SALT})[:20]
    instance_id = "in-" + sha256_json({"b": base_goal_id, "i": inst_key})[:20]
    return base_goal_id, instance_id


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


def _restricted(problem: ParsedProblem) -> bool:
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


def _rotate(items: list, n: int) -> list:
    """Deterministic rotation so families do not share anchor prefixes."""
    return items[n % len(items):] + items[: n % len(items)] if items else items


# ------------------------------------------------------- shared goal analysis

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


def _outermost_quants(goal: tuple) -> list[tuple]:
    """Quantifier nodes not nested inside another quantifier, textual order."""
    out: list[tuple] = []

    def walk(node: Any) -> None:
        if not isinstance(node, tuple) or not node:
            return
        h = node[0]
        if h in ("and", "or"):
            for c in node[1:]:
                walk(c)
            return
        if h == "not" and len(node) == 2:
            walk(node[1])
            return
        if h == "imply" and len(node) == 3:
            walk(node[1])
            walk(node[2])
            return
        if h in ("forall", "exists", "forn", "forpairs", "fornpairs"):
            out.append(node)
            return

    for c in goal:
        walk(c)
    return out


def _shape_tag(node: Any) -> str:
    """Root shape of a conjunct: 'atom' or the operator head token."""
    if isinstance(node, tuple) and node:
        if node[0] in engine.load_tables().predicate_map:
            return "atom"
        return str(node[0])
    return "malformed"


def _closure_refs(problem: ParsedProblem, conj: Any) -> set[str]:
    """Declared instances referenced by a conjunct.

    Direct ``?instance`` references plus every instance of each quantifier's
    iterable category (nested quantifiers included).
    """
    insts: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, tuple) or not node:
            return
        h = node[0]
        if h in ("and", "or"):
            for c in node[1:]:
                walk(c)
            return
        if h == "not" and len(node) == 2:
            walk(node[1])
            return
        if h == "imply" and len(node) == 3:
            walk(node[1])
            walk(node[2])
            return
        if h in ("forall", "exists"):
            for inst in problem.object_map.get(node[1][2], []):
                insts.add(inst)
            walk(node[2])
            return
        if h == "forn":
            for inst in problem.object_map.get(node[2][2], []):
                insts.add(inst)
            walk(node[3])
            return
        if h in ("forpairs", "fornpairs"):
            for it in node[1:-1]:
                if isinstance(it, tuple) and len(it) == 3:
                    for inst in problem.object_map.get(it[2], []):
                        insts.add(inst)
            walk(node[-1])
            return
        for arg in node[1:]:
            if isinstance(arg, str) and arg.startswith("?") \
                    and arg[1:] in problem.instances:
                insts.add(arg[1:])

    walk(conj)
    return insts


def _polarity_flags(goal: tuple) -> list[bool]:
    """Per-leaf negation polarity (True = negative), depth-first textual order."""
    out: list[bool] = []

    def walk(node: Any, neg: bool) -> None:
        if not isinstance(node, tuple) or not node:
            return
        h = node[0]
        if h == "not" and len(node) == 2:
            walk(node[1], not neg)
            return
        if h in ("and", "or"):
            for c in node[1:]:
                walk(c, neg)
            return
        if h == "imply" and len(node) == 3:
            walk(node[1], not neg)
            walk(node[2], neg)
            return
        if h in ("forall", "exists"):
            walk(node[2], neg)
            return
        if h == "forn":
            walk(node[3], neg)
            return
        if h in ("forpairs", "fornpairs"):
            walk(node[-1], neg)
            return
        out.append(neg)

    for c in goal:
        walk(c, False)
    return out


def _parsed(problem: ParsedProblem, goal: tuple,
            init: tuple | None = None) -> ParsedProblem:
    return ParsedProblem(
        name=problem.name, domain=problem.domain, objects=problem.objects,
        init=problem.init if init is None else tuple(init),
        goal=tuple(goal), goal_raw=tuple(goal))


def _obligations(problem: ParsedProblem, goal: tuple) -> list[tuple] | None:
    try:
        return evalgoal.obligation_atoms(_parsed(problem, goal))
    except evalgoal.EvalError:
        return None


def _contradictions(problem: ParsedProblem) -> set[tuple]:
    pos = set(textgen.positive_init_atoms(problem))
    neg = set()
    for lit in problem.init:
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            neg.add(lit[1])
    return pos & neg


def _duplicate_atoms(problem: ParsedProblem) -> set[str]:
    from collections import Counter
    pos = [textgen.atom_str(a) for a in textgen.positive_init_atoms(problem)]
    return {s for s, n in Counter(pos).items() if n > 1}


@lru_cache(maxsize=1)
def _init_predicate_index() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for p, prob in _clean_problems():
        out[p.parent.name] = frozenset(
            str(a[0]) for a in textgen.positive_init_atoms(prob))
    return out


def _predicate_partners(activity: str, goal_preds: set[str]) -> list[str]:
    idx = _init_predicate_index()
    return sorted(
        name for name, preds in idx.items()
        if name != activity and preds & goal_preds)


# ------------------------------------------------------- variant construction

def _variant_params(spec: FamilySpec20K, path: Path, problem: ParsedProblem,
                    variant: int) -> dict[str, Any] | None:
    """Family-specific instance parameters, or None if unsupported."""
    kind = spec.goal_kind
    if kind in _SUBGOAL_KINDS:
        specs = _subgoal_specs(problem, INSTANCES_PER_GOAL)
        if len(specs) < INSTANCES_PER_GOAL:
            return None
        return {"conjuncts": specs[variant]}

    if kind == "discriminating_state":
        n = len(problem.goal)
        if n < 2 or not _restricted(problem) \
                or evalgoal.goal_satisfied(problem):
            return None
        pairs: list[tuple[list[int], int]] = []
        for i in range(n):
            a_idx = [j for j in range(n) if j != i]
            goal_a = tuple(problem.goal[j] for j in a_idx)
            ob_a = _obligations(problem, goal_a)
            ob_b = _obligations(problem, (problem.goal[i],))
            if not ob_a or not ob_b:
                continue
            if set(ob_b) <= set(ob_a):
                continue  # B would stay satisfied over obligations(A)
            pairs.append((a_idx, i))
        if len(pairs) < INSTANCES_PER_GOAL:
            return None
        a_idx, b_index = pairs[variant % len(pairs)]
        return {"a_conjuncts": a_idx, "b_index": b_index}

    if kind == "contradiction_audit":
        pos = textgen.positive_init_atoms(problem)
        if len(pos) < INSTANCES_PER_GOAL or _contradictions(problem):
            return None
        return {"inject_index": variant % len(pos)}

    if kind == "duplicate_audit":
        pos = textgen.positive_init_atoms(problem)
        if len(pos) < INSTANCES_PER_GOAL or _duplicate_atoms(problem):
            return None
        i = variant % len(pos)
        j = (i + 1 + variant) % len(pos)
        return {"inject_indices": sorted({i, j})}

    if kind == "predicate_join":
        goal_preds = sorted({
            str(leaf[0])
            for _p, leaf in textgen.goal_leaf_positions(tuple(problem.goal))})
        if len(goal_preds) < 2:
            return None
        partners = _predicate_partners(path.parent.name, set(goal_preds))
        if len(partners) < INSTANCES_PER_GOAL:
            return None
        partner = partners[variant % len(partners)]
        partner_rel = ("bddl3/bddl/activity_definitions/%s/problem0.bddl"
                       % partner)
        digest = engine_sha256(engine.PINNED_ROOT / partner_rel)
        return {"partner_activity": partner, "partner_digest": digest}

    if kind == "room_grouping":
        inrooms = [lit for lit in problem.init
                   if isinstance(lit, tuple) and lit and lit[0] == "inroom"]
        if len(inrooms) < INSTANCES_PER_GOAL:
            return None
        i = variant % len(inrooms)
        rem = [a for j, a in enumerate(inrooms) if j != i]
        if len(rem) < 3 or len({a[1] for a in rem}) < 2:
            return None
        return {"drop_index": i}

    return None


# ------------------------------------------------------------------ proposal

def _anchor_pool(spec: FamilySpec20K, seed: int) -> list[tuple[Path, ParsedProblem]]:
    kind = spec.goal_kind
    pool = list(_clean_problems())
    if kind == "quant_inventory":
        pool = [(p, prob) for p, prob in pool
                if all(len([q for q in _outermost_quants(
                    tuple(prob.goal[i] for i in sp))
                    if q[0] in ("forall", "exists", "forn")]) >= 2
                    for sp in _subgoal_specs(prob, INSTANCES_PER_GOAL))]
    elif kind == "shape_census":
        pool = [(p, prob) for p, prob in pool
                if all(len({_shape_tag(prob.goal[i]) for i in sp}) >= 2
                       for sp in _subgoal_specs(prob, INSTANCES_PER_GOAL))]
    elif kind == "reference_closure":
        pool = [(p, prob) for p, prob in pool
                if _closure_ok(prob)]
    elif kind == "polarity_audit":
        pool = [(p, prob) for p, prob in pool
                if all(_polarity_ok(tuple(prob.goal[i] for i in sp))
                       for sp in _subgoal_specs(prob, INSTANCES_PER_GOAL))]
    elif kind == "discriminating_state":
        pool = [(p, prob) for p, prob in pool
                if _discriminating_ok(prob)]
    elif kind == "weakening_goal":
        pool = [(p, prob) for p, prob in pool
                if not evalgoal.goal_satisfied(prob) and _restricted(prob)
                and len(prob.goal) >= 2
                and _two_distinct_init_atoms(prob)]
    elif kind == "contradiction_audit":
        pool = [(p, prob) for p, prob in pool
                if not _contradictions(prob)
                and len(textgen.positive_init_atoms(prob))
                >= INSTANCES_PER_GOAL]
    elif kind == "duplicate_audit":
        pool = [(p, prob) for p, prob in pool
                if not _duplicate_atoms(prob)
                and len(textgen.positive_init_atoms(prob))
                >= INSTANCES_PER_GOAL]
    elif kind == "predicate_join":
        pool = [(p, prob) for p, prob in pool
                if len({str(leaf[0]) for _p, leaf in
                        textgen.goal_leaf_positions(tuple(prob.goal))}) >= 2
                and len(_predicate_partners(
                    p.parent.name,
                    {str(leaf[0]) for _p, leaf in
                     textgen.goal_leaf_positions(tuple(prob.goal))}))
                >= INSTANCES_PER_GOAL]
    elif kind == "room_grouping":
        pool = [(p, prob) for p, prob in pool
                if _room_ok(prob)]
    elif kind == "provenance_partition":
        pool = [(p, prob) for p, prob in pool
                if _provenance_ok(prob)]
    elif kind == "predicate_usage":
        pool = [(p, prob) for p, prob in pool
                if _usage_ok(prob)]
    return _rotate(pool, sum(ord(c) for c in spec.family_id) + seed)


def _closure_ok(problem: ParsedProblem) -> bool:
    for sp in _subgoal_specs(problem, INSTANCES_PER_GOAL):
        goal = tuple(problem.goal[i] for i in sp)
        if len(goal) < 2:
            return False
        closures = []
        for c in goal:
            refs = _closure_refs(problem, c)
            if not refs:
                return False
            closures.append(tuple(sorted(refs)))
        if len(set(closures)) < 2:
            return False
    return True


def _polarity_ok(goal: tuple) -> bool:
    flags = _polarity_flags(goal)
    return len(flags) >= 3 and any(flags) and not all(flags)


def _discriminating_ok(problem: ParsedProblem) -> bool:
    n = len(problem.goal)
    if n < 2 or not _restricted(problem) or evalgoal.goal_satisfied(problem):
        return False
    pairs = 0
    for i in range(n):
        goal_a = tuple(problem.goal[j] for j in range(n) if j != i)
        ob_a = _obligations(problem, goal_a)
        ob_b = _obligations(problem, (problem.goal[i],))
        if ob_a and ob_b and not set(ob_b) <= set(ob_a):
            pairs += 1
    return pairs >= INSTANCES_PER_GOAL


def _two_distinct_init_atoms(problem: ParsedProblem) -> bool:
    pos = textgen.positive_init_atoms(problem)
    return len(pos) >= 2 and textgen.atom_str(pos[0]) != textgen.atom_str(pos[1])


def _room_ok(problem: ParsedProblem) -> bool:
    inrooms = [lit for lit in problem.init
               if isinstance(lit, tuple) and lit and lit[0] == "inroom"]
    if len(inrooms) < INSTANCES_PER_GOAL:
        return False
    for i in range(len(inrooms)):
        rem = [a for j, a in enumerate(inrooms) if j != i]
        if len(rem) < 3 or len({a[1] for a in rem}) < 2:
            return False
    return True


def _provenance_ok(problem: ParsedProblem) -> bool:
    if evalgoal.goal_satisfied(problem) or not _restricted(problem) \
            or len(problem.goal) < 2:
        return False
    for sp in _subgoal_specs(problem, INSTANCES_PER_GOAL):
        goal = tuple(problem.goal[i] for i in sp)
        prov: dict[tuple, set[int]] = {}
        for ci, c in enumerate(goal):
            obs = _obligations(problem, (c,))
            if obs is None:
                return False
            for a in obs:
                prov.setdefault(a, set()).add(ci)
        if len(prov) < 3 or len({ci for cis in prov.values() for ci in cis}) < 2:
            return False
    return True


def _usage_ok(problem: ParsedProblem) -> bool:
    for sp in _subgoal_specs(problem, INSTANCES_PER_GOAL):
        goal = tuple(problem.goal[i] for i in sp)
        init_preds = {str(a[0]) for a in textgen.positive_init_atoms(problem)}
        goal_preds = {str(leaf[0]) for _p, leaf in
                      textgen.goal_leaf_positions(goal)}
        used = init_preds | goal_preds
        if len(used) < 3:
            return False
        tags = set()
        for pred in used:
            sections = sorted(
                ("init" if pred in init_preds else "") +
                ("goal" if pred in goal_preds else ""))
            tags.add(tuple(sections))
        if len(tags) < 2:
            return False
    return True


def engine_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _source_units(path: Path) -> tuple[str, ...]:
    return ("behavior-1k-bddl/%s" % str(path.relative_to(engine.PINNED_ROOT)),
            _DOMAIN_UNIT)


class BehaviorBddl20KTaskFamilies:
    """TaskFamilyPlugin for the L10 20K-expansion families (symbolic only)."""

    plugin_id = "l10.behavior-bddl-families-20k"

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        spec = NEW_FAMILY_BY_ID[family]
        pool = _anchor_pool(spec, seed)
        base_quota = int(constraints.get("base_goals", spec.base_goals))
        per_goal = int(constraints.get(
            "instances_per_goal", INSTANCES_PER_GOAL))
        made_bases = 0
        for path, problem in pool:
            if made_bases >= base_quota:
                break
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
            cands: list[CandidateTaskAST] = []
            for v, params in enumerate(variants):
                inst_key = {"v": v, "params": params}
                _bg, instance_id = _ids(family, base_key, inst_key)
                if instance_id in {c.instance_id for c in cands}:
                    break
                ast: dict[str, Any] = {
                    "family_id": family,
                    "goal_kind": spec.goal_kind,
                    "activity": path.parent.name,
                    "problem_rel": base_key,
                    "problem_digest": engine_sha256(path),
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
                    source_unit_ids=_source_units(path) + _extra_units(
                        spec, params),
                ))
            if len(cands) != per_goal:
                continue
            made_bases += 1
            yield from cands

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        oracle_ref = "l10x-oracle-%s" % sha256_json(
            [candidate.instance_id, ast["variant_key"], SEED_SALT])[:16]
        public = build_public_20k(ast)
        return GroundedTask(ast=candidate, public=public,
                            oracle_ref=oracle_ref, native_evidence={})


def _extra_units(spec: FamilySpec20K, params: dict[str, Any]) -> tuple[str, ...]:
    if spec.goal_kind == "predicate_join":
        return ("behavior-1k-bddl/bddl3/bddl/activity_definitions/%s/"
                "problem0.bddl" % params["partner_activity"],)
    return ()


# -------------------------------------------------------------- public build

def _domain_predicates() -> dict[str, int]:
    return dict(engine.load_tables().predicate_map)


def _observation_refs(ast: dict[str, Any]) -> list[dict[str, str]]:
    refs = [
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
    if ast["goal_kind"] == "predicate_join":
        refs.append({
            "path": "external_knowledge/pinned/behavior-1k-bddl/v3.9.3-post2/"
                    "bddl3/bddl/activity_definitions/%s/problem0.bddl"
                    % ast["params"]["partner_activity"],
            "sha256": ast["params"]["partner_digest"],
        })
    return refs


def _base(ast: dict[str, Any], user_goal: str) -> dict[str, Any]:
    spec = NEW_FAMILY_BY_ID[ast["family_id"]]
    return {
        "user_goal": user_goal,
        "allowed_tools": list(ALLOWED_TOOLS[spec.regime]),
        "budget": dict(BUDGETS[ast["complexity"]]),
        "observation_refs": _observation_refs(ast),
        "semantic_scope": SYMBOLIC_ONLY_NOTE,
    }


def _load_problem(ast: dict[str, Any]) -> ParsedProblem:
    path = engine.PINNED_ROOT / ast["problem_rel"]
    return engine.parse_problem_text(path.read_text(encoding="utf-8"))


def build_public_20k(ast: dict[str, Any]) -> dict[str, Any]:
    """Public task payload from ast params (recomputed; never contains gold)."""
    kind = ast["goal_kind"]
    params = ast["params"]
    problem = _load_problem(ast)
    obj_map, inits, goal = textgen.problem_parts(problem)
    name = ast["activity"] + "-%d" % ast["variant"]
    full_text = textgen.serialize_problem(
        name, obj_map, inits, tuple(problem.goal))

    if kind in ("quant_inventory", "shape_census", "reference_closure",
                "polarity_audit", "provenance_partition", "predicate_usage"):
        prompts = {
            "quant_inventory":
                "For the BDDL problem below, consider only the goal conjuncts "
                "at indices %s. List every OUTERMOST forall/exists/forn "
                "quantifier of that sub-goal in textual order as a list of "
                "[head, variable, category] triples (variable without the "
                "leading '?'; ignore quantifiers nested inside another "
                "quantifier and ignore forpairs/fornpairs). %s",
            "shape_census":
                "For the BDDL problem below, consider only the goal conjuncts "
                "at indices %s. Classify each selected conjunct by its ROOT "
                "expression shape: 'atom' if it is a bare predicate "
                "application, otherwise the operator head token (and/or/not/"
                "imply/forall/exists/forn/forpairs/fornpairs). Submit the "
                "exact mapping conjunct position (0-based within the selected "
                "sub-goal) -> shape tag. %s",
            "reference_closure":
                "For the BDDL problem below, consider only the goal conjuncts "
                "at indices %s. For each selected conjunct, compute the exact "
                "set of declared object instances it references: every "
                "?instance direct reference plus every instance of each "
                "quantifier's iterable category (nested quantifiers "
                "included). Submit the mapping conjunct position (0-based "
                "within the sub-goal) -> sorted instance list. %s",
            "polarity_audit":
                "For the BDDL problem below, consider only the goal conjuncts "
                "at indices %s. Order the sub-goal's predicate leaves "
                "depth-first in textual order (0-based). Submit the exact "
                "sorted index set of leaves that sit under an odd number of "
                "negations (negative polarity; the antecedent of imply counts "
                "as one negation). %s",
            "provenance_partition":
                "For the BDDL problem below, consider only the goal conjuncts "
                "at indices %s. Expand the sub-goal to its ground obligation "
                "atoms (closed-world; every leaf is universally quantified) "
                "and attribute each atom to the conjuncts that generate it. "
                "Submit the exact mapping atom string -> sorted list of "
                "generating conjunct positions (0-based within the sub-goal). "
                "%s",
            "predicate_usage":
                "For the BDDL problem below, consider the :init section and "
                "only the goal conjuncts at indices %s. For every domain "
                "predicate used in either place, submit the exact mapping "
                "predicate -> sorted list of sections ('goal' and/or 'init') "
                "in which it occurs. %s",
        }
        user_goal = prompts[kind] % (params["conjuncts"], SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        material: dict[str, Any] = {
            "problem_text": full_text,
            "subgoal_conjuncts": list(params["conjuncts"]),
        }
        if kind in ("quant_inventory", "provenance_partition"):
            material["domain_predicates"] = _domain_predicates()
        if kind == "polarity_audit":
            sub = tuple(problem.goal[i] for i in params["conjuncts"])
            material["leaf_count"] = len(
                textgen.goal_leaf_positions(sub))
        if kind == "reference_closure":
            material["objects"] = {c: list(ii) for c, ii in obj_map.items()}
        public["task_material"] = material
        return public

    if kind == "weakening_goal":
        sub = tuple(problem.goal[i] for i in params["conjuncts"])
        obligations = _obligations(problem, sub) or []
        s2 = textgen.positive_init_atoms(problem)
        s1 = list(s2) + list(obligations)
        user_goal = (
            "The reference goal G (the goal conjuncts at indices %s of the "
            "problem below) is satisfied under the closed-world state S1 but "
            "NOT under the state S2. Using only the domain predicates and "
            "declared objects, construct a goal expression that is satisfied "
            "under BOTH S1 and S2, and is NOT satisfied under the empty "
            "state. Submit the goal expression text (e.g. \"(and (predicate "
            "obj1 obj2))\"). %s" % (params["conjuncts"], SYMBOLIC_ONLY_NOTE))
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": full_text,
            "reference_conjuncts": list(params["conjuncts"]),
            "state_s1": [textgen.atom_str(a) for a in s1],
            "state_s2": [textgen.atom_str(a) for a in s2],
            "domain_predicates": _domain_predicates(),
            "objects": {c: list(ii) for c, ii in obj_map.items()},
        }
        return public

    if kind == "discriminating_state":
        a_idx = params["a_conjuncts"]
        b_index = params["b_index"]
        user_goal = (
            "Using only the domain predicates and the declared objects of "
            "the problem below, construct a set of ground atoms (a "
            "closed-world state) that SATISFIES sub-goal A (the goal "
            "conjuncts at indices %s) but FALSIFIES conjunct B (index %d). "
            "Submit the sorted atom list. %s"
            % (a_idx, b_index, SYMBOLIC_ONLY_NOTE))
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": full_text,
            "subgoal_a_conjuncts": list(a_idx),
            "subgoal_b_index": b_index,
            "domain_predicates": _domain_predicates(),
            "objects": {c: list(ii) for c, ii in obj_map.items()},
        }
        return public

    if kind == "contradiction_audit":
        pos = textgen.positive_init_atoms(problem)
        atom = pos[params["inject_index"] % len(pos)]
        insert_at = (params["inject_index"] * 3 + 1) % (len(inits) + 1)
        perturbed = list(inits)
        perturbed.insert(insert_at, ("not", atom))
        text = textgen.serialize_problem(
            name, obj_map, perturbed, tuple(problem.goal))
        user_goal = (
            "The :init section of the BDDL problem below asserts one atom "
            "both positively and negatively (an injected inconsistency). "
            "Submit the exact sorted list of atom strings that are asserted "
            "both ways. %s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": text,
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "duplicate_audit":
        pos = textgen.positive_init_atoms(problem)
        perturbed = list(inits)
        for k, idx in enumerate(params["inject_indices"]):
            atom = pos[idx % len(pos)]
            insert_at = (idx * 3 + 2 + k) % (len(perturbed) + 1)
            perturbed.insert(insert_at, atom)
        text = textgen.serialize_problem(
            name, obj_map, perturbed, tuple(problem.goal))
        user_goal = (
            "The :init section of the BDDL problem below contains duplicate "
            "atoms (some atoms were injected more than once). Submit the "
            "exact sorted list of atom strings that occur more than once in "
            ":init. %s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_text": text,
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "predicate_join":
        partner_rel = ("bddl3/bddl/activity_definitions/%s/problem0.bddl"
                       % params["partner_activity"])
        partner = engine.parse_problem_text(
            (engine.PINNED_ROOT / partner_rel).read_text(encoding="utf-8"))
        p_obj, p_init, _p_goal = textgen.problem_parts(partner)
        partner_text = textgen.serialize_problem(
            params["partner_activity"] + "-%d" % ast["variant"],
            p_obj, p_init, tuple(partner.goal))
        user_goal = (
            "Activity A's goal (the full problem A below) uses certain "
            "domain predicates. Join each such predicate against activity "
            "B's initial state (the full problem B below): submit the exact "
            "mapping predicate -> sorted list of B's positive :init atoms "
            "that use that predicate (empty list if B's init uses none). %s"
            % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "problem_a_text": full_text,
            "problem_b_text": partner_text,
            "domain_predicates": _domain_predicates(),
        }
        return public

    if kind == "room_grouping":
        inrooms = [lit for lit in problem.init
                   if isinstance(lit, tuple) and lit and lit[0] == "inroom"]
        rem = [a for j, a in enumerate(inrooms)
               if j != params["drop_index"] % len(inrooms)]
        rng = random.Random(sha256_json(
            [ast["variant_key"], "roomshuffle", SEED_SALT])[:16])
        shuffled = list(rem)
        rng.shuffle(shuffled)
        user_goal = (
            "The shuffled list below contains the ``inroom`` atoms of a "
            "BDDL problem (one atom removed). Organize it into the exact "
            "mapping room -> sorted list of instances located in that room. "
            "%s" % SYMBOLIC_ONLY_NOTE)
        public = _base(ast, user_goal)
        public["task_material"] = {
            "inroom_atoms": [textgen.atom_str(a) for a in shuffled],
            "objects": {c: list(ii) for c, ii in obj_map.items()},
        }
        return public

    raise ValueError("unknown 20k goal_kind: %s" % kind)


PLUGIN = BehaviorBddl20KTaskFamilies()

__all__ = [
    "BehaviorBddl20KTaskFamilies", "PLUGIN", "NEW_FAMILIES",
    "NEW_FAMILY_BY_ID", "build_public_20k", "SEED_SALT",
]
