"""L10 text utilities: s-expression serialization, mutation, position indexing.

Shared representation layer for the lane's generator (``propose.py``) and
verifier (``benchmark/plugins/verifier/l10/bddl_verifier.py``). It contains NO
semantic checks: every verdict is computed by ``engine`` (leg A) and the
vendored upstream ``bddl`` package (leg B). Corruptions applied here are task
*inputs*; the gold is always the pinned original problem bytes.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.task_family.l10.engine import ParsedProblem


# --------------------------------------------------------------- serialization

def atom_str(atom: tuple) -> str:
    """Canonical ground-atom string: ``(pred a b)``."""
    return "(%s)" % " ".join(str(x) for x in atom)


def sexpr(node: Any) -> str:
    """Serialize a canonical (tuple-tree) BDDL expression."""
    if isinstance(node, (tuple, list)):
        return "(%s)" % " ".join(sexpr(x) for x in node)
    return str(node)


def serialize_problem(
    name: str,
    objects: dict[str, list[str]] | tuple[tuple[str, tuple[str, ...]], ...],
    init: list[tuple] | tuple[tuple, ...],
    goal_conjuncts: list[tuple] | tuple[tuple, ...],
) -> str:
    """Serialize a BDDL problem from canonical structures.

    The outer ``and`` of ``:goal`` is always emitted; parsing strips it again,
    so the canonical conjunct list round-trips exactly.
    """
    if isinstance(objects, tuple):
        obj_map = {cat: list(insts) for cat, insts in objects}
    else:
        obj_map = {cat: list(insts) for cat, insts in objects.items()}
    lines = ["(define (problem %s)" % name, "    (:domain behavior-1k)", ""]
    lines.append("    (:objects")
    for cat in sorted(obj_map):
        insts = obj_map[cat]
        if insts:
            lines.append("        %s - %s" % (" ".join(insts), cat))
    lines.append("    )")
    lines.append("")
    lines.append("    (:init")
    for lit in init:
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            lines.append("        (not %s)" % sexpr(lit[1]))
        else:
            lines.append("        %s" % sexpr(lit))
    lines.append("    )")
    lines.append("")
    lines.append("    (:goal")
    lines.append("        (and")
    for conj in goal_conjuncts:
        lines.append("            %s" % sexpr(conj))
    lines.append("        )")
    lines.append("    )")
    lines.append(")")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------- position enumeration

def _walk_leaves(node: Any, path: tuple[int, ...], out: list[tuple[tuple, tuple]]) -> None:
    """Collect (path, leaf-atom) pairs; path is child indices from this root."""
    if not isinstance(node, tuple) or not node:
        return
    head = node[0]
    if head in ("and", "or"):
        for i, child in enumerate(node[1:]):
            _walk_leaves(child, path + (i + 1,), out)
        return
    if head == "not" and len(node) == 2:
        _walk_leaves(node[1], path + (1,), out)
        return
    if head == "imply" and len(node) == 3:
        _walk_leaves(node[1], path + (1,), out)
        _walk_leaves(node[2], path + (2,), out)
        return
    if head in ("forall", "exists"):
        _walk_leaves(node[2], path + (2,), out)
        return
    if head == "forn":
        _walk_leaves(node[3], path + (3,), out)
        return
    if head in ("forpairs", "fornpairs"):
        _walk_leaves(node[-1], path + (len(node) - 1,), out)
        return
    out.append((path, node))


def goal_leaf_positions(goal_conjuncts: tuple) -> list[tuple[tuple, tuple]]:
    """(conjunct_index + path, leaf) for every predicate leaf, in order."""
    out: list[tuple[tuple, tuple]] = []
    for ci, conj in enumerate(goal_conjuncts):
        sub: list[tuple[tuple, tuple]] = []
        _walk_leaves(conj, (), sub)
        for path, leaf in sub:
            out.append(((ci,) + path, leaf))
    return out


def _walk_quantifiers(node: Any, path: tuple[int, ...],
                      out: list[tuple[tuple, tuple]]) -> None:
    if not isinstance(node, tuple) or not node:
        return
    head = node[0]
    if head in ("forall", "exists", "forn", "forpairs", "fornpairs"):
        out.append((path, node))
        _walk_quantifiers(node[-1], path + (len(node) - 1,), out)
        return
    if head in ("and", "or"):
        for i, child in enumerate(node[1:]):
            _walk_quantifiers(child, path + (i + 1,), out)
    elif head == "not" and len(node) == 2:
        _walk_quantifiers(node[1], path + (1,), out)
    elif head == "imply" and len(node) == 3:
        _walk_quantifiers(node[1], path + (1,), out)
        _walk_quantifiers(node[2], path + (2,), out)


def quantifier_positions(goal_conjuncts: tuple) -> list[tuple[tuple, tuple]]:
    out: list[tuple[tuple, tuple]] = []
    for ci, conj in enumerate(goal_conjuncts):
        sub: list[tuple[tuple, tuple]] = []
        _walk_quantifiers(conj, (), sub)
        for path, node in sub:
            out.append(((ci,) + path, node))
    return out


def ref_positions(goal_conjuncts: tuple) -> list[tuple[tuple, int]]:
    """(leaf_path, arg_index) for every ``?``-prefixed goal argument, in order."""
    out: list[tuple[tuple, int]] = []
    for path, leaf in goal_leaf_positions(goal_conjuncts):
        for ai, arg in enumerate(leaf[1:]):
            if isinstance(arg, str) and arg.startswith("?"):
                out.append((path, ai + 1))
    return out


# ------------------------------------------------------------------- mutation

def _replace_at(node: Any, path: tuple[int, ...], new_node: Any) -> Any:
    """Rebuild a tuple tree with ``new_node`` at ``path`` (child indices)."""
    if not path:
        return new_node
    idx = path[0]
    lst = list(node)
    lst[idx] = _replace_at(lst[idx], path[1:], new_node)
    return tuple(lst)


def replace_goal_leaf(goal_conjuncts: tuple, leaf_path: tuple, new_leaf: tuple) -> tuple:
    """Replace the leaf at (conjunct_index, ...) with ``new_leaf``."""
    ci = leaf_path[0]
    conjuncts = list(goal_conjuncts)
    conjuncts[ci] = _replace_at(conjuncts[ci], leaf_path[1:], new_leaf)
    return tuple(conjuncts)


def replace_goal_quantifier(goal_conjuncts: tuple, q_path: tuple,
                            new_quant: tuple) -> tuple:
    ci = q_path[0]
    conjuncts = list(goal_conjuncts)
    conjuncts[ci] = _replace_at(conjuncts[ci], q_path[1:], new_quant)
    return tuple(conjuncts)


def positive_init_atoms(problem: ParsedProblem) -> list[tuple]:
    """Positive ground atoms of :init, in original order."""
    out = []
    for lit in problem.init:
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            continue
        out.append(lit)
    return out


def problem_parts(problem: ParsedProblem) -> tuple[dict[str, list[str]], list[tuple], tuple]:
    """(object_map, ALL init literals, goal_conjuncts) — serialization-faithful."""
    return (
        {cat: list(insts) for cat, insts in problem.objects},
        list(problem.init),
        tuple(problem.goal),
    )


def positive_init_positions(problem: ParsedProblem) -> list[tuple[int, tuple]]:
    """(full-init index, positive atom) pairs, in original order."""
    out = []
    for i, lit in enumerate(problem.init):
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            continue
        out.append((i, lit))
    return out


# ------------------------------------------------------------------ corruption

def corrupt_parts(
    problem: ParsedProblem, goal_kind: str, params: dict,
) -> tuple[dict[str, list[str]], list[tuple], tuple]:
    """Apply one family's corruption/perturbation to a parsed problem.

    Returns (object_map, init_atoms, goal_conjuncts) with the corruption
    applied. Purely structural: the pinned original bytes remain the gold.
    """
    obj_map, inits, goal = problem_parts(problem)
    inits = list(inits)
    goal = list(goal)
    leaves = goal_leaf_positions(tuple(goal))
    refs = ref_positions(tuple(goal))

    if goal_kind == "repair_predicate":
        i = params["leaf_index"] % len(leaves)
        path, leaf = leaves[i]
        goal = list(replace_goal_leaf(
            tuple(goal), path, (params["corrupt_token"],) + tuple(leaf[1:])))
    elif goal_kind == "repair_arity":
        i = params["leaf_index"] % len(leaves)
        path, leaf = leaves[i]
        goal = list(replace_goal_leaf(tuple(goal), path, tuple(leaf[:-1])))
    elif goal_kind == "repair_binding":
        leaf_path, arg_index = _leaf_arg_index(leaves, refs, params)
        path, leaf = next((p, l) for p, l in leaves if p == leaf_path)
        new_args = list(leaf)
        orig = new_args[arg_index]
        new_args[arg_index] = "?%s%s" % (orig[1:], params["suffix"])
        goal = list(replace_goal_leaf(tuple(goal), path, tuple(new_args)))
    elif goal_kind == "repair_init_predicate":
        pos = positive_init_positions(problem)
        full_i, atom = pos[params["init_index"] % len(pos)]
        inits[full_i] = (params["corrupt_token"],) + tuple(atom[1:])
    elif goal_kind == "repair_quantifier_category":
        quants = quantifier_positions(tuple(goal))
        q = params["quant_index"] % len(quants)
        path, node = quants[q]
        head = node[0]
        iter_idx = 2 if head == "forn" else 1
        if head == "fornpairs":
            iter_idx = 2
        new_iter = ("?%s" % node[iter_idx][0][1:], "-", params["corrupt_token"])
        new_node = list(node)
        new_node[iter_idx] = new_iter
        goal = list(replace_goal_quantifier(tuple(goal), path, tuple(new_node)))
    elif goal_kind == "defect_audit":
        for defect in params["defects"]:
            kind, section, index = defect["kind"], defect["section"], defect["index"]
            if kind == "predicate":
                path, leaf = leaves[index % len(leaves)]
                goal = list(replace_goal_leaf(
                    tuple(goal), path, ("defp%d" % index,) + tuple(leaf[1:])))
            elif kind == "arity":
                path, leaf = leaves[index % len(leaves)]
                goal = list(replace_goal_leaf(tuple(goal), path, tuple(leaf[:-1])))
            elif kind == "binding":
                leaf_path, arg_index = refs[index % len(refs)]
                path, leaf = next((p, l) for p, l in leaves if p == leaf_path)
                new_args = list(leaf)
                orig = new_args[arg_index]
                new_args[arg_index] = "?%sdefb%d" % (orig[1:], index)
                goal = list(replace_goal_leaf(tuple(goal), path, tuple(new_args)))
            elif kind == "init_predicate":
                i = index % len(inits)
                atom = inits[i]
                inits[i] = ("defi%d" % index,) + tuple(atom[1:])
    else:
        raise ValueError("not a corruption family: %s" % goal_kind)
    return obj_map, inits, tuple(goal)


def _leaf_arg_index(leaves, refs, params) -> tuple:
    leaf_i = params["leaf_index"] % len(leaves)
    leaf_path = leaves[leaf_i][0]
    arg_index = params["arg_index"]
    return leaf_path, arg_index
