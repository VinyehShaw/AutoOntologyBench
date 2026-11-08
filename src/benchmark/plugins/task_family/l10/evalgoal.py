"""L10 symbolic goal evaluation over an init atom set (leg A semantics).

Closed-world reading: a ground atom is true iff it is a positive literal of
``:init``. Quantifier semantics mirror the pinned upstream
``bddl.condition_evaluation`` classes (forall/exists/forn/forpairs/fornpairs);
``upstream.py`` recomputes every evaluation with the vendored package and the
two legs must agree for a task to qualify.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.task_family.l10.engine import ParsedProblem


class EvalError(ValueError):
    """Goal cannot be evaluated symbolically (unbound var, unknown category)."""


def init_truth_set(problem: ParsedProblem) -> frozenset[tuple[str, ...]]:
    """Positive ground atoms of :init as a truth set (closed world)."""
    out: set[tuple[str, ...]] = set()
    for lit in problem.init:
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            continue  # negative literals assert absence; closed world already does
        out.add(lit)
    return frozenset(out)


def _resolve(tok: str, env: dict[str, str], problem: ParsedProblem) -> str:
    if tok.startswith("?"):
        if tok in env:
            return env[tok]
        if tok[1:] in problem.instances:
            return tok[1:]
        raise EvalError("unbound variable %r" % tok)
    return tok


def _instances(problem: ParsedProblem, category: str) -> list[str]:
    obj = problem.object_map
    if category not in obj:
        raise EvalError("category %r not declared" % category)
    return list(obj[category])


def _pairs(matrix: list[list[bool]]) -> tuple[int, int]:
    """(rows_with_any, cols_with_any) cover counts for a boolean matrix."""
    if not matrix:
        return 0, 0
    rows = sum(1 for row in matrix if any(row))
    cols = sum(1 for j in range(len(matrix[0])) if any(matrix[i][j] for i in range(len(matrix))))
    return rows, cols


def eval_node(node: Any, env: dict[str, str], problem: ParsedProblem,
              truth: frozenset[tuple[str, ...]]) -> bool:
    if not isinstance(node, tuple) or not node:
        raise EvalError("non-compound goal node %r" % (node,))
    head = node[0]
    if head == "and":
        return all(eval_node(c, env, problem, truth) for c in node[1:])
    if head == "or":
        return any(eval_node(c, env, problem, truth) for c in node[1:])
    if head == "not":
        return not eval_node(node[1], env, problem, truth)
    if head == "imply":
        return (not eval_node(node[1], env, problem, truth)) or eval_node(
            node[2], env, problem, truth)
    if head in ("forall", "exists"):
        var, cat = _iterable(node[1])
        vals = [eval_node(node[2], {**env, var: inst}, problem, truth)
                for inst in _instances(problem, cat)]
        return all(vals) if head == "forall" else any(vals)
    if head == "forn":
        n = int(node[1][0])
        var, cat = _iterable(node[2])
        vals = [eval_node(node[3], {**env, var: inst}, problem, truth)
                for inst in _instances(problem, cat)]
        return sum(bool(v) for v in vals) == n
    if head in ("forpairs", "fornpairs"):
        idx = 2 if head == "fornpairs" else 1
        n = int(node[1][0]) if head == "fornpairs" else None
        var1, cat1 = _iterable(node[idx])
        var2, cat2 = _iterable(node[idx + 1])
        body = node[idx + 2]
        matrix = []
        for i1 in _instances(problem, cat1):
            row = []
            for i2 in _instances(problem, cat2):
                if i1 == i2:
                    continue
                row.append(eval_node(body, {**env, var1: i1, var2: i2}, problem, truth))
            matrix.append(row)
        rows, cols = _pairs(matrix)
        need = n if n is not None else min(len(matrix), len(matrix[0]) if matrix else 0)
        return rows >= need and cols >= need
    # leaf atom
    args = tuple(_resolve(tok, env, problem) for tok in node[1:])
    return (head,) + args in truth


def _iterable(form: Any) -> tuple[str, str]:
    if (isinstance(form, tuple) and len(form) == 3 and form[0].startswith("?")
            and form[1] == "-"):
        return form[0], form[2]
    raise EvalError("malformed iterable %r" % (form,))


def goal_satisfied(problem: ParsedProblem) -> bool:
    """True iff the goal already holds under the closed-world init state."""
    truth = init_truth_set(problem)
    return all(eval_node(c, {}, problem, truth) for c in problem.goal)


def obligation_atoms(problem: ParsedProblem) -> list[tuple[str, ...]]:
    """Fully-expanded positive obligation atoms of a restricted goal.

    Defined only for goals built from and/forall over leaf atoms (no or, not,
    imply, exists, forn, forpairs, fornpairs): every leaf is universally
    quantified, so the required atom set is unique. Raises EvalError otherwise.
    """
    out: list[tuple[str, ...]] = []

    def expand(node: Any, env: dict[str, str]) -> None:
        if not isinstance(node, tuple) or not node:
            raise EvalError("non-compound node")
        head = node[0]
        if head == "and":
            for c in node[1:]:
                expand(c, env)
            return
        if head == "forall":
            var, cat = _iterable(node[1])
            for inst in _instances(problem, cat):
                expand(node[2], {**env, var: inst})
            return
        if head in ("or", "not", "imply", "exists", "forn", "forpairs", "fornpairs"):
            raise EvalError("operator %r has no unique obligation expansion" % head)
        args = tuple(_resolve(tok, env, problem) for tok in node[1:])
        out.append((head,) + args)

    for conjunct in problem.goal:
        expand(conjunct, {})
    return sorted(set(out))
