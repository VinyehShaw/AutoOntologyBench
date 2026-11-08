"""L10 static checks: BDDL predicate/arity/binding/category/operator rules.

Every check is a pure function of (parsed problem, knowledge tables). The
tables come from the pinned snapshot; a perturbed table yields a perturbed
verdict — that flip is the lane's semantic-witness mechanism.

Finding codes:
  PRED_UNKNOWN   — atom head not declared in the domain :predicates
  ARITY          — atom argument count != domain arity
  UNBOUND        — ground argument not declared in :objects (and not a valid
                   room constant in inroom position 2)
  VAR_UNBOUND    — ?-variable used outside any quantifier that binds it
  OP_UNKNOWN     — compound-expression head is neither operator nor predicate
  QCAT_UNDECLARED — quantifier iterable category not declared in :objects
  CATEGORY_UNKNOWN — :objects category not in the pinned synset table
  INIT_NOT_GROUND — ?-variable inside :init
  MALFORMED      — structural defect (bad quantifier shape, empty group, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.plugins.task_family.l10.engine import (
    CONNECTIVES,
    OPERATORS,
    QUANTIFIERS,
    KnowledgeTables,
    ParsedProblem,
)


@dataclass(frozen=True)
class Finding:
    code: str
    section: str  # "objects" | "init" | "goal"
    detail: str
    node: Any = None  # canonical offending node (for defect localization)


def _is_var(tok: Any) -> bool:
    return isinstance(tok, str) and tok.startswith("?")


def _atom_finding(code: str, section: str, node: Any, detail: str) -> Finding:
    return Finding(code=code, section=section, detail=detail, node=node)


def _check_atom(node: Any, section: str, env: dict[str, str], problem: ParsedProblem,
                tables: KnowledgeTables, findings: list[Finding]) -> None:
    """Check one leaf atom (predicate application) in place."""
    pred = node[0]
    args = node[1:]
    arity = tables.predicate_map.get(pred)
    if arity is None:
        findings.append(_atom_finding(
            "PRED_UNKNOWN", section, node,
            "predicate %r not in domain :predicates" % pred))
        return
    if len(args) != arity:
        findings.append(_atom_finding(
            "ARITY", section, node,
            "predicate %r expects %d args, got %d" % (pred, arity, len(args))))
        return
    instances = problem.instances
    for pos, arg in enumerate(args):
        if _is_var(arg):
            if arg in env:
                continue
            # BDDL idiom: ?instance_name is a direct reference to a declared
            # instance (upstream strips "?" and resolves through the scope).
            if arg[1:] in instances:
                continue
            findings.append(_atom_finding(
                "VAR_UNBOUND", section, node,
                "variable %r is neither quantifier-bound nor a declared instance" % arg))
            continue
        if arg in instances:
            continue
        # inroom's second argument is a room constant, not an object instance.
        if pred == "inroom" and pos == 1 and arg in tables.rooms:
            continue
        findings.append(_atom_finding(
            "UNBOUND", section, node,
            "argument %r is not a declared object instance%s" % (
                arg, " or room constant" if pred == "inroom" and pos == 1 else "")))


def _walk_goal(node: Any, env: dict[str, str], problem: ParsedProblem,
               tables: KnowledgeTables, findings: list[Finding]) -> None:
    if not isinstance(node, tuple) or not node:
        findings.append(Finding("MALFORMED", "goal", "non-compound or empty node", node))
        return
    head = node[0]
    if head in CONNECTIVES:
        if head == "not":
            if len(node) != 2:
                findings.append(Finding("MALFORMED", "goal", "not expects 1 child", node))
                return
            _walk_goal(node[1], env, problem, tables, findings)
            return
        if head == "imply":
            if len(node) != 3:
                findings.append(Finding("MALFORMED", "goal", "imply expects 2 children", node))
                return
            _walk_goal(node[1], env, problem, tables, findings)
            _walk_goal(node[2], env, problem, tables, findings)
            return
        for child in node[1:]:
            _walk_goal(child, env, problem, tables, findings)
        return
    if head in QUANTIFIERS:
        _walk_quantifier(node, env, problem, tables, findings)
        return
    if head in tables.predicate_map:
        _check_atom(node, "goal", env, problem, tables, findings)
        return
    findings.append(Finding(
        "OP_UNKNOWN", "goal",
        "head %r is neither a BDDL operator nor a domain predicate" % (head,), node))


def _iterable_category(form: Any) -> tuple[str, str] | None:
    """(?var - category) -> (var, category); None when malformed."""
    if (isinstance(form, tuple) and len(form) == 3 and _is_var(form[0])
            and form[1] == "-" and isinstance(form[2], str)):
        return form[0], form[2]
    return None


def _walk_quantifier(node: Any, env: dict[str, str], problem: ParsedProblem,
                     tables: KnowledgeTables, findings: list[Finding]) -> None:
    head = node[0]
    declared = {cat for cat, _ in problem.objects}
    idx = 0
    count_ok = True
    if head in ("forn", "fornpairs"):
        if len(node) < 2 or not isinstance(node[1], tuple) or not node[1] \
                or not node[1][0].isdigit():
            findings.append(Finding("MALFORMED", "goal", "%s missing (N) count" % head, node))
            return
        idx = 1
    n_iterables = 2 if head in ("forpairs", "fornpairs") else 1
    if len(node) != 1 + idx + n_iterables + 1:
        findings.append(Finding("MALFORMED", "goal", "%s wrong shape" % head, node))
        return
    new_env = dict(env)
    for k in range(n_iterables):
        parsed = _iterable_category(node[1 + idx + k])
        if parsed is None:
            findings.append(Finding("MALFORMED", "goal",
                                    "%s iterable %d not (?var - category)" % (head, k), node))
            return
        var, cat = parsed
        if cat not in declared:
            findings.append(Finding(
                "QCAT_UNDECLARED", "goal",
                "quantifier category %r not declared in :objects" % cat, node))
        new_env[var] = cat
    body = node[1 + idx + n_iterables]
    _walk_goal(body, new_env, problem, tables, findings)


def check_problem(problem: ParsedProblem, tables: KnowledgeTables) -> list[Finding]:
    """Run all static checks; returns the (possibly empty) finding list."""
    findings: list[Finding] = []
    for cat, _insts in problem.objects:
        if cat != "object" and cat not in tables.synsets:
            findings.append(Finding(
                "CATEGORY_UNKNOWN", "objects",
                "category %r not in pinned synset table" % cat,
                (":objects", cat)))
    for lit in problem.init:
        atom = lit
        if isinstance(lit, tuple) and lit and lit[0] == "not":
            if len(lit) != 2:
                findings.append(Finding("MALFORMED", "init", "bad negated literal", lit))
                continue
            atom = lit[1]
        if not isinstance(atom, tuple) or not atom:
            findings.append(Finding("MALFORMED", "init", "non-compound literal", lit))
            continue
        if any(_is_var(tok) for tok in atom[1:]):
            findings.append(Finding("INIT_NOT_GROUND", "init",
                                    "init literal contains a ?-variable", lit))
            continue
        _check_atom(atom, "init", {}, problem, tables, findings)
    for conjunct in problem.goal:
        _walk_goal(conjunct, {}, problem, tables, findings)
    return findings


def count_goal_leaves(node: Any) -> int:
    """Number of predicate-atom leaves in a goal expression (structural size)."""
    if not isinstance(node, tuple) or not node:
        return 0
    head = node[0]
    if head in ("and", "or"):
        return sum(count_goal_leaves(c) for c in node[1:])
    if head == "not":
        return count_goal_leaves(node[1]) if len(node) == 2 else 0
    if head == "imply":
        return sum(count_goal_leaves(c) for c in node[1:]) if len(node) == 3 else 0
    if head in ("forall", "exists"):
        return count_goal_leaves(node[2]) if len(node) == 3 else 0
    if head == "forn":
        return count_goal_leaves(node[3]) if len(node) == 4 else 0
    if head in ("forpairs", "fornpairs"):
        return count_goal_leaves(node[-1])
    return 1


def count_quantifiers(node: Any) -> int:
    if not isinstance(node, tuple) or not node:
        return 0
    head = node[0]
    if head in QUANTIFIERS:
        sub = node[-1]
        return 1 + count_quantifiers(sub)
    if head in ("and", "or"):
        return sum(count_quantifiers(c) for c in node[1:])
    if head == "not":
        return count_quantifiers(node[1]) if len(node) == 2 else 0
    if head == "imply":
        return sum(count_quantifiers(c) for c in node[1:]) if len(node) == 3 else 0
    return 0


def complexity_of(problem: ParsedProblem) -> str:
    """structural-low/medium/high from goal size + quantifier depth."""
    leaves = sum(count_goal_leaves(c) for c in problem.goal)
    quants = sum(count_quantifiers(c) for c in problem.goal)
    if leaves >= 7 or quants >= 3:
        return "high"
    if leaves >= 3 or quants >= 1:
        return "medium"
    return "low"
