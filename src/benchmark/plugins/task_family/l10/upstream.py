"""L10 leg B: the vendored ``bddl`` package from the pinned BEHAVIOR-1K snapshot.

Everything here is a thin, cached adapter: the vendored package is imported
from ``external_knowledge/pinned/behavior-1k-bddl/.../bddl3`` (path-guarded so
no other ``bddl`` on sys.path can shadow it) and used to (1) re-parse problem
text and (2) re-evaluate goal satisfaction with the upstream expression-tree
semantics. ``engine.py`` (leg A) implements the same two computations
independently; the verifier requires agreement.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l10 import engine
from benchmark.plugins.task_family.l10.engine import ParsedProblem


class UpstreamError(ValueError):
    """The vendored bddl package failed to parse/compile/evaluate."""


_CACHE: dict[str, Any] = {}


def _load_upstream() -> Any:
    if "mod" in _CACHE:
        return _CACHE["mod"]
    bddl3_root = engine.BDDL_ROOT.parent
    if not (bddl3_root / "bddl" / "__init__.py").is_file():
        raise UpstreamError("vendored bddl package missing under %s" % bddl3_root)
    added = False
    if str(bddl3_root) not in sys.path:
        sys.path.insert(0, str(bddl3_root))
        added = True
    try:
        import bddl  # type: ignore
        mod_file = Path(bddl.__file__).resolve()
        if not str(mod_file).startswith(str(bddl3_root.resolve())):
            raise UpstreamError("bddl shadowed by %s (not the pinned tree)" % mod_file)
        from bddl import condition_evaluation, parsing, predicates  # type: ignore
    except ImportError as exc:
        raise UpstreamError("vendored bddl import failed: %s" % exc) from exc
    finally:
        if added:
            sys.path.remove(str(bddl3_root))
    token_to_cls = dict(predicates.TOKEN_TO_PREDICATE)
    cls_to_token = {v: k for k, v in token_to_cls.items()}
    token_mapping = dict(condition_evaluation.TOKEN_MAPPING)
    mod = {
        "parsing": parsing,
        "condition_evaluation": condition_evaluation,
        "token_mapping": token_mapping,
        "cls_to_token": cls_to_token,
        "predicate_arities": {tok: int(cls.arity) for tok, cls in token_to_cls.items()},
    }
    _CACHE["mod"] = mod
    return mod


def upstream_operator_tokens() -> frozenset[str]:
    """Operator tokens per the vendored TOKEN_MAPPING (table-consistency gate)."""
    return frozenset(_load_upstream()["token_mapping"])


def upstream_predicate_arities() -> dict[str, int]:
    """Predicate arities per the vendored predicate classes (Unary=1/Binary=2)."""
    return dict(_load_upstream()["predicate_arities"])


def parse_upstream(text: str) -> ParsedProblem:
    """Parse with the vendored parser; normalize into the leg-A shape."""
    mod = _load_upstream()
    parsing = mod["parsing"]
    try:
        name, objects, initial_state, goal_state = parsing.parse_problem(
            "", 0, "behavior-1k", predefined_problem=text)
    except Exception as exc:  # upstream raises bare Exception types
        raise UpstreamError("upstream parse failed: %s: %s"
                            % (type(exc).__name__, str(exc)[:200])) from exc
    init = tuple(engine.canon(lit) for lit in initial_state)
    goal = tuple(engine.canon(g) for g in goal_state)
    top = engine.scan_tokens(text)
    goal_raw = None
    for group in top[1:]:
        if isinstance(group, list) and group and group[0] == ":goal":
            goal_raw = engine.canon(group[1])
    return ParsedProblem(
        name=name,
        domain="behavior-1k",
        objects=tuple(sorted((c, tuple(insts)) for c, insts in objects.items())),
        init=init,
        goal=goal,
        goal_raw=goal_raw,
    )


def parse_agrees(text: str, mine: ParsedProblem) -> bool:
    """Leg A/B structural agreement: objects, init, goal conjuncts."""
    theirs = parse_upstream(text)
    return (
        theirs.objects == mine.objects
        and sorted(map(repr, theirs.init)) == sorted(map(repr, mine.init))
        and sorted(map(repr, theirs.goal)) == sorted(map(repr, mine.goal))
    )


def upstream_goal_satisfied(text: str) -> bool:
    """Goal satisfaction per the vendored compiler over the closed-world init."""
    mod = _load_upstream()
    ce = mod["condition_evaluation"]
    cls_to_token = mod["cls_to_token"]
    try:
        _name, objects, initial_state, goal_state = mod["parsing"].parse_problem(
            "", 0, "behavior-1k", predefined_problem=text)
        truth = frozenset(
            tuple(lit) for lit in initial_state
            if isinstance(lit, list) and lit and lit[0] != "not"
        )

        def evaluate_fn(predicate_cls: Any, *entities: str) -> bool:
            token = cls_to_token.get(predicate_cls)
            return token is not None and (token,) + tuple(entities) in truth

        scope = ce.create_scope(objects)
        compiled = ce.compile_state(goal_state, scope=scope, object_map=objects,
                                    generate_ground_options=False)
        satisfied, _results = ce.evaluate_state(compiled, evaluate_fn)
        return bool(satisfied)
    except UpstreamError:
        raise
    except Exception as exc:
        raise UpstreamError("upstream compile/eval failed: %s: %s"
                            % (type(exc).__name__, str(exc)[:200])) from exc
