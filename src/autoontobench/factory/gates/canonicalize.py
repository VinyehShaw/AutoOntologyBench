"""Structural AST canonicalization + signature hashing (owner: S11).

Implements FINAL-BUILD-ORDER-v2 §4.4:
- isomorphic rule networks: children of commutative ops are order-normalized
- alpha-renaming of scoped (bound) variables -> v0, v1, ... in first-use order
- irrelevant (opaque, instance-level) entity renaming -> e0, e1, ...
- relation synonym normalization via an injectable synonym table

AST conventions (JSON-shaped, from `benchmark.schema.plugin_protocols.CandidateTaskAST`):
- node          := {"op": <name>, ...}
- binder node   := {"op": "forall"|"exists"|"lambda"|"let", "var"|"vars", "body"}
- relation node := {"op": "rel", "name": <relation>}
- entity leaf   := {"entity": <id>, "entity_class": "opaque"}   (renamed)
                   {"entity": <id>, "entity_class": anything-else} (kept: source-anchored)

Three signature levels (a paraphrase must never change any of them):
- family_signature   : structural skeleton — every literal/identifier abstracted
- base_goal_signature: normalized AST minus instance-varying subtrees
- instance_signature : fully normalized AST (any semantic change flips it)

All digests use `contracts.models.canonical_sha` (the shared semantics hash).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from autoontobench.contracts.models import canonical_sha

BINDER_OPS: frozenset[str] = frozenset({"forall", "exists", "lambda", "let", "bind"})
COMMUTATIVE_OPS: frozenset[str] = frozenset(
    {"and", "or", "all_of", "any_of", "rule_set", "conjunction", "disjunction", "set"}
)
ORDERED_OPS: frozenset[str] = frozenset({"sequence", "then", "implies", "if_then_else"})

# Subtrees that vary per instance without changing the goal semantics.
INSTANCE_VARYING_KEYS: frozenset[str] = frozenset(
    {
        "initial_state",
        "start_state",
        "seed",
        "entities",
        "entity_bindings",
        "wording",
        "paraphrase",
        "surface",
        "user_goal_text",
    }
)

# Minimal built-in relation synonym table; lanes inject domain tables on top.
DEFAULT_RELATION_SYNONYMS: Mapping[str, str] = {
    "partOf": "part_of",
    "part-of": "part_of",
    "BFO:0000050": "part_of",
    "is_a": "is_a",
    "isA": "is_a",
    "subClassOf": "is_a",
    "rdfs:subClassOf": "is_a",
    "locatedIn": "located_in",
    "located-in": "located_in",
    "RO:0001025": "located_in",
}


@dataclass(frozen=True)
class CanonicalResult:
    normalized: dict[str, Any]
    variable_map: dict[str, str] = field(default_factory=dict)
    entity_map: dict[str, str] = field(default_factory=dict)
    relation_rewrites: dict[str, str] = field(default_factory=dict)


def canonical_dumps(node: Any) -> str:
    """Deterministic JSON serialization shared by all signature levels."""
    return json.dumps(node, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _Normalizer:
    """Single-pass normalizer; per-tree state (var/entity scopes)."""

    def __init__(self, synonyms: Mapping[str, str]) -> None:
        self.synonyms = synonyms
        self.scopes: list[dict[str, str]] = []
        self.var_counter = 0
        self.entity_map: dict[str, str] = {}
        self.variable_map: dict[str, str] = {}
        self.relation_rewrites: dict[str, str] = {}

    # -- variable scopes -------------------------------------------------
    def _push_scope(self, names: list[str]) -> dict[str, str]:
        scope: dict[str, str] = {}
        for name in names:
            canon = f"v{self.var_counter}"
            self.var_counter += 1
            scope[name] = canon
            self.variable_map.setdefault(name, canon)
        self.scopes.append(scope)
        return scope

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _lookup_var(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return name  # free variable: keep (unbound names are source-anchored)

    def _rename_entity(self, entity_id: str) -> str:
        canon = self.entity_map.get(entity_id)
        if canon is None:
            canon = f"e{len(self.entity_map)}"
            self.entity_map[entity_id] = canon
        return canon

    # -- main recursion ---------------------------------------------------
    def norm(self, node: Any) -> Any:
        if isinstance(node, dict):
            return self._norm_node(node)
        if isinstance(node, list):
            return [self.norm(v) for v in node]
        return node

    def _norm_node(self, node: dict[str, Any]) -> dict[str, Any]:
        # entity leaf
        if "entity" in node and isinstance(node.get("entity"), str):
            out = dict(node)
            if node.get("entity_class") == "opaque":
                out["entity"] = self._rename_entity(node["entity"])
            return out
        # relation node
        if node.get("op") == "rel" and isinstance(node.get("name"), str):
            name = node["name"]
            canon = self.synonyms.get(name, name)
            if canon != name:
                self.relation_rewrites[name] = canon
            out = {k: self.norm(v) for k, v in node.items() if k != "name"}
            out["name"] = canon
            return out
        # variable reference
        if "var_ref" in node and isinstance(node.get("var_ref"), str):
            out = dict(node)
            out["var_ref"] = self._lookup_var(node["var_ref"])
            return out
        # binder
        op = node.get("op")
        if op in BINDER_OPS:
            names: list[str] = []
            if isinstance(node.get("var"), str):
                names = [node["var"]]
            elif isinstance(node.get("vars"), list):
                names = [v for v in node["vars"] if isinstance(v, str)]
            scope = self._push_scope(names)
            try:
                out = {}
                for key, value in node.items():
                    if key == "var":
                        out[key] = scope[value]
                    elif key == "vars":
                        out[key] = [scope.get(v, v) for v in value]
                    else:
                        out[key] = self.norm(value)
            finally:
                self._pop_scope()
            return out
        # generic node; commutative children get order-normalized
        out = {key: self.norm(value) for key, value in node.items()}
        if op in COMMUTATIVE_OPS:
            for key, value in out.items():
                if isinstance(value, list):
                    out[key] = sorted(
                        value, key=lambda item: canonical_dumps(item)
                    )
        return out


def canonicalize_ast(
    ast: dict[str, Any], *, synonyms: Mapping[str, str] | None = None
) -> CanonicalResult:
    """Normalize one task AST. Pure function; deterministic across processes."""
    table = dict(DEFAULT_RELATION_SYNONYMS)
    if synonyms:
        table.update(synonyms)
    worker = _Normalizer(table)
    normalized = worker.norm(ast)
    return CanonicalResult(
        normalized=normalized,
        variable_map=worker.variable_map,
        entity_map=worker.entity_map,
        relation_rewrites=worker.relation_rewrites,
    )


def _skeleton(node: Any) -> Any:
    """Structural shape: keep ops + normalized relation names, abstract leaves."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("op",):
                out[key] = value
            elif key == "name" and node.get("op") == "rel":
                out[key] = value
            else:
                out[key] = _skeleton(value)
        return out
    if isinstance(node, list):
        return [_skeleton(v) for v in node]
    return "*"


def _strip_instance_varying(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: ("*" if key in INSTANCE_VARYING_KEYS else _strip_instance_varying(value))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_strip_instance_varying(v) for v in node]
    return node


def family_signature(ast: dict[str, Any]) -> str:
    """Structural template hash: identical mechanism shape => identical signature."""
    canon = canonicalize_ast(ast)
    return canonical_sha({"skeleton": _skeleton(canon.normalized)})


def base_goal_signature(ast: dict[str, Any]) -> str:
    """Goal-level hash: ignores start-state/entity/wording variation."""
    canon = canonicalize_ast(ast)
    return canonical_sha({"goal": _strip_instance_varying(canon.normalized)})


def instance_signature(ast: dict[str, Any]) -> str:
    """Instance-level hash: full normalized AST."""
    canon = canonicalize_ast(ast)
    return canonical_sha({"instance": canon.normalized})


def source_lineage_group(source_refs: list[tuple[str, str]]) -> str:
    """Stable group id over (source_id, rule_unit_id) lineage; order-insensitive."""
    members = sorted(f"{sid}:{rid}" for sid, rid in source_refs)
    blob = "\n".join(members).encode("utf-8")
    return "lineage-" + hashlib.sha256(blob).hexdigest()[:16]
