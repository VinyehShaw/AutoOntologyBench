"""L06 declared reasoning profile ``go-rel-closure/1`` over the GO graph.

The profile is a *bounded structural profile* (declared per FINAL-BUILD-ORDER
L06 "推理范围按 reasoner Profile 声明"), computed identically by the index
builder and recomputed identically by the native verifier. It is NOT a claim
of full OWL-RL entailment; owlrl is used separately as an independent
subClassOf-closure leg on task subgraphs.

Graph model (go-basic): nodes are non-obsolete GO classes; edges are
  - ``is_a``      (rdfs:subClassOf between two GO classes)
  - ``part_of``   (BFO_0000050), ``regulates`` (RO_0002211),
    ``positively_regulates`` (RO_0002213), ``negatively_regulates``
    (RO_0002212), ``has_part`` (BFO_0000051), ``occurs_in`` (BFO_0000066),
    ``capable_of`` (RO_0002215), ``capable_of_part_of`` (RO_0002216)
  extracted either from OBO ``relationship:`` stanzas or from OWL
  ``owl:Restriction owl:onProperty P owl:someValuesFrom T`` superclasses.

Profile rules (edge direction: s --p--> o means "s p o"):

  R1  is_a is transitive.
  R2  part_of is transitive.
  R3  s is_a o  ∧  o part_of t  ⇒  s part_of t   (subject-side is_a)
  R4  s part_of o  ∧  o is_a t  ⇒  s part_of t   (object-side is_a)
  R5  positively_regulates ⊆ regulates ; negatively_regulates ⊆ regulates
      (sub-property rollup for the generic "regulates" view)
  R6  s is_a o  ∧  o R t  ⇒  s R t  for R in {regulates, positively_regulates,
      negatively_regulates}  (subject-side is_a propagation of regulation)
  R7  s R o  ∧  o is_a t  ⇒  s R t  for R in {regulates, positively_regulates,
      negatively_regulates}  (object-side is_a propagation of regulation)
  R8  regulates is NOT transitive: no s regulates o ∧ o regulates t chain rule.

Closure operators (all strict, i.e. exclude the seed itself unless a cycle
through other nodes reaches it):

  isa_anc(t)      = { a : t is_a+ a }                                     (R1)
  partof_anc(t)   = { a : t (is_a ∪ part_of)+ a ∧ path uses ≥1 part_of }  (R1-R4)
  reg_anc(t)      = { s : s is_a* s' ∧ s' R o' ∧ o' is_a* t,
                        R ∈ {regulates, pos, neg} }                       (R5-R7)
  reg_pos_anc(t)  = same with R = positively_regulates only (R6,R7)
  reg_neg_anc(t)  = same with R = negatively_regulates only (R6,R7)

All operators are computed by deterministic fixpoint over the adjacency maps.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

REGULATES_KINDS = ("regulates", "positively_regulates", "negatively_regulates")


def transitive_ancestors(seed: str, parents: dict[str, list[str]]) -> frozenset[str]:
    """is_a transitive closure (strict ancestors)."""
    seen: set[str] = set()
    stack = list(parents.get(seed, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(p for p in parents.get(node, ()) if p not in seen)
    seen.discard(seed)
    return frozenset(seen)


def partof_ancestors(seed: str, isa_parents: dict[str, list[str]],
                     partof_targets: dict[str, list[str]]) -> frozenset[str]:
    """(is_a ∪ part_of)+ ancestors requiring ≥1 part_of edge (R1-R4)."""
    # state = (node, used_partof)
    seen: set[tuple[str, bool]] = set()
    out: set[str] = set()
    stack: list[tuple[str, bool]] = [(seed, False)]
    while stack:
        node, used = stack.pop()
        if (node, used) in seen:
            continue
        seen.add((node, used))
        if used:
            out.add(node)
        for p in isa_parents.get(node, ()):
            if (p, used) not in seen:
                stack.append((p, used))
        for t in partof_targets.get(node, ()):
            if (t, True) not in seen:
                stack.append((t, True))
    out.discard(seed)
    return frozenset(out)


def regulates_ancestors(seed: str, isa_parents: dict[str, list[str]],
                        rel_out: dict[str, dict[str, list[str]]],
                        kinds: Iterable[str] = REGULATES_KINDS,
                        isa_children: dict[str, list[str]] | None = None) -> frozenset[str]:
    """Regulators of ``seed`` under R5-R7 (non-transitive single R step).

    R7 object-side: regulating any is_a *descendant* of seed (or seed itself)
    entails regulating seed. R6 subject-side: any is_a *descendant* of a
    direct regulator also regulates seed. (Both directions follow the profile
    docstring and the owlrl propertyChainAxiom encoding; is_a edges point
    child -> parent, so propagation runs downward on both sides.)"""
    if isa_children is None:
        isa_children = _children_of(isa_parents)
    targets = {seed} | set(descendants(seed, isa_children))
    kinds = tuple(kinds)
    out: set[str] = set()
    for s, by_pred in rel_out.items():
        for k in kinds:
            if any(o in targets for o in by_pred.get(k, ())):
                out.add(s)
                break
    frontier = list(out)
    seen = set(out)
    while frontier:
        node = frontier.pop()
        for c in isa_children.get(node, ()):
            if c not in seen:
                seen.add(c)
                frontier.append(c)
    seen.discard(seed)
    return frozenset(seen)


def _children_of(isa_parents: dict[str, list[str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for s, ps in isa_parents.items():
        for p in ps:
            children.setdefault(p, []).append(s)
    return children


def regulates_ancestors_fast(seed: str, isa_desc_of: dict[str, frozenset[str]],
                              rel_in_reg: dict[str, list[str]],
                              isa_children: dict[str, list[str]]) -> frozenset[str]:
    """Same operator as ``regulates_ancestors`` but using precomputed isa
    descendant sets and a reverse regulation edge map (index-time fast path)."""
    targets = {seed} | set(isa_desc_of.get(seed, frozenset()))
    direct: set[str] = set()
    for o in targets:
        direct.update(rel_in_reg.get(o, ()))
    out = set(direct)
    stack = list(out)
    while stack:
        node = stack.pop()
        for c in isa_children.get(node, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    out.discard(seed)
    return frozenset(out)


def descendants(seed: str, isa_children: dict[str, list[str]]) -> frozenset[str]:
    """is_a transitive closure downward (strict descendants)."""
    seen: set[str] = set()
    stack = list(isa_children.get(seed, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(c for c in isa_children.get(node, ()) if c not in seen)
    seen.discard(seed)
    return frozenset(seen)


def compute_depths(isa_parents: dict[str, list[str]]) -> dict[str, int]:
    """Longest is_a path length to a root for every term (Kahn topological
    pass; the GO is_a graph over non-obsolete terms is a DAG)."""
    nodes = set(isa_parents)
    for ps in isa_parents.values():
        nodes.update(ps)
    depth: dict[str, int] = {}
    # iterate until fixpoint; DAG guarantees convergence in O(edges) rounds
    pending = set(nodes)
    while pending:
        progressed: list[str] = []
        for n in pending:
            ps = isa_parents.get(n, [])
            if all(p in depth for p in ps):
                depth[n] = 0 if not ps else 1 + max(depth[p] for p in ps)
                progressed.append(n)
        if not progressed:  # cycle guard: assign remaining 0 (should not happen)
            for n in pending:
                depth[n] = 0
            break
        pending.difference_update(progressed)
    return depth


def bfs_order(seed: str, children: dict[str, list[str]]) -> list[str]:
    """Deterministic BFS order (sorted neighbors) — for stable sampling docs."""
    out: list[str] = []
    seen = {seed}
    dq: deque[str] = deque([seed])
    while dq:
        node = dq.popleft()
        out.append(node)
        for c in sorted(children.get(node, ())):
            if c not in seen:
                seen.add(c)
                dq.append(c)
    return out
