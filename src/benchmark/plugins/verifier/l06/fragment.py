"""L06 fragment semantics: reference answer computation on a task fragment.

A *fragment* is the task universe: a set of GO term ids plus the induced edge
set (is_a / relation edges with both endpoints in the universe). All answers
are defined ON THE FRAGMENT under the declared profile go-rel-closure/1
(``benchmark.plugins.source.l06.closure``). The same operators run in the
proposer (screening) and in the native verifier (recompute); the verifier
additionally runs an independent owlrl leg (``owlrl_leg.py``).
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.source.l06.closure import (
    REGULATES_KINDS,
    compute_depths,
    descendants,
    partof_ancestors,
    regulates_ancestors,
    transitive_ancestors,
)

GO_ROOTS = ("GO:0008150", "GO:0003674", "GO:0005575")


def induce(universe: list[str], graph: dict[str, Any]) -> dict[str, Any]:
    """Fragment adjacency: edges of the full graph with both endpoints in universe."""
    u = frozenset(universe)
    isa_parents: dict[str, list[str]] = {}
    isa_children: dict[str, list[str]] = {}
    rel_out: dict[str, dict[str, list[str]]] = {}
    for s in universe:
        for p in graph["isa_parents"].get(s, []):
            if p in u:
                isa_parents.setdefault(s, []).append(p)
                isa_children.setdefault(p, []).append(s)
        for pred, targets in graph["rel_out"].get(s, {}).items():
            kept = [o for o in targets if o in u]
            if kept:
                rel_out.setdefault(s, {})[pred] = list(kept)
    return {"isa_parents": isa_parents, "isa_children": isa_children,
            "rel_out": rel_out, "universe": sorted(u)}


def fragment_edges(frag: dict[str, Any]) -> list[tuple[str, str, str]]:
    edges = [(s, "is_a", o) for s, ps in frag["isa_parents"].items() for o in ps]
    for s, by in frag["rel_out"].items():
        for pred, targets in by.items():
            edges.extend((s, pred, o) for o in targets)
    return sorted(set(edges))


def lowest_common_ancestors(anc_a: frozenset[str], anc_b: frozenset[str],
                            isa_parents: dict[str, list[str]]) -> frozenset[str]:
    ca = set(anc_a) & set(anc_b)
    if not ca:
        return frozenset()
    anc_of = {x: transitive_ancestors(x, isa_parents) for x in ca}
    lca = {x for x in ca
           if not any(x != y and x in anc_of[y] for y in ca)}
    return frozenset(lca)


def answer_query(ast: dict[str, Any], frag: dict[str, Any]) -> frozenset[str]:
    """Reference answer set for application families, computed on the fragment."""
    kind = ast["kind"]
    u = set(frag["universe"])
    ip, ic = frag["isa_parents"], frag["isa_children"]
    if kind == "isa_ancestors":
        return frozenset(x for x in transitive_ancestors(ast["seed"], ip) if x in u)
    if kind == "isa_descendants":
        return frozenset(x for x in descendants(ast["seed"], ic) if x in u)
    if kind == "common_ancestors":
        a = transitive_ancestors(ast["a"], ip) & frozenset(u)
        b = transitive_ancestors(ast["b"], ip) & frozenset(u)
        return frozenset(a & b)
    if kind == "lca":
        a = transitive_ancestors(ast["a"], ip)
        b = transitive_ancestors(ast["b"], ip)
        return frozenset(x for x in lowest_common_ancestors(a, b, ip) if x in u)
    if kind == "partof_ancestors":
        partof_targets = {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}
        return frozenset(x for x in partof_ancestors(ast["seed"], ip, partof_targets)
                         if x in u)
    if kind == "regulators":
        return frozenset(x for x in regulates_ancestors(
            ast["seed"], ip, frag["rel_out"], isa_children=ic) if x in u)
    if kind == "cross_isa_partof":
        partof_targets = {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}
        desc = descendants(ast["isa_root"], ic)
        po = partof_ancestors(ast["partof_target"], ip, partof_targets)
        return frozenset(x for x in (desc & po) if x in u)
    if kind == "branch_difference":
        big = descendants(ast["parent"], ic)
        small = descendants(ast["child"], ic)
        return frozenset(x for x in (big - small) if x in u)
    if kind == "depth_band":
        depth = compute_depths(ip)
        return frozenset(x for x in u if ast["lo"] <= depth.get(x, 0) <= ast["hi"])
    if kind == "multi_parent":
        return frozenset(x for x in u if len(ip.get(x, [])) >= 2)
    if kind == "annotation_closure":
        out: set[str] = set()
        partof_targets = {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}
        for t in ast["direct_terms"]:
            if t not in u:
                continue
            out.add(t)
            out.update(x for x in transitive_ancestors(t, ip) if x in u)
            out.update(x for x in partof_ancestors(t, ip, partof_targets) if x in u)
        return frozenset(out)
    if kind == "shared_ancestors":
        partof_targets = {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}
        inter: set[str] | None = None
        for gene, direct in sorted(ast["direct_terms_by_gene"].items()):
            prop: set[str] = set()
            for t in direct:
                if t not in u:
                    continue
                prop.add(t)
                prop.update(x for x in transitive_ancestors(t, ip) if x in u)
                prop.update(x for x in partof_ancestors(t, ip, partof_targets) if x in u)
            inter = prop if inter is None else inter & prop
        shared = (inter or set()) - set(GO_ROOTS)
        return frozenset(shared)
    raise ValueError(f"answer_query: unsupported kind {kind!r}")


def apply_repair(ast: dict[str, Any], frag: dict[str, Any],
                 fix: dict[str, Any]) -> dict[str, Any]:
    """Return a new fragment adjacency with ``fix`` applied to the *broken* fragment."""
    ip = {s: list(ps) for s, ps in frag["isa_parents"].items()}
    ro = {s: {p: list(ts) for p, ts in by.items()} for s, by in frag["rel_out"].items()}
    kind = ast["kind"]
    if kind == "repair_isa_parent_swap":
        t = ast["term"]
        ip[t] = [p for p in ip.get(t, []) if p != ast["wrong_parent"]]
        if fix["parent"] not in ip[t]:
            ip[t].append(fix["parent"])
            ip[t].sort()
    elif kind == "repair_missing_isa_edge":
        t = ast["term"]
        ip.setdefault(t, [])
        if fix["parent"] not in ip[t]:
            ip[t].append(fix["parent"])
            ip[t].sort()
    elif kind == "repair_orphan_reconnect":
        t = ast["term"]
        ip[t] = sorted(set(ip.get(t, [])) | set(fix["parents"]))
    elif kind == "repair_relation_type_confusion":
        s, o = ast["subject"], ast["object"]
        wrong = ast["wrong_pred"]
        if s in ro and wrong in ro[s] and o in ro[s][wrong]:
            ro[s][wrong] = [x for x in ro[s][wrong] if x != o]
            if not ro[s][wrong]:
                del ro[s][wrong]
        ro.setdefault(s, {}).setdefault(fix["predicate"], [])
        if o not in ro[s][fix["predicate"]]:
            ro[s][fix["predicate"]].append(o)
            ro[s][fix["predicate"]].sort()
    else:
        raise ValueError(f"apply_repair: unsupported kind {kind!r}")
    ic: dict[str, list[str]] = {}
    for s, ps in ip.items():
        for p in ps:
            ic.setdefault(p, []).append(s)
    for p in ic:
        ic[p].sort()
    return {"isa_parents": ip, "isa_children": ic, "rel_out": ro,
            "universe": frag["universe"]}


def apply_tamper(ast: dict[str, Any], frag: dict[str, Any]) -> dict[str, Any]:
    """The broken fragment: exactly the tampering the public edge list shows."""
    kind = ast["kind"]
    ip = {s: list(ps) for s, ps in frag["isa_parents"].items()}
    ro = {s: {p: list(ts) for p, ts in by.items()} for s, by in frag["rel_out"].items()}
    if kind == "repair_isa_parent_swap":
        t = ast["term"]
        ip[t] = [p for p in ip.get(t, []) if p != ast["correct_parent"]]
        ip[t].append(ast["wrong_parent"])
        ip[t].sort()
    elif kind == "repair_missing_isa_edge":
        t = ast["term"]
        ip[t] = [p for p in ip.get(t, []) if p != ast["removed_parent"]]
    elif kind == "repair_orphan_reconnect":
        ip[ast["term"]] = []
    elif kind == "repair_relation_type_confusion":
        s, o = ast["subject"], ast["object"]
        correct, wrong = ast["correct_pred"], ast["wrong_pred"]
        if s in ro and correct in ro.get(s, {}):
            ro[s][correct] = [x for x in ro[s][correct] if x != o]
            if not ro[s][correct]:
                del ro[s][correct]
        ro.setdefault(s, {}).setdefault(wrong, [])
        if o not in ro[s][wrong]:
            ro[s][wrong].append(o)
            ro[s][wrong].sort()
    else:
        raise ValueError(f"apply_tamper: unsupported kind {kind!r}")
    ic: dict[str, list[str]] = {}
    for s, ps in ip.items():
        for p in ps:
            ic.setdefault(p, []).append(s)
    for p in ic:
        ic[p].sort()
    return {"isa_parents": ip, "isa_children": ic, "rel_out": ro,
            "universe": frag["universe"]}


def closure_signature(frag: dict[str, Any], focus: list[str]) -> dict[str, dict[str, list[str]]]:
    """Closure fingerprint of the fragment restricted to focus terms — the
    quantity a repair must restore to the reference value."""
    partof_targets = {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}
    out: dict[str, dict[str, list[str]]] = {}
    for t in focus:
        out[t] = {
            "isa": sorted(transitive_ancestors(t, frag["isa_parents"])),
            "partof": sorted(partof_ancestors(t, frag["isa_parents"], partof_targets)),
            "reg": sorted(regulates_ancestors(
                t, frag["isa_parents"], frag["rel_out"],
                isa_children=frag["isa_children"])),
        }
    return out
