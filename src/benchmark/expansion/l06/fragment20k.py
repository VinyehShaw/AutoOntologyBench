"""L06 20K-expansion fragment semantics (leg 1): reference answers, tampering,
repair application and the polarity-aware closure signature.

A *fragment* is the task universe exactly as in the legacy verifier
(``benchmark/plugins/verifier/l06/fragment.py``): a set of GO term ids plus
the induced is_a / relation edges. All answers are defined ON THE FRAGMENT
under the declared profile ``go-rel-closure/1``. New operators introduced
here (each is a profile-consistent extension, and each closure kind has an
independent owlrl leg in ``owlrl20k.py`` that must agree exactly):

  posregulators(seed)      R6/R7 with R = positively_regulates only
  negregulators(seed)      R6/R7 with R = negatively_regulates only
  generic_only_regulators  regulators(seed) - pos - neg  (paths that use no
                           polarity edge at all)
  dual_regulators(a, b)    regulators(a) ∩ regulators(b)
  partof_isa_overlap       partof_ancestors(seed) ∩ isa_ancestors(seed)
  reg_subject_descendants  descendants(T) ∩ {x : x or an is_a-ancestor of x
                           has an outgoing regulation edge in the fragment}
                           (R6 subject-side inheritance)
  partof_depth             longest path to a fragment root over the merged
                           is_a ∪ part_of parent map (partonomy-aware depth)
  exp_annotation_closure   evidence-gated propagated annotation closure
  evidence_discriminated   prop(exp annotations) - prop(non-exp annotations)
  term_annotated_genes     reverse join: genes whose direct annotations
                           propagate (isa/part_of) to the given term
  exp_term_genes           same, gated by experimental evidence codes

New repair kinds (tamper / fix semantics):

  repair_partof_edge_deletion   tamper: delete part_of edge (s,o);
                                fix: re-add {'subject': s, 'object': o}
  repair_spurious_isa_edge      tamper: ADD spurious is_a edge (t, p*);
                                fix: remove {'term': t, 'remove_parent': p*}
  repair_regulation_polarity_swap  tamper: relabel pos<->neg on (s,o);
                                fix: {'subject', 'object', 'predicate'}
  repair_spurious_partof_edge   tamper: ADD spurious part_of edge (s, o*);
                                fix: remove {'subject': s, 'remove_object': o*}

The closure signature is polarity-aware: per focus term it records the isa,
partof, rolled-up reg, reg_pos and reg_neg closure lists; a repair is valid
iff applying the submitted fix to the tampered fragment reproduces the
reference signature exactly (and the gold identity matches).
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

POS_KINDS = ("positively_regulates",)
NEG_KINDS = ("negatively_regulates",)

REPAIR_KINDS_20K = {
    "repair_partof_edge_deletion", "repair_spurious_isa_edge",
    "repair_regulation_polarity_swap", "repair_spurious_partof_edge",
}
GENE_ANSWER_KINDS = {"term_annotated_genes", "exp_term_genes"}


def partof_targets_of(frag: dict[str, Any]) -> dict[str, list[str]]:
    return {s: by.get("part_of", []) for s, by in frag["rel_out"].items()}


def mixed_depths(frag: dict[str, Any]) -> dict[str, int]:
    """Longest path to a fragment root over the merged is_a ∪ part_of parent
    map (both edge types point 'upward': child->parent, part->whole)."""
    merged: dict[str, list[str]] = {}
    for s, ps in frag["isa_parents"].items():
        merged.setdefault(s, []).extend(ps)
    for s, ts in partof_targets_of(frag).items():
        merged.setdefault(s, []).extend(ts)
    return compute_depths(merged)


def answer_query20k(ast: dict[str, Any], frag: dict[str, Any],
                    evidence: dict[str, Any] | None = None) -> Any:
    """Reference answer for the 20K-expansion kinds, computed on the fragment.

    ``evidence`` is the verifier/proposer-shared evidence index rebuilt from
    the pinned GAF (never trusted from the stored AST): it maps
    ``gene -> {"exp": set, "other": set}`` of direct P-aspect terms split by
    evidence class. Organization kinds require it.
    """
    kind = ast["kind"]
    u = set(frag["universe"])
    ip, ic = frag["isa_parents"], frag["isa_children"]
    ro = frag["rel_out"]
    pt = partof_targets_of(frag)
    if kind == "posregulators":
        return frozenset(x for x in regulates_ancestors(
            ast["seed"], ip, ro, kinds=POS_KINDS, isa_children=ic) if x in u)
    if kind == "negregulators":
        return frozenset(x for x in regulates_ancestors(
            ast["seed"], ip, ro, kinds=NEG_KINDS, isa_children=ic) if x in u)
    if kind == "generic_only_regulators":
        allr = set(regulates_ancestors(ast["seed"], ip, ro,
                                       kinds=REGULATES_KINDS, isa_children=ic))
        pos = set(regulates_ancestors(ast["seed"], ip, ro,
                                      kinds=POS_KINDS, isa_children=ic))
        neg = set(regulates_ancestors(ast["seed"], ip, ro,
                                      kinds=NEG_KINDS, isa_children=ic))
        return frozenset(x for x in (allr - pos - neg) if x in u)
    if kind == "dual_regulators":
        ra = set(regulates_ancestors(ast["a"], ip, ro,
                                     kinds=REGULATES_KINDS, isa_children=ic))
        rb = set(regulates_ancestors(ast["b"], ip, ro,
                                     kinds=REGULATES_KINDS, isa_children=ic))
        return frozenset(x for x in (ra & rb) if x in u)
    if kind == "partof_isa_overlap":
        po = set(partof_ancestors(ast["seed"], ip, pt))
        isa = set(transitive_ancestors(ast["seed"], ip))
        return frozenset(x for x in (po & isa) if x in u)
    if kind == "reg_subject_descendants":
        reg_subjects = {z for z, by in ro.items()
                        if any(by.get(k) for k in REGULATES_KINDS)}
        anc_cache: dict[str, frozenset[str]] = {}

        def anc(z: str) -> frozenset[str]:
            if z not in anc_cache:
                anc_cache[z] = transitive_ancestors(z, ip)
            return anc_cache[z]

        qualifying = {x for x in u
                      if x in reg_subjects
                      or (set(anc(x)) & reg_subjects)}
        return frozenset(x for x in (qualifying & descendants(ast["root"], ic))
                         if x in u)
    if kind == "partof_depth_band":
        depth = mixed_depths(frag)
        return frozenset(x for x in u
                         if ast["lo"] <= depth.get(x, 0) <= ast["hi"])
    if kind == "exp_annotation_closure":
        ev = (evidence or {}).get(ast["gene"], {})
        out: set[str] = set()
        for t in ev.get("exp", ()):  # evidence recomputed from pinned GAF
            if t not in u:
                continue
            out.add(t)
            out.update(x for x in transitive_ancestors(t, ip) if x in u)
            out.update(x for x in partof_ancestors(t, ip, pt) if x in u)
        return frozenset(out)
    if kind == "evidence_discriminated":
        ev = (evidence or {}).get(ast["gene"], {})
        prop_exp: set[str] = set()
        for t in ev.get("exp", ()):
            if t not in u:
                continue
            prop_exp.add(t)
            prop_exp.update(x for x in transitive_ancestors(t, ip) if x in u)
            prop_exp.update(x for x in partof_ancestors(t, ip, pt) if x in u)
        prop_other: set[str] = set()
        for t in ev.get("other", ()):
            if t not in u:
                continue
            prop_other.add(t)
            prop_other.update(x for x in transitive_ancestors(t, ip) if x in u)
            prop_other.update(x for x in partof_ancestors(t, ip, pt) if x in u)
        return frozenset(x for x in (prop_exp - prop_other) if x in u)
    if kind in GENE_ANSWER_KINDS:
        term = ast["term"]
        out: set[str] = set()
        for g in ast["candidate_genes"]:
            direct = ((evidence or {}).get(g, {}).get("exp", ())
                      if kind == "exp_term_genes"
                      else (evidence or {}).get(g, {}).get("all", ()))
            for t in direct:
                if t not in u:
                    continue
                if (t == term or term in transitive_ancestors(t, ip)
                        or term in partof_ancestors(t, ip, pt)):
                    out.add(g)
                    break
        return frozenset(out)
    raise ValueError(f"answer_query20k: unsupported kind {kind!r}")


# ------------------------------------------------------------------ repairs

def apply_tamper20k(ast: dict[str, Any],
                    frag: dict[str, Any]) -> dict[str, Any]:
    """The broken fragment: exactly the tampering the public edge list shows."""
    kind = ast["kind"]
    ip = {s: list(ps) for s, ps in frag["isa_parents"].items()}
    ro = {s: {p: list(ts) for p, ts in by.items()} for s, by in frag["rel_out"].items()}
    if kind == "repair_partof_edge_deletion":
        s, o = ast["subject"], ast["object"]
        ro.setdefault(s, {})["part_of"] = [
            x for x in ro.get(s, {}).get("part_of", []) if x != o]
        if not ro[s]["part_of"]:
            del ro[s]["part_of"]
    elif kind == "repair_spurious_isa_edge":
        t, p = ast["term"], ast["spurious_parent"]
        ip.setdefault(t, [])
        if p not in ip[t]:
            ip[t].append(p)
            ip[t].sort()
    elif kind == "repair_regulation_polarity_swap":
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
    elif kind == "repair_spurious_partof_edge":
        s, o = ast["subject"], ast["spurious_target"]
        ro.setdefault(s, {}).setdefault("part_of", [])
        if o not in ro[s]["part_of"]:
            ro[s]["part_of"].append(o)
            ro[s]["part_of"].sort()
    else:
        raise ValueError(f"apply_tamper20k: unsupported kind {kind!r}")
    return _rebuild(ip, ro, frag["universe"])


def apply_repair20k(ast: dict[str, Any], frag: dict[str, Any],
                    fix: dict[str, Any]) -> dict[str, Any]:
    """Return a new fragment adjacency with ``fix`` applied to the *broken*
    fragment (``frag`` is the tampered one)."""
    kind = ast["kind"]
    ip = {s: list(ps) for s, ps in frag["isa_parents"].items()}
    ro = {s: {p: list(ts) for p, ts in by.items()} for s, by in frag["rel_out"].items()}
    if kind == "repair_partof_edge_deletion":
        s, o = fix.get("subject", ""), fix.get("object", "")
        if not s or not o:
            raise ValueError("fix requires subject and object")
        ro.setdefault(s, {}).setdefault("part_of", [])
        if o not in ro[s]["part_of"]:
            ro[s]["part_of"].append(o)
            ro[s]["part_of"].sort()
    elif kind == "repair_spurious_isa_edge":
        t, p = fix.get("term", ""), fix.get("remove_parent", "")
        if not t or not p:
            raise ValueError("fix requires term and remove_parent")
        ip.setdefault(t, [])
        ip[t] = [x for x in ip[t] if x != p]
    elif kind == "repair_regulation_polarity_swap":
        s, o, pred = (fix.get("subject", ""), fix.get("object", ""),
                      fix.get("predicate", ""))
        if not s or not o or not pred:
            raise ValueError("fix requires subject, object and predicate")
        wrong = ast["wrong_pred"]
        if s in ro and wrong in ro.get(s, {}) and o in ro[s][wrong]:
            ro[s][wrong] = [x for x in ro[s][wrong] if x != o]
            if not ro[s][wrong]:
                del ro[s][wrong]
        ro.setdefault(s, {}).setdefault(pred, [])
        if o not in ro[s][pred]:
            ro[s][pred].append(o)
            ro[s][pred].sort()
    elif kind == "repair_spurious_partof_edge":
        s, o = fix.get("subject", ""), fix.get("remove_object", "")
        if not s or not o:
            raise ValueError("fix requires subject and remove_object")
        if s in ro and "part_of" in ro.get(s, {}):
            ro[s]["part_of"] = [x for x in ro[s]["part_of"] if x != o]
            if not ro[s]["part_of"]:
                del ro[s]["part_of"]
    else:
        raise ValueError(f"apply_repair20k: unsupported kind {kind!r}")
    return _rebuild(ip, ro, frag["universe"])


def _rebuild(ip: dict[str, list[str]], ro: dict[str, dict[str, list[str]]],
             universe: list[str]) -> dict[str, Any]:
    ic: dict[str, list[str]] = {}
    for s, ps in ip.items():
        for p in ps:
            ic.setdefault(p, []).append(s)
    for p in ic:
        ic[p].sort()
    return {"isa_parents": ip, "isa_children": ic, "rel_out": ro,
            "universe": universe}


def closure_signature20k(frag: dict[str, Any],
                         focus: list[str]) -> dict[str, dict[str, list[str]]]:
    """Polarity-aware closure fingerprint restricted to focus terms — the
    quantity a 20K repair must restore to the reference value. Rows: isa,
    partof, reg (rolled up), reg_pos, reg_neg."""
    pt = partof_targets_of(frag)
    out: dict[str, dict[str, list[str]]] = {}
    for t in focus:
        out[t] = {
            "isa": sorted(transitive_ancestors(t, frag["isa_parents"])),
            "partof": sorted(partof_ancestors(t, frag["isa_parents"], pt)),
            "reg": sorted(regulates_ancestors(
                t, frag["isa_parents"], frag["rel_out"],
                kinds=REGULATES_KINDS, isa_children=frag["isa_children"])),
            "reg_pos": sorted(regulates_ancestors(
                t, frag["isa_parents"], frag["rel_out"],
                kinds=POS_KINDS, isa_children=frag["isa_children"])),
            "reg_neg": sorted(regulates_ancestors(
                t, frag["isa_parents"], frag["rel_out"],
                kinds=NEG_KINDS, isa_children=frag["isa_children"])),
        }
    return out


def gold_fix20k(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["kind"]
    if kind == "repair_partof_edge_deletion":
        return {"subject": ast["subject"], "object": ast["object"]}
    if kind == "repair_spurious_isa_edge":
        return {"term": ast["term"], "remove_parent": ast["spurious_parent"]}
    if kind == "repair_regulation_polarity_swap":
        return {"subject": ast["subject"], "object": ast["object"],
                "predicate": ast["correct_pred"]}
    if kind == "repair_spurious_partof_edge":
        return {"subject": ast["subject"],
                "remove_object": ast["spurious_target"]}
    raise ValueError(f"gold_fix20k: unsupported kind {kind!r}")
