"""L06 native verifier: deterministic rdflib/owlrl truth for GO tasks.

The verifier is the private oracle. Every answer is *recomputed* from the
pinned index; stored evidence is never trusted. Three independent legs:

  leg 1 (graph fixpoint): closure.py operators on the fragment induced from
       the indexed adjacency (``graph.json.gz``).
  leg 2 (owlrl): the fragment edges + profile axioms are materialized as RDF
       and closed under ``owlrl.OWLRL_Semantics`` (propertyChainAxiom,
       TransitiveProperty, subPropertyOf, subClassOf transitivity); answers
       read off the closed graph must equal leg 1 exactly
       (``owlrl_agree`` in evidence).
  leg 3 (parser differential, index-time): OBO-stanza extraction ==
       rdflib-OWL extraction, recorded in the index manifest and asserted
       here before any task is accepted.

Repairs verify by closure restoration: the submitted fix applied to the
tampered fragment must reproduce the reference fragment's full closure
signature AND match the gold edge identity exactly.

Gates: UNSOLVABLE (empty/undersized answer), TRIVIAL (answer == universe or
repair that changes nothing), PROFILE_UNSUPPORTED (leg disagreement),
NON_SEMANTIC (no counterfactual witness), LEAKAGE and DUPLICATE (builder).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.plugins.source.l06.plugin import L06SourcePlugin
from benchmark.plugins.verifier.l06.fragment import (
    answer_query,
    apply_repair,
    apply_tamper,
    closure_signature,
    fragment_edges,
    induce,
)
from benchmark.plugins.verifier.l06.owlrl_leg import (
    build_closed_graph,
    isa_ancestors_owlrl,
    partof_ancestors_owlrl,
    regulators_owlrl,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)

MAX_ANSWER_FRACTION = 0.8
ISA_ONLY_KINDS = {"isa_ancestors", "isa_descendants", "common_ancestors", "lca",
                  "branch_difference"}
PARTOF_KINDS = {"partof_ancestors", "cross_isa_partof"}
REG_KINDS = {"regulators"}
REPAIR_KINDS = {"repair_isa_parent_swap", "repair_missing_isa_edge",
                "repair_relation_type_confusion", "repair_orphan_reconnect"}
ORG_KINDS = {"align_gene_terms", "annotation_closure", "shared_ancestors"}


class L06NativeVerifier:
    def __init__(self, index_dir: str | Path,
                 index: dict[str, Any] | None = None) -> None:
        self.index = index if index is not None else L06SourcePlugin.load_index(
            Path(index_dir))
        self.graph = self.index["graph"]
        self.terms = self.index["terms"]
        self.gene_terms = self.index["gene_terms"]
        manifest = self.index["manifest"]
        diff = manifest["differential"]
        self.index_differential_ok = bool(diff.get("isa_equal") and diff.get("rel_equal"))
        self.release_id = manifest["release_id"]
        self.profile = manifest["profile"]

    # ------------------------------------------------------------- internals

    def _frag(self, ast: dict[str, Any]) -> dict[str, Any]:
        return induce(ast["universe_ids"], self.graph)

    def _owlrl_agreement(self, ast: dict[str, Any], frag: dict[str, Any],
                         reference: frozenset[str]) -> tuple[bool, str]:
        """Leg 2: owlrl entailment must reproduce leg 1 for closure kinds."""
        kind = ast["kind"]
        u = set(frag["universe"])
        edges = fragment_edges(frag)
        g = build_closed_graph(edges)
        if kind in ISA_ONLY_KINDS:
            anc_map = {t: isa_ancestors_owlrl(g, t, u) for t in u}
            desc_map: dict[str, set[str]] = {}
            for t, anc in anc_map.items():
                for a in anc:
                    desc_map.setdefault(a, set()).add(t)
            if kind == "isa_ancestors":
                got = anc_map[ast["seed"]]
            elif kind == "isa_descendants":
                got = frozenset(desc_map.get(ast["seed"], set()))
            elif kind == "common_ancestors":
                got = anc_map[ast["a"]] & anc_map[ast["b"]]
            elif kind == "lca":
                ca = anc_map[ast["a"]] & anc_map[ast["b"]]
                got = frozenset(x for x in ca
                                if not any(x != y and x in anc_map[y] for y in ca))
            else:  # branch_difference
                got = frozenset(desc_map.get(ast["parent"], set())
                                - desc_map.get(ast["child"], set()))
            return got == reference, "owlrl_isa"
        if kind in PARTOF_KINDS:
            if kind == "partof_ancestors":
                got = partof_ancestors_owlrl(g, ast["seed"], u)
            else:
                po = partof_ancestors_owlrl(g, ast["partof_target"], u)
                anc_map = {t: isa_ancestors_owlrl(g, t, u) for t in u}
                desc_map: dict[str, set[str]] = {}
                for t, anc in anc_map.items():
                    for a in anc:
                        desc_map.setdefault(a, set()).add(t)
                got = frozenset(x for x in desc_map.get(ast["isa_root"], set()) if x in po)
            return got == reference, "owlrl_partof"
        if kind in REG_KINDS:
            got = regulators_owlrl(g, ast["seed"], u)
            return got == reference, "owlrl_regulates"
        if kind == "annotation_closure":
            out: set[str] = set()
            for t in ast["direct_terms"]:
                if t not in u:
                    continue
                out.add(t)
                out.update(isa_ancestors_owlrl(g, t, u))
                out.update(partof_ancestors_owlrl(g, t, u))
            return frozenset(out) == reference, "owlrl_annotation_scope"
        if kind == "shared_ancestors":
            from benchmark.plugins.verifier.l06.fragment import GO_ROOTS
            inter: set[str] | None = None
            for gene, direct in sorted(ast["direct_terms_by_gene"].items()):
                prop: set[str] = set()
                for t in direct:
                    if t not in u:
                        continue
                    prop.add(t)
                    prop.update(isa_ancestors_owlrl(g, t, u))
                    prop.update(partof_ancestors_owlrl(g, t, u))
                inter = prop if inter is None else inter & prop
            got = frozenset((inter or set()) - set(GO_ROOTS))
            return got == reference, "owlrl_shared_scope"
        return True, "owlrl_not_applicable"  # structural / repair kinds

    def _reference(self, ast: dict[str, Any], frag: dict[str, Any]) -> Any:
        kind = ast["kind"]
        if kind == "align_gene_terms":
            return {g: sorted(self.gene_terms[g][ast["aspect"]]) for g in ast["genes"]}
        return answer_query(ast, frag)

    def _gold_fix(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["kind"]
        if kind == "repair_isa_parent_swap":
            return {"parent": ast["correct_parent"]}
        if kind == "repair_missing_isa_edge":
            return {"parent": ast["removed_parent"]}
        if kind == "repair_orphan_reconnect":
            return {"parents": sorted(ast["removed_parents"])}
        if kind == "repair_relation_type_confusion":
            return {"predicate": ast["correct_pred"]}
        raise ValueError(f"no gold fix for {kind!r}")

    def _broken_frag(self, ast: dict[str, Any], frag: dict[str, Any]) -> dict[str, Any]:
        return apply_tamper(ast, frag)

    # ------------------------------------------------------- plugin methods

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["kind"]
        ev: dict[str, Any] = {"profile": self.profile, "release_id": self.release_id,
                              "index_differential_ok": self.index_differential_ok}
        if not self.index_differential_ok:
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:index_differential")
        frag = self._frag(ast)
        ev["universe_size"] = len(frag["universe"])
        ev["fragment_edges"] = len(fragment_edges(frag))
        if kind in REPAIR_KINDS:
            broken = self._broken_frag(ast, frag)
            focus = sorted(frag["universe"])
            sig_ref = closure_signature(frag, focus)
            sig_broken = closure_signature(broken, focus)
            diff_terms = sorted(t for t in focus if sig_ref[t] != sig_broken[t])
            ev["closure_diff_terms"] = len(diff_terms)
            if not diff_terms:
                return ProofOrCounterexample(False, ev, "TRIVIAL:tamper_no_closure_effect")
            gold = self._gold_fix(ast)
            repaired = apply_repair(ast, broken, gold)
            sig_repaired = closure_signature(repaired, focus)
            if sig_repaired != sig_ref:
                return ProofOrCounterexample(False, ev, "UNSOLVABLE:gold_fix_mismatch")
            ev["gold_fix"] = gold
            ev["repair_restores_closure"] = True
            return ProofOrCounterexample(True, ev)
        ref = self._reference(ast, frag)
        if kind == "align_gene_terms":
            sets = [frozenset(v) for v in ref.values()]
            if len(set(sets)) != len(sets) or any(not s for s in sets):
                return ProofOrCounterexample(False, ev, "TRIVIAL:alignment_sets_degenerate")
            ev["reference_alignment"] = ref
            ev["genes"] = len(ref)
            return ProofOrCounterexample(True, ev)
        ev["answer_size"] = len(ref)
        if len(ref) < 1:
            return ProofOrCounterexample(False, ev, "UNSOLVABLE:empty_answer")
        if ref == frozenset(frag["universe"]):
            return ProofOrCounterexample(False, ev, "TRIVIAL:answer_is_universe")
        if len(ref) > MAX_ANSWER_FRACTION * len(frag["universe"]):
            return ProofOrCounterexample(False, ev, "TRIVIAL:answer_dominates_universe")
        agree, leg = self._owlrl_agreement(ast, frag, frozenset(ref))
        ev["owlrl_leg"] = leg
        ev["owlrl_agree"] = agree
        if not agree:
            return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:{leg}_mismatch")
        ev["reference_answer_ids"] = sorted(ref)
        return ProofOrCounterexample(True, ev)

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        kind = ast["kind"]
        frag = self._frag(ast)
        if kind in REPAIR_KINDS:
            if not isinstance(submitted, dict):
                return NativeVerdict(False, {"error": "missing submission"})
            gold = self._gold_fix(ast)
            identity_ok = True
            if kind == "repair_isa_parent_swap" or kind == "repair_missing_isa_edge":
                identity_ok = (submitted.get("term") == ast["term"]
                               and submitted.get("parent") == gold["parent"])
                fix = {"parent": submitted.get("parent", "")}
            elif kind == "repair_orphan_reconnect":
                identity_ok = (submitted.get("term") == ast["term"]
                               and sorted(submitted.get("parents", [])) == gold["parents"])
                fix = {"parents": list(submitted.get("parents", []))}
            else:
                identity_ok = (submitted.get("subject") == ast["subject"]
                               and submitted.get("object") == ast["object"]
                               and submitted.get("predicate") == gold["predicate"])
                fix = {"predicate": submitted.get("predicate", "")}
            broken = self._broken_frag(ast, frag)
            focus = sorted(frag["universe"])
            sig_ref = closure_signature(frag, focus)
            try:
                repaired = apply_repair(ast, broken, fix)
                sig_ok = closure_signature(repaired, focus) == sig_ref
            except (KeyError, TypeError, ValueError) as exc:
                return NativeVerdict(False, {"error": f"malformed_fix: {exc}"})
            return NativeVerdict(bool(identity_ok and sig_ok), {
                "identity_ok": identity_ok, "closure_restored": sig_ok,
                "profile": self.profile})
        if kind == "align_gene_terms":
            want = self._reference(ast, frag)
            got = submitted.get("alignment") if isinstance(submitted, dict) else None
            if not isinstance(got, dict):
                return NativeVerdict(False, {"error": "missing alignment"})
            ok = set(got) == set(want)
            per = {}
            for g, want_ids in want.items():
                got_ids = sorted(got.get(g, []))
                per[g] = got_ids == want_ids
                ok = ok and per[g]
            return NativeVerdict(ok, {"per_gene": per, "profile": self.profile})
        want = self._reference(ast, frag)
        got_raw = submitted.get("answer_ids") if isinstance(submitted, dict) else None
        if not isinstance(got_raw, list) or any(not isinstance(x, str) for x in got_raw):
            return NativeVerdict(False, {"error": "missing/invalid answer_ids"})
        got = frozenset(got_raw)
        unknown = sorted(got - frozenset(ast["universe_ids"]))
        return NativeVerdict(got == want and not unknown, {
            "expected": len(want), "received": len(got),
            "missing": sorted(want - got), "extra": sorted(got - want),
            "unknown_ids": unknown, "profile": self.profile})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        kind = ast["kind"]
        frag = self._frag(ast)
        if kind in REPAIR_KINDS:
            broken = self._broken_frag(ast, frag)
            focus = sorted(frag["universe"])
            sig_ref = closure_signature(frag, focus)
            sig_broken = closure_signature(broken, focus)
            diff = sum(1 for t in focus if sig_ref[t] != sig_broken[t])
            return SemanticWitness(diff > 0, {
                "counterfactual": "tampered_fragment", "closure_diff_terms": diff,
                "profile": self.profile})
        if kind == "align_gene_terms":
            ref = self._reference(ast, frag)
            a, b = sorted(ast["derangement"].items())[0]
            swapped = dict(ref)
            swapped[a], swapped[b] = ref[b], ref[a]
            diff = sum(1 for g in ref if sorted(swapped[g]) != sorted(ref[g]))
            return SemanticWitness(diff > 0, {
                "counterfactual": f"derangement {a}<->{b}", "rows_differing": diff,
                "profile": self.profile})
        # application + closure org kinds: perturb the fragment and require the
        # answer to change. Witnesses, cheapest first:
        #   1. remove one is_a edge (path bottleneck);
        #   2. remove ALL relation edges (regulation/part_of structure);
        #   3. remove ALL is_a parents of one term (redundant multi-parent
        #      structures, e.g. GO's doubly-linked regulation mirror, where no
        #      single edge is a bottleneck).
        ref = frozenset(self._reference(ast, frag))
        edges = fragment_edges(frag)
        ip = frag["isa_parents"]

        def _rebuild_children(mod_ip: dict[str, list[str]]) -> dict[str, list[str]]:
            mod_ic: dict[str, list[str]] = {}
            for cs, ps in mod_ip.items():
                for cp in ps:
                    mod_ic.setdefault(cp, []).append(cs)
            return mod_ic

        def _probe(mod_ip: dict[str, list[str]],
                   mod_ro: dict[str, dict[str, list[str]]]) -> frozenset[str] | None:
            mod_frag = {"isa_parents": mod_ip,
                        "isa_children": _rebuild_children(mod_ip),
                        "rel_out": mod_ro, "universe": frag["universe"]}
            try:
                return frozenset(answer_query(ast, mod_frag))
            except (KeyError, ValueError):
                return None

        for s, p, o in edges:
            if p != "is_a":
                continue
            mod_ip = {k: [x for x in v if not (k == s and x == o)]
                      for k, v in ip.items()}
            alt = _probe(mod_ip, frag["rel_out"])
            if alt is not None and alt != ref:
                return SemanticWitness(True, {
                    "counterfactual": f"remove is_a edge {s} -> {o}",
                    "answer_diff": len(ref ^ alt), "profile": self.profile})
        if any(p != "is_a" for _, p, _ in edges):
            empty_ro: dict[str, dict[str, list[str]]] = {}
            alt = _probe(ip, empty_ro)
            if alt is not None and alt != ref:
                return SemanticWitness(True, {
                    "counterfactual": "remove all relation edges",
                    "answer_diff": len(ref ^ alt), "profile": self.profile})
        multi = [t for t in sorted(ip) if len(ip[t]) >= 2]
        for t in multi[:200]:
            mod_ip = {k: ([] if k == t else list(v)) for k, v in ip.items()}
            alt = _probe(mod_ip, frag["rel_out"])
            if alt is not None and alt != ref:
                return SemanticWitness(True, {
                    "counterfactual": f"remove all is_a parents of {t}",
                    "answer_diff": len(ref ^ alt), "profile": self.profile})
        return SemanticWitness(False, {"error": "no counterfactual perturbation found",
                                       "profile": self.profile})
