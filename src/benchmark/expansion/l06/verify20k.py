"""L06 20K-expansion native verifier: deterministic recomputation for the 15
new-mechanism families.

The verifier is the private oracle. Every answer is *recomputed* from the
pinned GO index + the GAF evidence index (rebuilt from the pinned
annotations file); stored evidence is never trusted. Legs:

  leg 1 (graph fixpoint): ``fragment20k`` operators on the fragment induced
       from the indexed adjacency.
  leg 2 (owlrl): ``owlrl20k`` reads on the closed RDF graph (axioms injected
       by the legacy ``build_closed_graph``); answers must equal leg 1
       exactly (``owlrl_agree``) or the task is rejected
       PROFILE_UNSUPPORTED. For the four new repair kinds the dual-leg runs
       on the repair-signature quantity itself: the polarity-aware closure
       rows of the affected focus terms, computed by leg 1, must equal the
       owlrl rows read off the closed reference fragment.
  leg 3 (parser differential, index-time): OBO == OWL extraction, asserted
       per task via the index manifest.

Repairs verify by focused closure-signature restoration: the submitted fix
applied to the tampered fragment must reproduce the reference fragment's
polarity-aware closure signature over the affected focus terms AND match the
gold edge identity exactly. The focus set is the tamper's affected terms
(the edge subject + its fragment descendants, or the edge object + its
fragment ancestors), always containing the key term; single-edge fixes are
fully determined by the key term's parent/relation set, so the focused check
is complete for the fix space.

Gates: UNSOLVABLE (empty/undersized answer), TRIVIAL (answer == universe /
candidate panel, or repair that changes nothing), PROFILE_UNSUPPORTED (leg
disagreement), NON_SEMANTIC (no counterfactual witness), LEAKAGE and
DUPLICATE (builder).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.expansion.l06.evidence20k import load_evidence
from benchmark.expansion.l06.fragment20k import (
    GENE_ANSWER_KINDS,
    REPAIR_KINDS_20K,
    answer_query20k,
    apply_repair20k,
    apply_tamper20k,
    closure_signature20k,
    gold_fix20k,
)
from benchmark.expansion.l06.owlrl20k import (
    owlrl_agreement20k,
    signature_rows_owlrl,
)
from benchmark.plugins.source.l06.plugin import L06SourcePlugin
from benchmark.plugins.verifier.l06.fragment import fragment_edges, induce
from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)

MAX_ANSWER_FRACTION = 0.8
FOCUS_CAP = 60


class L06ExpansionVerifier:
    def __init__(self, index_dir: str | Path,
                 index: dict[str, Any] | None = None,
                 evidence: dict[str, Any] | None = None) -> None:
        self.index = index if index is not None else L06SourcePlugin.load_index(
            Path(index_dir))
        self.index_dir = Path(index_dir)
        self.graph = self.index["graph"]
        self.terms = self.index["terms"]
        self.evidence = evidence if evidence is not None else load_evidence(
            self.index_dir, self.index["closures"])
        manifest = self.index["manifest"]
        diff = manifest["differential"]
        self.index_differential_ok = bool(diff.get("isa_equal")
                                          and diff.get("rel_equal"))
        self.release_id = manifest["release_id"]
        self.profile = manifest["profile"]

    # ------------------------------------------------------------- internals

    def _frag(self, ast: dict[str, Any]) -> dict[str, Any]:
        return induce(ast["universe_ids"], self.graph)

    def _gene_ev(self) -> dict[str, Any]:
        return self.evidence["gene_ev"]

    def _reference(self, ast: dict[str, Any],
                   frag: dict[str, Any]) -> Any:
        return answer_query20k(ast, frag, self._gene_ev())

    def _focus_terms(self, ast: dict[str, Any],
                     frag: dict[str, Any]) -> list[str]:
        """Terms whose closure rows the tamper can affect, key term first,
        deterministically capped (single-edge fixes are decided by the key
        term's parent/relation set, so the cap cannot admit a wrong fix)."""
        from benchmark.plugins.source.l06.closure import descendants
        kind = ast["kind"]
        if kind == "repair_partof_edge_deletion":
            key = ast["subject"]
            rest = sorted(descendants(key, frag["isa_children"]))
        elif kind == "repair_spurious_isa_edge":
            key = ast["term"]
            rest = sorted(descendants(key, frag["isa_children"]))
        elif kind == "repair_regulation_polarity_swap":
            key = ast["object"]
            from benchmark.plugins.source.l06.closure import transitive_ancestors
            rest = sorted(transitive_ancestors(key, frag["isa_parents"]))
        elif kind == "repair_spurious_partof_edge":
            key = ast["subject"]
            rest = sorted(descendants(key, frag["isa_children"]))
        else:
            raise ValueError(f"no focus for kind {kind!r}")
        return [key] + [t for t in rest if t != key][:FOCUS_CAP - 1]

    # ------------------------------------------------------- plugin methods

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["kind"]
        ev: dict[str, Any] = {"profile": self.profile,
                              "release_id": self.release_id,
                              "index_differential_ok": self.index_differential_ok}
        if not self.index_differential_ok:
            return ProofOrCounterexample(
                False, ev, "PROFILE_UNSUPPORTED:index_differential")
        frag = self._frag(ast)
        ev["universe_size"] = len(frag["universe"])
        ev["fragment_edges"] = len(fragment_edges(frag))
        if kind in REPAIR_KINDS_20K:
            focus = self._focus_terms(ast, frag)
            broken = apply_tamper20k(ast, frag)
            sig_ref = closure_signature20k(frag, focus)
            sig_broken = closure_signature20k(broken, focus)
            diff_terms = [t for t in focus if sig_ref[t] != sig_broken[t]]
            ev["closure_focus_terms"] = len(focus)
            ev["closure_diff_terms"] = len(diff_terms)
            if not diff_terms:
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:tamper_no_closure_effect")
            gold = gold_fix20k(ast)
            repaired = apply_repair20k(ast, broken, gold)
            sig_repaired = closure_signature20k(repaired, focus)
            if sig_repaired != sig_ref:
                return ProofOrCounterexample(
                    False, ev, "UNSOLVABLE:gold_fix_mismatch")
            ev["gold_fix"] = gold
            ev["repair_restores_closure"] = True
            # dual-leg: the signature quantity itself must be owlrl-correct
            # on the reference fragment for the affected focus terms.
            from benchmark.plugins.verifier.l06.owlrl_leg import (
                build_closed_graph)
            g = build_closed_graph(fragment_edges(frag))
            rows_owlrl = signature_rows_owlrl(g, focus, set(frag["universe"]))
            ev["owlrl_leg"] = "owlrl_repair_signature"
            ev["owlrl_agree"] = rows_owlrl == sig_ref
            if not ev["owlrl_agree"]:
                return ProofOrCounterexample(
                    False, ev, "PROFILE_UNSUPPORTED:owlrl_signature_mismatch")
            return ProofOrCounterexample(True, ev)
        ref = self._reference(ast, frag)
        if kind in GENE_ANSWER_KINDS:
            candidates = set(ast["candidate_genes"])
            ev["answer_size"] = len(ref)
            ev["candidate_genes"] = len(candidates)
            if len(ref) < 1:
                return ProofOrCounterexample(False, ev, "UNSOLVABLE:empty_answer")
            if ref == frozenset(candidates):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:answer_is_candidate_panel")
            if len(ref) > MAX_ANSWER_FRACTION * len(candidates):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:answer_dominates_panel")
        else:
            ev["answer_size"] = len(ref)
            if len(ref) < 1:
                return ProofOrCounterexample(False, ev, "UNSOLVABLE:empty_answer")
            if ref == frozenset(frag["universe"]):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:answer_is_universe")
            if len(ref) > MAX_ANSWER_FRACTION * len(frag["universe"]):
                return ProofOrCounterexample(
                    False, ev, "TRIVIAL:answer_dominates_universe")
        agree, leg = owlrl_agreement20k(ast, frag, frozenset(ref),
                                        self._gene_ev())
        ev["owlrl_leg"] = leg
        ev["owlrl_agree"] = agree
        if not agree:
            return ProofOrCounterexample(
                False, ev, f"PROFILE_UNSUPPORTED:{leg}_mismatch")
        if kind in GENE_ANSWER_KINDS:
            ev["reference_answer_genes"] = sorted(ref)
        else:
            ev["reference_answer_ids"] = sorted(ref)
        return ProofOrCounterexample(True, ev)

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        kind = ast["kind"]
        frag = self._frag(ast)
        if kind in REPAIR_KINDS_20K:
            if not isinstance(submitted, dict):
                return NativeVerdict(False, {"error": "missing submission"})
            gold = gold_fix20k(ast)
            identity_ok = True
            if kind == "repair_partof_edge_deletion":
                identity_ok = (submitted.get("subject") == ast["subject"]
                               and submitted.get("object") == ast["object"])
                fix = {"subject": submitted.get("subject", ""),
                       "object": submitted.get("object", "")}
            elif kind == "repair_spurious_isa_edge":
                identity_ok = (submitted.get("term") == ast["term"]
                               and submitted.get("remove_parent")
                               == ast["spurious_parent"])
                fix = {"term": submitted.get("term", ""),
                       "remove_parent": submitted.get("remove_parent", "")}
            elif kind == "repair_regulation_polarity_swap":
                identity_ok = (submitted.get("subject") == ast["subject"]
                               and submitted.get("object") == ast["object"]
                               and submitted.get("predicate")
                               == gold["predicate"])
                fix = {"subject": submitted.get("subject", ""),
                       "object": submitted.get("object", ""),
                       "predicate": submitted.get("predicate", "")}
            else:  # repair_spurious_partof_edge
                identity_ok = (submitted.get("subject") == ast["subject"]
                               and submitted.get("remove_object")
                               == ast["spurious_target"])
                fix = {"subject": submitted.get("subject", ""),
                       "remove_object": submitted.get("remove_object", "")}
            broken = apply_tamper20k(ast, frag)
            focus = self._focus_terms(ast, frag)
            sig_ref = closure_signature20k(frag, focus)
            try:
                repaired = apply_repair20k(ast, broken, fix)
                sig_ok = closure_signature20k(repaired, focus) == sig_ref
            except (KeyError, TypeError, ValueError) as exc:
                return NativeVerdict(False, {"error": f"malformed_fix: {exc}"})
            return NativeVerdict(bool(identity_ok and sig_ok), {
                "identity_ok": identity_ok, "closure_restored": sig_ok,
                "profile": self.profile})
        if kind in GENE_ANSWER_KINDS:
            want = self._reference(ast, frag)
            got_raw = submitted.get("answer_genes") if isinstance(
                submitted, dict) else None
            if not isinstance(got_raw, list) or any(
                    not isinstance(x, str) for x in got_raw):
                return NativeVerdict(False, {"error": "missing/invalid answer_genes"})
            got = frozenset(got_raw)
            unknown = sorted(got - frozenset(ast["candidate_genes"]))
            return NativeVerdict(got == want and not unknown, {
                "expected": len(want), "received": len(got),
                "missing": sorted(want - got), "extra": sorted(got - want),
                "unknown_genes": unknown, "profile": self.profile})
        want = self._reference(ast, frag)
        got_raw = submitted.get("answer_ids") if isinstance(submitted, dict) \
            else None
        if not isinstance(got_raw, list) or any(
                not isinstance(x, str) for x in got_raw):
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
        if kind in REPAIR_KINDS_20K:
            focus = self._focus_terms(ast, frag)
            broken = apply_tamper20k(ast, frag)
            sig_ref = closure_signature20k(frag, focus)
            sig_broken = closure_signature20k(broken, focus)
            diff = sum(1 for t in focus if sig_ref[t] != sig_broken[t])
            return SemanticWitness(diff > 0, {
                "counterfactual": "tampered_fragment",
                "closure_diff_terms": diff, "profile": self.profile})
        # application + organization kinds: perturb the fragment and require
        # the answer to change. Witnesses, cheapest first (identical 3-tier
        # search contract as the legacy verifier):
        #   1. remove one is_a edge (path bottleneck);
        #   2. remove ALL relation edges (partonomy/regulation structure);
        #   3. remove ALL is_a parents of one term (redundant multi-parent
        #      structures, e.g. GO's doubly-linked regulation mirror).
        ref = self._reference(ast, frag)
        ref_set = frozenset(ref) if not isinstance(ref, frozenset) else ref
        edges = fragment_edges(frag)
        ip = frag["isa_parents"]
        gene_ev = self._gene_ev()

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
                return answer_query20k(ast, mod_frag, gene_ev)
            except (KeyError, ValueError):
                return None

        for s, p, o in edges:
            if p != "is_a":
                continue
            mod_ip = {k: [x for x in v if not (k == s and x == o)]
                      for k, v in ip.items()}
            alt = _probe(mod_ip, frag["rel_out"])
            if alt is not None and frozenset(alt) != ref_set:
                return SemanticWitness(True, {
                    "counterfactual": f"remove is_a edge {s} -> {o}",
                    "answer_diff": len(ref_set ^ frozenset(alt)),
                    "profile": self.profile})
        if any(p != "is_a" for _, p, _ in edges):
            empty_ro: dict[str, dict[str, list[str]]] = {}
            alt = _probe(ip, empty_ro)
            if alt is not None and frozenset(alt) != ref_set:
                return SemanticWitness(True, {
                    "counterfactual": "remove all relation edges",
                    "answer_diff": len(ref_set ^ frozenset(alt)),
                    "profile": self.profile})
        multi = [t for t in sorted(ip) if len(ip[t]) >= 2]
        for t in multi[:200]:
            mod_ip = {k: ([] if k == t else list(v)) for k, v in ip.items()}
            alt = _probe(mod_ip, frag["rel_out"])
            if alt is not None and frozenset(alt) != ref_set:
                return SemanticWitness(True, {
                    "counterfactual": f"remove all is_a parents of {t}",
                    "answer_diff": len(ref_set ^ frozenset(alt)),
                    "profile": self.profile})
        return SemanticWitness(False, {
            "error": "no counterfactual perturbation found",
            "profile": self.profile})
