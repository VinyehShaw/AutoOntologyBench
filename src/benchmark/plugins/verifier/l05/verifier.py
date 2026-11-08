"""L05 native verifier: RDKit substructure truth for query/repair/align tasks.

The verifier is the private oracle. Every method *recomputes* its answer from
the pinned index + pinned RDKit engine; stored ``native_evidence`` is only
compared against, never trusted. ``prove_exists`` applies the differential
gate (index evaluation == direct RDKit re-match on the universe) plus the
solvable/nontrivial gates. ``certify_semantic_relevance`` produces a
counterfactual witness: a concrete wrong rule/alignment whose answer set
differs on the task universe (recomputable; RDKit version pinned in evidence).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, __version__ as RDKIT_VERSION
from rdkit import RDLogger

from benchmark.plugins.source.l05.plugin import L05SourcePlugin
from benchmark.plugins.verifier.l05.query_eval import (
    GroupContext, QueryError, eval_index, eval_native, group_leaves, match_smarts,
    replace_leaf,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

RDLogger.DisableLog("rdApp.*")

SUPPORTED_PROFILE = "rdkit-smarts-native/1"


@dataclass(frozen=True)
class Recomputed:
    reference: frozenset
    index_answer: frozenset
    native_answer: frozenset
    differential_agree: bool


class RDKitNativeVerifier:
    def __init__(self, index_dir: str | Path) -> None:
        self.index = L05SourcePlugin.load_index(Path(index_dir))
        self.ctx = GroupContext.from_index(self.index)
        self.smiles_by_id = {m["chebi_id"]: m["canonical_smiles"]
                             for m in self.index["molecules"]}
        self.rdkit_version = RDKIT_VERSION
        self._mol_cache: dict[frozenset, dict[str, Chem.Mol]] = {}

    # ---- internals -----------------------------------------------------

    def universe_mols(self, ids: list[str] | frozenset) -> dict[str, Chem.Mol]:
        key = frozenset(ids)
        if key not in self._mol_cache:
            mols: dict[str, Chem.Mol] = {}
            for mid in sorted(key):
                smi = self.smiles_by_id.get(mid)
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None:
                    raise QueryError(f"universe molecule {mid} not reparseable")
                mols[mid] = mol
            self._mol_cache[key] = mols
        return self._mol_cache[key]

    def reference_set(self, ast: dict) -> frozenset:
        universe = frozenset(ast["universe_ids"])
        kind = ast["kind"]
        if kind == "query":
            return eval_index(ast["query"], self.ctx, universe)
        if kind == "repair":
            mols = self.universe_mols(universe)
            return frozenset(match_smarts(ast["golden_smarts"], mols))
        raise QueryError(f"reference_set: unsupported kind {kind!r}")

    def reference_alignment(self, ast: dict) -> dict[str, list[str]]:
        universe = frozenset(ast["universe_ids"])
        return {label: sorted(eval_index({"op": "group", "group": label}, self.ctx, universe))
                for label in ast["labels"]}

    def reference_submission(self, ast: dict) -> dict[str, Any]:
        kind = ast["kind"]
        if kind == "query":
            return {"answer_ids": sorted(self.reference_set(ast))}
        if kind == "repair":
            return {"repaired_smarts": ast["golden_smarts"]}
        return {"alignment": self.reference_alignment(ast)}

    def wrong_submission(self, ast: dict) -> dict[str, Any]:
        kind = ast["kind"]
        if kind == "query":
            want = self.reference_set(ast)
            universe = set(ast["universe_ids"])
            return {"answer_ids": sorted(universe - want)[: max(1, len(want))]}
        if kind == "repair":
            return {"repaired_smarts": ast["broken_smarts"]}
        ref = self.reference_alignment(ast)
        a, b = ast["misassign_pair"]
        wrong = dict(ref)
        wrong[a], wrong[b] = ref[b], ref[a]
        return {"alignment": wrong}

    def recompute_query(self, ast: dict) -> Recomputed:
        universe = frozenset(ast["universe_ids"])
        idx = eval_index(ast["query"], self.ctx, universe)
        nat = eval_native(ast["query"], self.ctx, self.universe_mols(universe))
        return Recomputed(reference=idx, index_answer=idx, native_answer=nat,
                          differential_agree=idx == nat)

    # ---- NativeVerifierPlugin ------------------------------------------

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        universe = frozenset(ast["universe_ids"])
        ev: dict[str, Any] = {"rdkit_version": self.rdkit_version}
        try:
            if ast["kind"] == "query":
                rec = self.recompute_query(ast)
                ev.update(reference_answer_ids=sorted(rec.reference),
                          differential_agree=rec.differential_agree,
                          answer_size=len(rec.reference), universe_size=len(universe))
                if not rec.differential_agree:
                    return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:differential_mismatch")
                if not rec.reference:
                    return ProofOrCounterexample(False, ev, "UNSOLVABLE:empty_answer")
                if rec.reference == universe:
                    return ProofOrCounterexample(False, ev, "TRIVIAL:answer_is_universe")
                if ast.get("require_interaction"):
                    margins = _interaction_margins(ast["query"], self.ctx, universe,
                                                   rec.reference)
                    ev["interaction_margins"] = margins
                    if not all(m > 0 for m in margins):
                        return ProofOrCounterexample(False, ev, "TRIVIAL:no_filtering_interaction")
                return ProofOrCounterexample(True, ev)
            if ast["kind"] == "repair":
                mols = self.universe_mols(universe)
                golden = frozenset(match_smarts(ast["golden_smarts"], mols))
                ev.update(reference_answer_ids=sorted(golden), answer_size=len(golden),
                          universe_size=len(universe))
                if not golden or golden == universe:
                    return ProofOrCounterexample(False, ev, "TRIVIAL:golden_set_degenerate")
                broken = ast["broken_smarts"]
                try:
                    broken_set: frozenset | None = frozenset(match_smarts(broken, mols))
                except QueryError:
                    broken_set = None  # syntax-level defect
                ev["broken_parses"] = broken_set is not None
                if broken_set is not None:
                    ev["broken_diff"] = len(golden ^ broken_set)
                    if broken_set == golden:
                        return ProofOrCounterexample(False, ev, "TRIVIAL:broken_matches_golden")
                return ProofOrCounterexample(True, ev)
            if ast["kind"] == "align":
                ref = self.reference_alignment(ast)
                ev["reference_alignment"] = ref
                sets = [frozenset(v) for v in ref.values()]
                if len(sets) < 2 or any(not s or s == universe for s in sets):
                    return ProofOrCounterexample(False, ev, "TRIVIAL:alignment_set_degenerate")
                if len(set(sets)) != len(sets):
                    return ProofOrCounterexample(False, ev, "TRIVIAL:duplicate_label_sets")
                return ProofOrCounterexample(True, ev)
        except QueryError as exc:
            return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:{exc}")
        return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:kind={ast.get('kind')}")

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        kind = ast["kind"]
        if kind == "query":
            want = self.reference_set(ast)
            got = _coerce_id_set(submitted, "answer_ids")
            if got is None:
                return NativeVerdict(False, {"error": "missing/invalid answer_ids"})
            unknown = sorted(got - frozenset(ast["universe_ids"]))
            return NativeVerdict(got == want, {
                "expected": len(want), "received": len(got),
                "missing": sorted(want - got), "extra": sorted(got - want),
                "unknown_ids": unknown, "rdkit_version": self.rdkit_version})
        if kind == "repair":
            want = self.reference_set(ast)
            smarts = (submitted or {}).get("repaired_smarts") if isinstance(submitted, dict) else None
            if not isinstance(smarts, str) or not smarts.strip():
                return NativeVerdict(False, {"error": "missing repaired_smarts"})
            try:
                got = frozenset(match_smarts(smarts, self.universe_mols(ast["universe_ids"])))
            except QueryError as exc:
                return NativeVerdict(False, {"error": f"smarts_unparsable: {exc}",
                                             "rdkit_version": self.rdkit_version})
            return NativeVerdict(got == want, {
                "expected": len(want), "received": len(got),
                "missing": sorted(want - got), "extra": sorted(got - want),
                "rdkit_version": self.rdkit_version})
        if kind == "align":
            want = self.reference_alignment(ast)
            got_raw = submitted.get("alignment") if isinstance(submitted, dict) else None
            if not isinstance(got_raw, dict):
                return NativeVerdict(False, {"error": "missing alignment mapping"})
            detail: dict[str, Any] = {"per_label": {}, "rdkit_version": self.rdkit_version}
            ok = set(got_raw) == set(want)
            for label, want_ids in want.items():
                got = frozenset(got_raw.get(label, []))
                ok_label = got == frozenset(want_ids)
                detail["per_label"][label] = {
                    "ok": ok_label, "missing": sorted(frozenset(want_ids) - got),
                    "extra": sorted(got - frozenset(want_ids))}
                ok = ok and ok_label
            return NativeVerdict(ok, detail)
        return NativeVerdict(False, {"error": f"unknown kind {kind!r}"})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        universe = frozenset(ast["universe_ids"])
        mols = self.universe_mols(universe)
        if ast["kind"] == "query":
            return self._witness_query(ast, universe, mols)
        if ast["kind"] == "repair":
            golden = frozenset(match_smarts(ast["golden_smarts"], mols))
            try:
                broken = frozenset(match_smarts(ast["broken_smarts"], mols))
                diff = len(golden ^ broken)
                return SemanticWitness(diff > 0, {
                    "counterfactual": "broken_smarts", "broken_smarts": ast["broken_smarts"],
                    "golden_smarts": ast["golden_smarts"], "answer_diff": diff,
                    "rdkit_version": self.rdkit_version})
            except QueryError:
                return SemanticWitness(True, {
                    "counterfactual": "broken_smarts_unparsable",
                    "broken_smarts": ast["broken_smarts"],
                    "golden_smarts": ast["golden_smarts"],
                    "rdkit_version": self.rdkit_version})
        if ast["kind"] == "align":
            ref = self.reference_alignment(ast)
            swapped = dict(ref)
            a, b = ast["misassign_pair"]
            swapped[a], swapped[b] = ref[b], ref[a]
            diff = sum(len(frozenset(ref[l]) ^ frozenset(swapped[l])) for l in (a, b))
            return SemanticWitness(diff > 0, {
                "counterfactual": f"swap {a} <-> {b}", "answer_diff": diff,
                "rdkit_version": self.rdkit_version})
        return SemanticWitness(False, {"error": f"unknown kind {ast.get('kind')}"})

    def _witness_query(self, ast: dict, universe: frozenset,
                       mols: dict[str, Chem.Mol]) -> SemanticWitness:
        """Swap each group leaf for a sibling group; first differing swap wins."""
        ref = eval_native(ast["query"], self.ctx, mols)
        for path, label in group_leaves(ast["query"]):
            siblings = sorted(self.ctx.children.get(self._parent_of(label), frozenset())
                              - {label})
            for alt in siblings or self._nearest_groups(label):
                mutated = replace_leaf(ast["query"], path, alt)
                cf = eval_native(mutated, self.ctx, mols)
                if cf != ref:
                    return SemanticWitness(True, {
                        "counterfactual": f"group {label} -> {alt}",
                        "mutated_query": mutated, "answer_diff": len(ref ^ cf),
                        "rdkit_version": self.rdkit_version})
        for mutated in _numeric_mutations(ast["query"]):
            cf = eval_native(mutated, self.ctx, mols)
            if cf != ref:
                return SemanticWitness(True, {
                    "counterfactual": "numeric_band_mutation",
                    "mutated_query": mutated, "answer_diff": len(ref ^ cf),
                    "rdkit_version": self.rdkit_version})
        return SemanticWitness(False, {"error": "no counterfactual found",
                                       "rdkit_version": self.rdkit_version})

    def _parent_of(self, label: str) -> str:
        return label.rsplit(".", 1)[0] if "." in label else ""

    def _nearest_groups(self, label: str) -> list[str]:
        size = len(self.ctx.membership.get(label, ()))
        others = [(abs(len(self.ctx.membership.get(lbl, ())) - size), lbl)
                  for lbl in self.ctx.smarts if lbl != label]
        return [lbl for _, lbl in sorted(others)[:5]]


def _numeric_mutations(query: dict) -> list[dict]:
    op = query.get("op")
    if op in ("prop_band", "element_band"):
        out = [{**query, "hi": query["hi"] + 1}]
        if query["hi"] > query["lo"]:
            out.append({**query, "hi": query["hi"] - 1})
        if query["lo"] > 0:
            out.append({**query, "lo": query["lo"] - 1})
        return out
    if op in ("and", "or", "xor"):
        out = []
        for i, arg in enumerate(query["args"]):
            for mutated in _numeric_mutations(arg):
                args = list(query["args"])
                args[i] = mutated
                out.append({**query, "args": args})
        return out
    if op == "not":
        return [{**query, "arg": m} for m in _numeric_mutations(query["arg"])]
    if op == "count_range_dual":
        out = []
        ranges = [list(r) for r in query["ranges"]]
        for i, (lo, hi) in enumerate(ranges):
            wider = [list(r) for r in ranges]
            wider[i][1] = hi + 1
            out.append({**query, "ranges": wider})
            if hi > lo:
                narrower = [list(r) for r in ranges]
                narrower[i][1] = hi - 1
                out.append({**query, "ranges": narrower})
        return out
    return []


def _coerce_id_set(submitted: Any, key: str) -> frozenset | None:
    if not isinstance(submitted, dict) or not isinstance(submitted.get(key), list):
        return None
    ids = submitted[key]
    if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        return None
    return frozenset(ids)


def _interaction_margins(query: dict, ctx: GroupContext, universe: frozenset,
                         answer: frozenset) -> list[int]:
    """For each positive group leaf: |leaf_set ∩ U| - |answer| (must be > 0)."""
    margins = []
    for _, label in group_leaves(query, positive_only=True):
        margins.append(len(ctx.membership.get(label, frozenset()) & universe) - len(answer))
    return margins

