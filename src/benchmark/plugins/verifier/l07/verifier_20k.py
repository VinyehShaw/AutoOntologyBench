"""Native verifier for the L07 20K-candidate mechanisms.

Subclasses the legacy libSBML/CVODE verifier and adds branches for six new
semantic predicates. Truth is always recomputed from the pinned SBML bytes;
the proposal-time index cache is only a differential cross-check. Structural
tasks additionally re-parse the pinned bytes and compare the index structure
to the fresh parse (index-vs-fresh differential).
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.source.l07.plugin import load_struct
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit
from benchmark.plugins.source.l07.sbml_struct import (
    LIBSBML_VERSION, parse_sbml_bytes,
)
from benchmark.plugins.source.l07.simulate import SimulationError, load_roadrunner
from benchmark.plugins.task_family.l07.families_20k import SIM_KINDS_20K
from benchmark.plugins.task_family.l07.goals_20k import (
    reaction_degrees, retention_of,
)
from benchmark.plugins.verifier.l07.verifier import (
    SbmlNativeVerifier, _coerce_float_set, _coerce_str_set,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

SET_KINDS_20K = ("retention_set", "pulse_set", "reaction_degree_query")


class SbmlNativeVerifier20K(SbmlNativeVerifier):
    # -- reference answers -----------------------------------------------------

    def _features(self, model_id: str, species: list[str]) -> dict[str, Any]:
        return self.sim.sim(model_id, tuple(sorted(species)))["species"]

    def _accepted_init(self, ast: dict[str, Any], pred: Any) -> list[float]:
        out: list[float] = []
        for value in ast["candidates"]:
            edit = ParameterEdit("", float(value), kind="species_initial",
                                 species_id=ast["species"])
            finals = self.sim.sim(ast["model_id"], (ast["target_species"],),
                                  edit)["finals"]
            if pred(finals[ast["target_species"]]):
                out.append(float(value))
        return sorted(out)

    def reference_submission(self, ast: dict[str, Any]) -> Any:
        kind = ast["kind"]
        if kind == "retention_set":
            feats = self._features(ast["model_id"], ast["universe_ids"])
            threshold = ast["retention_threshold"]
            answer = [s for s in ast["universe_ids"]
                      if (r := retention_of(feats[s])) is not None and r >= threshold]
            return {"answer_ids": sorted(answer)}
        if kind == "peak_time_ranking":
            feats = self._features(ast["model_id"], ast["universe_ids"])
            order = sorted(ast["universe_ids"],
                           key=lambda s: (-int(feats[s]["argmax_index"]),
                                          -float(feats[s]["last"]), s))
            return {"ranking": order[: ast["k"]]}
        if kind == "pulse_set":
            feats = self._features(ast["model_id"], ast["universe_ids"])
            level = ast["level"]
            return {"answer_ids": sorted(
                s for s in ast["universe_ids"]
                if feats[s]["max"] >= level and feats[s]["last"] < level)}
        if kind == "init_threshold_set":
            target = ast["target"]
            return {"accepted_values": self._accepted_init(
                ast, lambda f: f >= target)}
        if kind == "init_window_set":
            lo, hi = ast["window_low"], ast["window_high"]
            return {"accepted_values": self._accepted_init(
                ast, lambda f: lo <= f <= hi)}
        if kind == "reaction_degree_query":
            degrees = reaction_degrees(self.sim.struct(ast["model_id"]))
            threshold = ast["degree_threshold"]
            return {"answer_ids": sorted(
                r for r in ast["universe_reaction_ids"]
                if degrees[r] >= threshold)}
        return super().reference_submission(ast)

    # -- gates -----------------------------------------------------------------

    def _structural_differential(self, ast: dict[str, Any]) -> bool:
        rec = self.index["models"][ast["model_id"]]
        cached = load_struct(rec)
        fresh = parse_sbml_bytes(self.source.read_model(ast["model_id"]))
        if cached.sha256 != fresh.sha256:
            return False
        if [s.sid for s in cached.species] != [s.sid for s in fresh.species]:
            return False
        if [r.rid for r in cached.reactions] != [r.rid for r in fresh.reactions]:
            return False
        return reaction_degrees(cached) == reaction_degrees(fresh)

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["kind"]
        ev: dict[str, Any] = {
            "libsbml_version": LIBSBML_VERSION,
            "roadrunner_version": load_roadrunner().__version__,
            "protocol": dict(self._protocol()),
            "model_id": ast["model_id"],
            "model_sha256": self.sim.struct(ast["model_id"]).sha256,
        }
        try:
            reference = self.reference_submission(ast)
        except (SimulationError, ValueError, KeyError) as exc:
            return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:{exc}")
        ev["reference_submission"] = reference
        if kind in SIM_KINDS_20K:
            ev["differential_kind"] = "index_cache_vs_fresh_cvode"
            ev["differential_agree"] = self._differential(ast)
        else:
            ev["differential_kind"] = "index_struct_vs_fresh_libsbml"
            ev["differential_agree"] = self._structural_differential(ast)
        if not ev["differential_agree"]:
            return ProofOrCounterexample(
                False, ev, "PROFILE_UNSUPPORTED:differential_mismatch")
        reason = self._nontriviality(ast, reference)
        if reason:
            return ProofOrCounterexample(False, ev, reason)
        ev["solvable"] = True
        return ProofOrCounterexample(True, ev)

    def _protocol(self) -> dict[str, Any]:
        from benchmark.plugins.source.l07.simulate import PROTOCOL
        return PROTOCOL

    # -- semantics -------------------------------------------------------------

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        kind = ast["kind"]
        base = {"kind": kind, "model_id": ast["model_id"],
                "libsbml_version": LIBSBML_VERSION,
                "roadrunner_version": load_roadrunner().__version__}
        try:
            if kind == "retention_set":
                ref = set(self.reference_submission(ast)["answer_ids"])
                feats = self._features(ast["model_id"], ast["universe_ids"])
                for mult in (1.5, 0.5, 2.0, 0.25):
                    thr = ast["retention_threshold"] * mult
                    alt = {s for s in ast["universe_ids"]
                           if (r := retention_of(feats[s])) is not None and r >= thr}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"retention_threshold_multiplier:{mult}",
                            "alt_threshold": thr, "answer_diff": len(alt ^ ref)}))
                if ref:
                    values = [retention_of(feats[s]) for s in ref]
                    values = [v for v in values if v is not None]
                    if values:
                        boundary = min(values) * (1.0 + 1e-9)
                        alt = {s for s in ast["universe_ids"]
                               if (r := retention_of(feats[s])) is not None
                               and r >= boundary}
                        if alt != ref:
                            return SemanticWitness(True, dict(base, **{
                                "counterfactual": "retention_boundary",
                                "alt_threshold": boundary,
                                "answer_diff": len(alt ^ ref)}))
            elif kind == "peak_time_ranking":
                ref = list(self.reference_submission(ast)["ranking"])
                feats = self._features(ast["model_id"], ast["universe_ids"])
                universe = ast["universe_ids"]
                alt_sets = {
                    "earliest_peak": sorted(
                        universe, key=lambda s: (int(feats[s]["argmax_index"]),
                                                 float(feats[s]["last"]), s))[: ast["k"]],
                    "final_value_desc": sorted(
                        universe, key=lambda s: (-float(feats[s]["last"]),
                                                 int(feats[s]["argmax_index"]), s))[: ast["k"]],
                }
                for name, alt in alt_sets.items():
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": name,
                            "answer_diff": len(set(alt) ^ set(ref))}))
            elif kind == "pulse_set":
                ref = set(self.reference_submission(ast)["answer_ids"])
                feats = self._features(ast["model_id"], ast["universe_ids"])
                level = ast["level"]
                universe = ast["universe_ids"]
                alts = {
                    "drop_final_condition": {s for s in universe
                                             if feats[s]["max"] >= level},
                    "final_only": {s for s in universe
                                   if feats[s]["last"] < level},
                }
                for name, alt in alts.items():
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": name,
                            "answer_diff": len(alt ^ ref)}))
                for mult in (0.5, 2.0):
                    alt = {s for s in universe
                           if feats[s]["max"] >= level * mult
                           and feats[s]["last"] < level * mult}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"pulse_level_multiplier:{mult}",
                            "answer_diff": len(alt ^ ref)}))
            elif kind == "init_threshold_set":
                ref = set(self.reference_submission(ast)["accepted_values"])
                for mult in (1.5, 0.5, 2.0, 0.1):
                    target = ast["target"] * mult
                    alt = set(self._accepted_init(ast, lambda f, t=target: f >= t))
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"init_target_multiplier:{mult}",
                            "alt_target": target, "answer_diff": len(alt ^ ref)}))
            elif kind == "init_window_set":
                ref = set(self.reference_submission(ast)["accepted_values"])
                lo, hi = ast["window_low"], ast["window_high"]
                windows = ((lo * 1.5, hi * 1.5), (lo * 0.5, hi * 0.5),
                           (lo, hi * 2.0), (lo * 2.0, hi))
                for new_lo, new_hi in windows:
                    if not new_lo < new_hi:
                        continue
                    alt = set(self._accepted_init(
                        ast, lambda f, a=new_lo, b=new_hi: a <= f <= b))
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": "init_window_scaled",
                            "alt_window": [new_lo, new_hi],
                            "answer_diff": len(alt ^ ref)}))
            elif kind == "reaction_degree_query":
                ref = set(self.reference_submission(ast)["answer_ids"])
                struct = self.sim.struct(ast["model_id"])
                degrees = reaction_degrees(struct)
                threshold = ast["degree_threshold"]
                universe = ast["universe_reaction_ids"]
                alt_lt = {r for r in universe if degrees[r] < threshold}
                if alt_lt != ref:
                    return SemanticWitness(True, dict(base, **{
                        "counterfactual": "degree_predicate_reversed",
                        "answer_diff": len(alt_lt ^ ref)}))
                no_mod = _degrees_without_modifiers(struct)
                alt = {r for r in universe if no_mod[r] >= threshold}
                if alt != ref:
                    return SemanticWitness(True, dict(base, **{
                        "counterfactual": "degree_without_modifiers",
                        "answer_diff": len(alt ^ ref)}))
        except (SimulationError, ValueError, KeyError) as exc:
            return SemanticWitness(
                False, dict(base, error=f"{type(exc).__name__}: {exc}"))
        return SemanticWitness(False, dict(base, error="no_counterfactual_found"))

    # -- nontriviality ---------------------------------------------------------

    def _nontriviality(self, ast: dict[str, Any], reference: Any) -> str:
        kind = ast["kind"]
        if kind in SET_KINDS_20K:
            universe = ast.get("universe_ids") or ast["universe_reaction_ids"]
            answer = reference["answer_ids"]
            if not answer:
                return "UNSOLVABLE:empty_answer"
            if len(answer) == len(universe):
                return "TRIVIAL:answer_is_universe"
            return ""
        if kind == "peak_time_ranking":
            if len(reference["ranking"]) != ast["k"]:
                return "UNSOLVABLE:short_ranking"
            return ""
        if kind in ("init_threshold_set", "init_window_set"):
            accepted = reference["accepted_values"]
            if not accepted:
                return "UNSOLVABLE:empty_answer"
            if len(accepted) == len(ast["candidates"]):
                return "TRIVIAL:all_candidates_pass"
            return ""
        return super()._nontriviality(ast, reference)

    # -- terminal verification -------------------------------------------------

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        kind = ast["kind"]
        if kind not in SET_KINDS_20K + ("peak_time_ranking",
                                        "init_threshold_set", "init_window_set"):
            return super().verify_terminal(task, submitted)
        try:
            reference = self.reference_submission(ast)
        except (SimulationError, ValueError, KeyError) as exc:
            return NativeVerdict(False, {"error": f"reference_failed: {exc}"})
        detail: dict[str, Any] = {"kind": kind, "libsbml_version": LIBSBML_VERSION}
        if kind in SET_KINDS_20K:
            got = _coerce_str_set(submitted, "answer_ids")
            if got is None:
                return NativeVerdict(False, {"error": "missing/invalid answer_ids"})
            universe = set(ast.get("universe_ids")
                           or ast["universe_reaction_ids"])
            unknown = sorted(got - universe)
            want = set(reference["answer_ids"])
            detail.update(expected=sorted(want), received=sorted(got),
                          unknown=unknown)
            return NativeVerdict(got == want and not unknown, detail)
        if kind == "peak_time_ranking":
            got = submitted.get("ranking") if isinstance(submitted, dict) else None
            if not isinstance(got, list) or len(set(got)) != len(got):
                return NativeVerdict(False, {"error": "missing/invalid ranking"})
            detail.update(expected=reference["ranking"], received=got)
            return NativeVerdict(list(got) == reference["ranking"], detail)
        got = _coerce_float_set(submitted, "accepted_values")
        if got is None:
            return NativeVerdict(False, {"error": "missing/invalid accepted_values"})
        want = {repr(v) for v in reference["accepted_values"]}
        detail.update(expected=sorted(want), received=sorted(got))
        return NativeVerdict(got == want, detail)


def _degrees_without_modifiers(struct: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in struct.reactions:
        species = {s for s, _ in r.reactants}
        species.update(s for s, _ in r.products)
        out[r.rid] = len(species)
    return out