"""L07 native verifier: libSBML + pinned-CVODE truth, recomputed per call.

The verifier is the private oracle. Every answer is recomputed from the pinned
SBML bytes (fetcher pinlock path) with libsbml (structure) and libroadrunner
(CVODE, lane protocol, double-run bit-identical check). The proposal-time
index cache is used only as a differential cross-check
(``differential_agree`` in evidence), never as truth. No LLM output is gold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.plugins.source.l07.plugin import L07SourcePlugin, load_struct
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit, apply_edit
from benchmark.plugins.source.l07.sbml_struct import LIBSBML_VERSION
from benchmark.plugins.source.l07.simulate import (
    PROTOCOL, SimulationError, classify_shape, load_roadrunner, simulate_checked,
)
from benchmark.plugins.task_family.l07.goals_sim import swap_species_in_formula
from benchmark.plugins.task_family.l07.support import (
    SimRunner, rel_diff, relation_set, species_roles,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

ANSWER_RTOL = 1e-4
REPAIR_EFFECT_RTOL = 0.05


class SbmlNativeVerifier:
    def __init__(self, index_dir: str | Path) -> None:
        self.index = L07SourcePlugin.load_index(Path(index_dir))
        self.source = L07SourcePlugin()
        self.sim = SimRunner(self.index, self.source)

    # -- reference answers (recomputed) -----------------------------------------

    def reference_submission(self, ast: dict[str, Any]) -> Any:
        kind = ast["kind"]
        if kind == "threshold_set":
            finals = self._finals(ast["model_id"], ast["universe_ids"])
            return {"answer_ids": sorted(s for s in ast["universe_ids"]
                                         if finals[s] >= ast["threshold"])}
        if kind == "ranking":
            finals = self._finals(ast["model_id"], ast["universe_ids"])
            order = sorted(ast["universe_ids"],
                           key=lambda s: (-finals[s], s))[: ast["k"]]
            return {"ranking": order}
        if kind == "shape_org":
            feats = self._features(ast["model_id"], ast["universe_ids"])
            return {"assignment": {s: feats[s]["shape"] for s in ast["universe_ids"]}}
        if kind == "param_threshold":
            return {"accepted_values": self._accepted(ast)}
        if kind == "order_inversion":
            return {"accepted_values": self._inverted(ast)}
        if kind == "param_repair":
            p0 = self.sim.struct(ast["model_id"]).parameter_value(ast["parameter"])
            return {"parameter_value": p0}
        if kind == "init_repair":
            info = self.sim.struct(ast["model_id"]).species_info(ast["species"])
            v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
            assert v0 is not None
            return {"initial_value": v0}
        if kind == "law_repair":
            struct = self.sim.struct(ast["model_id"])
            return {"repaired_formula": struct.reaction_info(ast["reaction"]).kinetic_formula}
        if kind == "role_query":
            struct = self.sim.struct(ast["model_id"])
            hits = relation_set(struct, ast["relation"], ast["target_species"])
            return {"answer_ids": sorted(set(ast["universe_reaction_ids"]) & hits)}
        if kind == "species_role_org":
            roles = species_roles(self.sim.struct(ast["model_id"]))
            return {"assignment": {s: roles[s] for s in ast["universe_ids"]}}
        raise ValueError(f"unknown kind {kind!r}")

    def _finals(self, model_id: str, species: list[str]) -> dict[str, float]:
        return self.sim.sim(model_id, tuple(sorted(species)))["finals"]

    def _features(self, model_id: str, species: list[str]) -> dict[str, Any]:
        return self.sim.sim(model_id, tuple(sorted(species)))["species"]

    def _accepted(self, ast: dict[str, Any]) -> list[float]:
        out = []
        for v in ast["candidates"]:
            edit = ParameterEdit(ast["parameter"], float(v))
            f = self.sim.sim(ast["model_id"], (ast["target_species"],), edit)["finals"]
            if f[ast["target_species"]] >= ast["target"]:
                answer_append = float(v)
                out.append(answer_append)
        return sorted(out)

    def _inverted(self, ast: dict[str, Any]) -> list[float]:
        out = []
        probes = (ast["species_a"], ast["species_b"])
        for v in ast["candidates"]:
            edit = ParameterEdit(ast["parameter"], float(v))
            f = self.sim.sim(ast["model_id"], probes, edit)["finals"]
            if f[ast["species_a"]] < f[ast["species_b"]]:
                out.append(float(v))
        return sorted(out)

    # -- gates ------------------------------------------------------------------

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["kind"]
        ev: dict[str, Any] = {
            "libsbml_version": LIBSBML_VERSION,
            "roadrunner_version": load_roadrunner().__version__,
            "protocol": dict(PROTOCOL),
            "model_id": ast["model_id"],
            "model_sha256": self.sim.struct(ast["model_id"]).sha256,
        }
        try:
            reference = self.reference_submission(ast)
        except (SimulationError, ValueError, KeyError) as exc:
            return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:{exc}") 
        ev["reference_submission"] = reference
        if kind not in ("role_query", "species_role_org"):
            ev["differential_agree"] = self._differential(ast)
            if not ev["differential_agree"]:
                return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:differential_mismatch")
        reason = self._nontriviality(ast, reference)
        if reason:
            return ProofOrCounterexample(False, ev, reason)
        ev["solvable"] = True
        return ProofOrCounterexample(True, ev)

    def _differential(self, ast: dict[str, Any]) -> bool:
        """Verifier recompute must agree with the proposal-time index cache."""
        rec = self.index["models"][ast["model_id"]]
        cached = {sid: f["last"] for sid, f in (rec["screening"].get("species") or {}).items()}
        if not cached:
            return False
        probes = ast.get("probe_species") or ast.get("universe_ids") or [ast.get("target_species")]
        probes = [p for p in probes if p]
        if not any(p in cached for p in probes):
            probes = sorted(cached)
        fresh = self._finals(ast["model_id"], sorted(set(probes)))
        overlap = [s for s in fresh if s in cached]
        if not overlap:
            return False
        return all(rel_diff(cached[s], fresh[s]) <= 1e-9 for s in overlap)

    def _repair_witness(self, ast: dict[str, Any]) -> dict[str, Any]:
        probes = tuple(sorted(ast["probe_species"]))
        raw = self.source.read_model(ast["model_id"])
        kind = ast["kind"]
        if kind == "param_repair":
            p0 = self.sim.struct(ast["model_id"]).parameter_value(ast["parameter"])
            edit = ParameterEdit(ast["parameter"], p0 * ast["corrupt_factor"])
        elif kind == "init_repair":
            info = self.sim.struct(ast["model_id"]).species_info(ast["species"])
            v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
            assert v0 is not None
            edit = ParameterEdit("", v0 * ast["corrupt_factor"],
                                 kind="species_initial", species_id=ast["species"])
        else:
            struct = self.sim.struct(ast["model_id"])
            formula = struct.reaction_info(ast["reaction"]).kinetic_formula or ""
            corrupted = swap_species_in_formula(
                formula, [tuple(p) for p in ast["swap_pairs"]])
            edit = ParameterEdit("", 0.0, kind="kinetic_law",
                                 reaction_id=ast["reaction"], formula=corrupted)
        try:
            broken = simulate_checked(apply_edit(raw, edit).decode("utf-8"), list(probes))
            ref = self.sim.sim(ast["model_id"], probes)
        except (SimulationError, ValueError, KeyError) as exc:
            return {"effect": False, "error": f"{type(exc).__name__}: {exc}"}
        diffs = {s: rel_diff(ref["finals"][s], broken["finals"][s]) for s in probes}
        return {
            "effect": any(d >= REPAIR_EFFECT_RTOL for d in diffs.values()),
            "rel_diffs": diffs, "threshold": REPAIR_EFFECT_RTOL,
            "reference_finals": ref["finals"], "corrupted_finals": broken["finals"],
        }

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        kind = ast["kind"]
        base = {"kind": kind, "model_id": ast["model_id"],
                "libsbml_version": LIBSBML_VERSION,
                "roadrunner_version": load_roadrunner().__version__}
        try:
            if kind == "threshold_set":
                ref = set(self.reference_submission(ast)["answer_ids"])
                finals = self._finals(ast["model_id"], ast["universe_ids"])
                for mult in (1.5, 0.5, 2.0, 0.25):
                    alt = {s for s in ast["universe_ids"]
                           if finals[s] >= ast["threshold"] * mult}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"threshold_multiplier:{mult}",
                            "alt_threshold": ast["threshold"] * mult,
                            "answer_diff": len(alt ^ ref)}))
                # Boundary counterfactual: raising the cut just above the
                # smallest qualifying value drops that species, so the exact
                # threshold semantics always have an observable effect even
                # when the universe contains a single value cluster.
                answers = [finals[s] for s in ast["universe_ids"] if s in ref]
                if answers:
                    boundary = min(answers) * (1.0 + 1e-9)
                    alt = {s for s in ast["universe_ids"]
                           if finals[s] >= boundary}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": "threshold_boundary",
                            "alt_threshold": boundary,
                            "answer_diff": len(alt ^ ref)}))
            elif kind == "ranking":
                ref = list(self.reference_submission(ast)["ranking"])
                finals = self._finals(ast["model_id"], ast["universe_ids"])
                low = sorted(ast["universe_ids"],
                             key=lambda s: (finals[s], s))[: ast["k"]]
                if low != ref:
                    return SemanticWitness(True, dict(base, **{
                        "counterfactual": "ranking_direction_flipped",
                        "answer_diff": len(set(low) ^ set(ref))}))
            elif kind == "shape_org":
                ref = self.reference_submission(ast)["assignment"]
                alt = self._shape_under_edit(ast, ref)
                if alt is not None:
                    return SemanticWitness(True, dict(base, **alt))
            elif kind == "param_threshold":
                ref = set(self.reference_submission(ast)["accepted_values"])
                for mult in (1.5, 0.5, 2.0, 0.1):
                    alt = set(self._accepted({**ast, "target": ast["target"] * mult}))
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"target_multiplier:{mult}",
                            "alt_target": ast["target"] * mult,
                            "answer_diff": len(alt ^ ref)}))
            elif kind == "order_inversion":
                ref = set(self.reference_submission(ast)["accepted_values"])
                alt = set(ast["candidates"]) - ref
                if alt != ref:
                    return SemanticWitness(True, dict(base, **{
                        "counterfactual": "ordering_condition_reversed",
                        "answer_diff": len(alt ^ ref)}))
            elif kind in ("param_repair", "init_repair", "law_repair"):
                witness = self._repair_witness(ast)
                if witness.get("effect"):
                    return SemanticWitness(True, dict(base, **{
                        "counterfactual": "corruption_changes_reference_dynamics",
                        "max_rel_diff": max(witness["rel_diffs"].values()),
                        "threshold": witness["threshold"]}))
            elif kind == "role_query":
                struct = self.sim.struct(ast["model_id"])
                ref = set(self.reference_submission(ast)["answer_ids"])
                universe = set(ast["universe_reaction_ids"])
                for relation in ("consumes", "produces", "modulates"):
                    if relation == ast["relation"]:
                        continue
                    alt = {r for r in relation_set(struct, relation, ast["target_species"])
                           if r in universe}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"relation_swapped:{relation}",
                            "answer_diff": len(alt ^ ref)}))
                for info in struct.species:
                    if info.sid == ast["target_species"]:
                        continue
                    alt = {r for r in relation_set(struct, ast["relation"], info.sid)
                           if r in universe}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"target_species_swapped:{info.sid}",
                            "answer_diff": len(alt ^ ref)}))
            elif kind == "species_role_org":
                struct = self.sim.struct(ast["model_id"])
                ref = self.reference_submission(ast)["assignment"]
                for mode in ("reverse", "undirected"):
                    roles = self._role_assignment(struct, mode=mode)
                    alt = {s: roles[s] for s in ast["universe_ids"]}
                    if alt != ref:
                        return SemanticWitness(True, dict(base, **{
                            "counterfactual": f"reaction_edges_{mode}",
                            "changed_species": sorted(
                                s for s in ref if ref[s] != alt[s])}))
        except (SimulationError, ValueError, KeyError) as exc:
            return SemanticWitness(False, dict(base, error=f"{type(exc).__name__}: {exc}"))
        return SemanticWitness(False, dict(base, error="no_counterfactual_found"))

    def _shape_under_edit(self, ast: dict[str, Any],
                          ref: dict[str, str]) -> dict[str, Any] | None:
        struct = self.sim.struct(ast["model_id"])
        universe = ast["universe_ids"]
        params = [p for p in struct.settable_parameters
                  if abs(struct.parameter_value(p)) > 0]
        for pid in params[:6]:
            p0 = struct.parameter_value(pid)
            for mult in (0.5, 2.0, 0.1, 10.0):
                try:
                    feats = self.sim.sim(
                        ast["model_id"], tuple(sorted(universe)),
                        ParameterEdit(pid, p0 * mult))["species"]
                except (SimulationError, ValueError):
                    continue
                alt = {s: feats[s]["shape"] for s in universe}
                if alt != ref:
                    return {"counterfactual": f"parameter_perturbation:{pid}x{mult}",
                            "changed_species": sorted(
                                s for s in ref if ref[s] != alt[s])}
        for sid in struct.floating_species[:12]:
            info = struct.species_info(sid)
            v0 = (info.initial_amount if info.initial_amount is not None
                  else info.initial_concentration)
            if not v0:
                continue
            for mult in (0.1, 10.0, 100.0):
                try:
                    feats = self.sim.sim(
                        ast["model_id"], tuple(sorted(universe)),
                        ParameterEdit("", v0 * mult, kind="species_initial",
                                      species_id=sid))["species"]
                except (SimulationError, ValueError):
                    continue
                alt = {s: feats[s]["shape"] for s in universe}
                if alt != ref:
                    return {"counterfactual": f"initial_perturbation:{sid}x{mult}",
                            "changed_species": sorted(
                                s for s in ref if ref[s] != alt[s])}
        return None

    def _role_assignment(self, struct: Any, *, mode: str = "forward") -> dict[str, str]:
        reactant: set[str] = set()
        product: set[str] = set()
        modifier: set[str] = set()
        for r in struct.reactions:
            reac = [s for s, _ in r.reactants]
            prod = [s for s, _ in r.products]
            if mode == "reverse":
                reac, prod = prod, reac
            if mode == "undirected":
                reactant.update(reac)
                reactant.update(prod)
                product.update(reac)
                product.update(prod)
            else:
                reactant.update(reac)
                product.update(prod)
            modifier.update(r.modifiers)
        roles: dict[str, str] = {}
        for s in struct.species:
            sid = s.sid
            if s.boundary:
                roles[sid] = "boundary"
            elif sid in reactant and sid in product:
                roles[sid] = "intermediate"
            elif sid in reactant:
                roles[sid] = "source"
            elif sid in product:
                roles[sid] = "sink"
            elif sid in modifier:
                roles[sid] = "catalyst"
            else:
                roles[sid] = "isolated"
        return roles

    def _nontriviality(self, ast: dict[str, Any], reference: Any) -> str:
        kind = ast["kind"]
        if kind in ("threshold_set", "role_query"):
            universe = ast.get("universe_ids") or ast["universe_reaction_ids"]
            answer = reference["answer_ids"]
            if not answer:
                return "UNSOLVABLE:empty_answer"
            if len(answer) == len(universe):
                return "TRIVIAL:answer_is_universe"
        elif kind == "ranking":
            if len(reference["ranking"]) != ast["k"]:
                return "UNSOLVABLE:short_ranking"
        elif kind in ("shape_org", "species_role_org"):
            classes = set(reference["assignment"].values())
            if len(classes) < 2:
                return "TRIVIAL:single_class"
        elif kind in ("param_threshold", "order_inversion"):
            if not reference["accepted_values"]:
                return "UNSOLVABLE:empty_answer"
            if len(reference["accepted_values"]) == len(ast["candidates"]):
                return "TRIVIAL:all_candidates_pass"
        elif kind in ("param_repair", "init_repair", "law_repair"):
            witness = self._repair_witness(ast)
            if not witness.get("effect"):
                return "UNSOLVABLE:corruption_has_no_effect"
        return ""

    # -- terminal verification ----------------------------------------------------

    def verify_terminal(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        kind = ast["kind"]
        try:
            reference = self.reference_submission(ast)
        except (SimulationError, ValueError, KeyError) as exc:
            return NativeVerdict(False, {"error": f"reference_failed: {exc}"})
        detail: dict[str, Any] = {"kind": kind, "libsbml_version": LIBSBML_VERSION}
        if kind in ("threshold_set", "role_query"):
            got = _coerce_str_set(submitted, "answer_ids")
            if got is None:
                return NativeVerdict(False, {"error": "missing/invalid answer_ids"})
            universe = set(ast.get("universe_ids") or ast["universe_reaction_ids"])
            unknown = sorted(got - universe)
            want = set(reference["answer_ids"])
            detail.update(expected=sorted(want), received=sorted(got), unknown=unknown)
            return NativeVerdict(got == want and not unknown, detail)
        if kind == "ranking":
            got = submitted.get("ranking") if isinstance(submitted, dict) else None
            if not isinstance(got, list) or len(set(got)) != len(got):
                return NativeVerdict(False, {"error": "missing/invalid ranking"})
            detail.update(expected=reference["ranking"], received=got)
            return NativeVerdict(list(got) == reference["ranking"], detail)
        if kind in ("shape_org", "species_role_org"):
            got = submitted.get("assignment") if isinstance(submitted, dict) else None
            if not isinstance(got, dict):
                return NativeVerdict(False, {"error": "missing assignment"})
            want = reference["assignment"]
            detail.update(expected=want, received=got)
            return NativeVerdict(got == want, detail)
        if kind in ("param_threshold", "order_inversion"):
            got = _coerce_float_set(submitted, "accepted_values")
            if got is None:
                return NativeVerdict(False, {"error": "missing/invalid accepted_values"})
            want = {repr(v) for v in reference["accepted_values"]}
            detail.update(expected=sorted(want), received=sorted(got))
            return NativeVerdict(got == want, detail)
        if kind in ("param_repair", "init_repair", "law_repair"):
            return self._verify_repair(ast, submitted, reference, detail)
        return NativeVerdict(False, {"error": f"unknown kind {kind!r}"})

    def _verify_repair(self, ast: dict[str, Any], submitted: Any,
                       reference: Any, detail: dict[str, Any]) -> NativeVerdict:
        kind = ast["kind"]
        if not isinstance(submitted, dict):
            return NativeVerdict(False, {"error": "submission must be an object"})
        try:
            if kind == "param_repair":
                val = submitted.get("parameter_value")
                if not isinstance(val, (int, float)):
                    return NativeVerdict(False, {"error": "missing parameter_value"})
                edit = ParameterEdit(ast["parameter"], float(val))
            elif kind == "init_repair":
                val = submitted.get("initial_value")
                if not isinstance(val, (int, float)):
                    return NativeVerdict(False, {"error": "missing initial_value"})
                edit = ParameterEdit("", float(val), kind="species_initial",
                                     species_id=ast["species"])
            else:
                formula = submitted.get("repaired_formula")
                if not isinstance(formula, str) or not formula.strip():
                    return NativeVerdict(False, {"error": "missing repaired_formula"})
                edit = ParameterEdit("", 0.0, kind="kinetic_law",
                                     reaction_id=ast["reaction"], formula=formula)
            raw = self.source.read_model(ast["model_id"])
            edited = apply_edit(raw, edit)
        except (ValueError, KeyError) as exc:
            return NativeVerdict(False, {"error": f"edit_failed: {exc}"})
        probes = tuple(sorted(ast["probe_species"]))
        try:
            repaired = simulate_checked(edited.decode("utf-8"), list(probes))
            reference_sim = self.sim.sim(ast["model_id"], probes)
        except SimulationError as exc:
            return NativeVerdict(False, {"error": f"simulation_failed: {exc}"})
        mismatches = {
            s: {"submitted_final": repaired["finals"][s],
                "reference_final": reference_sim["finals"][s]}
            for s in probes
            if rel_diff(repaired["finals"][s], reference_sim["finals"][s]) > ANSWER_RTOL
        }
        detail.update(probe_species=list(probes), mismatches=mismatches,
                      answer_rtol=ANSWER_RTOL,
                      reference_finals=reference_sim["finals"],
                      repaired_result_sha256=repaired["result_sha256"])
        return NativeVerdict(not mismatches, detail)


def _coerce_str_set(submitted: Any, key: str) -> set[str] | None:
    """Validate ``{key: [str, ...]}`` and return the deduplicated set."""
    if not isinstance(submitted, dict):
        return None
    values = submitted.get(key)
    if not isinstance(values, (list, tuple)):
        return None
    out: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            return None
        out.add(value)
    if len(out) != len(values):
        return None
    return out


def _coerce_float_set(submitted: Any, key: str) -> set[str] | None:
    """Validate ``{key: [number, ...]}``; return canonical ``repr(float)`` set.

    Comparison is done on ``repr`` because reference candidate values are the
    exact floats emitted by the pinned ladder; a submission that names a value
    not on the ladder can never match, which is intended (exact-subset task).
    """
    if not isinstance(submitted, dict):
        return None
    values = submitted.get(key)
    if not isinstance(values, (list, tuple)):
        return None
    out: set[str] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        out.add(repr(float(value)))
    if len(out) != len(values):
        return None
    return out
