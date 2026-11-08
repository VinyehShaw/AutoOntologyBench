"""L07 task-family plugin: propose CandidateTaskASTs over the pinned snapshot.

Pipeline position: source index -> propose (this file) -> ground (ground.py)
-> native verifier gates -> shard writer (pipeline.py). One base goal per
(family, model); a base goal counts only when all of its instances
instantiate, so per-family yields are exactly ``base_goals *
instances_per_goal`` unless the capable-model pool is exhausted.
"""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any, Iterator

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.source.l07.plugin import L07SourcePlugin
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit
from benchmark.plugins.source.l07.simulate import SimulationError
from benchmark.plugins.task_family.l07 import goals_sim, goals_struct, ground
from benchmark.plugins.task_family.l07.families import BY_ID, LANE, PROFILE, FamilySpec
from benchmark.plugins.task_family.l07.goals_sim import CORRUPT_FACTORS
from benchmark.plugins.task_family.l07.support import SimRunner
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

_BASE_PROPOSERS = {
    "threshold_set": goals_sim.propose_threshold_set,
    "ranking": goals_sim.propose_ranking,
    "shape_org": goals_sim.propose_shape_org,
    "param_threshold": goals_sim.propose_param_threshold,
    "param_repair": goals_sim.propose_param_repair,
    "init_repair": goals_sim.propose_init_repair,
    "law_repair": goals_sim.propose_law_repair,
    "role_query": goals_struct.propose_role_query,
    "species_role_org": goals_struct.propose_species_role_org,
    "order_inversion": goals_sim.propose_order_inversion,
}


class L07TaskFamilyPlugin:
    """TaskFamilyPlugin for all 10 L07 families (see families.FAMILY_TABLE)."""

    def __init__(self, index_dir: str | Path, *, seed: int = 20261027) -> None:
        self.index = L07SourcePlugin.load_index(Path(index_dir))
        self.source = L07SourcePlugin()
        self.sim = SimRunner(self.index, self.source)
        self.seed = seed

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    # -- propose -------------------------------------------------------------

    def propose(self, snapshot: Any = None, *, family: str, seed: int | None = None,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = BY_ID[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get("max_base_goals", spec.base_goals)
        capable = self.sim.capable_models(
            needs_sim=spec.needs_sim, min_floating=spec.min_floating_species,
            min_reactions=spec.min_reactions, min_params=spec.min_settable_params)
        proposer = _BASE_PROPOSERS[spec.kind]
        goals_done = 0
        for model_id in capable:
            if goals_done >= max_goals:
                break
            rng = Random(f"{family}|base|{model_id}|{base_seed}")
            base = proposer(self.sim, model_id, rng)
            if base is None:
                continue
            base_goal_id = "l07b-" + canonical_sha(
                {"family": family, "params": base["goal_params"]})[:12]
            instances = [
                cand for inst_seq in range(spec.instances_per_goal)
                if (cand := self._instantiate(spec, base, base_goal_id,
                                              inst_seq, base_seed)) is not None
            ]
            if len(instances) != spec.instances_per_goal:
                continue
            if len({canonical_sha(i.ast) for i in instances}) != spec.instances_per_goal:
                continue
            goals_done += 1
            yield from instances

    # -- instantiation ---------------------------------------------------------

    def _instantiate(self, spec: FamilySpec, base: dict[str, Any], base_goal_id: str,
                     inst_seq: int, base_seed: int) -> CandidateTaskAST | None:
        rng = Random(f"{spec.family_id}|inst|{base_goal_id}|{inst_seq}|{base_seed}")
        gp = base["goal_params"]
        kind = spec.kind
        ast: dict[str, Any] = {"kind": kind, "model_id": gp["model_id"],
                               "complexity": spec.complexity}
        if kind in ("threshold_set", "ranking", "shape_org"):
            inst = self._universe_instance(spec, base, rng)
            if inst is None:
                return None
            ast.update(inst)
            if kind == "threshold_set":
                ast["threshold"] = gp["threshold"]
            elif kind == "ranking":
                ast["k"] = gp["k"]
        elif kind == "role_query":
            inst = goals_struct.instantiate_role_query(base, rng)
            if inst is None:
                return None
            ast.update(inst)
            ast.update(target_species=gp["target_species"], relation=gp["relation"])
        elif kind == "species_role_org":
            inst = goals_struct.instantiate_species_role_org(base, rng, spec.universe_size)
            if inst is None:
                return None
            ast.update(inst)
        elif kind in ("param_threshold", "order_inversion"):
            fn = (goals_sim.instantiate_param_threshold if kind == "param_threshold"
                  else goals_sim.instantiate_order_inversion)
            inst = fn(self.sim, base, inst_seq)
            if inst is None:
                return None
            ast.update(inst)
            ast.update({k: gp[k] for k in gp if k != "model_id"})
        elif kind in ("param_repair", "init_repair"):
            edit = self._repair_edit(gp, CORRUPT_FACTORS[inst_seq % len(CORRUPT_FACTORS)])
            if self._repair_weak(gp, edit):
                return None
            ast.update({k: gp[k] for k in gp if k != "model_id"})
            ast["corrupt_factor"] = CORRUPT_FACTORS[inst_seq % len(CORRUPT_FACTORS)]
        elif kind == "law_repair":
            variant = base["variants"][inst_seq % len(base["variants"])]
            struct = self.sim.struct(gp["model_id"])
            formula = struct.reaction_info(gp["reaction"]).kinetic_formula or ""
            corrupted = goals_sim.swap_species_in_formula(formula, [tuple(p) for p in variant])
            edit = ParameterEdit("", 0.0, kind="kinetic_law",
                                 reaction_id=gp["reaction"], formula=corrupted)
            if self._repair_weak(gp, edit):
                return None
            ast.update({k: gp[k] for k in gp if k != "model_id"})
            ast["swap_pairs"] = [list(p) for p in variant]
        else:
            raise ValueError(f"unknown kind {kind!r}")
        instance_id = "l07i-" + canonical_sha(
            {"family": spec.family_id, "base": base_goal_id, "ast": ast})[:16]
        return CandidateTaskAST(
            lane_id=LANE, family_id=spec.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime, semantic_profile=PROFILE,
            ast=ast,
            source_unit_ids=(f"biomodels/{gp['model_id']}",))

    def _universe_instance(self, spec: FamilySpec, base: dict[str, Any],
                           rng: Random) -> dict[str, Any] | None:
        if spec.kind == "threshold_set":
            return goals_sim.instantiate_threshold_set(base, rng, spec.universe_size)
        if spec.kind == "ranking":
            return goals_sim.instantiate_ranking(base, rng, spec.universe_size)
        return goals_sim.instantiate_shape_org(base, rng, spec.universe_size)

    def _repair_edit(self, gp: dict[str, Any], factor: float) -> ParameterEdit:
        if "parameter" in gp:
            p0 = self.sim.struct(gp["model_id"]).parameter_value(gp["parameter"])
            return ParameterEdit(gp["parameter"], p0 * factor)
        sid = gp["species"]
        info = self.sim.struct(gp["model_id"]).species_info(sid)
        v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
        assert v0 is not None
        return ParameterEdit("", v0 * factor, kind="species_initial", species_id=sid)

    def _repair_weak(self, gp: dict[str, Any], edit: ParameterEdit) -> bool:
        probes = tuple(gp["probe_species"])
        try:
            ref = self.sim.sim(gp["model_id"], probes)
            broken = self.sim.sim(gp["model_id"], probes, edit)
        except (SimulationError, ValueError):
            return True
        return not any(
            abs(ref["finals"][s] - broken["finals"][s])
            / max(abs(ref["finals"][s]), abs(broken["finals"][s]), 1e-30) >= 0.05
            for s in probes)

    # -- ground ------------------------------------------------------------------

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        return ground.ground(candidate, self.sim, self.source)
