"""L07 grounding: assemble the public view + oracle handle for a candidate.

The public view carries the working material (pinned SBML XML, or its corrupted
variant for repair families), the goal parameters, and the lane's simulation
protocol. It NEVER carries the answer set / original (pre-corruption) values;
the pipeline's leakage scan enforces that. All trajectories are in-silico
solver outputs — wording says so explicitly in every task.
"""
from __future__ import annotations

import hashlib
from typing import Any

from benchmark.plugins.source.l07.plugin import L07SourcePlugin
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit, apply_edit
from benchmark.plugins.source.l07.simulate import PROTOCOL
from benchmark.plugins.task_family.l07.families import PROFILE
from benchmark.plugins.task_family.l07.goals_sim import swap_species_in_formula
from benchmark.plugins.task_family.l07.support import SimRunner
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

SIM_TOOLS = ["sbml_model.read", "sbml_model.simulate", "sbml_model.edit"]
STRUCT_TOOLS = ["sbml_model.read"]
DISCLAIMER = ("All values are in-silico SBML simulation results under the "
              "stated pinned protocol, not experimental measurements.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _model_public(sim: SimRunner, source: L07SourcePlugin, model_id: str,
                  sbml_xml: str | None = None) -> dict[str, Any]:
    struct = sim.struct(model_id)
    entry = source.pinlock["models"][model_id]
    return {
        "model_id": model_id,
        "name": entry["name"],
        "sbml_level": entry["sbml_level"],
        "sbml_sha256": struct.sha256,
        "sbml_xml": sbml_xml if sbml_xml is not None else sim.raw(model_id).decode("utf-8"),
        "species_ids": [s.sid for s in struct.species],
        "reaction_ids": [r.rid for r in struct.reactions],
        "parameter_ids": [p for p, _, _ in struct.parameters],
    }


def _corrupted_sbml(sim: SimRunner, ast: dict[str, Any]) -> str:
    kind = ast["kind"]
    raw = sim.raw(ast["model_id"])
    if kind == "param_repair":
        p0 = sim.struct(ast["model_id"]).parameter_value(ast["parameter"])
        edit = ParameterEdit(ast["parameter"], p0 * ast["corrupt_factor"])
    elif kind == "init_repair":
        info = sim.struct(ast["model_id"]).species_info(ast["species"])
        v0 = info.initial_amount if info.initial_amount is not None else info.initial_concentration
        assert v0 is not None
        edit = ParameterEdit("", v0 * ast["corrupt_factor"],
                             kind="species_initial", species_id=ast["species"])
    else:
        struct = sim.struct(ast["model_id"])
        formula = struct.reaction_info(ast["reaction"]).kinetic_formula or ""
        corrupted = swap_species_in_formula(formula,
                                            [tuple(p) for p in ast["swap_pairs"]])
        edit = ParameterEdit("", 0.0, kind="kinetic_law",
                             reaction_id=ast["reaction"], formula=corrupted)
    return apply_edit(raw, edit).decode("utf-8")


def _reference_finals(sim: SimRunner, ast: dict[str, Any]) -> dict[str, float]:
    probes = tuple(ast["probe_species"])
    return sim.sim(ast["model_id"], probes)["finals"]


def ground(candidate: CandidateTaskAST, sim: SimRunner,
           source: L07SourcePlugin) -> GroundedTask:
    ast = candidate.ast
    kind = ast["kind"]
    is_repair = kind in ("param_repair", "init_repair", "law_repair")
    needs_sim = kind not in ("role_query", "species_role_org")
    sbml_xml = _corrupted_sbml(sim, ast) if is_repair else None
    public: dict[str, Any] = {
        "user_goal": _user_goal(ast),
        "model": _model_public(sim, source, ast["model_id"], sbml_xml),
        "allowed_tools": SIM_TOOLS if needs_sim else STRUCT_TOOLS,
        "budget": {"probes": 16, "actions": 40},
        "semantic_profile": PROFILE,
        "complexity": ast["complexity"],
        "in_silico": True,
    }
    if needs_sim:
        public["protocol"] = dict(PROTOCOL)
    if kind in ("threshold_set", "ranking", "shape_org", "species_role_org"):
        public["universe_ids"] = ast["universe_ids"]
    if kind == "threshold_set":
        public["threshold"] = ast["threshold"]
    if kind == "ranking":
        public["k"] = ast["k"]
    if kind == "role_query":
        public["universe_reaction_ids"] = ast["universe_reaction_ids"]
        public["target_species"] = ast["target_species"]
        public["relation"] = ast["relation"]
    if kind in ("param_threshold", "order_inversion"):
        public["parameter"] = ast["parameter"]
        public["candidates"] = ast["candidates"]
        if kind == "param_threshold":
            public["target_species"] = ast["target_species"]
            public["target"] = ast["target"]
        else:
            public["species_a"] = ast["species_a"]
            public["species_b"] = ast["species_b"]
    if is_repair:
        public["reference_finals"] = _reference_finals(sim, ast)
        if kind == "param_repair":
            corruption = {"field": "parameter", "parameter": ast["parameter"]}
        elif kind == "init_repair":
            corruption = {"field": ast["field"], "species": ast["species"]}
        else:
            corruption = {"field": "kinetic_law_formula", "reaction": ast["reaction"]}
        public["corruption"] = corruption
    oracle_ref = "orl07_" + _sha(candidate.instance_id + "|oracle")[:16]
    return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                        native_evidence={"profile": PROFILE, "protocol": dict(PROTOCOL)})


def _user_goal(ast: dict[str, Any]) -> str:
    kind = ast["kind"]
    if kind == "threshold_set":
        return (f"Simulate the provided SBML model under the pinned protocol and return "
                f"the exact set of species ids from the given universe whose final value "
                f"is >= {ast['threshold']!r}. Submit as answer_ids; partial or superset "
                f"answers are scored as failure. {DISCLAIMER}")
    if kind == "ranking":
        return (f"Simulate the provided SBML model under the pinned protocol and return "
                f"the top-{ast['k']} universe species by final value, exact order "
                f"(highest first). Submit as ranking. {DISCLAIMER}")
    if kind == "shape_org":
        return ("Simulate the provided SBML model under the pinned protocol and assign "
                "each universe species its trajectory-shape class "
                "(flat / increasing / decreasing / peaked) per the lane class rules. "
                f"Submit as assignment {{species_id: class}}. {DISCLAIMER}")
    if kind == "param_threshold":
        return (f"Choose every value from the candidate list for parameter "
                f"'{ast['parameter']}' that drives the final value of species "
                f"'{ast['target_species']}' to >= {ast['target']!r} when the model is "
                f"simulated under the pinned protocol. Submit the exact subset as "
                f"accepted_values. {DISCLAIMER}")
    if kind == "order_inversion":
        return (f"In the unmodified model, species '{ast['species_a']}' ends above "
                f"'{ast['species_b']}'. Choose every value from the candidate list for "
                f"parameter '{ast['parameter']}' that INVERTS this final ordering. "
                f"Submit the exact subset as accepted_values. {DISCLAIMER}")
    if kind == "param_repair":
        return (f"The global parameter '{ast['parameter']}' in the provided SBML model "
                f"was corrupted (its current value is wrong). Restore its value so the "
                f"probe species reach the stated reference final values under the pinned "
                f"protocol. Submit as parameter_value. {DISCLAIMER}")
    if kind == "init_repair":
        return (f"The {ast['field'].replace('_', ' ')} of species '{ast['species']}' in "
                f"the provided SBML model was corrupted. Restore it so the probe species "
                f"reach the stated reference final values under the pinned protocol. "
                f"Submit as initial_value. {DISCLAIMER}")
    if kind == "law_repair":
        return (f"The kinetic-law formula of reaction '{ast['reaction']}' in the "
                f"provided SBML model had two species identifiers swapped. Repair the "
                f"formula (SBML Level 3 infix) so the probe species reach the stated "
                f"reference final values under the pinned protocol. Submit as "
                f"repaired_formula. {DISCLAIMER}")
    if kind == "role_query":
        verb = {"consumes": "consume", "produces": "produce",
                "modulates": "list as modifier"}[ast["relation"]]
        return (f"Parse the provided SBML model and return the exact set of reaction "
                f"ids from the given universe that {verb} species "
                f"'{ast['target_species']}'. Submit as answer_ids.")
    if kind == "species_role_org":
        return ("Parse the provided SBML model's reaction graph and assign each "
                "universe species its role class (boundary / source / sink / "
                "intermediate / catalyst / isolated) per the lane role rules. "
                "Submit as assignment {species_id: role}.")
    raise ValueError(f"no wording for kind {kind!r}")
