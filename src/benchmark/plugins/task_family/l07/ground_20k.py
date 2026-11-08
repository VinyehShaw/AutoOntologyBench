"""Grounding for the L07 20K-candidate mechanisms.

The public view carries the pinned SBML, the goal parameters and the pinned
simulation protocol. It never carries the answer set; the qualification
leakage gate re-checks that. All task wording states that values are in-silico
solver outputs, not experimental measurements.
"""
from __future__ import annotations

import hashlib
from typing import Any

from benchmark.plugins.source.l07.plugin import L07SourcePlugin
from benchmark.plugins.source.l07.simulate import PROTOCOL
from benchmark.plugins.task_family.l07.families_20k import PROFILE
from benchmark.plugins.task_family.l07.ground import (
    DISCLAIMER, SIM_TOOLS, STRUCT_TOOLS, _model_public,
)
from benchmark.plugins.task_family.l07.support import SimRunner
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ground(candidate: CandidateTaskAST, sim: SimRunner,
           source: L07SourcePlugin) -> GroundedTask:
    ast = candidate.ast
    kind = ast["kind"]
    needs_sim = kind != "reaction_degree_query"
    public: dict[str, Any] = {
        "user_goal": _user_goal(ast),
        "model": _model_public(sim, source, ast["model_id"]),
        "allowed_tools": SIM_TOOLS if needs_sim else STRUCT_TOOLS,
        "budget": {"probes": 16, "actions": 40},
        "semantic_profile": PROFILE,
        "complexity": ast["complexity"],
        "in_silico": True,
    }
    if needs_sim:
        public["protocol"] = dict(PROTOCOL)
    if kind in ("retention_set", "peak_time_ranking", "pulse_set"):
        public["universe_ids"] = ast["universe_ids"]
    if kind == "retention_set":
        public["retention_threshold"] = ast["retention_threshold"]
    if kind == "peak_time_ranking":
        public["k"] = ast["k"]
    if kind == "pulse_set":
        public["level"] = ast["level"]
    if kind in ("init_threshold_set", "init_window_set"):
        public["species"] = ast["species"]
        public["field"] = ast["field"]
        public["target_species"] = ast["target_species"]
        public["candidates"] = ast["candidates"]
        if kind == "init_threshold_set":
            public["target"] = ast["target"]
        else:
            public["window_low"] = ast["window_low"]
            public["window_high"] = ast["window_high"]
    if kind == "reaction_degree_query":
        public["universe_reaction_ids"] = ast["universe_reaction_ids"]
        public["degree_threshold"] = ast["degree_threshold"]
    oracle_ref = "orl07_" + _sha(candidate.instance_id + "|oracle")[:16]
    return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                        native_evidence={"profile": PROFILE, "protocol": dict(PROTOCOL)})


def _user_goal(ast: dict[str, Any]) -> str:
    kind = ast["kind"]
    if kind == "retention_set":
        return (
            "Simulate the provided SBML model under the pinned protocol. For each "
            "species in the given universe compute retention = final value / maximum "
            f"value, and return the exact set of species with retention >= "
            f"{ast['retention_threshold']!r}. Submit as answer_ids; partial or "
            f"superset answers are scored as failure. {DISCLAIMER}")
    if kind == "peak_time_ranking":
        return (
            "Simulate the provided SBML model under the pinned protocol and rank the "
            f"top-{ast['k']} species from the given universe by the sample index of "
            "their trajectory maximum (later peak first). Break ties by higher final "
            "value, then by species id. Submit as ranking. "
            f"{DISCLAIMER}")
    if kind == "pulse_set":
        return (
            "Simulate the provided SBML model under the pinned protocol. Return the "
            "exact set of species from the given universe whose trajectory maximum "
            f"reaches {ast['level']!r} but whose final value is below {ast['level']!r} "
            f"(transient pulses). Submit as answer_ids. {DISCLAIMER}")
    if kind == "init_threshold_set":
        return (
            f"Choose every candidate initial value for species '{ast['species']}' "
            f"({ast['field']}) that makes the final value of species "
            f"'{ast['target_species']}' at least {ast['target']!r} when the model is "
            f"simulated under the pinned protocol. Submit the exact subset as "
            f"accepted_values. {DISCLAIMER}")
    if kind == "init_window_set":
        return (
            f"Choose every candidate initial value for species '{ast['species']}' "
            f"({ast['field']}) that keeps the final value of species "
            f"'{ast['target_species']}' inside the inclusive window "
            f"[{ast['window_low']!r}, {ast['window_high']!r}] when the model is "
            f"simulated under the pinned protocol. Submit the exact subset as "
            f"accepted_values. {DISCLAIMER}")
    if kind == "reaction_degree_query":
        return (
            "Parse the provided SBML reaction graph. For each reaction in the given "
            "universe compute its star degree as the number of distinct species "
            "appearing as a reactant, product or modifier, and return the exact set "
            f"of reactions whose degree is >= {ast['degree_threshold']!r}. Submit as "
            f"answer_ids.")
    raise ValueError(f"no wording for kind {kind!r}")