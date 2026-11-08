"""Bind a verified blueprint dict into a frozen TaskSpecV5 (T2 output -> T4 input).

Blueprints come from compose.py (deterministic, solver-grounded). Mock fixtures
are accepted only when explicitly labeled `mock: true` and forced to split=dev.
"""
from __future__ import annotations

from typing import Any, Mapping

from autoontobench.contracts.models import (
    Authorship, Budgets, Provenance, SourceRef, TaskFamily, TaskSpecV5, TaskStatus,
    TaskTrack,
)

REQUIRED_KEYS = (
    "task_id", "domain", "track", "task_family", "user_goal",
    "source_refs", "source_group_id", "mechanism_group_id",
    "initial_state_ref", "allowed_tools", "public_evidence_ref",
    "ontology_contract", "private_goal_verifier_ref", "private_native_truth_ref",
    "private_gold_artifact_ref", "private_control_specs_ref",
    "task_provenance", "split", "independent_native_checks",
)


class BindError(Exception):
    pass


def bind_blueprint(blueprint: Mapping[str, Any], *, seed: int = 0) -> TaskSpecV5:
    missing = [k for k in REQUIRED_KEYS if k not in blueprint]
    if missing:
        raise BindError(f"blueprint missing keys: {missing}")
    is_mock = bool(blueprint.get("mock"))
    split = "dev" if is_mock else str(blueprint["split"])
    try:
        return TaskSpecV5(
            task_id=str(blueprint["task_id"]),
            domain=str(blueprint["domain"]),
            track=TaskTrack(str(blueprint["track"])),
            task_family=TaskFamily(str(blueprint["task_family"])),
            task_regime=blueprint.get("task_regime"),
            source_refs=[SourceRef.model_validate(r) for r in blueprint["source_refs"]],
            source_group_id=str(blueprint["source_group_id"]),
            mechanism_group_id=str(blueprint["mechanism_group_id"]),
            combination_signature=blueprint.get("combination_signature"),
            user_goal=str(blueprint["user_goal"]),
            initial_state_ref=str(blueprint["initial_state_ref"]),
            allowed_tools=[str(t) for t in blueprint["allowed_tools"]],
            budgets=Budgets.model_validate(blueprint.get("budgets", {})),
            public_evidence_ref=str(blueprint["public_evidence_ref"]),
            ontology_contract=str(blueprint["ontology_contract"]),
            private_goal_verifier_ref=str(blueprint["private_goal_verifier_ref"]),
            private_native_truth_ref=str(blueprint["private_native_truth_ref"]),
            private_gold_artifact_ref=str(blueprint["private_gold_artifact_ref"]),
            private_control_specs_ref=str(blueprint["private_control_specs_ref"]),
            task_provenance=Provenance(str(blueprint["task_provenance"])),
            task_authorship=Authorship(
                blueprint.get("task_authorship", "deterministic_template")),
            split=split,
            independent_native_checks=[str(c) for c in blueprint["independent_native_checks"]],
            status=TaskStatus.CANDIDATE,
            seed=seed,
        )
    except ValueError as exc:
        raise BindError(f"blueprint failed TaskSpecV5 validation: {exc}") from exc
