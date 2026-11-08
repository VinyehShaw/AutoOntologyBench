"""SELF_BUILT_ONTOLOGY arm (owner: W6): agent builds/repairs O, then uses it.

Starts from a clean STUB artifact (never gold — Review-5B §5), may edit,
syntax-validate, and consult its own artifact via check_action_with_ontology;
construction cost is charged to the run.
"""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "ontology",
    representation="shacl-rule/1",
    initial_artifact="stub",
    artifact_editable=True,
    parity_note="E1 AGENT_ONT: self-built shacl-rule/1 artifact; full artifact toolset",
)
