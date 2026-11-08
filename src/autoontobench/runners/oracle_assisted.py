"""GOLD_ONTOLOGY_UPPER arm (owner: W6): oracle-assisted reference, NOT an
autonomous baseline (Review-5B §6 E1 / Shared Contract §E).

The artifact is preloaded from the private gold source by the harness and is
NOT editable; the arm may consult it. Gold receives extra information, so it
is excluded from paired causal fairness claims by construction.
"""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "oracle_assisted",
    representation="shacl-rule/1",
    initial_artifact="gold",
    artifact_editable=False,
    parity_note="GOLD_ONT upper reference: extra information, not a fair autonomous baseline",
)
