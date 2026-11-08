"""PROBE arm (owner: W6): discovery-track arm with charged native probes.

Tool surface identical to the ontology arm; the distinction is budgetary
(oracle_probes are the scarce resource, E4 B={0,1,2,4,8,16}).
"""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "probe",
    representation="shacl-rule/1",
    initial_artifact="stub",
    artifact_editable=True,
    parity_note="E4 active-discovery arm: same tools, probe budget is the axis",
)
