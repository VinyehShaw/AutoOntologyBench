"""CODE_DSL_MATCHED arm (owner: W6): agent-authored executable code/DSL
artifact with matched primitive operators (Review-5B §6 E1).

M1 status: thin config only. The representation profile ``code-dsl/1`` is
passed verbatim to the ontology consumer, which rejects it honestly
(``PROFILE_NOT_IMPLEMENTED`` / unknown profile) until a real DSL interpreter
with matched primitives is implemented. Tool surface otherwise identical to
the ontology arm (edit/validate/check/seal).
"""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "code_dsl",
    representation="code-dsl/1",
    initial_artifact=None,
    artifact_editable=True,
    parity_note="matched primitive executors; representation tool differs only",
)
