"""Arm configuration type (owner: W6). Arms are thin configs, not code forks.

Tool parity rule (Review-5B §6 E1): every arm exposes the SAME base action
channel; only the representation tool differs. The base tool set is shared;
representation arms add artifact tools.
"""
from __future__ import annotations

from dataclasses import dataclass

from autoontobench.runners.tool_api import TOOL_NAMES

#: Tools every arm shares (identical action channel + observation surface).
BASE_TOOLS: frozenset[str] = frozenset(
    {
        "inspect_schema",
        "inspect_state",
        "read_public_doc",
        "probe_native",
        "perform_action",
        "get_feedback",
        "submit_final",
    }
)

#: Representation (artifact) tools; only arms carrying a representation get them.
ARTIFACT_TOOLS: frozenset[str] = frozenset(
    {"edit_artifact", "validate_artifact", "check_action_with_ontology", "seal_artifact"}
)

assert BASE_TOOLS | ARTIFACT_TOOLS == frozenset(TOOL_NAMES)  # exhaustive partition


@dataclass(frozen=True)
class ArmConfig:
    """One experimental arm as data. ``representation`` is the ONLY axis on
    which tool surfaces differ; ``initial_artifact`` names a key into the
    harness's artifact source map (e.g. "stub" / "gold" / None)."""

    arm_id: str
    representation: str | None  # ontology profile id, "code-dsl/1", or None
    initial_artifact: str | None
    artifact_editable: bool
    tool_allowlist: frozenset[str]
    parity_note: str


def make_arm(
    arm_id: str,
    *,
    representation: str | None,
    initial_artifact: str | None = None,
    artifact_editable: bool = True,
    parity_note: str = "",
) -> ArmConfig:
    tools = set(BASE_TOOLS)
    if representation is not None:
        tools |= ARTIFACT_TOOLS
        if not artifact_editable:
            tools.discard("edit_artifact")
    return ArmConfig(
        arm_id=arm_id,
        representation=representation,
        initial_artifact=initial_artifact,
        artifact_editable=artifact_editable,
        tool_allowlist=frozenset(tools),
        parity_note=parity_note,
    )
