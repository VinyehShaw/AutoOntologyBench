"""DIRECT arm (owner: W6): raw tools/docs only, no representation artifact."""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "direct",
    representation=None,
    parity_note="baseline: same action channel, no representation tool (E1 DIRECT)",
)
