"""RAG_MATCHED arm (owner: W6): matched documentation retrieval.

Same tool surface as DIRECT (read_public_doc is the retrieval channel);
the matched-corpus wiring lands with real source docs in a later wave.
"""
from __future__ import annotations

from autoontobench.runners.arm_config import ArmConfig, make_arm

ARM: ArmConfig = make_arm(
    "rag",
    representation=None,
    parity_note="matched-docs retrieval via read_public_doc; tool surface identical to direct",
)
