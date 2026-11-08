"""T2 blueprint composition: sample admissible source-unit compositions (no LLM).

Real composition needs the native domains (W2/W3/W4) to check rule compatibility
and replay feasibility; those adapters are being built in parallel. The interface
below is the stable seam; the sampler raises until a NativeDomain is injected.
"""
from __future__ import annotations

from typing import Any, Protocol

from autoontobench.core.protocols import NativeDomain


class CompositionError(Exception):
    pass


class CompositionSampler(Protocol):
    def sample(self, source_id: str, n: int, seed: int) -> list[dict[str, Any]]: ...


class NativeComposeSampler:
    """Samples blueprint dicts from admissible source-rule compositions."""

    def __init__(self, domain: NativeDomain, source_id: str) -> None:
        self.domain = domain
        self.source_id = source_id

    def sample(self, source_id: str, n: int, seed: int) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "NOT_IMPLEMENTED_DEP: blueprint composition needs the source NativeDomain "
            "adapters (W2/W3/W4) to enumerate admissible rule-unit compositions; "
            "interface frozen, sampler lands with the first native domain (M2)")


def sample_blueprints(domain: NativeDomain | None, source_id: str, n: int,
                      seed: int) -> list[dict[str, Any]]:
    if domain is None:
        raise NotImplementedError(
            "NOT_IMPLEMENTED_DEP: compose.sample_blueprints requires a NativeDomain "
            f"for source {source_id!r} (owner: W2/W3/W4); none injected")
    return NativeComposeSampler(domain, source_id).sample(source_id, n, seed)
