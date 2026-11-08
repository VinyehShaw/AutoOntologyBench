"""Ontology-intervention replay accounting (owner: W6; Review-5B §6 E2).

Deterministic replay intervention: frozen public-fact/action steps, ONLY the
artifact O varies (gold vs wrong vs repaired vs controls). This module
computes the decision sequence per O variant via the ontology consumer and
counts ``decision_changes_due_to_O`` partitioned into governed vs unaffected
steps. Instrument-level ONLY — whole-task causality additionally requires
free-planning paired runs (never inferred from replay alone).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from autoontobench.core.ontology_consumer import OntologyConsumer


@dataclass(frozen=True)
class InterventionStep:
    """One frozen replay step: PUBLIC facts + the proposed action."""

    public_facts: Mapping[str, Any]
    action: Mapping[str, Any]
    governed: bool  # True iff the fault is designed to govern this step


@dataclass(frozen=True)
class StepDecision:
    step_index: int
    governed: bool
    decision: str | None
    status: str


@dataclass(frozen=True)
class InterventionComparison:
    """Paired comparison of one O variant against the baseline (gold)."""

    baseline: str
    variant: str
    governed_changes: int
    governed_total: int
    unaffected_changes: int
    unaffected_total: int
    changed_steps: tuple[int, ...] = field(default_factory=tuple)


def replay_ontology_intervention(
    consumer: OntologyConsumer,
    o_variants: Mapping[str, bytes],
    steps: Sequence[InterventionStep],
    *,
    profile: str,
) -> dict[str, list[StepDecision]]:
    """Replay the frozen steps once per O variant (deterministic)."""
    decisions: dict[str, list[StepDecision]] = {}
    for name, artifact_bytes in o_variants.items():
        artifact = consumer.load(artifact_bytes, profile)
        variant_decisions: list[StepDecision] = []
        for index, step in enumerate(steps):
            prediction = consumer.query_or_check(
                artifact, step.public_facts, {"action": dict(step.action)}
            )
            variant_decisions.append(
                StepDecision(
                    step_index=index,
                    governed=step.governed,
                    decision=(
                        prediction.prediction
                        if isinstance(prediction.prediction, str)
                        else None
                    ),
                    status=str(prediction.detail.get("status", "INFRA_ERROR")),
                )
            )
        decisions[name] = variant_decisions
    return decisions


def count_decision_changes(
    decisions: Mapping[str, list[StepDecision]],
    *,
    baseline: str,
    variant: str,
) -> InterventionComparison:
    """Count decision changes of ``variant`` vs ``baseline``, partitioned by
    governed/unaffected steps. ``decision_changes_due_to_O`` = governed_changes."""
    base = decisions[baseline]
    other = decisions[variant]
    if len(base) != len(other):
        raise ValueError("decision sequences differ in length (replay must be paired)")
    governed_changes = governed_total = unaffected_changes = unaffected_total = 0
    changed_steps: list[int] = []
    for b, v in zip(base, other):
        changed = b.decision != v.decision
        if b.governed:
            governed_total += 1
            governed_changes += int(changed)
        else:
            unaffected_total += 1
            unaffected_changes += int(changed)
        if changed:
            changed_steps.append(b.step_index)
    return InterventionComparison(
        baseline=baseline,
        variant=variant,
        governed_changes=governed_changes,
        governed_total=governed_total,
        unaffected_changes=unaffected_changes,
        unaffected_total=unaffected_total,
        changed_steps=tuple(changed_steps),
    )
