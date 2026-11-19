"""Label provenance checks: replay-sourced labels, spec versions, taxonomy codes."""

from __future__ import annotations

from .contract import LABEL_SOURCE_PATTERN, SPLIT_NAMES
from .errors import Violation


def check_labels(
    by_split: dict[str, list[dict[str, object]]],
    provenance: dict[str, object],
    spec: dict[str, object],
    violations: list[Violation],
) -> None:
    taxonomy = spec.get("failure_taxonomy")
    taxonomy_keys = set(taxonomy) if isinstance(taxonomy, dict) else set()
    spec_block = provenance.get("spec")
    expected_version = spec_block.get("spec_version") if isinstance(spec_block, dict) else None
    for split in SPLIT_NAMES:
        for record in by_split[split]:
            task_id = str(record.get("task_id"))
            if record.get("spec_version") != expected_version:
                violations.append(
                    Violation(
                        "spec_version_mismatch",
                        task_id,
                        f"record spec_version != {expected_version!r}",
                    )
                )
            if not LABEL_SOURCE_PATTERN.match(str(record.get("label_source", ""))):
                violations.append(
                    Violation(
                        "label_source_invalid",
                        task_id,
                        "label_source is not <family>.<engine>.replay",
                    )
                )
            outcomes = record.get("outcomes")
            reasons = [record.get("expected_reason")]
            if isinstance(outcomes, list):
                reasons.extend(
                    outcome.get("reason") if isinstance(outcome, dict) else None
                    for outcome in outcomes
                )
            for reason in reasons:
                if taxonomy_keys and reason not in taxonomy_keys:
                    violations.append(
                        Violation(
                            "reason_out_of_taxonomy", task_id, f"reason {reason!r} not in spec"
                        )
                    )