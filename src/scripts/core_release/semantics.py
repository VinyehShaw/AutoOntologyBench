"""Semantic label-leak gate for the release projection.

The structural leakage check (``leakage.py``) scans pack public files for
hidden ids, label keys, private paths, and gold hashes. This module guards the
assembled release rows against the semantic channels found in the final QA
reports:

1. A public token that restates the private verdict (``features`` carrying
   ``polarity:accepted|rejected``) or a prompt that names the private
   polarity-marked state variant (L01 ``gk_validate_bundle`` quoting ``cf_*``
   counterfactual ids).
2. A public identity field (``template``, ``template_family``, ``family:``
   feature value, ``actions[].variant``) that either equals an original
   private name (template/family/counterfactual/scenario) or carries a
   label-word pattern such as ``canonical``/``cf_``/``valid``/``missing``.
   The release projection replaces those values with opaque IDs, so the gate
   is an independent post-condition: any future projection regression fails
   assembly instead of shipping.

The gate runs on the *sanitized* release rows. Prompt/content vocabulary is
deliberately not pattern-gated; only the four identity channels above are.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Final

from .contract import (
    LABEL_FEATURE_KEY_GATE,
    LABEL_FEATURE_VALUE_PATTERN,
    SPLIT_NAMES,
)
from .errors import Violation

PRIVATE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "template",
    "template_family",
    "counterfactual",
    "scenario",
)
# Label-word vocabulary at token boundaries (``_`` and punctuation separate
# tokens). ``validate`` does not match ``valid``; ``cf_bad_image`` matches both
# the literal ``cf_`` prefix and the ``bad`` token.
IDENTITY_LABEL_WORD_PATTERN: Final = re.compile(
    r"(?i)(?:^|[^a-z0-9])"
    r"(?:canonical|counterfactual|accepted|rejected|valid|invalid|pass|fail|"
    r"good|bad|missing|positive|negative)"
    r"(?=$|[^a-z0-9])"
    r"|cf_"
)


def private_identity_names(records: Iterable[dict[str, object]]) -> frozenset[str]:
    """Every original identity name a release must not publish verbatim."""
    names: set[str] = set()
    for record in records:
        for field in PRIVATE_IDENTITY_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value:
                names.add(value)
    return frozenset(names)


def has_label_word(value: str) -> bool:
    """True when ``value`` carries label-word vocabulary at a token boundary."""
    return bool(IDENTITY_LABEL_WORD_PATTERN.search(value))


def public_identity_values(row: dict[str, object]) -> Iterator[tuple[str, str]]:
    """Yield ``(channel, value)`` for the four public identity channels."""
    template = row.get("template")
    if isinstance(template, str) and template:
        yield "template", template
    family = row.get("template_family")
    if isinstance(family, str) and family:
        yield "template_family", family
    features = row.get("features")
    if isinstance(features, list):
        for token in features:
            key, separator, value = str(token).partition(":")
            if separator and key == "family" and value:
                yield "family_feature", value
    actions = row.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            variant = action.get("variant")
            if isinstance(variant, str) and variant:
                yield "variant", variant


def check_semantic_leakage(
    release_rows: list[dict[str, object]],
    by_split: dict[str, list[dict[str, object]]],
    violations: list[Violation],
) -> None:
    """Fail assembly when a release row still leaks the private label."""
    private_records = [
        record for split in SPLIT_NAMES for record in by_split[split]
    ]
    private_by_id = {str(record.get("task_id")): record for record in private_records}
    identity_names = private_identity_names(private_records)
    for row in release_rows:
        task_id = str(row.get("task_id"))
        features = row.get("features")
        if isinstance(features, list):
            for token in features:
                text = str(token)
                key, _, value = text.partition(":")
                if key in LABEL_FEATURE_KEY_GATE or LABEL_FEATURE_VALUE_PATTERN.match(
                    value
                ):
                    violations.append(
                        Violation(
                            "public_semantic_leak",
                            task_id,
                            f"release feature token {text!r} restates the verdict",
                        )
                    )
        for channel, value in public_identity_values(row):
            if value in identity_names:
                violations.append(
                    Violation(
                        "public_semantic_leak",
                        task_id,
                        f"release {channel} {value!r} names a private identity",
                    )
                )
            if has_label_word(value):
                violations.append(
                    Violation(
                        "public_semantic_leak",
                        task_id,
                        f"release {channel} {value!r} contains a label word",
                    )
                )
        record = private_by_id.get(task_id)
        if record is None:
            continue
        prompt = str(row.get("prompt"))
        counterfactual = record.get("counterfactual")
        if isinstance(counterfactual, str) and counterfactual and counterfactual in prompt:
            violations.append(
                Violation(
                    "public_semantic_leak",
                    task_id,
                    f"release prompt names the private variant {counterfactual!r}",
                )
            )
        scenario = str(record.get("scenario") or "")
        if scenario and scenario in prompt:
            violations.append(
                Violation(
                    "public_semantic_leak",
                    task_id,
                    f"release prompt names the private scenario {scenario!r}",
                )
            )


__all__ = [
    "IDENTITY_LABEL_WORD_PATTERN",
    "PRIVATE_IDENTITY_FIELDS",
    "check_semantic_leakage",
    "has_label_word",
    "private_identity_names",
    "public_identity_values",
]