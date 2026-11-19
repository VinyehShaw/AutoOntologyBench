"""Public-surface leakage checks (hidden ids, label keys, private paths, gold hashes)."""

from __future__ import annotations

from pathlib import Path

from .contract import GOLD_ARTIFACTS, LABEL_KEY_TOKENS, PUBLIC_FILES, SPLIT_NAMES
from .errors import Violation


def check_leakage(
    pack: Path,
    by_split: dict[str, list[dict[str, object]]],
    provenance: dict[str, object],
    violations: list[Violation],
) -> None:
    hidden_ids = [
        str(record.get("task_id"))
        for split in SPLIT_NAMES
        if split != "dev"
        for record in by_split[split]
    ]
    artifacts = provenance.get("artifacts")
    artifact_hashes = artifacts if isinstance(artifacts, dict) else {}
    needles = [
        *hidden_ids,
        "private/",
        *(str(artifact_hashes[path]) for path in GOLD_ARTIFACTS if path in artifact_hashes),
    ]
    for relative in PUBLIC_FILES:
        text = (pack / relative).read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle and needle in text:
                violations.append(
                    Violation("public_leak", relative, f"public file contains {needle!r}")
                )
        for token in LABEL_KEY_TOKENS:
            if token in text:
                violations.append(
                    Violation("public_leak", relative, f"public file contains label key {token}")
                )