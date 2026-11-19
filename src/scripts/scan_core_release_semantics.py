#!/usr/bin/env python3
"""Correlation scan: label-equivalent public tokens in an assembled release.

Joins ``release_core/tasks/tasks.jsonl`` (and each world's shipped
``public/examples_dev.jsonl``) to the source packs' private records and
reports every public token channel that could restate the private verdict:

- feature tokens in the verdict namespace (``polarity:*`` keys or
  accepted/rejected/canonical/counterfactual value vocabulary) — the channel
  the 2026-11-18 final QA found 1:1 with ``expected_ok``;
- prompts quoting their record's private variant (``counterfactual``) or
  ``scenario`` id (the gated prompt channel);
- public identity fields (``template``, ``template_family``, ``family:``
  feature values, ``actions[].variant``) that equal an original private name or
  carry label-word vocabulary, plus each world's ``metadata.json`` family list
  and ``structural_transfer`` entry;
- informationally, prompt wording that differs by polarity inside a
  mixed-polarity ``(world, template_family, paraphrase_variant)`` group —
  instance-value differences are ordinary task content, so this is reported
  for transparency, not gated.

Content metadata that merely correlates with the label (family/model/taxon
tokens, instance names in prompts) is reported separately as informational:
it is task content, not a restated verdict. Opaque identity groups may still be
single-label by construction; the scan reports that as informational too.

Usage:
    python3.11 src/scripts/scan_core_release_semantics.py \
        --release release_core [--json reports/core_semantic_neutralization/scan_after.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_release.contract import (  # noqa: E402
    LABEL_FEATURE_KEY_GATE,
    LABEL_FEATURE_VALUE_PATTERN,
    SPLIT_NAMES,
    read_json,
    read_jsonl,
)
from scripts.core_release.semantics import (  # noqa: E402
    has_label_word,
    private_identity_names,
    public_identity_values,
)


def _verdict_token(token: str) -> bool:
    key, _, value = token.partition(":")
    return key in LABEL_FEATURE_KEY_GATE or bool(
        LABEL_FEATURE_VALUE_PATTERN.match(value)
    )


@dataclass
class ScanHits:
    """Mutable accumulator shared by the task, example, and metadata surfaces."""

    joined: int = 0
    verdict_tokens: Counter[str] = field(default_factory=Counter)
    feature_label: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"ok": 0, "bad": 0})
    )
    prompt_variant_hits: list[str] = field(default_factory=list)
    prompt_scenario_hits: list[str] = field(default_factory=list)
    identity_name_hits: Counter[str] = field(default_factory=Counter)
    identity_label_word_hits: Counter[str] = field(default_factory=Counter)
    identity_samples: list[str] = field(default_factory=list)

    def record_identity(
        self, surface: str, channel: str, value: str, names: frozenset[str]
    ) -> None:
        key = f"{surface}:{channel}"
        if value in names:
            self.identity_name_hits[key] += 1
            if len(self.identity_samples) < 10:
                self.identity_samples.append(f"{key}:private-name")
        if has_label_word(value):
            self.identity_label_word_hits[key] += 1
            if len(self.identity_samples) < 10:
                self.identity_samples.append(f"{key}:label-word")


def _load_private(packs: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    private: dict[tuple[str, str], dict[str, object]] = {}
    for pack in packs:
        difficulty = read_json(pack / "manifests/difficulty.json")
        world_id = str(difficulty["world_id"])
        for split in SPLIT_NAMES:
            name = "dev_labels" if split == "dev" else split
            for record in read_jsonl(pack / "private" / f"{name}.jsonl"):
                private[(world_id, str(record["task_id"]))] = record
    return private


def _check_surface(
    surface_rows: list[tuple[str, dict[str, object]]],
    private: dict[tuple[str, str], dict[str, object]],
    names: frozenset[str],
    source: str,
    hits: ScanHits,
) -> None:
    for world_id, row in surface_rows:
        record = private.get((world_id, str(row.get("task_id"))))
        if record is None:
            continue
        hits.joined += 1
        for token in row.get("features", []):
            text = str(token)
            if _verdict_token(text):
                hits.verdict_tokens[text] += 1
            bucket = hits.feature_label[text]
            bucket["ok" if record.get("expected_ok") else "bad"] += 1
        for channel, value in public_identity_values(row):
            hits.record_identity(source, channel, value, names)
        prompt = str(row.get("prompt"))
        counterfactual = record.get("counterfactual")
        if (
            isinstance(counterfactual, str)
            and counterfactual
            and counterfactual in prompt
        ):
            hits.prompt_variant_hits.append(f"{source}:{world_id}:{row.get('task_id')}")
        scenario = str(record.get("scenario") or "")
        if scenario and scenario in prompt:
            hits.prompt_scenario_hits.append(f"{source}:{world_id}:{row.get('task_id')}")


def _check_metadata(
    release: Path,
    world_ids: list[str],
    names: frozenset[str],
    hits: ScanHits,
) -> None:
    for world_id in world_ids:
        path = release / "worlds" / world_id / "metadata.json"
        if not path.is_file():
            continue
        block = read_json(path).get("template_families")
        if not isinstance(block, dict):
            continue
        families = block.get("families")
        values = (
            [("metadata_family", str(family)) for family in families]
            if isinstance(families, list)
            else []
        )
        structural = block.get("structural_transfer")
        if isinstance(structural, str):
            values.append(("metadata_structural_transfer", structural))
        for channel, value in values:
            hits.record_identity("metadata", channel, value, names)


def _polarity_group_summary(
    task_rows: list[tuple[str, dict[str, object]]],
    private: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    """Informational: prompts that differ by polarity inside one family group."""
    groups: dict[tuple[str, str, int], dict[bool, set[str]]] = defaultdict(
        lambda: {True: set(), False: set()}
    )
    single_label_groups: dict[bool, int] = defaultdict(int)
    for world_id, row in task_rows:
        record = private.get((world_id, str(row["task_id"])))
        if record is None:
            continue
        group = (
            world_id,
            str(row["template_family"]),
            int(row["paraphrase_variant"]),
        )
        groups[group][bool(record["expected_ok"])].add(str(row["prompt"]))
    for sides in groups.values():
        if sides[True] and not sides[False]:
            single_label_groups[True] += 1
        elif sides[False] and not sides[True]:
            single_label_groups[False] += 1
    return {
        "polarity_split_prompt_groups": {
            f"{group[0]}:{group[1]}:pv{group[2]}": {
                "accepted_prompts": sorted(sides[True]),
                "rejected_prompts": sorted(sides[False]),
            }
            for group, sides in sorted(groups.items())
            if sides[True] and sides[False] and sides[True] != sides[False]
        },
        "single_label_identity_groups": {
            "accepted_only": single_label_groups[True],
            "rejected_only": single_label_groups[False],
            "total_groups": len(groups),
        },
    }


def scan(release: Path, packs: list[Path]) -> dict[str, object]:
    private = _load_private(packs)
    names = private_identity_names(private.values())
    task_rows = [
        (str(row["world_id"]), row) for row in read_jsonl(release / "tasks/tasks.jsonl")
    ]
    world_ids = sorted({world_id for world_id, _ in task_rows})
    example_rows: list[tuple[str, dict[str, object]]] = []
    for world_id in world_ids:
        path = release / "worlds" / world_id / "public/examples_dev.jsonl"
        if path.is_file():
            example_rows.extend((world_id, row) for row in read_jsonl(path))

    hits = ScanHits()
    _check_surface(task_rows, private, names, "tasks.jsonl", hits)
    _check_surface(example_rows, private, names, "examples_dev.jsonl", hits)
    _check_metadata(release, world_ids, names, hits)

    informational = _polarity_group_summary(task_rows, private)
    single_polarity_content_tokens = {
        token: counts
        for token, counts in sorted(hits.feature_label.items())
        if (counts["ok"] == 0) != (counts["bad"] == 0) and not _verdict_token(token)
    }
    identity_leaks = bool(
        hits.identity_name_hits or hits.identity_label_word_hits
    )
    return {
        "release": str(release),
        "task_rows": len(task_rows),
        "example_rows": len(example_rows),
        "private_records": len(private),
        "joined_rows": hits.joined,
        "verdict_feature_tokens": dict(hits.verdict_tokens),
        "prompt_variant_id_hits": hits.prompt_variant_hits,
        "prompt_scenario_id_hits": hits.prompt_scenario_hits,
        "identity_original_name_hits": sum(hits.identity_name_hits.values()),
        "identity_original_name_by_channel": dict(hits.identity_name_hits),
        "identity_label_word_hits": sum(hits.identity_label_word_hits.values()),
        "identity_label_word_by_channel": dict(hits.identity_label_word_hits),
        "identity_samples": hits.identity_samples,
        **informational,
        "single_polarity_content_token_count": len(single_polarity_content_tokens),
        "ok": not (
            hits.verdict_tokens
            or hits.prompt_variant_hits
            or hits.prompt_scenario_hits
            or identity_leaks
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="release_core", metavar="DIR")
    parser.add_argument(
        "--packs",
        default="pilot_worlds",
        metavar="GLOBROOT",
        help="directory holding world_L*_scaleup/pack",
    )
    parser.add_argument("--json", metavar="PATH", help="write the scan summary as JSON")
    args = parser.parse_args(argv)
    packs = sorted(Path(args.packs).glob("world_L*_scaleup/pack"))
    summary = scan(Path(args.release), packs)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"rows joined: {summary['joined_rows']}")
    print(f"verdict feature tokens (gated): {summary['verdict_feature_tokens']}")
    print(
        "prompts naming private variant id (gated): "
        f"{len(summary['prompt_variant_id_hits'])}"
    )
    print(
        "prompts naming private scenario id (gated): "
        f"{len(summary['prompt_scenario_id_hits'])}"
    )
    print(
        "identity fields equal to a private name (gated): "
        f"{summary['identity_original_name_hits']}"
    )
    print(
        "identity fields containing a label word (gated): "
        f"{summary['identity_label_word_hits']}"
    )
    print(
        "informational: mixed-polarity groups with polarity-split prompts: "
        f"{len(summary['polarity_split_prompt_groups'])}"
    )
    print(
        "informational: single-label opaque identity groups: "
        f"{summary['single_label_identity_groups']}"
    )
    print(
        "informational: single-polarity content feature tokens: "
        f"{summary['single_polarity_content_token_count']}"
    )
    print("SCAN", "PASS" if summary["ok"] else "FAIL")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())