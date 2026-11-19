"""Task-record checks: schema, split membership, counts, families, tags."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .contract import (
    DIFFICULTY_FILE,
    SPLIT_FILES,
    SPLIT_NAMES,
    WORLD_TAG_3_PATTERN,
    WORLD_TAG_4_PATTERN,
    PackTaskPrivate,
    PackTaskPublic,
    public_projection,
    read_jsonl,
)
from .errors import Violation


def parse_tasks(
    pack: Path, violations: list[Violation]
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]] | None:
    by_split: dict[str, list[dict[str, object]]] = {}
    for split in SPLIT_NAMES:
        relative = SPLIT_FILES[split]
        try:
            records = read_jsonl(pack / relative)
        except (ValueError, json.JSONDecodeError) as exc:
            violations.append(Violation("json_parse_error", relative, str(exc)))
            return None
        for index, record in enumerate(records, 1):
            try:
                PackTaskPrivate.model_validate(record)
            except ValidationError as exc:
                first = exc.errors()[0]
                violations.append(
                    Violation(
                        "task_schema",
                        f"{relative}:{index}",
                        f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}",
                    )
                )
            if record.get("split") != split:
                violations.append(
                    Violation(
                        "split_field_mismatch",
                        f"{relative}:{index}",
                        f"record split {record.get('split')!r} != file split {split!r}",
                    )
                )
        by_split[split] = records
    try:
        public_records = read_jsonl(pack / "public/examples_dev.jsonl")
    except (ValueError, json.JSONDecodeError) as exc:
        violations.append(Violation("json_parse_error", "public/examples_dev.jsonl", str(exc)))
        return None
    for index, record in enumerate(public_records, 1):
        try:
            PackTaskPublic.model_validate(record)
        except ValidationError as exc:
            first = exc.errors()[0]
            violations.append(
                Violation(
                    "task_schema",
                    f"public/examples_dev.jsonl:{index}",
                    f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}",
                )
            )
    return by_split, public_records


def check_splits(
    by_split: dict[str, list[dict[str, object]]],
    public_records: list[dict[str, object]],
    difficulty: dict[str, object],
    violations: list[Violation],
) -> None:
    seen: dict[str, str] = {}
    for split, records in by_split.items():
        for record in records:
            task_id = str(record.get("task_id"))
            if task_id in seen:
                violations.append(
                    Violation("duplicate_task_id", task_id, f"also in {seen[task_id]}")
                )
            seen[task_id] = split
    dev_projection = [public_projection(record) for record in by_split["dev"]]
    if dev_projection != public_records:
        violations.append(
            Violation(
                "public_dev_projection_mismatch",
                "public/examples_dev.jsonl",
                "public dev records are not the label-stripped dev split, in order",
            )
        )
    scenarios: dict[str, set[str]] = {}
    for split, records in by_split.items():
        for record in records:
            scenarios.setdefault(str(record.get("scenario")), set()).add(split)
    for scenario, splits in sorted(scenarios.items()):
        if len(splits) > 1:
            violations.append(
                Violation("paraphrase_split_leak", scenario, f"scenario split across {sorted(splits)}")
            )
    counts = difficulty.get("task_counts")
    if not isinstance(counts, dict):
        violations.append(Violation("count_mismatch", DIFFICULTY_FILE, "task_counts absent"))
        return
    declared = counts.get("per_split")
    expected_splits = declared if isinstance(declared, dict) else {}
    for split in SPLIT_NAMES:
        actual = len(by_split[split])
        if expected_splits.get(split) != actual:
            violations.append(
                Violation(
                    "split_count_mismatch",
                    split,
                    f"difficulty {expected_splits.get(split)!r} != records {actual}",
                )
            )
    total = len(seen)
    if counts.get("total") != total:
        violations.append(
            Violation("count_mismatch", DIFFICULTY_FILE, f"declared total != {total} records")
        )
    variants = counts.get("paraphrase_variants_per_scenario")
    per_scenario: dict[str, int] = {}
    variants_by_scenario: dict[str, list[object]] = {}
    for records in by_split.values():
        for record in records:
            key = str(record.get("scenario"))
            per_scenario[key] = per_scenario.get(key, 0) + 1
            variants_by_scenario.setdefault(key, []).append(record.get("paraphrase_variant"))
    if not isinstance(variants, int) or variants < 1:
        violations.append(
            Violation("count_formula_mismatch", DIFFICULTY_FILE, "paraphrase_variants absent")
        )
        return
    if counts.get("scenarios") != len(per_scenario) or total != len(per_scenario) * variants:
        violations.append(
            Violation(
                "count_formula_mismatch",
                DIFFICULTY_FILE,
                "total != scenarios x paraphrase_variants_per_scenario",
            )
        )
    for scenario, record_count in sorted(per_scenario.items()):
        if record_count != variants:
            violations.append(
                Violation(
                    "count_formula_mismatch",
                    scenario,
                    f"{record_count} variants != declared {variants}",
                )
            )
        variants_seen = variants_by_scenario[scenario]
        if len(set(variants_seen)) != len(variants_seen):
            violations.append(
                Violation(
                    "paraphrase_variant_duplicate",
                    scenario,
                    f"paraphrase variants not unique: {variants_seen}",
                )
            )


def check_families_and_tags(
    by_split: dict[str, list[dict[str, object]]],
    difficulty: dict[str, object],
    spec: dict[str, object],
    violations: list[Violation],
) -> None:
    records = [record for split in SPLIT_NAMES for record in by_split[split]]
    declared = difficulty.get("template_families")
    families = declared.get("families") if isinstance(declared, dict) else None
    used = {str(record.get("template_family")) for record in records}
    if not isinstance(families, list) or not families:
        violations.append(
            Violation("template_family_undeclared", DIFFICULTY_FILE, "families list absent")
        )
    else:
        for family in sorted(used - set(families)):
            violations.append(
                Violation("template_family_undeclared", str(family), "not declared in difficulty")
            )
        if declared.get("count") != len(families):
            violations.append(
                Violation(
                    "template_family_count_mismatch",
                    DIFFICULTY_FILE,
                    f"count {declared.get('count')!r} != {len(families)} families",
                )
            )
        transfer = declared.get("structural_transfer")
        if transfer is not None and transfer not in families:
            violations.append(
                Violation("template_family_undeclared", str(transfer), "transfer family undeclared")
            )
    generation = spec.get("generation")
    width = generation.get("instance_tag_width") if isinstance(generation, dict) else None
    tags = {str(record.get("world_tag")) for record in records}
    widths: set[int] = set()
    for tag in sorted(tags):
        if WORLD_TAG_3_PATTERN.match(tag):
            widths.add(3)
        elif WORLD_TAG_4_PATTERN.match(tag):
            widths.add(4)
        else:
            violations.append(Violation("world_tag_invalid", tag, "expected wNNN or wNNNN"))
    if len(widths) > 1:
        violations.append(Violation("world_tag_width_mismatch", "world_tag", "mixed tag widths"))
    if isinstance(width, int) and widths and widths != {width}:
        violations.append(
            Violation("world_tag_width_mismatch", "world_tag", f"expected width {width}")
        )