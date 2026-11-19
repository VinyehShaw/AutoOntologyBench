"""Render the generated release documents (README + contract) deterministically."""

from __future__ import annotations

from pathlib import Path

from .contract import PACK_CONTRACT_VERSION, SPLIT_NAMES

_TEMPLATE_DIR = Path(__file__).with_name("templates")
_TARGET_TASK_COUNT = 19_947

CHECK_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("missing_pack", "pack directory does not exist"),
    ("missing_file", "a required pack file is absent"),
    ("unexpected_file", "a file outside the 17-file contract layout is present"),
    ("artifact_hash_missing", "a pack file is unlisted in provenance.artifacts"),
    ("artifact_hash_unexpected", "provenance.artifacts lists a non-contract path"),
    ("artifact_hash_mismatch", "a declared sha256 does not match the bytes on disk"),
    ("json_parse_error", "a manifest or JSONL file is not valid JSON"),
    ("spec_unresolvable", "provenance.spec.path does not resolve to a file"),
    ("spec_hash_mismatch", "the frozen spec bytes drifted from provenance.spec.sha256"),
    ("spec_not_frozen", "spec status is not 'frozen'"),
    ("spec_parse_error", "the spec file is not a JSON object"),
    ("spec_version_mismatch", "spec_version disagrees across spec/provenance/difficulty"),
    ("taxonomy_missing", "the spec has no non-empty failure_taxonomy"),
    ("world_id_invalid", "world_id is not world_LNN"),
    ("world_id_mismatch", "difficulty and provenance world ids disagree"),
    ("lane_id_mismatch", "spec provenance lane_id disagrees with world_LNN"),
    ("task_schema", "a task record violates the public/private task schema"),
    ("split_field_mismatch", "a record's split field disagrees with its split file"),
    ("duplicate_task_id", "a task id occurs more than once in the pack"),
    ("public_dev_projection_mismatch", "public dev examples are not the stripped dev split"),
    ("paraphrase_split_leak", "paraphrase siblings of one scenario cross splits"),
    ("count_mismatch", "declared total or per-split counts disagree with the files"),
    ("split_count_mismatch", "a split's declared count disagrees with its record count"),
    ("count_formula_mismatch", "total != scenarios x paraphrase variants"),
    ("paraphrase_variant_duplicate", "one scenario repeats a paraphrase variant index"),
    ("label_source_invalid", "label_source is not <family>.<engine>.replay"),
    ("reason_out_of_taxonomy", "a reason code is outside the spec failure_taxonomy"),
    ("template_family_undeclared", "a used or declared transfer family is not in the family list"),
    ("template_family_count_mismatch", "declared family count disagrees with the family list"),
    ("world_tag_invalid", "world_tag is neither wNNN nor wNNNN"),
    ("world_tag_width_mismatch", "tag widths are mixed or disagree with instance_tag_width"),
    ("public_leak", "a public file leaks a hidden id, label key, private path, or gold hash"),
    ("public_semantic_leak", "a release row carries a verdict-namespace feature token, names a private variant/scenario id, or publishes a private/label-word identity value"),
    ("duplicate_world_id", "the same world id was supplied more than once"),
    ("duplicate_release_task_id", "a release_task_id collides across worlds"),
    ("release_exists", "output directory is non-empty and --force was not given"),
    ("release_path_invalid", "output directory is the repo root, filesystem root, or a pack"),
    ("manifest_drift", "a released file's hash/bytes disagree with MANIFEST.sha256"),
    ("unlisted_file", "a file exists in the release tree but not in the manifest"),
    ("forbidden_path", "the release tree contains a private-oracle path"),
    (
        "license_source_missing",
        "a world has no license registry entry, or a declared license text file is absent",
    ),
)


def _check_table() -> str:
    lines = ["| code | meaning |", "|---|---|"]
    lines.extend(f"| `{code}` | {meaning} |" for code, meaning in CHECK_DESCRIPTIONS)
    return "\n".join(lines)


def _split_totals(per_split: dict[str, int], total: int) -> str:
    lines = ["| split | records |", "|---|---:|"]
    lines.extend(f"| `{split}` | {per_split.get(split, 0)} |" for split in SPLIT_NAMES)
    lines.append(f"| **total** | **{total}** |")
    return "\n".join(lines)


def _world_table(worlds: list[dict[str, object]]) -> str:
    header = (
        "| world_id | lane | spec_version | dev | hidden_iid | hidden_comp | "
        "hidden_transfer | total | floors |"
    )
    lines = [header, "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for world in worlds:
        splits = world["per_split"]
        lines.append(
            "| {world_id} | {lane_id} | {spec_version} | {dev} | {iid} | {comp} | "
            "{transfer} | {total} | {floors} |".format(
                world_id=world["world_id"],
                lane_id=world["lane_id"],
                spec_version=world["spec_version"],
                dev=splits["dev"],
                iid=splits["hidden_iid"],
                comp=splits["hidden_comp"],
                transfer=splits["hidden_transfer"],
                total=world["task_count"],
                floors=world["split_floors_status"],
            )
        )
    return "\n".join(lines)


def _render(template_name: str, replacements: dict[str, str]) -> str:
    text = (_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    for token, value in replacements.items():
        text = text.replace(f"{{{{{token}}}}}", value)
    if "{{" in text:
        raise ValueError(f"unrendered template token in {template_name}")
    return text


def render_readme(
    worlds: list[dict[str, object]], per_split: dict[str, int], total: int
) -> str:
    return _render(
        "release_README.md",
        {
            "CONTRACT_VERSION": PACK_CONTRACT_VERSION,
            "WORLD_COUNT": str(len(worlds)),
            "TASK_COUNT": str(total),
            "SPLIT_TOTALS": _split_totals(per_split, total),
            "WORLD_TABLE": _world_table(worlds),
        },
    )


def render_contract() -> str:
    return _render(
        "release_contract.md",
        {
            "CONTRACT_VERSION": PACK_CONTRACT_VERSION,
            "CHECK_TABLE": _check_table(),
        },
    )


TARGET_TASK_COUNT = _TARGET_TASK_COUNT