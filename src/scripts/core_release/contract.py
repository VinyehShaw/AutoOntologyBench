"""Core-track benchmark release contract v1: pack layout, task schema, IO.

The input side is the world-pack contract documented in
``docs/CORE_TRACK_SCALE_PLAN.md`` section 1.1 (17 files, filenames exact). This
module owns the machine-readable version of that contract plus the release-side
record shapes; it performs no filesystem mutation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_RELEASE = "aob-core-release/1.0"
SCHEMA_WORLD_METADATA = "aob-core-world-metadata/1.0"
SCHEMA_WORLD_INDEX = "aob-core-world-index/1.0"
PACK_CONTRACT_VERSION: Final = "1"

SPLIT_NAMES: Final[tuple[str, ...]] = (
    "dev",
    "hidden_iid",
    "hidden_comp",
    "hidden_transfer",
)
SPLIT_FILES: Final[dict[str, str]] = {
    "dev": "private/dev_labels.jsonl",
    "hidden_iid": "private/hidden_iid.jsonl",
    "hidden_comp": "private/hidden_comp.jsonl",
    "hidden_transfer": "private/hidden_transfer.jsonl",
}

PUBLIC_FILES: Final[tuple[str, ...]] = (
    "public/README.md",
    "public/examples_dev.jsonl",
    "public/observations.jsonl",
    "public/ontology_init.ttl",
    "public/probe_api_spec.json",
    "public/shapes_init.ttl",
    "public/source_schema.json",
)
PRIVATE_FILES: Final[tuple[str, ...]] = (
    "private/corruption_manifest.json",
    "private/dev_labels.jsonl",
    "private/gold_ontology.ttl",
    "private/gold_shapes.ttl",
    "private/hidden_comp.jsonl",
    "private/hidden_iid.jsonl",
    "private/hidden_transfer.jsonl",
    "private/world_rules.py",
)
MANIFEST_FILES: Final[tuple[str, ...]] = (
    "manifests/difficulty.json",
    "manifests/provenance.json",
)
PACK_FILES: Final[tuple[str, ...]] = tuple(
    sorted(PUBLIC_FILES + PRIVATE_FILES + MANIFEST_FILES)
)
PROVENANCE_FILE: Final = "manifests/provenance.json"
DIFFICULTY_FILE: Final = "manifests/difficulty.json"
GOLD_ARTIFACTS: Final[tuple[str, ...]] = (
    "private/gold_ontology.ttl",
    "private/gold_shapes.ttl",
)

PUBLIC_TASK_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "world_tag",
    "spec_version",
    "template",
    "template_family",
    "actions",
    "prompt",
    "features",
    "instance_ids",
    "paraphrase_variant",
)
RELEASE_TASK_FIELDS: Final[tuple[str, ...]] = (
    "release_task_id",
    "world_id",
    "lane_id",
    "split",
) + PUBLIC_TASK_FIELDS
PRIVATE_TASK_FIELDS: Final[tuple[str, ...]] = (
    "split",
    "scenario",
    "expected",
    "expected_ok",
    "expected_reason",
    "outcomes",
    "counterfactual",
    "label_source",
)
# Quoted JSON keys: a raw phrase such as "scenario" may legitimately appear in
# prose, but a serialized label field in a public file is always a leak.
LABEL_KEY_TOKENS: Final[tuple[str, ...]] = tuple(
    f'"{field}"' for field in PRIVATE_TASK_FIELDS
)

WORLD_ID_PATTERN: Final = re.compile(r"^world_L(?P<lane>\d{2})$")
WORLD_TAG_3_PATTERN: Final = re.compile(r"^w\d{3}$")
WORLD_TAG_4_PATTERN: Final = re.compile(r"^w\d{4}$")

# --- Release-only opaque identities ---------------------------------------
#
# Public task identity fields (``template``, ``template_family``, ``family:``
# feature values, ``actions[].variant``) carry names chosen by the world
# generator, and those names restate the private verdict (``validate_canonical``,
# ``cf_bad_image`` and similar). The release therefore publishes deterministic
# opaque IDs instead of the original words. The digest scope binds the domain,
# the identifier kind, the world's ``spec_version``, the ``world_id``, and the
# original name: the same original always projects to the same public ID, and
# unrelated originals cannot collide in practice. Only 16 hex digits of the
# SHA-256 digest are published, so the original name is not recoverable from
# the release, while the private pack keeps the name and the mapping.
OPAQUE_ID_DOMAIN: Final = "aob-core-release/opaque-id/v1"
OpaqueIdKind = Literal["template", "family", "variant"]
OPAQUE_ID_PREFIXES: Final[dict[str, str]] = {
    "template": "tpl",
    "family": "fam",
    "variant": "var",
}
OPAQUE_ID_PATTERN: Final = re.compile(r"^(?:tpl|fam|var)_[0-9a-f]{16}$")
LABEL_SOURCE_PATTERN: Final = re.compile(
    r"^[a-z0-9_]+\.[a-z0-9_]+(?:\.[a-z0-9_]+)*\.replay$"
)
TASK_ID_PATTERN: Final = r"^task_[0-9]{4}$"


class PackTaskPublic(BaseModel):
    """Public task record; ``extra='forbid'`` is the label-leak gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=TASK_ID_PATTERN)
    world_tag: str = Field(min_length=4, max_length=5)
    spec_version: str = Field(min_length=1)
    template: str = Field(min_length=1)
    template_family: str = Field(min_length=1)
    actions: list[dict[str, object]] = Field(min_length=1)
    prompt: str = Field(min_length=1)
    features: list[str]
    instance_ids: list[str] = Field(min_length=1)
    paraphrase_variant: int = Field(ge=0)


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    reason: str = Field(min_length=1)
    payload: object | None = None


class PackTaskPrivate(PackTaskPublic):
    """Pack-private record: public fields plus replay-derived labels."""

    split: Literal["dev", "hidden_iid", "hidden_comp", "hidden_transfer"]
    scenario: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    expected_ok: bool
    expected_reason: str = Field(min_length=1)
    outcomes: list[Outcome] = Field(min_length=1)
    counterfactual: object | None = None
    label_source: str = Field(min_length=1)


def read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(loaded)
    return records


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(record: dict[str, object]) -> str:
    """One sorted-key, compact JSON line (release task records are canonical)."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    return "".join(f"{canonical_json(record)}\n" for record in records).encode("utf-8")


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def repo_relative(path: Path, repo_root: Path) -> str:
    """Repo-relative POSIX path when possible; absolute otherwise."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tree_entries(root: Path, exclude: frozenset[str]) -> list[dict[str, object]]:
    """Sorted ``{path, sha256, bytes}`` for every regular file except ``exclude``."""
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        entries.append(
            {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return sorted(entries, key=lambda entry: str(entry["path"]))


def public_projection(record: dict[str, object]) -> dict[str, object]:
    """Strip every non-public field from a task record."""
    return {field: record[field] for field in PUBLIC_TASK_FIELDS}


# --- Release-side semantic sanitization -----------------------------------
#
# The pack contract keeps full private fidelity; the release projection must
# additionally drop public tokens that restate the replay verdict. Two
# channels are covered:
#
# 1. Feature tokens whose key or value names the label axis: the
#    ``polarity:accepted|rejected`` namespace (1:1 with ``expected_ok`` on the
#    2026-11-18 QA scan) and ``variant:`` values drawn from the
#    canonical/counterfactual polarity-naming vocabulary.
# 2. Prompts that name a polarity-marked state variant (e.g. L01
#    ``gk_validate_bundle`` prompts quoting ``cf_*`` ids or asserting the
#    "canonical compliant" polarity). ``NEUTRAL_PROMPT_TABLE`` replaces them
#    with a per-paraphrase neutral wording; the variant identity stays
#    available in the public ``actions`` parameters, which the world schema
#    declares and the harness requires.

LABEL_FEATURE_KEYS: Final[frozenset[str]] = frozenset({"polarity"})
LABEL_FEATURE_VALUE_PATTERN: Final = re.compile(
    r"^(?:accepted|rejected|canonical|counterfactual(?:_.*)?|cf_.*)$"
)
# Gate-only superset: future verdict namespaces are not stripped silently;
# they fail assembly (``public_semantic_leak``) until a human extends the
# projection. Keyed by feature-token key (the part before the first colon).
LABEL_FEATURE_KEY_GATE: Final[frozenset[str]] = frozenset(
    {"polarity", "variant", "expected", "label", "labels", "verdict", "outcome", "answer"}
)

# (spec_version, template_family) -> paraphrase_variant -> neutral prompt.
NEUTRAL_PROMPT_TABLE: Final[dict[tuple[str, str], dict[int, str]]] = {
    ("gatekeeper_policy/1.0.0", "gk_validate_bundle"): {
        0: (
            "Validate the cluster state variant named in the task's action list "
            "against the shared SHACL shapes bundle. Report whether the projected "
            "state conforms."
        ),
        1: (
            "Run SHACL Core validation of the cluster state variant named in the "
            "task's action list with the current shapes bundle and report the "
            "conformance verdict."
        ),
    },
}


def is_label_feature_token(token: str) -> bool:
    """True when a feature token restates the verdict or its polarity naming."""
    key, _, value = token.partition(":")
    return key in LABEL_FEATURE_KEYS or bool(LABEL_FEATURE_VALUE_PATTERN.match(value))


def opaque_release_id(
    kind: OpaqueIdKind, original: str, *, spec_version: str, world_id: str = ""
) -> str:
    """Deterministic opaque release ID for one private identity name.

    ``kind`` is one of ``template``/``family``/``variant`` and selects the
    public prefix (``tpl``/``fam``/``var``). The SHA-256 scope includes the
    domain tag, the kind, ``spec_version``, ``world_id``, and the original
    name, so IDs are stable across rebuilds and distinct per identity. The
    original words never appear in the result.
    """
    prefix = OPAQUE_ID_PREFIXES.get(kind)
    if prefix is None:
        raise ValueError(f"unknown opaque id kind: {kind!r}")
    material = "\x1f".join((OPAQUE_ID_DOMAIN, kind, spec_version, world_id, original))
    return f"{prefix}_{sha256_bytes(material.encode('utf-8'))[:16]}"


def release_projection(
    record: dict[str, object], *, world_id: str = ""
) -> dict[str, object]:
    """Public projection with label-equivalent tokens and identities removed.

    This is the release boundary: private packs are unchanged, but the public
    row never carries a feature token that is 1:1 with ``expected_ok``, never
    names a private state variant in its prompt, and never exposes the
    generator's label-semantic identity names. ``template``, ``template_family``,
    ``family:`` feature values, and string ``actions[].variant`` values are
    replaced with opaque release IDs (``tpl_…``/``fam_…``/``var_…``). The input
    record is not mutated: nested ``features``/``actions`` are copied.
    """
    row = public_projection(record)
    spec_version = str(record.get("spec_version") or "")
    scope = {"spec_version": spec_version, "world_id": world_id}

    template = row.get("template")
    if isinstance(template, str):
        row["template"] = opaque_release_id("template", template, **scope)

    family = row.get("template_family")
    if isinstance(family, str):
        row["template_family"] = opaque_release_id("family", family, **scope)

    features = row["features"]
    assert isinstance(features, list)
    projected_features: list[str] = []
    for token in features:
        text = str(token)
        if is_label_feature_token(text):
            continue
        key, separator, value = text.partition(":")
        if separator and key == "family" and value:
            projected_features.append(
                f"family:{opaque_release_id('family', value, **scope)}"
            )
            continue
        projected_features.append(text)
    row["features"] = projected_features

    actions = row["actions"]
    assert isinstance(actions, list)
    projected_actions: list[dict[str, object]] = []
    for action in actions:
        assert isinstance(action, dict)
        copied = dict(action)
        variant = copied.get("variant")
        if isinstance(variant, str):
            copied["variant"] = opaque_release_id("variant", variant, **scope)
        projected_actions.append(copied)
    row["actions"] = projected_actions

    table = NEUTRAL_PROMPT_TABLE.get(
        (spec_version, str(record.get("template_family")))
    )
    if table is not None:
        variant = record.get("paraphrase_variant")
        neutral = table.get(variant) if isinstance(variant, int) else None
        if neutral is not None:
            row["prompt"] = neutral
    return row


def release_task_row(
    record: dict[str, object],
    *,
    world_id: str,
    lane_id: str,
    split: str,
) -> dict[str, object]:
    row = release_projection(record, world_id=world_id)
    row["release_task_id"] = f"{world_id}:{record['task_id']}"
    row["world_id"] = world_id
    row["lane_id"] = lane_id
    row["split"] = split
    return row