"""L13 expansion task families (20K candidate).

These families are *additive* to the nine legacy L13 families. Each one opens a
contract surface the legacy builders never exercise:

  x01 pattern/format param repair       (regex-constrained request parameters)
  x02 additionalProperties body repair  (undeclared body property under `false`)
  x03 nested required body repair       (depth-2 required sub-property)
  x04 response status selection         (choose status codes that accept an instance)
  x06 response header construction      (header schema, not body/param)
  x07 union response construction       (oneOf/anyOf branch discrimination)
  x08 response version migration        (response payload across pinned versions)
  x09 array parameter repair            (item/count/uniqueItems on array params)
  x11 compound param repair             (two independent faults at once)
  x12 array-constrained body repair     (array property min/max/unique/items)

No family is a renamed copy of a legacy family: each selects a different spec
site predicate and/or a different contract kind, and the verifier recomputes the
new kinds from pinned bytes.

The pin is the same as the legacy lane, but new tasks are restricted to spec
units the legacy 240 did not use (`exclude_units`), so the expansion adds new
API surfaces rather than re-instantiating legacy specs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.plugins.source.l13.apis_guru import ApisGuruSourcePlugin, load_index_units
from benchmark.plugins.task_family.l13.families import (
    ALLOWED_TOOLS, BUDGETS, build_ast, instance_rng, make_id,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot
from source_registry.fetchers import l13_apis_guru as fetcher

LANE = "L13"
SEMANTIC_PROFILE = "openapi-3.0-contract"
PINNED_REF_ROOT = "external_knowledge/pinned/apis-guru/commit-f04b8d0bcd39"
MAX_SPEC_BYTES = 300 * 1024


@dataclass(frozen=True)
class ExpFamily:
    family_id: str
    goal_kind: str
    regime: str
    complexity: str
    target_records: int
    instances_per_goal: int
    mechanism: str
    builder: str


FAMILIES: tuple[ExpFamily, ...] = (
    ExpFamily("l13-x01-pattern-parameter-repair", "param_contract", "repair", "medium",
              170, 7, "regex_constraint", "generate.pattern_param"),
    ExpFamily("l13-x02-additional-property-body-repair", "body_contract", "repair", "medium",
              60, 6, "additional_properties_false", "generate.additional_property_body"),
    ExpFamily("l13-x03-nested-required-body-repair", "body_contract", "repair", "high",
              140, 6, "nested_required", "generate.nested_required_body"),
    ExpFamily("l13-x04-response-status-selection", "status_selection", "construction", "high",
              120, 5, "status_code_selection", "generate.response_status_selection"),
    ExpFamily("l13-x06-response-header-construction", "header_construction", "construction", "high",
              110, 5, "response_header_schema", "generate.response_header_construction"),
    ExpFamily("l13-x07-union-response-construction", "response_construction", "construction", "high",
              30, 5, "union_branch", "generate.union_response_construction"),
    ExpFamily("l13-x08-response-version-migration", "version_migration", "repair", "high",
              12, 6, "response_version_contrast", "generate.response_version_migration"),
    ExpFamily("l13-x09-array-parameter-repair", "param_contract", "repair", "medium",
              125, 6, "array_parameter", "generate.array_param"),
    ExpFamily("l13-x11-compound-parameter-repair", "param_contract", "repair", "high",
              60, 6, "compound_fault", "generate.compound_param"),
    ExpFamily("l13-x12-array-constrained-body-repair", "body_contract", "repair", "medium",
              83, 5, "array_body_property", "generate.array_body"),
)
FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}
TARGET_TOTAL = sum(f.target_records for f in FAMILIES)  # 910

_GOAL_TEXT = {
    "l13-x01-pattern-parameter-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): a value violates its declared regular "
        "expression (pattern) constraint. Repair the parameter set so every value matches the "
        "declared pattern. Do not change values that already conform; do not add undeclared "
        "parameters."
    ),
    "l13-x02-additional-property-body-repair": (
        "The JSON request body below violates the requestBody schema of {method} {path} in the "
        "pinned OpenAPI spec '{title}' ({api_path} {version}) because it contains a property that "
        "`additionalProperties: false` forbids. Remove only the offending property; keep every "
        "declared field unchanged."
    ),
    "l13-x03-nested-required-body-repair": (
        "The JSON request body below violates the requestBody schema of {method} {path} in the "
        "pinned OpenAPI spec '{title}' ({api_path} {version}): a nested object is missing a "
        "required property. Add the missing nested property; keep every other field unchanged."
    ),
    "l13-x04-response-status-selection": (
        "Given the response instance below and the pinned OpenAPI spec '{title}' "
        "({api_path} {version}), list exactly the response status codes of {method} {path} whose "
        "response schema accepts the instance. Return the status codes sorted."
    ),
    "l13-x06-response-header-construction": (
        "Construct a value for the response header '{header}' of the {code} response of "
        "{method} {path} in the pinned OpenAPI spec '{title}' ({api_path} {version}) that "
        "validates against the declared header schema."
    ),
    "l13-x07-union-response-construction": (
        "Construct a JSON instance that validates against the {code} response schema of "
        "{method} {path} in the pinned OpenAPI spec '{title}' ({api_path} {version}). The schema "
        "is a union (oneOf/anyOf); the instance must validate against exactly one branch."
    ),
    "l13-x08-response-version-migration": (
        "The response payload below is valid for {method} {path} under '{api_path}' version "
        "{version} but the API is being migrated to version {version_b}. Produce the migrated "
        "payload that satisfies the version {version_b} response contract, changing only what "
        "the newer contract breaks or newly requires."
    ),
    "l13-x09-array-parameter-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): an array-valued parameter violates its "
        "item/count/uniqueness constraint. Repair that parameter; keep all other values "
        "unchanged and do not add undeclared parameters."
    ),
    "l13-x11-compound-parameter-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}) in two independent ways: a required "
        "parameter is missing and another value is invalid. Repair both faults; keep every "
        "other value unchanged and do not add undeclared parameters."
    ),
    "l13-x12-array-constrained-body-repair": (
        "The JSON request body below violates the requestBody schema of {method} {path} in the "
        "pinned OpenAPI spec '{title}' ({api_path} {version}): an array property violates its "
        "minItems/maxItems/uniqueItems/item constraint. Repair only that property; keep every "
        "other field unchanged."
    ),
}


def eligible_units() -> list[dict[str, Any]]:
    """OpenAPI 3.0.x units under the legacy size cap, sorted by unit_id.

    Reads the persisted pinned index (no re-hashing of the 727 MB tree); each
    spec's digest is still verified lazily on first load.
    """
    out: list[dict[str, Any]] = []
    for unit in load_index_units():
        if unit.get("kind") != "openapi_spec":
            continue
        if not str(unit.get("metadata", {}).get("spec_version", "")).startswith("3.0"):
            continue
        path = fetcher.PINNED_ROOT / unit["payload_ref"]
        try:
            if path.stat().st_size > MAX_SPEC_BYTES:
                continue
        except OSError:
            continue
        out.append({
            "unit_id": unit["unit_id"],
            "payload_ref": unit["payload_ref"],
            "digest": unit["digest"],
            "metadata": dict(unit.get("metadata") or {}),
        })
    out.sort(key=lambda u: u["unit_id"])
    return out


def observation_refs(ast: dict[str, Any]) -> list[str]:
    refs = [PINNED_REF_ROOT + "/" + ast["unit_id"].split("/", 1)[1]]
    if ast.get("unit_id_b"):
        refs.append(PINNED_REF_ROOT + "/" + ast["unit_id_b"].split("/", 1)[1])
    return refs


def render_user_goal(ast: dict[str, Any]) -> str:
    meta = ast["unit_meta"]
    coord = ast["coordinate"]
    meta_b = ast.get("unit_meta_b") or {}
    return _GOAL_TEXT[ast["family_id"]].format(
        method=str(coord.get("method", "")).upper(),
        path=coord.get("path", coord.get("schema", "")),
        title=meta.get("title") or meta.get("api_path"),
        api_path=meta.get("api_path"),
        provider=meta.get("provider"),
        version=meta.get("api_version"),
        version_b=meta_b.get("api_version", ""),
        api_path_b=meta_b.get("api_path", ""),
        provider_b=meta_b.get("provider", ""),
        code=coord.get("code", "200"),
        header=coord.get("header", ""),
        schema_a=coord.get("schema_a", ""),
        schema_b=coord.get("schema_b", ""),
    )


class L13ExpansionFamilies:
    """TaskFamilyPlugin for the additive 20K L13 expansion."""

    plugin_id = "l13.expansion"

    def __init__(self) -> None:
        self.source = ApisGuruSourcePlugin()
        self._units_cache: list[dict[str, Any]] | None = None

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def units(self, snapshot: SourceSnapshot | None = None) -> list[dict[str, Any]]:
        if self._units_cache is None:
            self._units_cache = eligible_units()
        return self._units_cache

    def propose(self, snapshot: SourceSnapshot | None, *, family: str, seed: int,
                target_records: int, exclude_units: set[str] | None = None):
        from benchmark.plugins.task_family.l13_exp import generate

        spec = FAMILY_BY_ID[family]
        builder = getattr(generate, spec.builder.split(".", 1)[1])
        yield from builder(self, spec, seed=seed, target_records=target_records,
                           exclude_units=exclude_units or set())

    def ground(self, candidate: CandidateTaskAST):
        ast = candidate.ast
        oracle_ref = "l13x-oracle-%s" % make_id("o", candidate.instance_id, ast["unit_id"],
                                                ast["goal_kind"])[2:]
        public: dict[str, Any] = {
            "user_goal": render_user_goal(ast),
            "spec": ast["unit_meta"] | {"unit_id": ast["unit_id"], "digest": ast["unit_digest"]},
            "coordinate": ast["coordinate"],
            "contract": ast["contract"],
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": dict(BUDGETS[ast["complexity"]]),
            "observation_refs": observation_refs(ast),
        }
        if ast.get("unit_id_b"):
            public["spec_b"] = ast["unit_meta_b"] | {
                "unit_id": ast["unit_id_b"], "digest": ast["unit_digest_b"]}
        if ast.get("contract_b") is not None:
            public["contract_b"] = ast["contract_b"]
        if ast.get("initial") is not None:
            public["initial"] = ast["initial"]
        if ast.get("constraints"):
            public["constraints"] = ast["constraints"]
        from benchmark.schema.plugin_protocols import GroundedTask
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={})


def emit(family: ExpFamily, unit: dict[str, Any], coordinate: dict[str, Any],
         contract: dict[str, Any], variants: list[dict[str, Any]], seed: int,
         *, unit_b: dict[str, Any] | None = None,
         contract_b: dict[str, Any] | None = None,
         base_goal: Any = None):
    """Yield one CandidateTaskAST per variant dict.

    Variant keys: initial, reference, guards, constraints, witness.
    """
    base_goal_id = base_goal if base_goal is not None else make_id(
        "bg", family.family_id, unit["unit_id"], coordinate)
    unit_ids = (unit["unit_id"],) if unit_b is None else (unit["unit_id"], unit_b["unit_id"])
    for variant, v in enumerate(variants):
        instance_id = make_id("in", base_goal_id, variant, seed)
        ast = build_ast(family, unit=unit, coordinate=coordinate, contract=contract,
                        initial=v.get("initial"), reference=v["reference"],
                        guards=v.get("guards") or [], constraints=v.get("constraints") or {},
                        witness=v["witness"], variant=variant,
                        unit_b=unit_b, contract_b=contract_b)
        yield CandidateTaskAST(
            lane_id=LANE, family_id=family.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=family.regime,
            semantic_profile=SEMANTIC_PROFILE, ast=ast, source_unit_ids=unit_ids)


__all__ = [
    "FAMILIES", "FAMILY_BY_ID", "TARGET_TOTAL", "ExpFamily",
    "L13ExpansionFamilies", "eligible_units", "observation_refs",
    "render_user_goal", "emit", "instance_rng", "make_id",
]