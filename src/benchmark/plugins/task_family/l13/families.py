"""L13 task families: 9 families over pinned OpenAPI 3.0.x contracts.

Family registry + the TaskFamilyPlugin entry points. Per-regime candidate
builders live in ``gen_repair.py`` (f01/f02/f03/f07), ``gen_construction.py``
(f04/f08/f09) and ``gen_cross.py`` (f05/f06). All generation is deterministic:
sites are found by sorted scans of the pinned snapshot; instances are seeded
via sha256-derived RNGs. No LLM is involved anywhere in candidate or gold
construction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

from benchmark.plugins.source.l13 import oas
from benchmark.plugins.source.l13.apis_guru import ApisGuruSourcePlugin
from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)

LANE = "l13"
SEMANTIC_PROFILE = "openapi-3.0-contract"
ALLOWED_TOOLS = ["openapi_spec_inspect", "contract_validate"]
BUDGETS = {"low": {"probes": 4, "actions": 6},
           "medium": {"probes": 6, "actions": 10},
           "high": {"probes": 8, "actions": 14}}


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    goal_kind: str
    regime: str
    complexity: str
    base_goals: int
    builder: str  # dotted builder name in the per-regime module


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("l13-f01-required-parameter-repair", "param_contract", "repair", "low", 5, "repair.required_param"),
    FamilySpec("l13-f02-parameter-type-repair", "param_contract", "repair", "medium", 5, "repair.type_param"),
    FamilySpec("l13-f03-request-body-repair", "body_contract", "repair", "high", 5, "repair.body"),
    FamilySpec("l13-f04-response-instance-construction", "response_construction", "construction", "high", 4, "construction.response_instance"),
    FamilySpec("l13-f05-api-version-migration", "version_migration", "repair", "high", 4, "cross.version_migration"),
    FamilySpec("l13-f06-cross-api-schema-mapping", "schema_mapping", "organization", "high", 4, "mapping.schema_mapping"),
    FamilySpec("l13-f07-enum-constraint-repair", "param_contract", "repair", "low", 5, "repair.enum_param"),
    FamilySpec("l13-f08-operation-selection", "operation_selection", "construction", "medium", 4, "construction.operation_selection"),
    FamilySpec("l13-f09-security-requirement-construction", "security_requirement", "construction", "medium", 4, "construction.security_requirement"),
)
FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

INSTANCES_PER_GOAL = 6


def make_id(prefix: str, *parts: Any) -> str:
    return "%s-%s" % (prefix, oas.sha256_json([str(p) for p in parts])[:20])


def instance_rng(seed: int, instance_id: str) -> random.Random:
    return random.Random(int(oas.sha256_json([seed, instance_id])[:16], 16))


def build_ast(family: FamilySpec, *, unit: dict[str, Any], coordinate: dict[str, Any],
              contract: dict[str, Any], initial: Any, reference: Any,
              guards: list[dict[str, Any]], constraints: dict[str, Any],
              witness: dict[str, Any], variant: int,
              unit_b: dict[str, Any] | None = None,
              contract_b: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "goal_kind": family.goal_kind,
        "family_id": family.family_id,
        "regime": family.regime,
        "complexity": family.complexity,
        "unit_id": unit["unit_id"],
        "unit_digest": unit["digest"],
        "unit_meta": {k: unit["metadata"].get(k) for k in ("provider", "api_path", "api_version", "title", "license_name")},
        "unit_id_b": unit_b["unit_id"] if unit_b else None,
        "unit_digest_b": unit_b["digest"] if unit_b else None,
        "unit_meta_b": ({k: unit_b["metadata"].get(k) for k in ("provider", "api_path", "api_version", "title", "license_name")}
                        if unit_b else None),
        "coordinate": coordinate,
        "contract": contract,
        "contract_b": contract_b,
        "initial": initial,
        "reference": reference,
        "guards": guards,
        "constraints": constraints,
        "witness": witness,
        "variant": variant,
    }


class L13TaskFamilies:
    """TaskFamilyPlugin for lane L13 (OpenAPI contract tasks)."""

    plugin_id = "l13.families"

    def __init__(self) -> None:
        self.source = ApisGuruSourcePlugin()
        self._units_cache: list[dict[str, Any]] | None = None

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def units(self, snapshot: SourceSnapshot) -> list[dict[str, Any]]:
        if self._units_cache is None:
            self._units_cache = self.source.task_units(snapshot)
        return self._units_cache

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int, constraints: dict[str, Any]
    ) -> Iterator[CandidateTaskAST]:
        spec = FAMILY_BY_ID[family]
        from benchmark.plugins.task_family.l13 import (
            gen_construction, gen_cross, gen_mapping, gen_repair,
        )
        modules = {"repair": gen_repair, "construction": gen_construction,
                   "cross": gen_cross, "mapping": gen_mapping}
        builder_name = spec.builder.split(".", 1)[1]
        builder = getattr(modules[spec.builder.split(".", 1)[0]], builder_name)
        instances_per_goal = int(constraints.get("instances_per_goal", INSTANCES_PER_GOAL))
        yield from builder(self, snapshot, spec, seed=seed, instances_per_goal=instances_per_goal)

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        oracle_ref = "l13-oracle-%s" % oas.sha256_json(
            [candidate.instance_id, ast["unit_id"], ast["goal_kind"]]
        )[:16]
        public = {
            "user_goal": render_user_goal(ast),
            "spec": ast["unit_meta"] | {"unit_id": ast["unit_id"], "digest": ast["unit_digest"]},
            "coordinate": ast["coordinate"],
            "contract": ast["contract"],
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": dict(BUDGETS[ast["complexity"]]),
            "observation_refs": observation_refs(ast),
        }
        if ast.get("unit_id_b"):
            public["spec_b"] = ast["unit_meta_b"] | {"unit_id": ast["unit_id_b"], "digest": ast["unit_digest_b"]}
        if ast.get("contract_b") is not None:
            public["contract_b"] = ast["contract_b"]
        if ast.get("initial") is not None:
            public["initial"] = ast["initial"]
        if ast.get("constraints"):
            public["constraints"] = ast["constraints"]
        if ast["goal_kind"] == "security_requirement":
            public["contract"] = {"kind": ast["contract"]["kind"],
                                  "security_schemes": ast["contract"]["security_schemes"]}
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref, native_evidence={})


_GOAL_TEXT = {
    "l13-f01-required-parameter-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): required parameters are missing. Supply a "
        "corrected parameter set that satisfies every declared parameter constraint. Do not change "
        "values that already conform; do not add undeclared parameters."
    ),
    "l13-f02-parameter-type-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): one value violates its declared type/bound "
        "constraint. Repair the parameter set so every value validates against the declared schema. "
        "Do not change conforming values; do not add undeclared parameters."
    ),
    "l13-f03-request-body-repair": (
        "The JSON request body below violates the requestBody schema of {method} {path} in the "
        "pinned OpenAPI spec '{title}' ({api_path} {version}). Repair the body so it validates; "
        "keep every field that already conforms unchanged."
    ),
    "l13-f04-response-instance-construction": (
        "Construct a JSON instance that is valid against the {code} response schema of "
        "{method} {path} in the pinned OpenAPI spec '{title}' ({api_path} {version}), and that "
        "additionally satisfies the stated extra constraints."
    ),
    "l13-f05-api-version-migration": (
        "The payload below is valid for {method} {path} under '{api_path}' version {version} but "
        "the API is being migrated to version {version_b}. Produce the migrated payload that "
        "satisfies the version {version_b} contract, changing only what the newer contract breaks "
        "or newly requires."
    ),
    "l13-f06-cross-api-schema-mapping": (
        "Align the source instance of schema {schema_a} from '{api_path}' ({provider}) into schema "
        "{schema_b} from '{api_path_b}' ({provider_b}) using the given field mapping. The result "
        "must validate against the target schema and contain exactly the mapped fields."
    ),
    "l13-f07-enum-constraint-repair": (
        "The request parameter set below fails the contract of {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): a value is outside its declared enum. "
        "Repair it by choosing a declared enum value; keep all other values unchanged and do not "
        "add undeclared parameters."
    ),
    "l13-f08-operation-selection": (
        "List exactly the operations of the pinned OpenAPI spec '{title}' ({api_path} {version}) "
        "that satisfy ALL of the stated predicate constraints. Return the operation keys sorted."
    ),
    "l13-f09-security-requirement-construction": (
        "Enumerate every credential alternative that satisfies {method} {path} in the pinned "
        "OpenAPI spec '{title}' ({api_path} {version}): each alternative is the set of security "
        "schemes (with their required scopes) that together authorize the call."
    ),
}


PINNED_REF_ROOT = "external_knowledge/pinned/apis-guru/commit-f04b8d0bcd39"


def observation_refs(ast: dict[str, Any]) -> list[str]:
    """Agent-visible pointers into the pinned snapshot for the task's spec unit(s).

    Repo-relative paths only (no absolute host paths). The referenced
    definition documents are the public evidence the solver may inspect; they
    do not contain the reference answer, which is derived from them at
    verification time.
    """
    refs = [PINNED_REF_ROOT + "/" + ast["unit_id"].split("/", 1)[1]]
    if ast.get("unit_id_b"):
        refs.append(PINNED_REF_ROOT + "/" + ast["unit_id_b"].split("/", 1)[1])
    return refs


def render_user_goal(ast: dict[str, Any]) -> str:
    meta = ast["unit_meta"]
    coord = ast["coordinate"]
    meta_b = ast.get("unit_meta_b") or {}
    text = _GOAL_TEXT[ast["family_id"]].format(
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
        schema_a=coord.get("schema_a", ""),
        schema_b=coord.get("schema_b", ""),
    )
    return text
