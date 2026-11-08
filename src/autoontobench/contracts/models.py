"""AutoOntologyBench Review-5 canonical contracts (frozen interface; owner: orchestrator).

Schema version: aob-task/5.0 (Implementation §3 canonical task schema; Shared Contract §D/F).

Enforced rules:
- `private_*` fields must never leave the private root. Any agent-visible serialization must go
  through `public_projection()`; `assert_no_private_keys()` is the recursive leak guard.
- Canonical goal identity for dedup = `canonical_sha()` over a semantics-only payload
  (paraphrases must NOT change it; see Implementation §4 T4 dedup rule).
- Run rows use None for "not measured"; PENDING/BLOCKED/NOT_APPLICABLE and true 0 are distinct.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_TASK_V5 = "aob-task/5.0"


class TaskFamily(str, Enum):
    repair = "repair"
    organization = "organization"
    construction = "construction"
    application = "application"


class TaskTrack(str, Enum):
    stateful_action = "stateful_action"   # Gatekeeper / CEL / DMN / BDDL-style
    analytical = "analytical"             # RDKit / Rhea+ChEBI / GO query-set style
    logical_ontology = "logical_ontology" # GO entailment
    numeric = "numeric"                   # SBML
    clinical = "clinical"                 # CQL / WHO


class TaskStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    NATIVE_VERIFIED = "NATIVE_VERIFIED"
    TASK_FEASIBLE = "TASK_FEASIBLE"
    SEMANTICALLY_GROUNDED = "SEMANTICALLY_GROUNDED"
    LEAK_FREE = "LEAK_FREE"
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class Provenance(str, Enum):
    source_exact = "source_exact"
    source_parameterized = "source_parameterized"
    source_derived_composition = "source_derived_composition"
    counterfactual_derived = "counterfactual_derived"


class Authorship(str, Enum):
    deterministic_template = "deterministic_template"
    dashscope_paraphrase = "dashscope_paraphrase"
    dashscope_candidate_validated = "dashscope_candidate_validated"


class SourceRef(BaseModel):
    source_id: str
    source_rule_unit_id: str
    commit: str
    file_sha256: str


class Budgets(BaseModel):
    oracle_probes: int = 8
    total_tool_calls: int = 32
    max_actions: int = 12
    max_output_tokens: int = 12000


class QC(BaseModel):
    native_pass: bool | None = None
    feasible: bool | None = None
    nontrivial: bool | None = None
    agent_view_clean: bool | None = None


class TaskSpecV5(BaseModel):
    schema_version: Literal["aob-task/5.0"] = SCHEMA_TASK_V5
    task_id: str
    domain: str
    track: TaskTrack
    task_family: TaskFamily
    task_regime: str | None = None
    source_refs: list[SourceRef]
    source_group_id: str
    mechanism_group_id: str
    combination_signature: str | None = None
    user_goal: str
    initial_state_ref: str
    allowed_tools: list[str]
    budgets: Budgets = Field(default_factory=Budgets)
    public_evidence_ref: str
    ontology_contract: str  # named profile, e.g. "shacl-rule/1"
    private_goal_verifier_ref: str
    private_native_truth_ref: str
    private_gold_artifact_ref: str
    private_control_specs_ref: str
    task_provenance: Provenance
    task_authorship: Authorship
    split: str
    independent_native_checks: list[str]
    status: TaskStatus = TaskStatus.CANDIDATE
    qc: QC = Field(default_factory=QC)
    seed: int = 0


PRIVATE_PREFIX = "private_"


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not k.startswith(PRIVATE_PREFIX)}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def public_projection(spec: TaskSpecV5) -> dict[str, Any]:
    """Agent-visible projection: recursively removes private_* keys."""
    return _strip_private(spec.model_dump(mode="json"))


def assert_no_private_keys(payload: Any) -> None:
    """Recursive leak guard; raises ValueError on any surviving `private_*` key."""

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k.startswith(PRIVATE_PREFIX):
                    raise ValueError(f"private field leaked at {path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(payload, "$")


def canonical_sha(payload: dict[str, Any]) -> str:
    """Deterministic semantics hash (dedup key; paraphrases must not change it)."""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class WordingCandidate(BaseModel):
    """DashScope wording output (Implementation §4 T4). NEVER contains gold/rule text."""

    task_id: str
    user_goal_en: str
    user_goal_zh: str
    plausible_reason: str | None = None
    named_entities_in_public_state: list[str] = Field(default_factory=list)
    required_actions_hint_without_solution: str | None = None
    surface_style: str | None = None
    seed: int = 0
    author_model: str
    prompt_hash: str


class GoalResult(BaseModel):
    goal_reached: bool
    evidence: dict[str, Any] = Field(default_factory=dict)


class RunRowV5(BaseModel):
    """Run-row skeleton (Implementation §11.6). None = not measured (never 0, never fake SIM)."""

    run_id: str
    task_id: str
    base_goal_id: str | None = None
    source_rule_units: list[str] = Field(default_factory=list)
    independent_source_group: str | None = None
    domain: str | None = None
    split: str | None = None
    arm: str | None = None
    model_exact_id: str | None = None
    provider: str = "dashscope"
    launched: bool = False
    status: str | None = None  # SCORED | PENDING | BLOCKED | NOT_APPLICABLE | ...
    sealed_ontology_sha: str | None = None
    task_success: bool | None = None
    valid_task_success: bool | None = None
    unsafe_action_count: int | None = None
    native_terminal_verifier_sha: str | None = None
    native_terminal_evidence_sha: str | None = None
    initial_state_sha: str | None = None
    terminal_state_sha: str | None = None
    tool_calls: int | None = None
    probe_cost: int | None = None
    api_calls: int | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    wall_seconds: float | None = None
    ontology_build_cost: float | None = None
    ontology_reuse_count: int | None = None
    source_seed: int | None = None
    repeat_seed: int | None = None
    fail_reason: str | None = None