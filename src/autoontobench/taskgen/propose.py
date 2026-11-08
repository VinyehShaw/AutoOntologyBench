"""T4 Mode A: DashScope bilingual wording proposal from a VERIFIED blueprint.

DashScope is a proposer/paraphraser, never a source of truth (Shared Contract
section D.5). Guards enforced here:
- blueprint is checked with assert_no_private_keys BEFORE anything is sent;
- the model only sees public schema/entities/tools from the blueprint;
- each returned candidate is validated against WordingCandidate AND against the
  blueprint's public entity list (invented entities -> WordingValidationError);
- response payload is scanned for private_* keys.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import (
    TaskSpecV5, WordingCandidate, assert_no_private_keys, public_projection,
)
from autoontobench.models.dashscope import CallRequest, CallParams, DashScopeClient, run_batch
from autoontobench.taskgen.bind import bind_blueprint
from autoontobench.taskgen.dedupe import DedupeResult, dedupe_by_goal

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You write realistic user requests for a benchmark. You are NOT a source of truth. "
    "Never invent new entities, identifiers, policies, or constraints. The supplied "
    "verified blueprint defines exactly what exists. Produce JSON only. Do not reveal "
    "solutions or any hidden evaluation data."
)

USER_TEMPLATE = (
    "Given PUBLIC entities/tools and a VERIFIED task-goal schema, write one clear user "
    "goal that requires accomplishing the objective, not answering what the rule is. "
    "When the blueprint carries a user_goal_template field, paraphrase it faithfully: "
    "preserve every required target and restriction verbatim, but never state hidden "
    "required values or solution steps. Vary wording and user "
    "context without changing semantics. Do not mention RDF/ontology unless this task "
    "explicitly asks an operator to maintain it.\n"
    "Return JSON with keys: task_id, user_goal_en, user_goal_zh, "
    "named_entities_in_public_state (list), rephrased_constraints (list), "
    "plausible_reason, surface_style, reasoning_note_for_internal_qc.\n"
    "Requested surface_style for this candidate: {style_hint}\n"
    "BLUEPRINT (JSON):\n{blueprint_json}"
)

STYLE_HINTS = (
    "concise operator request",
    "detailed SRE ticket",
    "terse CLI-style instruction",
    "compliance audit phrasing",
    "incident-response phrasing",
    "junior engineer asking for help",
    "managerial request with deadline framing",
    "checklist-driven phrasing",
    "post-incident remediation task",
    "routine change-request phrasing",
)


class WordingValidationError(Exception):
    pass


def strip_private_keys(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {k: strip_private_keys(v) for k, v in payload.items()
                if not str(k).startswith("private_")}
    if isinstance(payload, list):
        return [strip_private_keys(v) for v in payload]
    return payload


def build_mode_a_messages(blueprint: Mapping[str, Any], *, style_hint: str) -> list[dict[str, str]]:
    """Prompt contains ONLY the blueprint's public projection (leak-guarded)."""
    assert_no_private_keys(dict(blueprint))
    public_view = strip_private_keys(blueprint)
    user = (USER_TEMPLATE
            .replace("{style_hint}", style_hint)
            .replace("{blueprint_json}", json.dumps(public_view, ensure_ascii=False, indent=1)))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def validate_wording_candidate(raw: Mapping[str, Any], blueprint: Mapping[str, Any],
                               *, author_model: str, prompt_hash: str) -> WordingCandidate:
    try:
        assert_no_private_keys(dict(raw))
    except ValueError as exc:
        raise WordingValidationError(f"candidate leaks hidden key: {exc}") from exc
    allowed_entities = {str(e) for e in blueprint.get("public_entities", [])}
    named = {str(e) for e in raw.get("named_entities_in_public_state", [])}
    invented = named - allowed_entities
    if invented:
        raise WordingValidationError(
            f"candidate names entities absent from public blueprint: {sorted(invented)}")
    merged = {**dict(raw), "author_model": author_model, "prompt_hash": prompt_hash}
    bound_task_id = blueprint.get("task_id") or blueprint.get("blueprint_id")
    if bound_task_id is not None:
        # Binding is fixed by construction (one batch per blueprint); echo is advisory.
        merged["task_id"] = str(bound_task_id)
    try:
        return WordingCandidate.model_validate(merged)
    except ValueError as exc:
        raise WordingValidationError(f"schema violation: {exc}") from exc


@dataclass(frozen=True)
class ProposeResult:
    candidates: list[WordingCandidate] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    dedupe: DedupeResult | None = None
    n_api_calls: int = 0
    usage_totals: dict[str, int] = field(default_factory=dict)

    @property
    def n_semantically_distinct(self) -> int:
        return self.dedupe.n_unique_goals if self.dedupe else 0


def candidate_semantics(cand: WordingCandidate, blueprint: Mapping[str, Any]) -> dict[str, Any]:
    """The blueprint owns semantics; wording never changes the goal identity."""
    return {k: blueprint.get(k) for k in
            ("blueprint_id", "domain", "track", "goal_type", "public_entities", "constraints")}


def is_real_blueprint(blueprint: Mapping[str, Any]) -> bool:
    return isinstance(blueprint.get("spec"), Mapping) and not blueprint.get("mock")


def public_entities_from_state(spec: TaskSpecV5, repo_root: Path) -> list[str]:
    """Public entity ids = object names in the public initial-state file."""
    state_path = repo_root / spec.initial_state_ref
    data = json.loads(state_path.read_text(encoding="utf-8"))
    objects = data.get("objects") if isinstance(data, Mapping) else None
    if not isinstance(objects, Mapping) or not objects:
        raise WordingValidationError(f"initial state {state_path} has no public objects")
    return sorted(str(k) for k in objects)


def public_wording_input(blueprint: Mapping[str, Any],
                         *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """PUBLIC Mode-A input from a real (W2-style) blueprint.

    The spec block is schema-bound via bind_blueprint, projected through
    public_projection() (contract: private_* stripped), and leak-guarded again
    before returning. The blueprint's private goal/plan/replay/search blocks are
    never read into the payload; public entities come only from the public
    initial-state file referenced by the spec.
    """
    if not is_real_blueprint(blueprint):
        raise WordingValidationError("not a real blueprint (missing 'spec' block or mock)")
    spec_dict = blueprint["spec"]
    spec = bind_blueprint(spec_dict, seed=int(spec_dict.get("seed", 0) or 0))
    projection = public_projection(spec)
    assert_no_private_keys(projection)
    payload: dict[str, Any] = {
        "blueprint_id": str(blueprint.get("blueprint_id", spec.task_id)),
        "task_id": spec.task_id,
        "domain": spec.domain,
        "track": spec.track.value,
        "task_family": spec.task_family.value,
        "goal_type": f"{spec.track.value}:{spec.task_family.value}",
        "user_goal_template": spec.user_goal,
        "public_entities": public_entities_from_state(spec, repo_root),
        "public_tools": list(spec.allowed_tools),
        "constraints": [
            "every guarded resource must remain byte-identical",
            "use only the listed tools",
        ],
        "budgets": projection["budgets"],
        "seed": spec.seed,
    }
    assert_no_private_keys(payload)
    return payload


async def propose_wording_candidates(
    client: DashScopeClient,
    blueprint: Mapping[str, Any],
    *,
    n: int,
    model: str,
    seed: int = 0,
    workers: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> ProposeResult:
    blueprint_id = str(blueprint.get("blueprint_id", "unknown-blueprint"))
    requests: list[CallRequest] = []
    prompt_hashes: list[str] = []
    for i in range(n):
        messages = build_mode_a_messages(blueprint,
                                         style_hint=STYLE_HINTS[i % len(STYLE_HINTS)])
        req = CallRequest(
            model=model, messages=messages,
            params=CallParams(temperature=temperature, max_tokens=max_tokens,
                              json_mode=True, seed=seed + i),
            custom_id=f"modeA-{blueprint_id}-{i:03d}")
        requests.append(req)
        prompt_hashes.append(req.prompt_sha)

    results = await run_batch(client, requests, workers=workers)

    candidates: list[WordingCandidate] = []
    rejected: list[dict[str, Any]] = []
    usage = {"input": 0, "output": 0, "reasoning": 0}
    for i, res in enumerate(results):
        for field_name in usage:
            usage[field_name] += getattr(res.usage, field_name) or 0
        if not res.ok or res.content is None:
            rejected.append({"index": i, "stage": "api_call", "reason": res.error})
            continue
        try:
            raw = json.loads(res.content)
        except json.JSONDecodeError as exc:
            rejected.append({"index": i, "stage": "json_parse", "reason": str(exc)[:200]})
            continue
        try:
            cand = validate_wording_candidate(raw, blueprint, author_model=res.model_exact_id,
                                              prompt_hash=prompt_hashes[i])
        except WordingValidationError as exc:
            rejected.append({"index": i, "stage": "validate", "reason": str(exc)[:300]})
            continue
        candidates.append(cand.model_copy(update={"seed": seed + i}))

    dedupe = dedupe_by_goal(candidates, lambda c: candidate_semantics(c, blueprint))
    return ProposeResult(candidates=candidates, rejected=rejected, dedupe=dedupe,
                         n_api_calls=len(results), usage_totals=usage)
