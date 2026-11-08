"""Real-model agent adapter for the Review-5B harness (owner: W11 / G2 dev pilot).

``DashScopeAgentModel`` implements the same duck-typed ``AgentModel`` surface as
``MockModel``: ``next_action(observation) -> {"tool": <name>, "args": {...}}``.
It is the FIRST real-model consumer of the frozen harness loop
(``runners/harness.py``); G0 is signed closed and the M1/M2 evidence runs stay
on the scripted MockModel (which remains the default for tests).

Design contract (binding for the G2 dev pilot):

- **All calls route through ``DashScopeClient``** (AIMD limiter, 429/5xx
  exponential backoff with jitter, idempotent request ids, sanitized append-only
  api_logs). The API key comes from the environment only and is never written
  to any file or log line.
- **System prompt** = the qualified task's PUBLIC projection
  (``public_projection(spec)``; ``private_*`` stripped and asserted) + the
  11-tool catalog with per-arm availability + the strict JSON tool-call
  protocol. ``assert_no_private_keys`` guards both the construction-time spec
  and every observation before it leaves the machine.
- **Observation mapping**: each harness observation becomes one user message;
  the model's raw replies become assistant messages. W12: the harness echoes
  EVERY executed tool result on the next turn (``last_tool_result`` =
  ``{"tool", "ok", "error", "charge", "payload"}``); ``last_feedback`` remains
  as the dedicated perform_action shortcut.
- **Strict parse with a documented cap**: a reply must be a JSON object
  ``{"tool": <non-empty str>, "args": <object>}``. Malformed replies are
  retried in-loop up to ``config.max_parse_retries`` (default 2) with an
  explicit PROTOCOL_ERROR note; persistent failure degrades to a
  ``submit_final`` carrying ``MODEL_PARSE_FAILURE`` (the run still seals and is
  scored honestly; the launched-denominator rule of preregistration applies).
- **Provider-error policy**: the client already retries 429/5xx with backoff;
  if a turn still errors, the adapter retries the turn up to
  ``config.max_api_retries`` (default 1) and then degrades to
  ``submit_final(MODEL_FAILURE)``. The model id is fixed at construction and
  NEVER silently substituted; ``model_exact_id`` is refreshed from each
  response and recorded.
- **Loop fuses**: ``max_turns`` (default = the run's total_tool_calls budget)
  forces a final submit so the harness loop always terminates;
  ``max_calls_per_run`` caps total API spend per run.
- **Action-shape guard** (W11 pilot finding): the frozen GatekeeperDomain
  raises uncaught ``KeyError``/``IndexError`` on actions whose ``set``/``remove``
  paths walk into missing intermediates, or on ``create_object`` without
  ``value`` — one such action kills the entire run mid-flight. The adapter
  refuses exactly those shapes client-side (``check_action_shape``), tells the
  model why, and re-asks within the same documented per-turn retry cap;
  refusals are counted (``guard_retries``) and visible in the saved transcript.
  Legality decisions remain entirely with the native oracle — the guard only
  blocks shapes the domain cannot even evaluate.
- **Accounting**: per-call records (custom_id, prompt_sha, response_id,
  model_exact_id, usage incl. reasoning tokens, latency, attempts, error) are
  kept on ``.records`` for the run-dir ``model_log.jsonl``; cumulative
  ``api_calls`` / ``tokens_input`` / ``tokens_output`` / ``tokens_reasoning``
  feed the RunRowV5 token columns.

Sync/async: the harness loop is synchronous while ``DashScopeClient`` is
async. The adapter drives a single dedicated event loop on a daemon thread
(``_LoopThread``); the client's loop-bound locks (logger/limiter) live entirely
on that loop. One adapter = one run; do not share an adapter across runs.

allow: SIZE_OK — W11 owns exactly one new runners file; this module is one
cohesive unit (adapter + its prompt data + its crash guard) and ~40% of its
bulk is prompt/catalog text.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from autoontobench.contracts.models import assert_no_private_keys
from autoontobench.models.dashscope import CallParams, CallRequest, CallResult, DashScopeClient
from autoontobench.runners.tool_api import TOOL_NAMES

#: Tool catalog rendered into the system prompt (all 11 tools; per-arm
#: availability is marked from the ArmConfig allowlist). Descriptions mirror
#: runners/tool_api.py — harness documentation, never task gold.
TOOL_CATALOG: dict[str, str] = {
    "inspect_schema": "inspect_schema {} -> the public task spec (goal text, budgets, allowed tools)",
    "inspect_state": "inspect_state {} -> the current public state (also included in every observation)",
    "read_public_doc": 'read_public_doc {"name": "<relative path>"} -> read a public document',
    "probe_native": "probe_native {" + '"action": ACTION' + "} -> ask the native oracle whether ACTION is legal "
    "(costs 1 oracle probe; the verdict arrives as next turn's last_tool_result.payload)",
    "edit_artifact": "edit_artifact {" + '"files": {"<name>": "<text>", ...}' + "} -> replace artifact files "
    "(artifact arms only; editable names: ontology.ttl, shapes.ttl, mappings.json, phi.json, manifest.json)",
    "validate_artifact": "validate_artifact {} -> syntax-validate the working artifact (artifact arms only)",
    "check_action_with_ontology": "check_action_with_ontology {" + '"action": ACTION' + "} -> the working "
    "artifact's opinion on ACTION (artifact arms only; an opinion, never native truth)",
    "perform_action": "perform_action {" + '"action": ACTION' + "} -> apply ACTION to the native state "
    "(costs 1 action; its feedback IS echoed as last_feedback; a natively-illegal action counts as unsafe)",
    "get_feedback": "get_feedback {} -> last perform_action feedback + remaining budgets + unsafe count",
    "seal_artifact": "seal_artifact {} -> seal the working artifact snapshot (artifact arms only; single shot)",
    "submit_final": "submit_final {" + '"answer": {...}' + "} -> TERMINAL: submit the final answer and end the run",
}

#: The W6 action envelope (runners/policy allowlist keys on "kind"; the
#: gatekeeper domain keys on "op" — bridged actions carry both, identical).
ACTION_SCHEMA_DOC = (
    'ACTION = {"op": "patch_object"|"create_object"|"delete_object", '
    '"kind": <same value as "op">, "object": "<kind>/<name>" e.g. "pod/web-1", '
    '"idempotence_key": "<unique string>", '
    '"set": {"<dotted.path>": <value>, ...} (patch/create), '
    '"remove": ["<dotted.path>", ...] (optional, patch)}'
)

_PROTOCOL = """\
You are an autonomous researcher agent operating a frozen tool API for exactly one benchmark task.

# Action protocol (STRICT)
Every reply MUST be exactly one JSON object: {"tool": "<tool name>", "args": {...}}.
No prose, no markdown fences, no extra keys. Any other shape is a protocol error and is retried up to a cap; persistent protocol errors terminate the run as MODEL_PARSE_FAILURE.

# Observation channel (read carefully)
Each turn you receive one JSON observation with:
- "observation": the current PUBLIC state (post all applied actions),
- "budgets_remaining": {"tool_calls", "oracle_probes", "actions"},
- "last_feedback": the result of your most recent perform_action ({"applied": bool, "legal": bool|null, "detail": ...}),
- "last_tool_result": the FULL result of your most recent executed tool call ({"tool", "ok", "error", "charge", "payload"}) — every tool's result (inspect_*, probe_native, edit/validate/check/seal_artifact, read_public_doc, get_feedback, perform_action) is echoed here on the turn after it executes; null before your first call,
- "submitted": whether the run has ended.

# Task action schema
%s

# Strategy notes
- The goal is verified at the END by an independent native verifier against exact target values; you get NO goal-progress signal during the run, so commit to your best-supported plan.
- A natively-illegal perform_action increments unsafe_action_count and voids run validity; prefer actions you are confident are legal.
- When you believe the state satisfies the user goal, call submit_final with a short summary answer.
- Do not mutate objects the goal does not concern.\
"""

_ONTOLOGY_ARM_NOTE = """\
# Artifact arm (shacl-rule/1, SELF_BUILT)
You additionally own a working ontology artifact that starts as a STUB. You may edit it (edit_artifact), syntax-validate it (validate_artifact), ask its opinion on candidate actions (check_action_with_ontology), and must seal it before finishing (seal_artifact). Construction cost (tool calls) is charged to this run. The bundle carries files ontology.ttl / shapes.ttl / mappings.json / phi.json / manifest.json; the check path serializes a proposed patch action into flattened fields under base urn:aob:gk# (e.g. gk:set_spec_containers_0_image, gk:object, gk:idempotence_key) and the public state into per-object records (type/name/namespace/label_<k>/image_<i>/replicas/service_type). The manifest's contract_hash is recomputed harness-side; never hand-compute hashes.\
"""


@dataclass(frozen=True)
class RealModelConfig:
    """Tunables for one real-model run. Defaults bound the dev pilot's spend."""

    model: str = "qwen3.8-max"
    max_tokens: int = 4096  # reasoning model; prereg requires >=2048
    temperature: float = 0.0
    seed: int | None = None
    max_parse_retries: int = 2  # malformed-envelope retries per turn (documented cap)
    max_api_retries: int = 1  # extra turn-level retries after client-side backoff
    max_turns: int = 32  # forced submit beyond this (harness-loop fuse)
    max_calls_per_run: int = 40  # hard API-spend fuse per run
    request_timeout_s: float = 900.0  # per-call fuse; qwen3.8-max reasoning calls run minutes under load


@dataclass(frozen=True)
class AgentPromptContext:
    """Everything the system prompt is built from (public-only, asserted).

    ``arm_note`` (W13): per-arm construction note; when None and the arm
    carries artifact tools (``seal_artifact`` in the allowlist), the default
    SELF_BUILT ontology note is used. ``retrieval_digest`` (W13 rag arm):
    the deterministic retriever's top-k chunk listing for the task goal.
    """

    public_spec: Mapping[str, Any]
    arm_id: str
    tool_allowlist: frozenset[str]
    arm_note: str | None = None
    retrieval_digest: str | None = None

    def __post_init__(self) -> None:
        assert_no_private_keys(self.public_spec)


class _LoopThread:
    """One dedicated asyncio loop on a daemon thread (client locks stay loop-bound)."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._main, name="dashscope-agent-loop", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


class ToolCallParseError(ValueError):
    """The model reply is not the strict {"tool", "args"} envelope."""


def parse_tool_call(content: str) -> dict[str, Any]:
    """Strict envelope parse: a JSON object with non-empty str ``tool`` and
    object ``args`` (default {}). Extra keys are dropped (recorded by caller)."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError("reply is not JSON: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise ToolCallParseError("reply JSON is not an object")
    tool = payload.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ToolCallParseError("reply missing non-empty string field 'tool'")
    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise ToolCallParseError("reply field 'args' must be an object")
    return {"tool": tool, "args": args}


class ActionShapeError(ValueError):
    """The proposed action would crash the frozen domain (uncaught KeyError on
    missing intermediates / missing create value). The adapter refuses to emit
    it and re-asks; every refusal is counted and visible in the transcript."""


#: Real-object path prefix -> public-projection prefix (GatekeeperDomain.observe
#: projects a documented subset; only these subtrees are checkable client-side).
_PROJECTION_PREFIXES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("metadata", "labels"), ("labels",)),
    (("spec", "containers"), ("containers",)),
    (("spec", "initContainers"), ("initContainers",)),
    (("spec", "ephemeralContainers"), ("ephemeralContainers",)),
    (("spec", "replicas"), ("replicas",)),
    (("spec", "type"), ("service_type",)),
)

_PATH_SEGMENT = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")


def _parse_dotted(path: str) -> list[Any] | None:
    parts: list[Any] = []
    for seg in path.split("."):
        match = _PATH_SEGMENT.match(seg)
        if not match:
            return None
        parts.append(match.group(1))
        if match.group(2) is not None:
            parts.append(int(match.group(2)))
    return parts


def _walkable(container: Any, parts: list[Any]) -> bool:
    cur = container
    for part in parts:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return False
            cur = cur[part]
        else:
            if not isinstance(cur, Mapping) or part not in cur:
                return False
            cur = cur[part]
    return True


def _translate(parts: list[Any], observed_entry: Mapping[str, Any]) -> list[Any] | None:
    for real_prefix, projected_prefix in _PROJECTION_PREFIXES:
        n = len(real_prefix)
        if tuple(parts[:n]) == real_prefix:
            return list(projected_prefix) + parts[n:]
    return None


#: Real-object top-level keys (k8s-style objects in this domain family). A
#: set/remove path starting with anything else mimics the flattened PUBLIC
#: PROJECTION shape (e.g. "containers[0].image" instead of
#: "spec.containers[0].image") and crashes the frozen domain's set_path.
_REAL_TOP_LEVEL_KEYS = frozenset({"apiVersion", "kind", "metadata", "spec"})


def _check_patch_paths(observed_entry: Mapping[str, Any], paths: Any, *, full: bool) -> str | None:
    for path in paths:
        if not isinstance(path, str):
            return "set/remove path must be a string, got %r" % type(path).__name__
        parts = _parse_dotted(path)
        if parts is None:
            return "unparseable dotted path %r" % path
        if not parts or parts[0] not in _REAL_TOP_LEVEL_KEYS:
            return ("path %r is not a real object path; use full paths like "
                    "metadata.labels.<key>, spec.containers[0].image, spec.replicas, spec.type" % path)
        translated = _translate(parts, observed_entry)
        if translated is None:
            continue  # unprojected subtree: uncheckable client-side, passes through
        walk = translated if full else translated[:-1]
        if walk and not _walkable(observed_entry, walk):
            return "path %r has no existing intermediate in the current state of this object" % path
    return None


def check_action_shape(call: Mapping[str, Any], observed_objects: Mapping[str, Any]) -> str | None:
    """Client-side guard for the one crash class the frozen domain does not
    contain (W11 pilot finding). Returns a human reason, or None when safe.
    Only shapes that would raise inside ``legal_action``/``step`` are refused;
    legality itself stays with the native oracle."""
    if call.get("tool") not in ("perform_action", "probe_native"):
        return None
    args = call.get("args") or {}
    action = args.get("action")
    if not isinstance(action, Mapping):
        return "args.action must be an object"
    op = action.get("op")
    if op not in ("patch_object", "create_object", "delete_object"):
        return "action.op must be one of patch_object/create_object/delete_object"
    if action.get("kind") != op:
        return "action.kind must equal action.op (W6 envelope mirror)"
    oid = action.get("object")
    if not isinstance(oid, str) or not oid:
        return "action.object must be a non-empty string like 'pod/web-1'"
    if not action.get("idempotence_key"):
        return "action.idempotence_key is required"
    if op == "create_object":
        if oid in observed_objects:
            return "create_object target %r already exists" % oid
        if not isinstance(action.get("value"), Mapping):
            return "create_object requires an object-valued 'value' (the full resource body)"
        return None
    if oid not in observed_objects:
        return "object %r is not in the current public state" % oid
    if op == "delete_object":
        return None
    entry = observed_objects[oid]
    if not isinstance(entry, Mapping):
        return None
    set_map = action.get("set") or {}
    if not isinstance(set_map, Mapping):
        return "action.set must be an object of dotted-path -> value"
    reason = _check_patch_paths(entry, list(set_map), full=False)
    if reason:
        return reason
    remove = action.get("remove") or []
    if not isinstance(remove, list):
        return "action.remove must be a list of dotted paths"
    return _check_patch_paths(entry, remove, full=True)



class DashScopeAgentModel:
    """Duck-typed AgentModel backed by a real DashScope reasoning model."""

    provider = "dashscope"

    def __init__(
        self,
        client: DashScopeClient,
        config: RealModelConfig,
        context: AgentPromptContext,
    ) -> None:
        self._client = client
        self._config = config
        self._context = context
        self.model_exact_id = config.model
        self._loop = _LoopThread()
        self._closed = False
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        self._turn = 0
        self._submitted = False
        self._observed_objects: Mapping[str, Any] = {}
        # accounting (read by the pilot driver for RunRowV5 + model_log.jsonl)
        self.records: list[dict[str, Any]] = []
        self.api_calls = 0
        self.api_errors = 0
        self.parse_retries = 0
        self.parse_failures = 0
        self.guard_retries = 0
        self.tokens_input = 0
        self.tokens_output = 0
        self.tokens_reasoning = 0
        self.latency_total_s = 0.0

    # -- prompt ----------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        spec_lines = json.dumps(self._context.public_spec, ensure_ascii=False, sort_keys=True, indent=1)
        catalog = []
        for name in TOOL_NAMES:
            mark = "available" if name in self._context.tool_allowlist else "NOT available in this arm"
            catalog.append("- [%s] %s" % (mark, TOOL_CATALOG[name]))
        arm_note = self._context.arm_note
        if arm_note is None and "seal_artifact" in self._context.tool_allowlist:
            arm_note = _ONTOLOGY_ARM_NOTE  # default: self-built shacl-rule/1 arm
        prompt = (
            _PROTOCOL
            % ACTION_SCHEMA_DOC
            + ("\n\n" + arm_note if arm_note else "")
            + (
                "\n\n# Retrieved public documentation (deterministic lexical retrieval for THIS task's goal)\n"
                + self._context.retrieval_digest
                if self._context.retrieval_digest
                else ""
            )
            + "\n\n# Tools (11-tool catalog; availability for THIS run marked)\n"
            + "\n".join(catalog)
            + "\n\n# Public task spec (authoritative; private fields are stripped harness-side)\n"
            + spec_lines
        )
        assert_no_private_keys(prompt)  # fail-closed: nothing private reaches the API
        return prompt

    # -- harness surface ---------------------------------------------------------

    def next_action(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        assert_no_private_keys(observation)
        self._turn += 1
        if self._submitted:
            return self._forced_submit("ALREADY_SUBMITTED")
        if self._turn > self._config.max_turns or self.api_calls >= self._config.max_calls_per_run:
            return self._forced_submit("BUDGET_FUSE")
        objects = observation.get("observation")
        if isinstance(objects, Mapping) and isinstance(objects.get("objects"), Mapping):
            self._observed_objects = objects["objects"]
        self._messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"turn": self._turn, "observation": observation},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        return self._produce_call()

    # -- internals ----------------------------------------------------------------

    def _produce_call(self) -> dict[str, Any]:
        api_attempts = 0
        bad_replies = 0  # parse + guard failures share the documented per-turn cap
        while True:
            if self.api_calls >= self._config.max_calls_per_run:
                return self._forced_submit("BUDGET_FUSE")
            result = self._call_api()
            if result.error or result.content is None:
                self.api_errors += 1
                api_attempts += 1
                if api_attempts > self._config.max_api_retries:
                    return self._forced_submit("MODEL_FAILURE", detail=result.error)
                self._messages.append(
                    {
                        "role": "user",
                        "content": "SYSTEM NOTE: the previous API call failed (%s). "
                        "Emit exactly one JSON tool call now." % (result.error or "empty content"),
                    }
                )
                continue
            self._messages.append({"role": "assistant", "content": result.content})
            try:
                call = parse_tool_call(result.content)
            except ToolCallParseError as exc:
                self.parse_retries += 1
                bad_replies += 1
                if bad_replies > self._config.max_parse_retries:
                    self.parse_failures += 1
                    return self._forced_submit("MODEL_PARSE_FAILURE", detail=str(exc))
                self._messages.append(
                    {
                        "role": "user",
                        "content": "PROTOCOL_ERROR: %s. Reply with EXACTLY one JSON object "
                        '{"tool": "<name>", "args": {...}}} and nothing else.' % exc,
                    }
                )
                continue
            shape_error = check_action_shape(call, self._observed_objects)
            if shape_error is not None:
                self.guard_retries += 1
                bad_replies += 1
                if bad_replies > self._config.max_parse_retries:
                    self.parse_failures += 1
                    return self._forced_submit("MODEL_PARSE_FAILURE", detail=shape_error)
                self._messages.append(
                    {
                        "role": "user",
                        "content": "ACTION_GUARD: your action was NOT executed (%s). "
                        "Correct it against the current public state and reply with "
                        "exactly one JSON tool call." % shape_error,
                    }
                )
                continue
            if call["tool"] == "submit_final":
                self._submitted = True
            return call

    def _call_api(self) -> CallResult:
        cfg = self._config
        custom_id = "w11pilot-t%04d-c%04d" % (self._turn, self.api_calls + 1)
        req = CallRequest(
            model=cfg.model,
            messages=list(self._messages),
            custom_id=custom_id,
            params=CallParams(
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                json_mode=True,
                seed=cfg.seed,
            ),
        )
        result = self._loop.run(self._client.call(req), timeout=cfg.request_timeout_s + 60.0)
        self.model_exact_id = result.model_exact_id
        self.api_calls += 1
        self.latency_total_s += result.latency_s
        usage = result.usage
        self.tokens_input += usage.input or 0
        self.tokens_output += usage.output or 0
        self.tokens_reasoning += usage.reasoning or 0
        self.records.append(
            {
                "ts": result.ts,
                "turn": self._turn,
                "custom_id": result.custom_id,
                "prompt_sha": result.prompt_sha,
                "response_id": result.response_id,
                "model_exact_id": result.model_exact_id,
                "usage": usage.model_dump(),
                "latency_s": round(result.latency_s, 3),
                "attempts": result.attempts,
                "finish_reason": result.finish_reason,
                "error": result.error,
            }
        )
        return result

    def _forced_submit(self, reason: str, detail: str | None = None) -> dict[str, Any]:
        self._submitted = True
        answer: dict[str, Any] = {"error": reason}
        if detail:
            answer["detail"] = detail[:300]
        return {"tool": "submit_final", "args": {"answer": answer}}

    # -- run-scoped summaries (pilot driver reads these) -----------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def health(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "parse_retries": self.parse_retries,
            "parse_failures": self.parse_failures,
            "guard_retries": self.guard_retries,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_reasoning": self.tokens_reasoning,
            "latency_total_s": round(self.latency_total_s, 3),
            "turns": self._turn,
            "model_exact_id": self.model_exact_id,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._loop.close()

    def __enter__(self) -> "DashScopeAgentModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
