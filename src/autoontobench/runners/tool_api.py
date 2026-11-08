"""Review-5 minimum tool API session (owner: W6; Review-5B §5).

Eleven tools, one session per run: ``inspect_schema`` / ``inspect_state`` /
``read_public_doc`` (public), ``probe_native`` (charged native oracle probe),
``edit_artifact`` / ``validate_artifact`` (syntax-only, leak-closed) /
``seal_artifact`` (single snapshot), ``check_action_with_ontology``
(O-consumer opinion, never native truth), ``perform_action`` (native
legality + state mutation), ``get_feedback``, ``submit_final``.
Implementations live in ``tool_handlers.ToolHandlers``; this module owns
session state, dispatch, budget/allowlist enforcement, and the typed
append-only JSONL tool log (``tool_call_id``, action, ``observation_sha``,
charge, ``wall_ms``, ``o_sha``, native/public ``channel``).

W12 observation channel (finding A): every executed tool call's result is
recorded on ``self._last_tool_result`` (``tool``/``ok``/``error``/``charge``/
``payload``) and echoed to the model on the NEXT turn via
``model_observation()["last_tool_result"]`` — identical for all arms; budgets
and allowlist semantics are unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import TaskSpecV5, assert_no_private_keys
from autoontobench.core.ontology_consumer import (
    NotImplementedYet,
    OntologyConsumer,
    OntologyRejected,
)
from autoontobench.core.policy_hooks import (
    ActionAllowlist,
    BudgetMeter,
    FileEditPolicy,
    PolicyViolation,
)
from autoontobench.core.protocols import (
    Budget,
    NativeDomain,
    NativeState,
    PermissionSet,
)
from autoontobench.runners.tool_handlers import ToolHandlers
from autoontobench.runners.tool_log import (
    ToolLogRow,
    ToolLogWriter,
    canonical_sha256,
    sanitize_args,
)

TOOL_NAMES = (
    "inspect_schema",
    "inspect_state",
    "read_public_doc",
    "probe_native",
    "edit_artifact",
    "validate_artifact",
    "check_action_with_ontology",
    "perform_action",
    "get_feedback",
    "seal_artifact",
    "submit_final",
)

_CHANNELS = {
    "inspect_schema": "public",
    "inspect_state": "native",
    "read_public_doc": "public",
    "probe_native": "native",
    "edit_artifact": "artifact",
    "validate_artifact": "artifact",
    "check_action_with_ontology": "ontology",
    "perform_action": "native",
    "get_feedback": "session",
    "seal_artifact": "artifact",
    "submit_final": "session",
}


class ToolSession(ToolHandlers):
    """One run's tool surface. Constructed by the harness; never by the model."""

    def __init__(
        self,
        *,
        domain: NativeDomain,
        spec: TaskSpecV5,
        state: NativeState,
        rights: PermissionSet,
        budget: Budget,
        action_allowlist: ActionAllowlist,
        file_policy: FileEditPolicy,
        ontology: OntologyConsumer | None,
        ontology_profile: str | None,
        run_id: str,
        tool_allowlist: frozenset[str] | None = None,
        initial_files: Mapping[str, str] | None = None,
        artifact_editable: bool = True,
        forced_artifact_bytes: bytes | None = None,
        public_root: Path | None = None,
        log_path: Path | None = None,
    ):
        self._domain = domain
        self._spec = spec
        self._state = state
        self._rights = rights
        self._meter = BudgetMeter(budget)
        self._allowlist = action_allowlist
        self._file_policy = file_policy
        self._ontology = ontology
        self._ontology_profile = ontology_profile
        self._tool_allowlist = tool_allowlist
        self._files: dict[str, str] = dict(initial_files or {})
        self._artifact_editable = artifact_editable
        # E2 hook (Review-5B §6): replace ONLY the O used at check time.
        self._forced_artifact_bytes = forced_artifact_bytes
        self._public_root = public_root
        self._run_id = run_id
        self._log = ToolLogWriter(log_path)
        self.rows: list[ToolLogRow] = []
        self._seq = 0
        self._sealed_sha: str | None = None
        self._submitted = False
        self._final_answer: Any = None
        self._last_feedback: dict[str, Any] | None = None
        self._last_tool_result: dict[str, Any] | None = None
        self.unsafe_action_count = 0
        self._ontology_cache: dict[str, Any] = {}

    # -- read-only accessors used by the harness/scorer ----------------------

    @property
    def state(self) -> NativeState:
        return self._state

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def final_answer(self) -> Any:
        return self._final_answer

    @property
    def sealed_ontology_sha(self) -> str | None:
        return self._sealed_sha

    @property
    def meter(self) -> BudgetMeter:
        return self._meter

    @property
    def working_files(self) -> dict[str, str]:
        return dict(self._files)

    @property
    def has_forced_artifact(self) -> bool:
        return self._forced_artifact_bytes is not None

    # -- dispatch -------------------------------------------------------------

    def execute(self, call: Mapping[str, Any]) -> dict[str, Any]:
        tool = str(call.get("tool", ""))
        args = call.get("args") or {}
        handler = getattr(self, "_tool_" + tool, None) if tool in TOOL_NAMES else None
        started = time.monotonic()
        error: str | None = None
        try:
            self._meter.charge("tool")
            if handler is None:
                raise PolicyViolation("UNKNOWN_TOOL", "tool %r is not available" % tool)
            if tool not in self._spec.allowed_tools:
                raise PolicyViolation(
                    "TOOL_NOT_IN_TASK", "tool %r is not in the task allowed_tools" % tool
                )
            if self._tool_allowlist is not None and tool not in self._tool_allowlist:
                raise PolicyViolation(
                    "TOOL_NOT_IN_ARM", "tool %r is not available in this arm" % tool
                )
            payload = handler(args)
            ok = True
        except PolicyViolation as exc:
            payload, ok, error = {"error": exc.code, "detail": str(exc)}, False, exc.code
        except OntologyRejected as exc:
            payload, ok, error = (
                {"error": "ARTIFACT_REJECTED", "reasons": exc.reasons},
                False,
                "ARTIFACT_REJECTED",
            )
        except NotImplementedYet as exc:
            payload, ok, error = (
                {"error": "PROFILE_NOT_IMPLEMENTED", "profile": exc.profile},
                False,
                "PROFILE_NOT_IMPLEMENTED",
            )
        wall_ms = (time.monotonic() - started) * 1000.0
        assert_no_private_keys(payload)  # fail-closed leak guard, never reaches agent
        row = ToolLogRow(
            tool_call_id="%s:%04d" % (self._run_id, self._seq),
            seq=self._seq,
            tool=tool or "<missing>",
            action=sanitize_args({"tool": tool, "args": args}),
            ok=ok,
            error=error,
            charge=_charge_of(tool),
            observation_sha=canonical_sha256(payload),
            o_sha=self._current_o_sha(),
            channel=_CHANNELS.get(tool, "session"),
            wall_ms=round(wall_ms, 3),
        )
        self._seq += 1
        self.rows.append(row)
        self._log.append(row)
        self._last_tool_result = {
            "tool": row.tool,
            "ok": ok,
            "error": error,
            "charge": dict(row.charge),
            "payload": payload,
        }
        return payload


def _charge_of(tool: str) -> dict[str, int]:
    charge = {"tool": 1}
    if tool == "probe_native":
        charge["probe"] = 1
    if tool == "perform_action":
        charge["action"] = 1
    return charge
