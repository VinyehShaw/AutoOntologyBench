"""Policy hooks for the Review-5 tool API (owner: W6).

The enforcement seam between the frozen ``Budget``/allowlist contracts and
the tool session: action allowlists (manifest-level), artifact file-edit
policy, and budget metering. Pure logic, no I/O; the tool session
(``autoontobench.runners.tool_api``) is the only caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from autoontobench.core.protocols import Budget
from autoontobench.runners.ontology_bundle import OPTIONAL_FILES, REQUIRED_FILES


class PolicyViolation(ValueError):
    """An agent request violated a harness policy (allowlist/file/budget)."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class ActionAllowlist:
    """Manifest-level allowlist for ``perform_action``/``probe_native`` kinds.

    Actions outside the allowlist are rejected before any native call
    (protocols.py contract: "actions outside the manifest allowlist are
    rejected").
    """

    kinds: frozenset[str]

    def check(self, action: Mapping[str, Any]) -> None:
        kind = action.get("kind")
        if not isinstance(kind, str) or not kind:
            raise PolicyViolation("ACTION_MALFORMED", "action must carry a non-empty 'kind'")
        if kind not in self.kinds:
            raise PolicyViolation(
                "ACTION_NOT_ALLOWED",
                "action kind %r is outside the manifest allowlist" % kind,
            )


@dataclass(frozen=True)
class FileEditPolicy:
    """Which artifact files the agent may write, and per-file byte caps."""

    allowed_names: frozenset[str]
    max_bytes_per_file: int = 262144  # mirrors erule-core/1.0 max_bytes

    def check(self, files: Mapping[str, Any]) -> None:
        if not isinstance(files, Mapping) or not files:
            raise PolicyViolation("EDIT_MALFORMED", "files must be a non-empty mapping")
        for name, content in files.items():
            if name not in self.allowed_names:
                raise PolicyViolation(
                    "FILE_NOT_ALLOWED", "artifact file %r is not editable" % (name,)
                )
            if not isinstance(content, str):
                raise PolicyViolation("EDIT_MALFORMED", "file %r content must be text" % (name,))
            size = len(content.encode("utf-8"))
            if size > self.max_bytes_per_file:
                raise PolicyViolation(
                    "FILE_TOO_LARGE",
                    "file %r is %d bytes (cap %d)" % (name, size, self.max_bytes_per_file),
                )


DEFAULT_FILE_POLICY = FileEditPolicy(
    allowed_names=frozenset(set(REQUIRED_FILES) | set(OPTIONAL_FILES))
)


@dataclass
class BudgetMeter:
    """Mutable charge meter over the frozen ``Budget`` (per-run)."""

    budget: Budget
    tool_calls: int = 0
    oracle_probes: int = 0
    actions: int = 0

    def charge(self, kind: str) -> None:
        """Charge one unit of ``kind`` ("tool"|"probe"|"action").

        Raises PolicyViolation(BUDGET_EXCEEDED) before any execution.
        """
        if kind == "tool":
            if self.tool_calls + 1 > self.budget.total_tool_calls:
                raise PolicyViolation("BUDGET_EXCEEDED", "total_tool_calls exhausted")
            self.tool_calls += 1
        elif kind == "probe":
            if self.oracle_probes + 1 > self.budget.oracle_probes:
                raise PolicyViolation("BUDGET_EXCEEDED", "oracle_probes exhausted")
            self.oracle_probes += 1
        elif kind == "action":
            if self.actions + 1 > self.budget.max_actions:
                raise PolicyViolation("BUDGET_EXCEEDED", "max_actions exhausted")
            self.actions += 1
        else:  # pragma: no cover - internal misuse guard
            raise ValueError("unknown charge kind %r" % kind)

    def remaining(self) -> dict[str, int]:
        return {
            "tool_calls": max(0, self.budget.total_tool_calls - self.tool_calls),
            "oracle_probes": max(0, self.budget.oracle_probes - self.oracle_probes),
            "actions": max(0, self.budget.max_actions - self.actions),
        }
