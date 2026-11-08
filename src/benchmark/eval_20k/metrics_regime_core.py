"""Shared constants, errors, and input values for the regime reporter."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "aob-eval-20k/regime-breakdown/1"

#: Canonical v3 regimes, sorted for deterministic JSON keys.
REGIMES: tuple[str, ...] = (
    "application",
    "construction",
    "organization",
    "repair",
)

#: Groups that make a complete 4 x 4 model sweep.
EXPECTED_GROUPS: tuple[str, ...] = ("g1", "g2", "g3", "g4")

#: Cost fields requested by the regime breakdown.  ``tool_calls`` is omitted:
#: it is present in the reports but is always zero in v3 and not requested.
COST_KEYS: tuple[str, ...] = (
    "api_calls",
    "tokens_input",
    "tokens_output",
    "wall_seconds",
)

COUNT_FIELDS: tuple[str, ...] = (
    "tasks",
    "scored",
    "success",
    "verified_success",
    "error",
    "missing",
    "not_supported",
    "oracle_missing",
)

STATUS_SCORED = "SCORED"
STATUS_MISSING = "MISSING"
STATUS_NOT_SUPPORTED = "NOT_SUPPORTED"
STATUS_ORACLE_MISSING = "ORACLE_MISSING"
STATUS_ERROR = "ERROR"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_GLOB = str(
    REPO_ROOT / "runs" / "eval_20k_full2-*" / "report.json")
DEFAULT_TASKS_DIR = str(REPO_ROOT / "runs" / "full_v3_models" / "tasks")
DEFAULT_OUT = str(
    REPO_ROOT / "reports" / "extra_metrics" / "regime_breakdown.json")

RUN_DIR_RE = re.compile(
    r"^eval_20k_full2-(?P<group>g[0-9]+)-(?P<model>.+?)_(?P<suffix>[A-Za-z0-9]+)$")


class RegimeError(ValueError):
    """Base class for hard, user-facing regime aggregation errors."""


class RegimeJoinError(RegimeError):
    """A prediction/scored/task row cannot be joined to a task regime."""


class RegimeValidationError(RegimeError):
    """Per-regime totals do not reconcile with ``report.metrics.overall``."""


@dataclass(frozen=True, slots=True)
class TaskRow:
    """Minimal task projection needed for the regime join."""

    task_id: str
    task_regime: str


@dataclass(frozen=True, slots=True)
class RunInputs:
    """Everything one completed run contributes, with provenance."""

    run_id: str
    group: str
    model: str
    report_path: str
    report_sha256: str
    report_overall: Mapping[str, Any]
    tasks_path: str
    tasks_sha256: str
    task_rows: tuple[TaskRow, ...]
    predictions_path: str
    predictions_sha256: str
    prediction_rows: tuple[Mapping[str, Any], ...]
    rows_path: str
    rows_sha256: str
    scored_rows: tuple[Mapping[str, Any], ...]