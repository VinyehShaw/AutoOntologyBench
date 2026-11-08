"""Camunda DMN adapter (Review 5B W2/M1 smoke; EDUCATIONAL source).

Wraps the pinned native pipeline — ``native_runners/dmn_runner.run_native_eval``
(camunda-engine-dmn 7.24.0 + feel-engine 1.19.3 scala-shaded, JDK25 at
/opt/conda/envs/jena, jar sha256 pins verified by ``verify_jar_pins``) — behind a
minimal typed interface: load one decision table, evaluate typed fixtures
natively, get typed decisions back.

LICENSE/SCOPE: the pinned tables are Camunda *teaching examples*
(camunda-bpm-examples @ 6c7f4c4a…, Apache-2.0); per Implementation §11.2 they are
labelled **EDUCATIONAL** — a protocol pilot, not enterprise-realism evidence.
Every evaluation lands a .vars.json + .log under the caller's log dir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from native_runners.dmn_runner import (
    DEFAULT_BUILD_DIR,
    DEFAULT_JAR_DIR,
    toolchain_available,
    verify_jar_pins,
)
from native_runners.dmn_runner import run_native_eval as _run_native_eval
from rule_ingestion.dmn_adapter import (
    DISH_DECISION_FILE,
    DmnRuleUnit,
    external_input_types,
    parse_dmn_file,
)

CAMUNDA_REPO = (
    Path(__file__).resolve().parents[3] / "references" / "sources" / "camunda-bpm-examples"
)
CAMUNDA_COMMIT = "6c7f4c4adb4beb9a2f5d4c5e49fc1ddfc6dab3fc"
CAMUNDA_VERSION = "7.24.0"
SOURCE_LABEL = "EDUCATIONAL"


@dataclass(frozen=True)
class DmnDecisionOutcome:
    """Typed native decision outcome (never fabricated; empty is a legal outcome)."""

    unit_id: str
    decision_id: str
    hit_policy: str
    empty: bool
    entries: Tuple[Dict[str, Any], ...]
    determined_output: Any  # str for UNIQUE, sorted list for COLLECT, None if empty
    input_sha256: str
    log_path: str
    exit_code: int


def load_decision_table(rel_path: str = DISH_DECISION_FILE) -> Tuple[DmnRuleUnit, ...]:
    """Parse one pinned DMN file into typed rule units (fail-closed fragment)."""
    return parse_dmn_file(CAMUNDA_REPO, rel_path)


def evaluate_decision(
    unit: DmnRuleUnit,
    variables: Dict[str, Any],
    *,
    input_id: str,
    log_dir: Path,
    all_units: Tuple[DmnRuleUnit, ...] | None = None,
) -> DmnDecisionOutcome:
    """Evaluate one decision natively on typed fixture variables.

    For DRG members pass ``all_units`` (the file's full unit tuple) so the
    input-variable typeRefs are taken from the required-decision closure.
    """
    input_types = external_input_types(unit, all_units) if all_units else None
    outcome = _run_native_eval(
        unit, variables, input_id=input_id, log_dir=Path(log_dir),
        jar_dir=DEFAULT_JAR_DIR, build_dir=DEFAULT_BUILD_DIR, input_types=input_types,
    )
    return DmnDecisionOutcome(
        unit_id=outcome.unit_id, decision_id=outcome.decision_id,
        hit_policy=outcome.hit_policy, empty=outcome.empty, entries=outcome.entries,
        determined_output=outcome.determined_output, input_sha256=outcome.input_sha256,
        log_path=str(outcome.log_path), exit_code=outcome.exit_code,
    )


def dmn_toolchain_status() -> Dict[str, Any]:
    """Pinned toolchain self-check: java binary, jar sha256 manifest, compiled driver."""
    ok = toolchain_available()
    pins: Dict[str, str] = {}
    if ok:
        pins = verify_jar_pins(DEFAULT_JAR_DIR)
    return {
        "available": ok,
        "camunda_version": CAMUNDA_VERSION,
        "jar_count": len(pins),
        "source_label": SOURCE_LABEL,
        "source_commit": CAMUNDA_COMMIT,
    }
