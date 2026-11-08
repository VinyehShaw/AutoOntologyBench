"""CQL / WHO SMART-ANC clinical-rule domain adapter (W4, Review 5B).

Track: TaskTrack.clinical, profile ``cql-guide/1``.

What is RUNNABLE here (verified M1, logs under data/erule/review5/clinical_physical/logs/):
- CQL -> ELM translation via pinned info.cqframework:cql-to-elm-cli:3.29.0
  (vendored classpath under sources/cql/toolchain/jars, huaweicloud Maven mirror).
- ELM evaluation for System-model (retrieve-free) libraries via pinned
  info.cqframework:engine:3.29.0 and the RunElm driver in sources/cql/.

What is NOT runnable: patient-data (QUICK/FHIR) retrieval evaluation and the WHO
SMART-ANC layer. Those need engine-fhir wiring, a terminology provider over the local
ValueSets, and synthetic FHIR R4 patient fixtures. See SmartAncDomain.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.protocols import (
    NativeDecision,
    NativeState,
    NativeStateAndFeedback,
    PermissionSet,
    PublicObservation,
    TypedAction,
)

PROFILE = "cql-guide/1"
TOOLCHAIN = Path(__file__).resolve().parents[3] / "sources" / "cql" / "toolchain"
MAIN_CLASS = "org.cqframework.cql.cql2elm.cli.Main"

BLOCKED_SMART_ANC_REASON = (
    "SMART-ANC evaluation BLOCKED: dependency closure is complete locally (38 PlanDefinition, "
    "64 Library with 63 embedded ELM, 613 ValueSet, 1 CodeSystem, 6 ConceptMap, 63 CQL), but "
    "runnable evaluation still needs (1) engine-fhir 3.29.0 wiring, (2) a TerminologyProvider "
    "backed by the local ValueSets, (3) canonical-URL aliasing (ELM refs "
    "'http://smart.who.int/anc/<Name>' vs local 'http://smart.who.int/anc/Library/<Name>'), "
    "and (4) synthetic FHIR R4 patient fixtures. Next step logged in "
    "data/erule/review5/clinical_physical/REPORT.md."
)


class NotRunnableYet(RuntimeError):
    """Raised when this adapter's native dependency is unavailable; message carries the blocker."""


def _java() -> str:
    java_home = os.environ.get("JAVA_HOME", "/opt/conda/envs/jena")
    return str(Path(java_home) / "bin" / "java")


def _javac() -> str:
    java_home = os.environ.get("JAVA_HOME", "/opt/conda/envs/jena")
    return str(Path(java_home) / "bin" / "javac")


def _classpath() -> str:
    jars = TOOLCHAIN / "jars"
    if not jars.is_dir() or not list(jars.glob("*.jar")):
        raise NotRunnableYet(
            "BLOCKED: CQL toolchain jars missing; rebuild with "
            "sources/cql/toolchain/build_toolchain.sh (huaweicloud Maven mirror)"
        )
    return f"{jars}/*"


def _ensure_driver_compiled() -> Path:
    classes = TOOLCHAIN / "classes"
    if not (classes / "RunElm.class").exists():
        classes.mkdir(parents=True, exist_ok=True)
        src = TOOLCHAIN.parent / "RunElm.java"
        subprocess.run(
            [_javac(), "-cp", _classpath(), "-d", str(classes), str(src)],
            check=True, capture_output=True, text=True, timeout=300,
        )
    return classes


def translate_cql(cql_path: str, out_json: str) -> dict[str, Any]:
    """Run the pinned native translator. NOTE: cql-to-elm-cli 3.29.0 exits 0 even on

    semantic failure, so success is detected from the output file + stderr marker.
    """
    proc = subprocess.run(
        [_java(), "-cp", _classpath(), MAIN_CLASS,
         "--input", cql_path, "--output", out_json, "--format", "JSON"],
        capture_output=True, text=True, timeout=600,
    )
    out = Path(out_json)
    failed = "Translation failed" in proc.stderr or not out.exists()
    result = {
        "exit_code": proc.returncode,
        "stderr": proc.stderr,
        "translation_failed": failed,
    }
    if out.exists():
        elm = json.loads(out.read_text())
        result["library_id"] = elm.get("library", {}).get("identifier", {})
        result["n_statements"] = len(elm.get("library", {}).get("statements", {}).get("def", []))
        result["elm_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return result


def evaluate_library(cql_dir: str, lib_name: str) -> dict[str, Any]:
    """Evaluate all expressions of a System-model library via the pinned engine."""
    classes = _ensure_driver_compiled()
    proc = subprocess.run(
        [_java(), "-cp", f"{_classpath()}:{classes}", "RunElm", cql_dir, lib_name],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"engine evaluation failed: {proc.stderr[-2000:]}")
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("EXPR "):
            name, _, rest = line[5:].partition(" = ")
            values[name] = rest
    return {"exit_code": proc.returncode, "values": values}


class CqlWhoDomain:
    """NativeDomain for CQL translation + System-model ELM evaluation.

    Analytical/clinical-decision track: the only state-changing action is
    ``translate`` (CQL -> ELM); ``verify_terminal`` re-evaluates the named library
    natively and compares against the submitted expression values.
    """

    profile = PROFILE

    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        cfg = json.loads(Path(spec.initial_state_ref).read_text())
        cql_path = cfg["cql_path"]
        return NativeState(payload={
            "cql_path": cql_path,
            "cql_sha256": hashlib.sha256(Path(cql_path).read_bytes()).hexdigest(),
            "elm": None,
            "seed": seed,
        })

    def observe(self, state: NativeState, rights: PermissionSet) -> PublicObservation:
        return PublicObservation(payload={
            "cql_sha256": state.payload["cql_sha256"],
            "translated": state.payload["elm"] is not None,
            "track": "clinical",
        })

    def legal_action(self, state: NativeState, action: TypedAction) -> NativeDecision:
        if action.get("type") in {"translate", "submit_answer"}:
            return NativeDecision(legal=True, detail={"profile": PROFILE})
        return NativeDecision(legal=False, detail={"reason": f"unknown action {action.get('type')!r}"})

    def step(self, state: NativeState, action: TypedAction) -> NativeStateAndFeedback:
        decision = self.legal_action(state, action)
        if not decision.legal:
            return NativeStateAndFeedback(state=state, feedback={"error": decision.detail}, cost=0.0)
        if action["type"] == "translate":
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "out.json")
                res = translate_cql(state.payload["cql_path"], out)
            new_payload = {**state.payload, "elm": res_summary(res)}
            return NativeStateAndFeedback(
                state=NativeState(payload=new_payload),
                feedback={"translation_failed": res["translation_failed"]},
                cost=1.0,
            )
        return NativeStateAndFeedback(state=state, feedback={"submitted": True}, cost=0.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        if not isinstance(submitted_answer, dict) or "expressions" not in submitted_answer:
            return GoalResult(goal_reached=False, evidence={"reason": "no expressions submitted"})
        lib_name = submitted_answer.get("library") or Path(state.payload["cql_path"]).stem
        truth = evaluate_library(str(Path(state.payload["cql_path"]).parent), lib_name)
        mismatches = {
            k: {"claimed": v, "actual": truth["values"].get(k)}
            for k, v in submitted_answer["expressions"].items()
            if str(truth["values"].get(k)) != str(v)
        }
        return GoalResult(
            goal_reached=not mismatches,
            evidence={"engine_values": truth["values"], "mismatches": mismatches},
        )


def res_summary(res: dict[str, Any]) -> dict[str, Any]:
    return {k: res.get(k) for k in ("library_id", "n_statements", "elm_sha256", "translation_failed")}


class SmartAncDomain:
    """WHO SMART-ANC track placeholder. Always raises with the recorded blocker."""

    profile = "cql-guide/1"

    def __init__(self) -> None:
        raise NotRunnableYet(BLOCKED_SMART_ANC_REASON)
