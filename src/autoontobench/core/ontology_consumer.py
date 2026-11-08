"""Ontology consumer for Review-5 (owner: W6). Implements the frozen
``OntologyTool`` protocol from ``autoontobench/core/protocols.py``.

Profile dispatch (Implementation §2 named profiles):
- ``shacl-rule/1``  : implemented. Reuses, READ-ONLY, the erule contract
  validator (``erule.contract.validator.load_artifact``) and the erule rule
  interpreter (``erule.runtime.rule_interpreter.evaluate``). Native truth is
  NEVER consulted here: this consumer only interprets the agent-authored
  artifact over PUBLIC facts handed to it by the caller.
- ``linked-data/1``, ``smarts-executable/1``, ``dmn-decision/1``,
  ``cel-policy/1``, ``cql-guide/1``, ``sbml-executable/1``, ``bddl-goal/1``:
  honest ``NotImplementedYet`` dispatch points (M1).

Execution isolation (Review-5B §5): artifact parsing and SHACL/SPARQL
evaluation run in a SEPARATE PROCESS via
``autoontobench.safety.sandbox.run_sandboxed`` (W1; RLIMIT memory/CPU/
file-size caps + wall-clock process-group kill). The child entry point is
``autoontobench.runners.ontology_worker`` — the only W6 module importing the
heavy erule/rdflib/pySHACL stack. Fallback: if the sandbox module is not
importable, the same job runs in-process (``executor="inprocess_fallback"``;
TODO: drop once the sandbox is mandatory). Bundle format:
``autoontobench.runners.ontology_bundle``.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoontobench.core.protocols import OntologyPrediction, ValidatedOntology
from autoontobench.runners.ontology_bundle import (
    BundleRejected,
    bundle_content_hash,
    parse_bundle,
)
from autoontobench.runners.ontology_worker import (
    RESULT_MARKER,
    sanitize_reasons,
)

PROFILE_SHACL_RULE = "shacl-rule/1"
ERULE_PROFILE = "erule-core/1.0"
IMPLEMENTED_PROFILES = frozenset({PROFILE_SHACL_RULE})
DECLARED_PROFILES = frozenset(
    {
        PROFILE_SHACL_RULE,
        "linked-data/1",
        "smarts-executable/1",
        "dmn-decision/1",
        "cel-policy/1",
        "cql-guide/1",
        "sbml-executable/1",
        "bddl-goal/1",
    }
)

_WORKER_MEMORY_BYTES = 512 * 1024 * 1024  # measured: full import ~40 MiB RSS
_WORKER_WALL_SECONDS = 30.0  # interpreter-level SHACL timeout stays 5 s
_WORKER_OUTPUT_BYTES = 8 * 1024 * 1024


class NotImplementedYet(NotImplementedError):
    """Honest dispatch marker for declared-but-unimplemented profiles (M1)."""

    def __init__(self, profile: str):
        self.profile = profile
        super().__init__(
            "ontology profile %r is declared but not implemented in M1 "
            "(dispatch point; see Review-5B Implementation §2)" % (profile,)
        )


class OntologyRejected(ValueError):
    """The artifact failed contract validation (reasons path-sanitized)."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("artifact rejected: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class _ShaclHandle:
    ontology_nt: str
    shapes_nt: str
    mappings: dict[str, Any] | None
    manifest: dict[str, Any]
    contract_hash: str
    phi: dict[str, Any] | None
    executor: str  # "sandbox" | "inprocess_fallback"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sandbox_worker(mode: str, job: dict[str, Any]) -> dict[str, Any] | None:
    """Run the worker under hard process limits; None => sandbox unavailable."""
    try:
        from autoontobench.safety.sandbox import (  # type: ignore[import-not-found]
            SandboxLimits,
            run_sandboxed,
        )
    except Exception:
        return None
    limits = SandboxLimits(
        max_wall_seconds=_WORKER_WALL_SECONDS,
        max_cpu_seconds=int(_WORKER_WALL_SECONDS),
        max_memory_bytes=_WORKER_MEMORY_BYTES,
        max_output_bytes=_WORKER_OUTPUT_BYTES,
    )
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONPATH": str(_repo_root())}
    with tempfile.TemporaryDirectory(prefix="aob-otool-") as jobdir:
        job_path = Path(jobdir) / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        result = run_sandboxed(
            [
                sys.executable,
                "-m",
                "autoontobench.runners.ontology_worker",
                "--mode",
                mode,
                str(job_path),
            ],
            limits=limits,
            env=env,
        )
    if result.killed_reason is not None:
        return {
            "ok": False,
            "error": "TIMEOUT" if "timeout" in result.killed_reason else "INFRA_ERROR",
            "reasons": ["sandbox killed the worker: %s" % result.killed_reason],
        }
    marker_line = next(
        (
            line
            for line in result.stdout.decode("utf-8", "replace").splitlines()
            if line.startswith(RESULT_MARKER)
        ),
        None,
    )
    if result.exit_code != 0 or marker_line is None:
        tail = result.stderr.decode("utf-8", "replace")[-300:]
        return {
            "ok": False,
            "error": "INFRA_ERROR",
            "reasons": sanitize_reasons(["worker failed (exit %s): %s" % (result.exit_code, tail)]),
        }
    try:
        return json.loads(marker_line[len(RESULT_MARKER):])
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": "INFRA_ERROR",
            "reasons": ["worker result unparseable: %s" % exc],
        }


class OntologyConsumer:
    """Frozen ``OntologyTool`` implementation (see module docstring)."""

    def __init__(self, *, force_inprocess_fallback: bool = False):
        self._force_fallback = force_inprocess_fallback

    def _run(self, mode: str, job: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not self._force_fallback:
            result = _sandbox_worker(mode, job)
            if result is not None:
                return result, "sandbox"
        try:
            from autoontobench.runners.ontology_worker import run_job

            return run_job(mode, job), "inprocess_fallback"
        except BundleRejected as exc:
            return {"ok": False, "reasons": sanitize_reasons(exc.reasons)}, "inprocess_fallback"
        except Exception as exc:  # boundary: report, never crash mute
            return (
                {
                    "ok": False,
                    "error": "INFRA_ERROR",
                    "reasons": sanitize_reasons(["%s: %s" % (type(exc).__name__, exc)]),
                },
                "inprocess_fallback",
            )

    def load(self, public_artifact_bytes: bytes, named_profile: str) -> ValidatedOntology:
        if named_profile not in DECLARED_PROFILES:
            raise OntologyRejected(["unknown ontology profile %r" % (named_profile,)])
        if named_profile not in IMPLEMENTED_PROFILES:
            raise NotImplementedYet(named_profile)
        try:
            parse_bundle(public_artifact_bytes)  # fail-closed bundle gate
        except BundleRejected as exc:
            raise OntologyRejected(sanitize_reasons(exc.reasons)) from exc
        job = {"bundle_b64": base64.b64encode(public_artifact_bytes).decode("ascii")}
        result, executor = self._run("load", job)
        if not result.get("ok"):
            raise OntologyRejected(sanitize_reasons(result.get("reasons", ["load failed"])))
        payload = result["payload"]
        handle = _ShaclHandle(
            ontology_nt=payload["ontology_nt"],
            shapes_nt=payload["shapes_nt"],
            mappings=payload.get("mappings"),
            manifest=payload["manifest"],
            contract_hash=payload["contract_hash"],
            phi=payload.get("phi"),
            executor=executor,
        )
        return ValidatedOntology(
            profile=named_profile,
            artifact_bytes=public_artifact_bytes,
            content_hash=bundle_content_hash(public_artifact_bytes),
            handle=handle,
        )

    def query_or_check(
        self, artifact: ValidatedOntology, public_facts: object, request: object
    ) -> OntologyPrediction:
        """Evaluate ``request={"action": {...}}`` over PUBLIC facts only."""
        if artifact.profile != PROFILE_SHACL_RULE:
            raise NotImplementedYet(artifact.profile)
        if not isinstance(request, Mapping) or not isinstance(request.get("action"), Mapping):
            raise OntologyRejected(["request must be a mapping carrying an 'action' object"])
        if not isinstance(public_facts, Mapping):
            raise OntologyRejected(["public_facts must be a state mapping (public facts only)"])
        job = {
            "bundle_b64": base64.b64encode(artifact.artifact_bytes).decode("ascii"),
            "state": dict(public_facts),
            "action": dict(request["action"]),
        }
        result, executor = self._run("check", job)
        if not result.get("ok"):
            return OntologyPrediction(
                prediction=None,
                detail={
                    "status": result.get("error", "INFRA_ERROR"),
                    "reasons": sanitize_reasons(result.get("reasons", [])),
                    "executor": executor,
                    "o_sha": artifact.content_hash,
                },
            )
        return OntologyPrediction(
            prediction=result["decision"],
            detail={
                "status": result["status"],
                "violations": sanitize_reasons(
                    [json.dumps(v, sort_keys=True) for v in result.get("violations", [])]
                ),
                "unknowns": sanitize_reasons(result.get("unknowns", [])),
                "report_hash": result["report_hash"],
                "contract_hash": result["contract_hash"],
                "executor": executor,
                "o_sha": artifact.content_hash,
            },
        )

    def seal(self, artifact: ValidatedOntology) -> str:
        """Single content-addressed snapshot hash of the exact bundle bytes."""
        return artifact.content_hash
