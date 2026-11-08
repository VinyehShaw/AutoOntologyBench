"""Dual-profile ontology consumer (owner: W13): ``shacl-rule/1`` + ``code-dsl/1``.

Wraps the W6 ``OntologyConsumer`` verbatim for ``shacl-rule/1`` and adds the
CODE_DSL arm's ``code-dsl/1`` profile: the agent-authored ``program.json``
(gk-dsl/1, ``dsl_program``) is schema-validated and evaluated in a SEPARATE
PROCESS under ``autoontobench.safety.sandbox.run_sandboxed`` (RLIMIT memory/
CPU/file-size caps + wall-clock process-group kill — the same sandbox the
SHACL worker runs under), child entry point
``autoontobench.runners.dsl_worker``. Fallback: if the sandbox module is not
importable, the same job runs in-process (``executor="inprocess_fallback"``),
mirroring ``core.ontology_consumer``.

Duck-types the frozen ``OntologyTool`` protocol (load / query_or_check /
seal); ``core/ontology_consumer.py`` and the frozen contracts are untouched.
Native truth is NEVER consulted here: ``query_or_check`` only interprets the
agent-authored program over the PUBLIC facts handed to it by the caller.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoontobench.core.ontology_consumer import OntologyConsumer, OntologyRejected
from autoontobench.core.protocols import OntologyPrediction, ValidatedOntology
from autoontobench.runners.dsl_worker import DSL_PROFILE, RESULT_MARKER
from autoontobench.runners.ontology_bundle import bundle_content_hash

_DSL_MEMORY_BYTES = 128 * 1024 * 1024  # stdlib-only interpreter
_DSL_WALL_SECONDS = 10.0
_DSL_OUTPUT_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class _DslHandle:
    program: dict[str, Any]
    program_sha: str
    executor: str  # "sandbox" | "inprocess_fallback"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sandbox_dsl(mode: str, job: dict[str, Any]) -> dict[str, Any] | None:
    """Run the DSL worker under hard process limits; None => sandbox unavailable."""
    try:
        from autoontobench.safety.sandbox import (
            SandboxLimits,
            run_sandboxed,
        )
    except Exception:
        return None
    limits = SandboxLimits(
        max_wall_seconds=_DSL_WALL_SECONDS,
        max_cpu_seconds=int(_DSL_WALL_SECONDS),
        max_memory_bytes=_DSL_MEMORY_BYTES,
        max_output_bytes=_DSL_OUTPUT_BYTES,
    )
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONPATH": str(_repo_root())}
    with tempfile.TemporaryDirectory(prefix="aob-dsl-") as jobdir:
        job_path = Path(jobdir) / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        result = run_sandboxed(
            [sys.executable, "-m", "autoontobench.runners.dsl_worker",
             "--mode", mode, str(job_path)],
            limits=limits,
            env=env,
        )
    if result.killed_reason is not None:
        return {
            "ok": False,
            "error": "TIMEOUT" if "timeout" in result.killed_reason else "INFRA_ERROR",
            "reasons": ["sandbox killed the dsl worker: %s" % result.killed_reason],
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
            "reasons": ["dsl worker failed (exit %s): %s" % (result.exit_code, tail)],
        }
    try:
        return json.loads(marker_line[len(RESULT_MARKER):])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "INFRA_ERROR", "reasons": ["dsl result unparseable: %s" % exc]}


class DualProfileConsumer:
    """Frozen ``OntologyTool`` duck: shacl-rule/1 via the W6 consumer, code-dsl/1 native."""

    def __init__(self, *, inner: OntologyConsumer | None = None,
                 force_inprocess_fallback: bool = False):
        self._inner = inner if inner is not None else OntologyConsumer()
        self._force_fallback = force_inprocess_fallback

    def _run(self, mode: str, job: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not self._force_fallback:
            result = _sandbox_dsl(mode, job)
            if result is not None:
                return result, "sandbox"
        try:
            from autoontobench.runners.dsl_worker import DslBundleRejected, run_job

            return run_job(mode, job), "inprocess_fallback"
        except DslBundleRejected as exc:
            return {"ok": False, "reasons": exc.reasons}, "inprocess_fallback"
        except Exception as exc:  # boundary: report, never crash mute
            return (
                {"ok": False, "error": "INFRA_ERROR",
                 "reasons": ["%s: %s" % (type(exc).__name__, str(exc)[:300])]},
                "inprocess_fallback",
            )

    def load(self, public_artifact_bytes: bytes, named_profile: str) -> ValidatedOntology:
        if named_profile != DSL_PROFILE:
            return self._inner.load(public_artifact_bytes, named_profile)
        job = {"bundle_b64": base64.b64encode(public_artifact_bytes).decode("ascii")}
        result, executor = self._run("load", job)
        if not result.get("ok"):
            raise OntologyRejected(list(result.get("reasons", ["dsl load failed"])))
        payload = result["payload"]
        handle = _DslHandle(
            program=payload["program"],
            program_sha=payload["program_sha"],
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
        if artifact.profile != DSL_PROFILE:
            return self._inner.query_or_check(artifact, public_facts, request)
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
                    "reasons": list(result.get("reasons", [])),
                    "executor": executor,
                    "o_sha": artifact.content_hash,
                },
            )
        return OntologyPrediction(
            prediction=result["decision"],
            detail={
                "status": result["status"],
                "violations": [json.dumps(v, sort_keys=True) for v in result.get("violations", [])],
                "unknowns": list(result.get("unknowns", [])),
                "report_hash": result["report_hash"],
                "program_sha": artifact.handle.program_sha
                if isinstance(artifact.handle, _DslHandle)
                else None,
                "executor": executor,
                "o_sha": artifact.content_hash,
            },
        )

    def seal(self, artifact: ValidatedOntology) -> str:
        """Single content-addressed snapshot hash of the exact bundle bytes."""
        if artifact.profile != DSL_PROFILE:
            return self._inner.seal(artifact)
        return artifact.content_hash
