"""BYOK native verifier plugin (example 1): real OPA eval against pinned Rego.

prove_exists runs `opa eval` on the violating manifest (must be non-empty:
task is non-trivial) and on the repaired manifest (must be empty: a solution
exists). verify_terminal re-evaluates a submitted repaired manifest.
certify_semantic_relevance requires the disallowed/allowed violation sets to
differ — a source-native semantic witness.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness)


def _opa_bin() -> str:
    env = os.environ.get("AOB_OPA_BIN")
    if env:
        return env
    return str(Path(__file__).resolve().parents[4] / "tools" / "bin" / "opa")


def _eval_violations(regos: list[str], package: str, obj: Any, params: Any,
                     timeout_s: float = 30.0) -> list[dict[str, Any]] | None:
    """None = opa error; [] = pass; non-empty list = violations."""
    with tempfile.TemporaryDirectory(prefix="aob_opa_") as td:
        args = [_opa_bin(), "eval", "--format", "json"]
        for i, rego in enumerate(regos):
            rego_path = Path(td) / f"policy{i}.rego"
            rego_path.write_text(rego, encoding="utf-8")
            args += ["-d", str(rego_path)]
        input_path = Path(td) / "input.json"
        input_path.write_text(json.dumps({"review": {"object": obj},
                                          "parameters": params}), encoding="utf-8")
        args += ["-i", str(input_path), f"data.{package}.violation"]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        return None
    out = json.loads(proc.stdout)
    results = out.get("result") or []
    if not results:
        return []
    value = results[0].get("expressions", [{}])[0].get("value")
    return list(value) if value else []


class GatekeeperOpaVerifier:
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ev = grounded_task.native_evidence
        bad = _eval_violations(ev["regos"], ev["package"], ev["disallowed"], ev["parameters"])
        if not bad:
            return ProofOrCounterexample(
                exists=False, reason="disallowed sample produced no violation "
                "(task trivial or opa failed)")
        good = _eval_violations(ev["regos"], ev["package"], ev["allowed"], ev["parameters"])
        if good is None:
            return ProofOrCounterexample(exists=False, reason="opa eval failed on allowed sample")
        if good:
            return ProofOrCounterexample(
                exists=False, reason=f"allowed sample still violates: {good[:1]}")
        return ProofOrCounterexample(exists=True, evidence={
            "violations": bad, "violation_count": len(bad),
            "opa": _opa_bin(), "engine": "opa eval data." + ev["package"] + ".violation"})

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        ev = task.native_evidence
        submitted = submitted_state_or_answer
        if isinstance(submitted, dict) and "repaired_manifest" in submitted:
            verdict = _eval_violations(ev["regos"], ev["package"],
                                       submitted["repaired_manifest"], ev["parameters"])
            return NativeVerdict(valid=verdict == [], detail={"residual_violations": verdict})
        return NativeVerdict(valid=False, detail={"reason": "expected repaired_manifest object"})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ev = task.native_evidence
        bad = _eval_violations(ev["regos"], ev["package"], ev["disallowed"], ev["parameters"])
        good = _eval_violations(ev["regos"], ev["package"], ev["allowed"], ev["parameters"])
        differs = bad is not None and good is not None and bad != good
        return SemanticWitness(ok=bool(differs), witness={
            "disallowed_violations": len(bad or []), "allowed_violations": len(good or []),
            "kind": "verdict-differs-on-knowledge-edit"})


PLUGIN = GatekeeperOpaVerifier()
