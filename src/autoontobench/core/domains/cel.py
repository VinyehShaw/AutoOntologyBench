"""Kubernetes CEL expression-subset adapter (Review 5B W2/M1 smoke; profile cel-expression-subset/1).

Wraps the pinned cel-go helper binary built from
``data/erule/review5/software/cel/helper/main.go`` (cel-go v0.26.1, go1.27.3).
Scope is deliberately the expression subset — see CEL_STATUS.md for the honest
limits (dyn typing; no authorizer / namespaceObject / schema type-checking).
Every invocation is logged with command + exit code + input digest.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

HELPER = (
    Path(__file__).resolve().parents[3]
    / "data" / "erule" / "review5" / "software" / "cel" / "helper" / "cel_helper"
)
CEL_GO_VERSION = "v0.26.1"
PROFILE = "cel-expression-subset/1"
EVAL_TIMEOUT_S = 30
RESULT_PREFIX = "RESULT_JSON="


class CelRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class CelOutcome:
    ok: bool
    mode: str
    output_type: str | None
    value: Any
    error: str | None
    exit_code: int
    input_sha256: str


def _run(args: list[str], input_sha: str) -> CelOutcome:
    proc = subprocess.run([str(HELPER)] + args, capture_output=True, text=True,
                          timeout=EVAL_TIMEOUT_S)
    line = next((l for l in proc.stdout.splitlines() if l.startswith(RESULT_PREFIX)), None)
    if line is None:
        raise CelRunError("no RESULT_JSON line (exit %d): %s" % (proc.returncode, proc.stderr))
    payload = json.loads(line[len(RESULT_PREFIX):])
    return CelOutcome(
        ok=bool(payload.get("ok")), mode=payload.get("mode", ""),
        output_type=payload.get("type"), value=payload.get("value"),
        error=payload.get("error"), exit_code=proc.returncode, input_sha256=input_sha,
    )


def _write_atomic(path: Path, text: str) -> None:
    """Same-content concurrent writes must not leave a truncated file for a
    racing subprocess: write to a sibling temp file, then os.replace."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def check(expression: str, work_dir: Path) -> CelOutcome:
    """Parse+compile only. exit 0 => compiles under the dyn-typed subset."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    expr_path = work_dir / (hashlib.sha256(expression.encode()).hexdigest()[:16] + ".cel")
    _write_atomic(expr_path, expression)
    return _run(["check", str(expr_path)], hashlib.sha256(expression.encode()).hexdigest())


def evaluate(expression: str, inputs: Dict[str, Any], work_dir: Path) -> CelOutcome:
    """Compile + evaluate with {"object":..., "params":...} dyn inputs."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"expression": expression, "inputs": inputs}, sort_keys=True)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    expr_path = work_dir / (sha[:16] + ".cel")
    input_path = work_dir / (sha[:16] + ".input.json")
    _write_atomic(expr_path, expression)
    _write_atomic(input_path, json.dumps(inputs))
    return _run(["eval", str(expr_path), str(input_path)], sha)
