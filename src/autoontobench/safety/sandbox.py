"""Hard process sandbox for SHACL/SPARQL/DSL execution (owner: W1, G0 gate).

Runs an untrusted child in a *separate process* (never a thread timeout —
Review-5B Implementation §5) with:

- `RLIMIT_AS`   : address-space cap (memory)
- `RLIMIT_CPU`  : CPU seconds (kernel delivers SIGKILL past the hard limit)
- `RLIMIT_FSIZE`: per-file byte cap applied to the captured stdout/stderr files
  (kernel enforces truncation; child receives SIGXFSZ past the cap)
- wall-clock kill: SIGKILL to the whole process group after `max_wall_seconds`
- working-dir isolation: child runs in a fresh empty temp dir
- env scrub: minimal environment unless one is explicitly supplied
- `no_network` is a DECLARED, best-effort flag (env scrub only); true network
  namespace / container isolation is scheduled for M2 and recorded as residual.

Returns a structured `SandboxResult`; never raises for child misbehaviour.
"""
from __future__ import annotations

import os
import resource
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB
DEFAULT_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MiB (host-safe ceiling)

SCRUBBED_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


@dataclass(frozen=True)
class SandboxLimits:
    max_wall_seconds: float = 5.0
    max_cpu_seconds: int = 5
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    no_network: bool = True


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None  # negative => killed by signal N
    stdout: bytes
    stderr: bytes
    truncated: bool
    killed_reason: str | None  # wall_timeout|cpu_timeout|memory_limit|output_limit|None
    wall_seconds: float
    no_network: bool


def _apply_limits(limits: SandboxLimits) -> None:
    """preexec_fn: runs in the child between fork and exec."""
    resource.setrlimit(resource.RLIMIT_AS, (limits.max_memory_bytes,) * 2)
    cpu_soft = max(1, limits.max_cpu_seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_output_bytes,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_sandboxed(
    argv: list[str],
    *,
    limits: SandboxLimits | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute `argv` under hard limits. Captured output is truncated by the kernel."""
    limits = limits or SandboxLimits()
    child_env = dict(SCRUBBED_ENV if env is None else env)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aob-sandbox-") as workdir:
        out_path = Path(workdir) / "stdout.bin"
        err_path = Path(workdir) / "stderr.bin"
        with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
            proc = subprocess.Popen(
                argv,
                stdout=out_f,
                stderr=err_f,
                cwd=workdir,
                env=child_env,
                start_new_session=True,  # own process group => group SIGKILL
                preexec_fn=lambda: _apply_limits(limits),
            )
            killed_reason: str | None = None
            try:
                proc.wait(timeout=limits.max_wall_seconds)
            except subprocess.TimeoutExpired:
                killed_reason = "wall_timeout"
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        wall = time.monotonic() - started
        exit_code = proc.returncode
        stdout = out_path.read_bytes()
        stderr = err_path.read_bytes()

    truncated = (
        len(stdout) >= limits.max_output_bytes or len(stderr) >= limits.max_output_bytes
    )
    if killed_reason is None:
        if exit_code == -signal.SIGXFSZ:
            killed_reason = "output_limit"
            truncated = True
        elif exit_code in (-signal.SIGXCPU, -signal.SIGKILL):
            killed_reason = "cpu_timeout"
        elif exit_code == -signal.SIGSEGV or b"MemoryError" in stderr:
            killed_reason = "memory_limit"
    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        truncated=truncated,
        killed_reason=killed_reason,
        wall_seconds=wall,
        no_network=limits.no_network,
    )
