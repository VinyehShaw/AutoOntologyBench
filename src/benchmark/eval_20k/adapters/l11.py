"""L11 (OmniGibson embodied replay) adapter for the AutoOntologyBench 20K
harness.

L11 is not a text-only lane: a submission ``answer`` is an *action program*.

Accepted answer schema (both modes)::

    {"primitive_plan": [["GRASP", "can__of__soda.n.01_1"],
                        ["PLACE_INSIDE", "ashcan.n.01_1"], ...]}

i.e. a JSON object with a ``primitive_plan`` list of symbolic primitive calls
(``GRASP`` / ``PLACE_INSIDE`` / ``PLACE_ON_TOP`` / ``OPEN`` / ``CLOSE`` /
``TOGGLE_ON`` / ``TOGGLE_OFF``) whose remaining elements are BDDL object
instance names from the task scope, exactly as ``public.user_goal`` of every
L11 release row specifies. ``submission.actions`` is ignored by this lane.

Eval modes (``--lane-kw L11.mode=<mode>`` or env ``AOB_L11_MODE``):

* ``plan-match`` (default, CPU-only): the lane's own native verifier
  ``benchmark.plugins.verifier.l11.replay_verifier.OmnigibsonReplayVerifier``
  — the registered NativeVerifierPlugin for L11 — scores a submission by
  exact program equality against the oracle action program, where that oracle
  program was itself natively re-executed in OmniGibson at qualification time
  and satisfied every terminal BDDL predicate (audit-v3 Gate B). Only oracle
  *resolution* is adapted to the 20K mounted oracle store
  (:mod:`.l11_verifier`); the scoring logic is inherited untouched.
  Consequence: a physically-equivalent but different program scores False —
  use ``sim-replay`` for open-ended program evaluation.

* ``sim-replay`` (GPU, restricted to CUDA devices 6/7 by default): the
  submitted plan is *physically executed* in OmniGibson after a fresh restore
  of the pre-sampled task instance, and the terminal BDDL predicates are
  checked natively — the audit-v3 Gate B machinery generalized to arbitrary
  submitted programs (:mod:`.l11_sim_eval`, run as a subprocess under
  ``.envs/omnigibson/bin/python``). Requires: the OmniGibson env, the
  BEHAVIOR-1K data tree, caller-accepted ``OMNI_KIT_ACCEPT_EULA=YES`` (this
  adapter never accepts licenses on the caller's behalf), and enough free
  VRAM on an allowed GPU. If any precondition is missing, ``support()``
  reports NOT_SUPPORTED with the precise blocker — no faked pass.

Runtime: the harness process itself is the plain repo venv (no simulator
imports at adapter import/construction time). ``sim-replay`` delegates to the
pinned OmniGibson interpreter via subprocess.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported, Verdict, _json_safe
from benchmark.eval_20k.oracle import OracleNotFound, OracleStore
from benchmark.eval_20k.submission import Prediction
from benchmark.eval_20k.tasks import TaskRecord

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

MODE_PLAN_MATCH = "plan-match"
MODE_SIM_REPLAY = "sim-replay"
MODES = (MODE_PLAN_MATCH, MODE_SIM_REPLAY)

KNOWN_PRIMITIVES = (
    "GRASP", "PLACE_INSIDE", "PLACE_ON_TOP", "OPEN", "CLOSE",
    "TOGGLE_ON", "TOGGLE_OFF",
)

#: sim-replay defaults (shared cluster: GPU 6/7 only).
DEFAULT_SIM_GPUS = "6,7"
DEFAULT_OMNI_PYTHON = REPO_ROOT / ".envs/omnigibson/bin/python"
DEFAULT_OMNI_DATA = REPO_ROOT / ".envs/omnigibson-data"
DEFAULT_OMNI_APPDATA = REPO_ROOT / ".envs/omnigibson-appdata"
L11_EXEC_SCRIPT = (REPO_ROOT / "data/erule/review5/build8000/expansion/lanes/L11"
                   "/scripts/l11_exec.py")
#: Isaac Sim 5.1 headless needs several GiB just to boot; below this we do
#: not even attempt a launch on the shared cluster.
DEFAULT_MIN_FREE_VRAM_MIB = 8192
DEFAULT_SIM_TIMEOUT_S = 900


def answer_schema_error(answer: Any) -> str | None:
    """Format diagnostics for an L11 answer (annotation only; the lane
    verifier remains the scoring authority)."""
    if not isinstance(answer, dict):
        return "answer must be a JSON object with a 'primitive_plan' list"
    if "primitive_plan" not in answer:
        return "missing required key 'primitive_plan'"
    plan = answer["primitive_plan"]
    if not isinstance(plan, list) or not plan:
        return "'primitive_plan' must be a non-empty list of primitive calls"
    for i, step in enumerate(plan):
        if not isinstance(step, (list, tuple)) or not step:
            return "step %d must be a non-empty list [PRIMITIVE, obj, ...]" % i
        if not all(isinstance(x, str) for x in step):
            return "step %d must contain only strings" % i
        if step[0] not in KNOWN_PRIMITIVES:
            return "step %d: unknown primitive %r (known: %s)" % (
                i, step[0], ", ".join(KNOWN_PRIMITIVES))
    return None


class _SimReplayBackend:
    """GPU backend: run l11_sim_eval as a subprocess per task."""

    backend_id = "l11.sim-replay/omnigibson-subprocess"

    def __init__(self, omni_python: pathlib.Path, data_dir: pathlib.Path,
                 appdata_dir: pathlib.Path, gpu: str, timeout_s: int) -> None:
        self.omni_python = omni_python
        self.data_dir = data_dir
        self.appdata_dir = appdata_dir
        self.gpu = gpu
        self.timeout_s = timeout_s

    def execute(self, oracle_path: str, plan: Any) -> dict[str, Any]:
        """Run the worker; return its verdict dict or raise RuntimeError."""
        with tempfile.TemporaryDirectory(prefix="l11_sim_eval_") as tmp:
            tmp_dir = pathlib.Path(tmp)
            plan_path = tmp_dir / "plan.json"
            out_path = tmp_dir / "verdict.json"
            plan_path.write_text(json.dumps(plan))
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
            env["OMNIGIBSON_HEADLESS"] = "1"
            env["OMNIGIBSON_DATA_PATH"] = str(self.data_dir)
            env["OMNIGIBSON_APPDATA_PATH"] = str(self.appdata_dir)
            cmd = [
                str(self.omni_python), "-m",
                "benchmark.eval_20k.adapters.l11_sim_eval",
                "--oracle", oracle_path, "--plan", str(plan_path),
                "--out", str(out_path), "--timeout-s", str(self.timeout_s),
            ]
            try:
                proc = subprocess.run(
                    cmd, env=env, cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.timeout_s + 900,  # env boot budget on top
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "sim-replay worker timeout after %ds (gpu=%s)"
                    % (self.timeout_s, self.gpu)) from exc
            if out_path.exists():
                return json.loads(out_path.read_text())
            raise RuntimeError(
                "sim-replay worker produced no verdict (rc=%d): %s"
                % (proc.returncode, proc.stderr.decode(errors="replace")[-400:]))


class Adapter(LaneAdapter):
    """20K harness adapter for L11 (OmniGibson embodied replay)."""

    lane_id = "L11"

    def __init__(self, lane_kw: dict[str, str] | None = None) -> None:
        super().__init__(lane_kw)
        mode = (self.lane_kw.get("mode")
                or os.environ.get("AOB_L11_MODE")
                or MODE_PLAN_MATCH).strip().lower()
        self._mode = mode
        if mode not in MODES:
            self._support_error = (
                "L11 adapter: unknown mode %r (lane_kw L11.mode or "
                "AOB_L11_MODE); expected one of %s" % (mode, ", ".join(MODES)))

    # -- construction --------------------------------------------------------

    def build_verifier(self) -> Any:
        if self._mode == MODE_SIM_REPLAY:
            return self._build_sim_backend()
        try:
            from benchmark.eval_20k.adapters.l11_verifier import (
                StoreBackedReplayVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L11 native replay verifier not importable "
                "(benchmark.plugins.verifier.l11.replay_verifier): %s" % exc) from exc
        try:
            return StoreBackedReplayVerifier()
        except Exception as exc:
            raise LaneNotSupported(
                "L11 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    def verifier_id(self, verifier: Any) -> str:
        if isinstance(verifier, _SimReplayBackend):
            return verifier.backend_id
        return super().verifier_id(verifier)

    # -- sim-replay support probe ---------------------------------------------

    def _sim_config(self) -> dict[str, Any]:
        kw = self.lane_kw
        return {
            "omni_python": pathlib.Path(
                kw.get("omnigibson_python")
                or os.environ.get("AOB_L11_OMNI_PYTHON")
                or DEFAULT_OMNI_PYTHON),
            "data_dir": pathlib.Path(
                kw.get("data_dir") or os.environ.get("OMNIGIBSON_DATA_PATH")
                or DEFAULT_OMNI_DATA),
            "appdata_dir": pathlib.Path(
                kw.get("appdata_dir") or os.environ.get("OMNIGIBSON_APPDATA_PATH")
                or DEFAULT_OMNI_APPDATA),
            "gpus": (kw.get("gpu") or os.environ.get("AOB_L11_GPU")
                     or DEFAULT_SIM_GPUS),
            "min_free_mib": int(kw.get("min_free_vram_mib")
                                or os.environ.get("AOB_L11_MIN_FREE_VRAM_MIB")
                                or DEFAULT_MIN_FREE_VRAM_MIB),
            "timeout_s": int(kw.get("timeout_s")
                             or os.environ.get("AOB_L11_SIM_TIMEOUT_S")
                             or DEFAULT_SIM_TIMEOUT_S),
        }

    def _pick_gpu(self, cfg: dict[str, Any]) -> tuple[str | None, str]:
        """Return (gpu_id, blocker_reason): the first allowed GPU with enough
        free VRAM, else (None, precise blocker)."""
        allowed = [g.strip() for g in str(cfg["gpus"]).split(",") if g.strip()]
        try:
            proc = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,memory.free,memory.total",
                 "--format=csv,noheader,nounits", "-i", ",".join(allowed)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, "nvidia-smi probe failed: %s: %s" % (type(exc).__name__, exc)
        if proc.returncode != 0:
            return None, "nvidia-smi -i %s failed: %s" % (
                ",".join(allowed), proc.stderr.decode(errors="replace")[:200])
        free: dict[str, tuple[int, int]] = {}
        for line in proc.stdout.decode().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                free[parts[0]] = (int(parts[1]), int(parts[2]))
        for gpu in allowed:
            if gpu in free and free[gpu][0] >= cfg["min_free_mib"]:
                return gpu, ""
        detail = ", ".join(
            "GPU%s free=%dMiB/%dMiB" % (g, free.get(g, (0, 0))[0], free.get(g, (0, 0))[1])
            for g in allowed)
        return None, (
            "insufficient free VRAM on allowed GPUs (need >= %d MiB): %s — "
            "shared cluster busy; rerun when a GPU drains or pass "
            "L11.gpu/L11.min_free_vram_mib" % (cfg["min_free_mib"], detail))

    def _build_sim_backend(self) -> _SimReplayBackend:
        cfg = self._sim_config()
        problems = []
        if not cfg["omni_python"].is_file():
            problems.append("OmniGibson interpreter missing: %s" % cfg["omni_python"])
        if not cfg["data_dir"].is_dir():
            problems.append("OMNIGIBSON_DATA_PATH missing: %s" % cfg["data_dir"])
        if not L11_EXEC_SCRIPT.is_file():
            problems.append("lane executor missing: %s" % L11_EXEC_SCRIPT)
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in ("yes", "1", "true"):
            problems.append(
                "OMNI_KIT_ACCEPT_EULA not set: the caller must accept the "
                "NVIDIA Omniverse EULA and the BEHAVIOR-1K data license "
                "themselves; the adapter never accepts terms on their behalf")
        if problems:
            raise LaneNotSupported(
                "L11 mode=sim-replay unavailable: " + "; ".join(problems))
        gpu, blocker = self._pick_gpu(cfg)
        if gpu is None:
            raise LaneNotSupported("L11 mode=sim-replay unavailable: %s" % blocker)
        return _SimReplayBackend(cfg["omni_python"], cfg["data_dir"],
                                 cfg["appdata_dir"], gpu, cfg["timeout_s"])

    # -- oracle ---------------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        raise LaneNotSupported(
            "L11 oracle payloads carry no task AST by design (embodied lane); "
            "the adapter dispatches on the replay oracle payload directly")

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        plan = oracle_payload.get("primitive_plan")
        if not isinstance(plan, list) or not plan:
            raise LaneNotSupported(
                "L11 oracle payload exposes no reference primitive_plan")
        return {"primitive_plan": plan}

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """Negative control mirroring audit-v3 mutation m_action_order
        (reversed program), falling back to an object-binding mutation for
        single-step/palindromic plans."""
        plan = oracle_payload.get("primitive_plan")
        if not isinstance(plan, list) or not plan:
            raise LaneNotSupported(
                "L11 oracle payload exposes no reference primitive_plan")
        mutated = list(reversed(plan))
        if mutated == plan:
            step = list(plan[0])
            if len(step) > 1:
                step[1] = str(step[1]) + "__mutated"
            else:
                step = ["GRASP", "nonexistent__object.n.01_0"]
            mutated = [step]
        return {"primitive_plan": mutated}

    # -- scoring ---------------------------------------------------------------

    def evaluate(self, task: TaskRecord, prediction: Prediction,
                 oracle_store: OracleStore) -> Verdict:
        try:
            verifier = self.verifier
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        try:
            oracle_path = str(oracle_store.locate(
                task.lane_id, task.instance_id, task.oracle_handle))
            payload = oracle_store.load(
                task.lane_id, task.instance_id, task.oracle_handle)
        except OracleNotFound as exc:
            return Verdict(status="ORACLE_MISSING", valid=None, reason=str(exc))
        except (OSError, ValueError) as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="oracle payload unreadable: %s" % exc)
        if not isinstance(payload_plan := payload.get("primitive_plan"), list) \
                or not payload_plan:
            return Verdict(
                status="ORACLE_MISSING", valid=None,
                reason="L11 oracle payload %s carries no primitive_plan "
                       "(cannot score an embodied task without its replayed "
                       "oracle program)" % oracle_path,
                oracle_path=oracle_path)
        if self._mode == MODE_SIM_REPLAY:
            return self._evaluate_sim(verifier, payload, oracle_path,
                                      prediction)
        return self._evaluate_plan_match(verifier, task, payload, oracle_path,
                                         prediction)

    def _evaluate_plan_match(self, verifier: Any, task: TaskRecord,
                             payload: dict[str, Any], oracle_path: str,
                             prediction: Prediction) -> Verdict:
        from benchmark.eval_20k.adapters.l11_verifier import ORACLE_PAYLOAD_KEY

        task_dict = {
            "task_id": task.task_id,
            "instance_id": task.instance_id,
            "oracle_handle": task.oracle_handle,
            "native_evidence": {
                "replay": {
                    "activity": payload.get("activity"),
                    "instance_id": payload.get("instance_id"),
                    "scene": payload.get("scene") or "Rs_int",
                },
            },
            ORACLE_PAYLOAD_KEY: payload,
        }
        try:
            result = verifier.verify_terminal(task_dict, prediction.answer)
        except Exception as exc:  # native engine failure is data, not a crash
            return Verdict(status="ERROR", valid=None,
                           reason="verify_terminal raised %s: %s"
                                  % (type(exc).__name__, exc),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=oracle_path)
        if not isinstance(result, dict) or "valid" not in result:
            return Verdict(status="ERROR", valid=None,
                           reason="L11 verifier returned a non-verdict: %r"
                                  % (result,),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=oracle_path)
        detail = dict(result.get("detail") or {})
        detail["eval_mode"] = MODE_PLAN_MATCH
        detail["scoring"] = (
            "exact program equality vs the oracle action program that was "
            "natively replayed in OmniGibson at qualification time "
            "(audit-v3 Gate B); physically-equivalent alternative programs "
            "score False in this mode — use L11.mode=sim-replay for physical "
            "execution of arbitrary submitted plans")
        shape_note = answer_schema_error(prediction.answer)
        if shape_note is not None:
            detail["submission_shape"] = shape_note
        return Verdict(status="SCORED", valid=bool(result.get("valid")),
                       detail=_json_safe(detail),
                       verifier_id=self.verifier_id(verifier),
                       oracle_path=oracle_path)

    def _evaluate_sim(self, backend: _SimReplayBackend, payload: dict[str, Any],
                      oracle_path: str, prediction: Prediction) -> Verdict:
        shape_note = answer_schema_error(prediction.answer)
        if shape_note is not None:
            return Verdict(status="SCORED", valid=False,
                           detail={"eval_mode": MODE_SIM_REPLAY,
                                   "reason": "malformed submission: " + shape_note},
                           verifier_id=self.verifier_id(backend),
                           oracle_path=oracle_path)
        try:
            result = backend.execute(oracle_path,
                                     prediction.answer["primitive_plan"])
        except Exception as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="sim-replay execution failed: %s: %s"
                                  % (type(exc).__name__, exc),
                           verifier_id=self.verifier_id(backend),
                           oracle_path=oracle_path)
        detail = dict(result.get("detail") or {})
        detail.setdefault("eval_mode", MODE_SIM_REPLAY)
        return Verdict(status="SCORED", valid=bool(result.get("valid")),
                       detail=_json_safe(detail),
                       verifier_id=self.verifier_id(backend),
                       oracle_path=oracle_path)


#: Alias matching the per-lane naming convention elsewhere in the harness.
L11Adapter = Adapter

__all__ = ["Adapter", "L11Adapter", "MODE_PLAN_MATCH", "MODE_SIM_REPLAY",
           "answer_schema_error"]
