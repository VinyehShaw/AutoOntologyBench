"""L11 sim-replay worker: physically execute a *submitted* primitive plan in
OmniGibson and check the terminal BDDL goal predicates natively.

This is the submitted-plan analogue of the lane's own reverify CLI
(``benchmark.plugins.verifier.l11.reverify``) and of audit-v3 Gate B
(``bank/audits/aob-20000-audit-v3/l11/scripts/audit_v3.py``): fresh restore of
the pre-sampled BEHAVIOR-1K task instance, symbolic primitive execution via
the lane's shared ``l11_exec.execute_plan``, then a native goal check. The
robot/controller configuration (28-DOF R1Pro ``RESET_JOINT_POS``,
``CONTROLLERS``, ``env_config``) is imported from the lane's reverify module
so the replay conditions are byte-identical to the audited ones.

Must be run with the OmniGibson interpreter, GPU-pinned (shared cluster:
GPU 6/7 only), headless::

  OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=6 OMNIGIBSON_HEADLESS=1 \
  OMNIGIBSON_DATA_PATH=$PWD/.envs/omnigibson-data \
  OMNIGIBSON_APPDATA_PATH=$PWD/.envs/omnigibson-appdata \
  .envs/omnigibson/bin/python -m benchmark.eval_20k.adapters.l11_sim_eval \
      --oracle <oracle_payload.json> --plan <submitted_plan.json> \
      --out <result.json>

Exit codes: 0 = evaluation completed (verdict JSON written to --out; check
its ``valid`` field); 1 = engine/restore failure (no verdict or verdict with
``engine_error``); 2 = EULA not accepted by the caller; 42 = wall-clock
timeout. The caller's EULA acceptance is never set by this tool.

No simulator imports at module load.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Lane's own replay constants (import-safe: reverify.py is stdlib-only at
# module level). Do not duplicate them here; a drifted reset pose is what
# broke audit-v3 once already.
from benchmark.plugins.verifier.l11.reverify import (  # noqa: E402
    CONTROLLERS,
    RESET_JOINT_POS,
    env_config,
)

L11_EXEC_SCRIPTS = REPO / "data/erule/review5/build8000/expansion/lanes/L11/scripts"

KNOWN_PRIMITIVES = (
    "GRASP", "PLACE_INSIDE", "PLACE_ON_TOP", "OPEN", "CLOSE",
    "TOGGLE_ON", "TOGGLE_OFF",
)

ENGINE = "omnigibson-3.9.3+isaacsim-5.1.0.0-replay/1"


def plan_shape_error(plan: object) -> str | None:
    """Format check of a submitted primitive plan (semantics stay native)."""
    if not isinstance(plan, list) or not plan:
        return "primitive_plan must be a non-empty JSON list of primitive calls"
    for i, step in enumerate(plan):
        if not isinstance(step, (list, tuple)) or not step:
            return "step %d must be a non-empty list [PRIMITIVE, obj, ...]" % i
        if not all(isinstance(x, str) for x in step):
            return "step %d must contain only strings" % i
        if step[0] not in KNOWN_PRIMITIVES:
            return "step %d: unknown primitive %r (known: %s)" % (
                i, step[0], ", ".join(KNOWN_PRIMITIVES))
    return None


def _write(out_path: pathlib.Path, verdict: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=1, sort_keys=True))
    print(json.dumps(verdict))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, help="oracle payload JSON")
    parser.add_argument("--plan", required=True,
                        help="submitted primitive_plan JSON (list of calls)")
    parser.add_argument("--out", required=True, help="verdict JSON output path")
    parser.add_argument("--timeout-s", type=int, default=900,
                        help="wall-clock budget for the replay (default 900)")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)

    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in ("yes", "1", "true"):
        print(
            "REFUSED: OMNI_KIT_ACCEPT_EULA is not set. The NVIDIA Omniverse "
            "EULA and the BEHAVIOR-1K data license must be accepted by the "
            "caller; this tool never accepts terms on your behalf.",
            file=sys.stderr,
        )
        return 2

    oracle = json.loads(pathlib.Path(args.oracle).read_text())
    plan = json.loads(pathlib.Path(args.plan).read_text())

    shape_err = plan_shape_error(plan)
    if shape_err is not None:
        _write(out_path, {
            "valid": False,
            "detail": {"reason": "malformed_plan: %s" % shape_err,
                       "engine": ENGINE, "eval_mode": "sim-replay"},
        })
        return 0

    activity = oracle["activity"]
    instance_id = str(oracle["instance_id"])
    scene_model = oracle.get("scene") or "Rs_int"
    seed = oracle.get("seed", 20261031)

    t0 = time.time()
    import random

    import numpy as np
    import torch as th

    import omnigibson as og
    from omnigibson import macros as gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True
    gm.RENDER_VIEWER_CAMERA = False

    def seed_all(s: int) -> None:
        random.seed(s)
        np.random.seed(s)
        th.manual_seed(s)

    seed_all(seed)

    env = og.Environment(configs=env_config(activity, scene=scene_model))
    robot = env.scene.robots[0]
    robot.reload_controllers(CONTROLLERS)
    og.sim.stop()
    robot.base_footprint_link.mass = 250.0
    og.sim.play()

    from omnigibson.tasks.behavior_task import get_presampled_robot_poses
    from omnigibson.utils.python_utils import recursively_convert_to_torch
    from omnigibson.utils.asset_utils import get_task_instance_path

    task = env.task
    scene = env.scene
    robot.reset()

    # Restore the pre-sampled task instance exactly like reverify.py: prefer
    # the oracle-recorded tro_state path, fall back to the asset lookup.
    tro_path = None
    recorded = oracle.get("tro_path")
    if recorded and pathlib.Path(recorded).exists():
        tro_path = recorded
    else:
        tro_name = "%s_task_%s_0_%s_template" % (scene_model, activity, instance_id)
        tro_path = get_task_instance_path(
            scene_model,
            "%s_task_%s_instances/%s-tro_state" % (scene_model, activity, tro_name),
        )
    if tro_path is None or not pathlib.Path(str(tro_path)).exists():
        print("FAIL: tro_state not found for %s/%s instance %s"
              % (scene_model, activity, instance_id), file=sys.stderr)
        og.shutdown()
        return 1
    with open(str(tro_path)) as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for tro_key, state in tro_state.items():
        if tro_key == "robot_poses":
            available_poses = get_presampled_robot_poses(
                state, robot_model=robot.model, pose_key=None)
            robot.set_position_orientation(
                available_poses[0]["position"], available_poses[0]["orientation"],
                frame="scene")
            scene.write_task_metadata(key=tro_key, data=state)
        else:
            obj = task.object_scopes[0][tro_key]
            if obj is not None:
                obj.load_state(state, serialized=False)
    og.sim.update_handles()
    for _ in range(10):
        og.sim.step_physics()
        for _, entity in task.object_scopes[0].items():
            if entity is not None and hasattr(entity, "keep_still"):
                entity.keep_still()
    scene.update_initial_file()

    def goal_check():
        done, status = task.compiled_task.check_goal(
            lambda predicate_name, *entities:
                task._evaluate_predicate(0, predicate_name, *entities))
        return bool(done), {
            "n_satisfied": len(status["satisfied"]),
            "n_unsatisfied": len(status["unsatisfied"]),
        }

    done0, status0 = goal_check()

    sys.path.insert(0, str(L11_EXEC_SCRIPTS))
    from l11_exec import execute_plan

    seed_all(seed)
    step_counter = {"n": 0}

    def _timeout_guard(n):
        step_counter["n"] = n
        if time.time() - t0 > args.timeout_s:
            print("TIMEOUT: exceeded %ds after %d steps" % (args.timeout_s, n),
                  file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(42)

    exec_error = None
    try:
        # step_modes=None on purpose: recorded drop modes belong to the oracle
        # program; a submitted plan executes with the generic semantics
        # (sampled placement, deterministic drop fallback on SAMPLING_ERROR).
        execute_plan(env, robot, plan, record_actions=False,
                     step_modes=None, on_step=_timeout_guard)
    except Exception as exc:  # an inexecutable plan is an invalid submission
        done_f, status_f = goal_check()
        _write(out_path, {
            "valid": False,
            "detail": {
                "reason": "execution_error: %s: %s"
                          % (type(exc).__name__, str(exc)[:300]),
                "engine": ENGINE, "eval_mode": "sim-replay",
                "activity": activity, "instance_id": instance_id,
                "scene": scene_model,
                "goal_initial": status0, "goal_after_plan": status_f,
                "n_steps": step_counter["n"],
                "elapsed_s": round(time.time() - t0, 1),
            },
        })
        og.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    done_final, status_final = goal_check()
    _write(out_path, {
        "valid": bool(done_final),
        "detail": {
            "engine": ENGINE, "eval_mode": "sim-replay",
            "check": "submitted plan physically executed after fresh instance "
                     "restore; terminal BDDL predicates checked natively",
            "activity": activity, "instance_id": instance_id,
            "scene": scene_model,
            "goal_initial": status0,
            "goal_after_plan": status_final,
            "oracle_goal_after_replay": oracle.get("goal_after_replay"),
            "n_steps": step_counter["n"],
            "elapsed_s": round(time.time() - t0, 1),
        },
    })
    exit_code = 0
    og.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
