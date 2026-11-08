"""L11 per-task native reverify CLI.

Loads the stored oracle for one task, restores the pre-sampled BEHAVIOR-1K
task instance in a fresh OmniGibson environment, replays the symbolic action
program, and checks the terminal BDDL goal predicates natively. Exits 0 when
the replay satisfies the goal, 1 otherwise (or on any failure).

Usage (repo root; caller must have accepted the NVIDIA Omniverse EULA and the
BEHAVIOR-1K data license — this tool never accepts terms on your behalf):

  OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 OMNIGIBSON_HEADLESS=1 \
  OMNIGIBSON_DATA_PATH=$PWD/.envs/omnigibson-data \
  OMNIGIBSON_APPDATA_PATH=$PWD/.envs/omnigibson-appdata \
  .envs/omnigibson/bin/python -m benchmark.plugins.verifier.l11.reverify \
      --task-id l11-in-<hash>

No simulator imports at module load (plugin discovery imports this file with
the plain repo venv).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path("/mnt/faster_nas/benyou.mys/git/oob")
# Search order: 20k expansion staging first, then the frozen pilot staging.
# Both hold the same oracle bytes for shared (copied) task ids; new 20k tasks
# exist only in the first root.
STAGING_ROOTS = [
    REPO / "bank/staging_20k/l11",
    REPO / "bank/staging/l11",
]
STAGING = STAGING_ROOTS[0]
RESULT_DIR = REPO / "data/erule/review5/build8000/expansion/lanes/L11/artifacts/reverify"
LEGACY_RESULT_DIR = REPO / "data/erule/review5/build8000/lanes/L11/artifacts/reverify"

RESET_JOINT_POS = [
    0.0, -0.0, 0.0247, 0.0009, 0.0004, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.03, 0.03, 0.03, 0.03,
]

CONTROLLERS = {
    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
    "trunk": {"name": "JointController", "motor_type": "position", "command_input_limits": None, "use_delta_commands": False, "use_impedances": False},
    "arm_left": {"name": "JointController", "motor_type": "position", "command_input_limits": None, "use_delta_commands": False, "use_impedances": False},
    "arm_right": {"name": "JointController", "motor_type": "position", "command_input_limits": None, "use_delta_commands": False, "use_impedances": False},
    "gripper_left": {"name": "JointController", "motor_type": "position", "command_input_limits": None, "use_delta_commands": False, "use_impedances": False},
    "gripper_right": {"name": "JointController", "motor_type": "position", "command_input_limits": None, "use_delta_commands": False, "use_impedances": False},
}


def load_record(task_id: str) -> dict:
    for root in STAGING_ROOTS:
        for shard in sorted((root / "qualified").glob("shard-*.jsonl")):
            for line in shard.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("task_id") == task_id:
                        rec["_staging_root"] = str(root)
                        return rec
    raise SystemExit(f"task_id {task_id} not found in {[str(r) for r in STAGING_ROOTS]}/qualified")


def load_oracle(record: dict) -> dict:
    root = pathlib.Path(record.get("_staging_root", STAGING))
    oracle_path = root / "oracle" / f"{record['oracle_handle']}.json"
    if not oracle_path.exists():
        raise SystemExit(f"oracle not found: {oracle_path}")
    return json.loads(oracle_path.read_text())


def env_config(activity: str, scene: str = "Rs_int") -> dict:
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "automatic_reset": False,
                "flatten_action_space": False, "flatten_obs_space": False, "use_external_obs": False},
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene,
                  "trav_map_resolution": 0.1, "default_erosion_radius": 0.0,
                  "trav_map_with_objects": True, "num_waypoints": 1, "waypoint_resolution": 0.2,
                  "load_task_relevant_only": True, "not_load_object_categories": ["ceilings"],
                  "seg_map_resolution": 1.0, "include_robots": False},
        "robots": [{
            "type": "R1Pro", "obs_modalities": [], "scale": 1.0, "self_collisions": True,
            "action_normalize": False, "action_type": "continuous", "grasping_mode": "sticky",
            "reset_joint_pos": RESET_JOINT_POS, "controller_config": CONTROLLERS,
        }],
        "objects": [],
        "task": {"type": "BehaviorTask", "activity_name": activity, "activity_definition_id": 0,
                 "activity_instance_id": 0, "online_object_sampling": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--timeout-s", type=int, default=175,
                        help="wall-clock budget for the whole reverify (default 170)")
    args = parser.parse_args()

    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in ("yes", "1", "true"):
        print(
            "REFUSED: OMNI_KIT_ACCEPT_EULA is not set. The NVIDIA Omniverse EULA and the "
            "BEHAVIOR-1K data license must be accepted by the caller (PI has accepted both); "
            "this tool never accepts terms on your behalf.",
            file=sys.stderr,
        )
        return 2

    record = load_record(args.task_id)
    oracle = load_oracle(record)
    replay_meta = record.get("native_evidence", {}).get("replay", {})
    activity = oracle.get("activity") or replay_meta.get("activity")
    instance_id = oracle.get("instance_id") or replay_meta.get("instance_id")
    scene_model = oracle.get("scene") or replay_meta.get("scene") or "Rs_int"
    if activity is None or instance_id is None:
        print("FAIL: record/oracle missing activity or instance_id", file=sys.stderr)
        return 1
    plan = oracle["primitive_plan"]
    seed = oracle.get("seed", 20261031)

    t0 = time.time()
    import random

    import numpy as np
    import torch as th

    import omnigibson as og
    from omnigibson import macros as gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True
    # No camera observations are needed for predicate verification; skip the
    # viewer camera (and its render product) entirely.
    gm.RENDER_VIEWER_CAMERA = False

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    env = og.Environment(configs=env_config(activity, scene=scene_model))
    robot = env.scene.robots[0]
    # The pre-sampled task instance re-instantiates the robot with DEFAULT
    # controllers (trunk use_delta=True, IK arms); reload the primitives
    # controller config so q_to_action's JointController assertion holds.
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
    tro_name = f"{scene_model}_task_{activity}_0_{instance_id}_template"
    tro_path = get_task_instance_path(
        scene_model, f"{scene_model}_task_{activity}_instances/{tro_name}-tro_state"
    )
    if tro_path is None:
        print(f"FAIL: tro_state not found for {scene_model}/{activity} instance {instance_id}", file=sys.stderr)
        og.shutdown()
        return 1
    with open(tro_path) as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for tro_key, state in tro_state.items():
        if tro_key == "robot_poses":
            available_poses = get_presampled_robot_poses(state, robot_model=robot.model, pose_key=None)
            robot.set_position_orientation(
                available_poses[0]["position"], available_poses[0]["orientation"], frame="scene"
            )
            scene.write_task_metadata(key=tro_key, data=state)
        else:
            obj = task.object_scopes[0][tro_key]
            if obj is not None:
                obj.load_state(state, serialized=False)
    og.sim.update_handles()
    for _ in range(10):
        og.sim.step_physics()
        for inst, entity in task.object_scopes[0].items():
            if entity is not None and hasattr(entity, "keep_still"):
                entity.keep_still()
    scene.update_initial_file()

    def goal_check():
        done, status = task.compiled_task.check_goal(
            lambda predicate_name, *entities: task._evaluate_predicate(0, predicate_name, *entities)
        )
        return bool(done), {
            "n_satisfied": len(status["satisfied"]),
            "n_unsatisfied": len(status["unsatisfied"]),
        }

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    sys.path.insert(0, str(
        REPO / "data/erule/review5/build8000/expansion/lanes/L11/scripts"
    ))
    from l11_exec import execute_plan

    step_counter = {"n": 0}

    def _timeout_guard(_n):
        step_counter["n"] = _n
        if time.time() - t0 > args.timeout_s:
            print(f"FAIL: exceeded {args.timeout_s}s budget after {_n} steps", file=sys.stderr)
            og.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

    _, _modes = execute_plan(
        env, robot, plan, record_actions=False,
        step_modes=oracle.get("step_modes"), on_step=_timeout_guard,
    )
    n_steps = step_counter["n"]

    done_final, status_final = goal_check()
    elapsed = round(time.time() - t0, 1)

    result = {
        "task_id": args.task_id,
        "activity": activity,
        "instance_id": instance_id,
        "scene": scene_model,
        "goal_initial": {"note": "skipped for time budget; initial unsatisfaction recorded at generation time"},
        "goal_after_replay": status_final,
        "replay_steps": n_steps,
        "elapsed_s": elapsed,
        "verdict": "PASS" if done_final else "FAIL",
        "engine": "omnigibson-3.9.3+isaacsim-5.1.0.0-replay/1",
        "world": "task-relevant subset (load_task_relevant_only, upstream solve_behavior_task pattern)",
    }
    result_dir = RESULT_DIR
    if str(REPO / "bank/staging/l11") == record.get("_staging_root"):
        result_dir = LEGACY_RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{args.task_id}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    exit_code = 0 if done_final else 1
    og.shutdown()
    # og.shutdown() tears down via app.close(), which can call exit(0) and
    # swallow our verdict; force the intended exit code afterwards.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
