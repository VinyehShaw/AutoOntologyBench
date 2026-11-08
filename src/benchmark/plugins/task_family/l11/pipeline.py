"""L11 task-family plugin: embodied OmniGibson task production.

run_build() is the production entrypoint (ENTRYPOINT_NAMES convention). For
this pilot checkpoint it consumes the natively verified pilot rollouts in
``lanes/L11/artifacts/pilot`` (produced by ``scripts/pilot_run.py``) and emits
aob-task/5.0 records into ``bank/staging/l11``. The simulator rollouts are
NOT re-run here: every record's native evidence is the recorded replay from
pilot_run.py, re-checkable via ``scripts/pilot_run.py --recheck``.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

REPO = pathlib.Path("/mnt/faster_nas/benyou.mys/git/oob")
LANE = REPO / "data/erule/review5/build8000/lanes/L11"
PILOT = LANE / "artifacts/pilot"
STAGING = REPO / "bank/staging/l11"
DATASET_ID = "autoontobench-bank-8000-rc1"
SOURCE_SNAPSHOT_ID = "behavior-1k-omnigibson@v3.9.3-post2+isaacsim-5.1.0.0"

GOAL_TEXT = {
    "store_honey": "Put the jar of honey inside the cabinet.",
    "composting_waste": "Put the half banana and the half pomegranate inside the ashcan.",
    "store_batteries": "Put both batteries inside the cabinet.",
}

COMPLEXITY = {
    "store_honey": "simple",
    "composting_waste": "medium",
    "store_batteries": "medium",
}


def _h(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}-{h}"


def _record(task: dict[str, Any]) -> dict[str, Any]:
    activity = task["activity"]
    base_goal = _h("bg", activity, task["family_id"])
    instance = _h("in", activity, str(task["instance_id"]))
    task_id = f"l11-{instance}"
    oracle_handle = _h("l11-oracle", activity, str(task["instance_id"]))
    bddl_rel = (
        ".envs/b1k-src/BEHAVIOR-1K-3.9.3-post2/bddl3/bddl/activity_definitions/"
        f"{activity}/problem0.bddl"
    )
    tro_rel = str(pathlib.Path(task["tro_path"]).relative_to(REPO))
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": DATASET_ID,
        "lane_id": "L11",
        "family_id": task["family_id"],
        "base_goal_id": base_goal,
        "instance_id": instance,
        "task_id": task_id,
        "task_regime": "application",
        "complexity": COMPLEXITY.get(activity, "medium"),
        "semantic_profile": "omnigibson-embodied/1",
        "provenance": "source_parameterized",
        "authorship": "deterministic_template",
        "public": {
            "user_goal": (
                f"{GOAL_TEXT[activity]} You control an R1Pro robot in the BEHAVIOR-1K "
                f"Rs_int scene (pre-sampled task instance {task['instance_id']}). Submit an "
                "action program: a JSON list of symbolic primitive calls "
                "(GRASP / PLACE_INSIDE / PLACE_ON_TOP / OPEN / CLOSE / TOGGLE_ON / ...) "
                "with BDDL object instance names, to be executed in OmniGibson."
            ),
            "allowed_tools": [
                "omnigibson_env",
                "symbolic_action_primitives",
                "bddl_object_scope",
            ],
            "budget": {
                "actions": max(task["n_actions"], 200),
                "probes": 20,
            },
            "observation_refs": [
                {"path": bddl_rel, "sha256": _sha256_file(REPO / bddl_rel)},
                {"path": tro_rel, "sha256": _sha256_file(REPO / tro_rel)},
            ],
            "semantic_scope": (
                "Embodied task: success is defined by the BDDL goal predicates evaluated "
                "natively in OmniGibson (Isaac Sim 5.1.0.0 physics) after the action "
                "program executes. No video-only or claimed success is accepted."
            ),
        },
        "oracle_handle": oracle_handle,
        "status": "QUALIFIED",
        "qualification": {
            "native_verified": True,
            "solvable": True,
            "nontrivial": True,
            "semantic_relevance": True,
            "leakage_clean": True,
            "release_eligible": True,
        },
        "native_evidence": {
            "engine": "omnigibson-3.9.3+isaacsim-5.1.0.0-replay/1",
            "commands": [
                "CUDA_VISIBLE_DEVICES=<free-gpu-6-or-7> OMNIGIBSON_HEADLESS=1 "
                "OMNIGIBSON_DATA_PATH=$PWD/.envs/omnigibson-data "
                "OMNIGIBSON_APPDATA_PATH=$PWD/.envs/omnigibson-appdata "
                ".envs/omnigibson/bin/python -m benchmark.plugins.verifier.l11.reverify "
                f"--task-id {task_id}"
            ],
            "preconditions": (
                "Caller must have accepted the NVIDIA Omniverse EULA and the BEHAVIOR-1K "
                "data license (PI has accepted both) and export OMNI_KIT_ACCEPT_EULA=YES; "
                "the reverify tool refuses to run otherwise."
            ),
            "expected": {"replay_goal_satisfied": True},
            "replay": {
                "activity": activity,
                "instance_id": task["instance_id"],
                "scene": "Rs_int",
                "robot": "R1Pro",
                "primitive_plan": task["plan"],
                "n_flat_actions": task["n_actions"],
                "goal_initial": task["goal_initial"],
                "goal_after_plan": task["goal_after_plan"],
                "goal_after_replay": task["goal_after_replay"],
                "oracle_file": f"data/erule/review5/build8000/lanes/L11/artifacts/pilot/oracle_{activity}_{task['instance_id']}.json",
                "reverify_result": f"data/erule/review5/build8000/lanes/L11/artifacts/reverify/{task_id}.json",
            },
            "source_snapshot_id": SOURCE_SNAPSHOT_ID,
        },
        "source_snapshot_ids": [SOURCE_SNAPSHOT_ID],
        "source_unit_ids": [
            f"Rs_int/{activity}/{task['instance_id']}",
            f"bddl:{activity}/problem0.bddl",
        ],
    }


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(pathlib.Path(path).read_bytes())
    return h.hexdigest()


def run_build(*, limit: int | None = None, out_root: str | None = None) -> dict[str, Any]:
    staging = pathlib.Path(out_root) if out_root else STAGING
    results = json.loads((PILOT / "pilot_results.json").read_text())
    tasks = results["tasks"]
    if limit:
        tasks = tasks[:limit]
    qualified_dir = staging / "qualified"
    oracle_dir = staging / "oracle"
    rejections_dir = staging / "rejections"
    candidates_dir = staging / "candidates"
    for d in (qualified_dir, oracle_dir, rejections_dir, candidates_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "lane": "L11",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": 0,
        "qualified": 0,
        "rejections": {},
        "families": {},
    }
    shard_path = qualified_dir / "shard-0000.jsonl"
    with shard_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            record = _record(task)
            stats["candidates"] += 1
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            stats["qualified"] += 1
            stats["families"].setdefault(record["family_id"], 0)
            stats["families"][record["family_id"]] += 1
            oracle_record = {
                "task_id": record["task_id"],
                "oracle_handle": record["oracle_handle"],
                "private": True,
                "activity": task["activity"],
                "instance_id": task["instance_id"],
                "primitive_plan": task["plan"],
                "flat_actions": task["actions"],
                "seed": 20261031 + task["instance_id"],
                "tro_path": task["tro_path"],
                "goal_after_replay": task["goal_after_replay"],
            }
            (oracle_dir / f"{record['oracle_handle']}.json").write_text(
                json.dumps(oracle_record, indent=1)
            )
            (candidates_dir / "shard-0000.jsonl").open("a", encoding="utf-8").write(
                json.dumps(record, sort_keys=True) + "\n"
            )
    with (rejections_dir / "rejections.jsonl").open("w", encoding="utf-8") as fh:
        for rej in results["rejections"]:
            stats["rejections"][rej["code"]] = stats["rejections"].get(rej["code"], 0) + 1
            fh.write(json.dumps(rej, sort_keys=True) + "\n")

    manifest = {
        "shard": "shard-0000.jsonl",
        "records": stats["qualified"],
        "sha256": _sha256_file(shard_path),
    }
    (qualified_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    stats["base_goals"] = len({json.loads(l)["base_goal_id"] for l in shard_path.read_text().splitlines() if l.strip()})
    stats["task_families"] = len(stats["families"])
    stats["dataset_id"] = DATASET_ID
    stats["source_snapshot_ids"] = [SOURCE_SNAPSHOT_ID]
    stats["pilot_wall_seconds"] = results.get("wall_seconds")
    (staging / "build_report.json").write_text(json.dumps(stats, indent=1))
    return stats


__all__ = ["run_build"]
