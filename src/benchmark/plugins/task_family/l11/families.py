"""L11 task-family §3.2 facade: embodied OmniGibson (activity, instance) pairs.

Exposes the TaskFamilyPlugin surface (supported_profiles / propose / ground)
for plugin discovery and factory dispatch. The production batch entrypoint
remains ``pipeline.run_build`` (consumes natively verified pilot rollouts);
this facade proposes (activity, tro-instance) candidates from the pinned
source snapshot and grounds them with the per-task reverify evidence command.
No simulator imports at module load time (discovery imports this file with
the plain repo venv).
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask, SourceSnapshot
from benchmark.plugins.source.l11.omnigibson_embodied import (
    B1K_SRC,
    CH26,
    SCENE,
    SOURCE_ID,
    RELEASE_ID,
)

SEMANTIC_PROFILE = "omnigibson-embodied/1"
LANE_ID = "L11"
DATASET_ID = "autoontobench-bank-8000-rc1"
SOURCE_SNAPSHOT_ID = f"{SOURCE_ID}@{RELEASE_ID}"

REPO = pathlib.Path("/mnt/faster_nas/benyou.mys/git/oob")

FAMILY_ACTIVITIES: dict[str, list[str]] = {
    "l11-f01-embodied-stow-single": ["store_honey"],
    "l11-f02-embodied-stow-multi": ["store_batteries"],
    "l11-f03-embodied-dispose": ["composting_waste"],
    "l11-f04-embodied-assemble": ["installing_a_modem", "make_cabinet_doors"],
    "l11-f05-embodied-pack": ["make_gift_bags_for_baby_showers"],
}

GOAL_TEXT = {
    "store_honey": "Put the jar of honey inside the cabinet.",
    "composting_waste": "Put the half banana and the half pomegranate inside the ashcan.",
    "store_batteries": "Put both batteries inside the cabinet.",
    "installing_a_modem": "Set up the modem next to the TV on the cabinet and switch it on.",
    "make_cabinet_doors": "Attach the cabinet door to the cabinet base.",
    "make_gift_bags_for_baby_showers": "Put the gift bags on the coffee table and fill them with the toy dice and wafers.",
}

COMPLEXITY = {
    "store_honey": "simple",
    "composting_waste": "medium",
    "store_batteries": "medium",
}


def _h(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _iter_units() -> Iterator[dict[str, Any]]:
    bddl_root = B1K_SRC / "bddl3/bddl/activity_definitions"
    json_root = CH26 / "scenes" / SCENE / "json"
    for activity_dir in sorted(bddl_root.iterdir()):
        bddl_file = activity_dir / "problem0.bddl"
        if not bddl_file.exists():
            continue
        tro_dir = json_root / f"{SCENE}_task_{activity_dir.name}_instances"
        if not tro_dir.is_dir():
            continue
        for tro_file in sorted(tro_dir.glob("*-tro_state.json")):
            stem = tro_file.name.replace("-tro_state.json", "")
            try:
                instance_id = int(stem.split("_")[-2])
            except (ValueError, IndexError):
                continue
            yield {
                "activity": activity_dir.name,
                "instance_id": instance_id,
                "bddl_file": bddl_file,
                "tro_file": tro_file,
            }


class OmnigibsonEmbodiedTaskFamilies:
    """TaskFamilyPlugin for lane L11 (OmniGibson embodied)."""

    plugin_id = "l11.behavior-1k-omnigibson-families"

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def propose(
        self,
        snapshot: SourceSnapshot,
        *,
        family: str,
        seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        activities = FAMILY_ACTIVITIES.get(family)
        if not activities:
            return
        per_goal = int(constraints.get("instances_per_goal", 5))
        base_quota = int(constraints.get("base_goals", 1))
        made_bases = 0
        for activity in activities:
            if made_bases >= base_quota:
                break
            pool = [u for u in _iter_units() if u["activity"] == activity]
            if len(pool) < per_goal:
                continue
            start = seed % (len(pool) - per_goal + 1)
            instances = pool[start : start + per_goal]
            base_goal_id = _h("bg", activity, family)
            for unit in instances:
                instance_id = _h("in", activity, str(unit["instance_id"]))
                ast = {
                    "activity": activity,
                    "instance_id": unit["instance_id"],
                    "scene": SCENE,
                    "robot": "R1Pro",
                    "bddl_rel": str(unit["bddl_file"].relative_to(REPO)),
                    "tro_rel": str(unit["tro_file"].relative_to(REPO)),
                }
                yield CandidateTaskAST(
                    lane_id=LANE_ID,
                    family_id=family,
                    base_goal_id=base_goal_id,
                    instance_id=instance_id,
                    regime="application",
                    semantic_profile=SEMANTIC_PROFILE,
                    ast=ast,
                    source_unit_ids=(
                        f"{SCENE}/{activity}/{unit['instance_id']}",
                        f"bddl:{activity}/problem0.bddl",
                    ),
                )
            made_bases += 1

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        activity = ast["activity"]
        task_id = f"l11-{candidate.instance_id}"
        oracle_handle = _h("l11-oracle", activity, str(ast["instance_id"]))
        bddl_path = REPO / ast["bddl_rel"]
        tro_path = REPO / ast["tro_rel"]
        command = (
            "CUDA_VISIBLE_DEVICES=<free-gpu-6-or-7> OMNIGIBSON_HEADLESS=1 "
            "OMNIGIBSON_DATA_PATH=$PWD/.envs/omnigibson-data "
            "OMNIGIBSON_APPDATA_PATH=$PWD/.envs/omnigibson-appdata "
            f".envs/omnigibson/bin/python -m benchmark.plugins.verifier.l11.reverify "
            f"--task-id {task_id}"
        )
        public = {
            "user_goal": (
                f"{GOAL_TEXT[activity]} You control an R1Pro robot in the BEHAVIOR-1K "
                f"{SCENE} scene (pre-sampled task instance {ast['instance_id']}). Submit an "
                "action program: a JSON list of symbolic primitive calls "
                "(GRASP / PLACE_INSIDE / PLACE_ON_TOP / OPEN / CLOSE / TOGGLE_ON / ...) "
                "with BDDL object instance names, to be executed in OmniGibson."
            ),
            "allowed_tools": [
                "omnigibson_env",
                "symbolic_action_primitives",
                "bddl_object_scope",
            ],
            "budget": {"actions": 200, "probes": 20},
            "observation_refs": [
                {"path": ast["bddl_rel"], "sha256": _sha256_file(bddl_path)},
                {"path": ast["tro_rel"], "sha256": _sha256_file(tro_path)},
            ],
            "semantic_scope": (
                "Embodied task: success is defined by the BDDL goal predicates evaluated "
                "natively in OmniGibson (Isaac Sim 5.1.0.0 physics) after the action "
                "program executes. No video-only or claimed success is accepted."
            ),
        }
        native_evidence = {
            "engine": "omnigibson-3.9.3+isaacsim-5.1.0.0-replay/1",
            "commands": [command],
            "expected": {"replay_goal_satisfied": True},
            "source_snapshot_id": SOURCE_SNAPSHOT_ID,
        }
        return GroundedTask(
            ast=candidate,
            public=public,
            oracle_ref=oracle_handle,
            native_evidence=native_evidence,
        )


PLUGIN = OmnigibsonEmbodiedTaskFamilies()

__all__ = ["OmnigibsonEmbodiedTaskFamilies", "PLUGIN", "SEMANTIC_PROFILE", "FAMILY_ACTIVITIES"]
