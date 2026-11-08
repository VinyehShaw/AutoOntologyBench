"""L11 source plugin: pinned BEHAVIOR-1K + OmniGibson embodied stack snapshot.

The snapshot is disk-pinned (no network at plugin time):
- .envs/b1k-src/BEHAVIOR-1K-3.9.3-post2   (source: BDDL defs, OmniGibson 3.9.3)
- .envs/omnigibson-data/behavior-1k-assets (minimal subset of assets 3.9.0,
  extracted via HTTP range requests from the official zipped dataset)
- .envs/omnigibson-data/2026-challenge-task-instances (pre-sampled instances)
- .envs/omnigibson-data/omnigibson-robot-assets (3.8.2)
Units are (activity, tro-instance) pairs usable as embodied task instances.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import SourceSnapshot, SourceUnit

REPO = Path(__file__).resolve().parents[4]
B1K_SRC = REPO / ".envs/b1k-src/BEHAVIOR-1K-3.9.3-post2"
DATA = REPO / ".envs/omnigibson-data"
CH26 = DATA / "2026-challenge-task-instances"
SCENE = "Rs_int"

SOURCE_ID = "behavior-1k-omnigibson"
RELEASE_ID = "v3.9.3-post2+isaacsim-5.1.0.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class OmnigibsonEmbodiedSourcePlugin:
    """SourcePlugin facade (build-order §3.2) for the pinned embodied stack."""

    plugin_id = "l11.behavior-1k-omnigibson"

    def pin_and_index(self, source_config: dict[str, Any]) -> SourceSnapshot:
        missing = []
        for p in [
            B1K_SRC / "OmniGibson/omnigibson",
            B1K_SRC / "bddl3/bddl/activity_definitions",
            DATA / "behavior-1k-assets/scenes" / SCENE,
            DATA / "behavior-1k-assets/objects",
            CH26 / "scenes" / SCENE / "json",
            DATA / "omnigibson-robot-assets/models/r1pro",
            DATA / "omnigibson.key",
        ]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(f"pinned embodied snapshot incomplete, missing: {missing}")
        descriptor = {
            "source_id": SOURCE_ID,
            "official_url": "https://behavior.stanford.edu",
            "download_url_or_api": "https://huggingface.co/datasets/behavior-1k/zipped-datasets",
            "source_type": "embodied_simulation",
            "upstream_version": "BEHAVIOR-1K 3.9.3-post2 / OmniGibson 3.9.3 / Isaac Sim 5.1.0.0",
            "commit_or_release": RELEASE_ID,
            "retrieved_at": "2026-10-31",
            "sha256": "",
            "native_engine": "omnigibson+isaacsim",
            "license_url": "https://behavior.stanford.edu/agreement",
            "redistribution_policy": "assets encrypted, non-commercial academic research, no redistribution",
            "is_complete_snapshot": False,
            "coverage_fraction_if_known": None,
            "attribution": "StanfordVL BEHAVIOR-1K; NVIDIA Isaac Sim",
            "usage_constraints": "minimal asset subset (range-extracted); scenes limited to Rs_int",
        }
        return SourceSnapshot(
            source_id=SOURCE_ID,
            release_id=RELEASE_ID,
            root=DATA,
            descriptor=descriptor,  # type: ignore[arg-type]
        )

    def enumerate_units(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
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
                yield SourceUnit(
                    unit_id=f"{SCENE}/{activity_dir.name}/{instance_id}",
                    kind="embodied_task_instance",
                    payload_ref=str(tro_file),
                    digest=_sha256(tro_file),
                    metadata={
                        "scene": SCENE,
                        "activity": activity_dir.name,
                        "instance_id": instance_id,
                        "bddl_file": str(bddl_file),
                        "bddl_sha256": _sha256(bddl_file),
                    },
                )


__all__ = ["OmnigibsonEmbodiedSourcePlugin", "SOURCE_ID", "RELEASE_ID"]
