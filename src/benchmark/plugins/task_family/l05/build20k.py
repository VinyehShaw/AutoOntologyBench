"""Build the L05 20K expansion shards (250 new tasks, new mechanisms).

Writes only ``bank/staging_20k/l05``; the legacy ``bank/staging/l05`` bank is
never opened for writing. Uses the plus-family table (descriptor bands, dual
count ranges, parent-minus-descendants, and three new repair modes) with the
20K dataset id and seed.

CLI (repo root, ``.venv/bin/python``)::

    .venv/bin/python -m benchmark.plugins.task_family.l05.build20k \
      --index external_knowledge/indexes/rdkit-molecules/rdkit-Release_2026_09_1+chebi-rel255 \
      --out bank/staging_20k/l05 --seed 20261101 --target 250
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_registry.fetchers.l05_rdkit import RELEASE_ID

from benchmark.plugins.task_family.l05.build import build_shards, write_build_report
from benchmark.plugins.task_family.l05.plus_families import PLUS_FAMILY_TABLE

DATASET_ID_20K = "autoontobench-bank-20000-candidate-dev1"
PROFILE = "rdkit-smarts-native/1"
BANK_REL_20K = "bank/staging_20k/l05"
QUALIFIED_TARGET = 250
LEGACY_COUNT = 1050
TARGET_TOTAL = 1300


def quota() -> dict:
    return {
        "qualified_instances": QUALIFIED_TARGET,
        "legacy_qualified": LEGACY_COUNT,
        "target_total": TARGET_TOTAL,
        "new_needed": QUALIFIED_TARGET,
        "task_families": len(PLUS_FAMILY_TABLE),
        "seed": 20261101,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L05 20K expansion shards")
    ap.add_argument("--index", default=str(
        Path("external_knowledge/indexes/rdkit-molecules") / RELEASE_ID))
    ap.add_argument("--out", default=BANK_REL_20K)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--target", type=int, default=QUALIFIED_TARGET)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    stats = build_shards(
        Path(args.index), Path(args.out), seed=args.seed,
        shard_size=args.shard_size, resume=not args.no_resume,
        family_table=PLUS_FAMILY_TABLE, dataset_id=DATASET_ID_20K,
        profile=PROFILE, bank_rel=BANK_REL_20K, qualified_target=args.target)
    report = write_build_report(
        Path(args.out), seed=args.seed, index_dir=Path(args.index),
        wall_seconds=stats["wall_seconds"], dataset_id=DATASET_ID_20K,
        profile=PROFILE, quota=quota())
    stats["build_report"] = report
    print(json.dumps({k: stats[k] for k in ("candidates", "qualified",
                                            "rejections", "wall_seconds")},
                     indent=1, sort_keys=True))
    return 0 if stats["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())