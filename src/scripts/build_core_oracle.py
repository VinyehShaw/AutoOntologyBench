#!/usr/bin/env python3
"""Build the private Core-track answer oracle from world packs.

The public ``release_core`` intentionally contains no labels. This script joins
every public row to its private pack record and writes a minimal private JSONL
oracle keyed by ``release_task_id``. It is an evaluator-side tool and must not
be shipped in the public dataset bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {
    "dev": "dev_labels.jsonl",
    "hidden_iid": "hidden_iid.jsonl",
    "hidden_comp": "hidden_comp.jsonl",
    "hidden_transfer": "hidden_transfer.jsonl",
}


def _jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no}: row is not an object")
        yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        default=None,
        type=Path,
        help="public release root (default: data/core, else release_core)",
    )
    parser.add_argument("--packs", default="pilot_worlds", type=Path)
    parser.add_argument(
        "--out",
        default="release_candidates/AutoOntologyBench-Core-private",
        type=Path,
    )
    args = parser.parse_args(argv)
    release = args.release or (
        Path("data/core") if Path("data/core/tasks/tasks.jsonl").is_file() else Path("release_core")
    )

    public_rows = list(_jsonl(release / "tasks/tasks.jsonl"))
    by_task: dict[tuple[str, str], dict] = {}
    for row in public_rows:
        by_task[(str(row["world_id"]), str(row["task_id"]))] = row

    output: list[dict] = []
    missing: list[str] = []
    for world_dir in sorted(args.packs.glob("world_L*_scaleup")):
        world_id = world_dir.name.replace("_scaleup", "")
        pack = world_dir / "pack"
        if not (pack / "manifests/difficulty.json").is_file():
            continue
        for split, filename in SPLIT_FILES.items():
            path = pack / "private" / filename
            if not path.is_file():
                raise SystemExit(f"missing private split: {path}")
            for private in _jsonl(path):
                task_id = str(private.get("task_id"))
                public = by_task.get((world_id, task_id))
                if public is None or public.get("split") != split:
                    missing.append(f"{world_id}:{task_id}:{split}")
                    continue
                outcomes = private.get("outcomes")
                if not isinstance(outcomes, list) or not outcomes:
                    raise SystemExit(f"{path}:{task_id}: outcomes missing")
                output.append(
                    {
                        "release_task_id": f"{world_id}:{task_id}",
                        "world_id": world_id,
                        "task_id": task_id,
                        "split": split,
                        "expected": private.get("expected"),
                        "expected_ok": private.get("expected_ok"),
                        "expected_reason": private.get("expected_reason"),
                        "outcomes": outcomes,
                    }
                )

    if missing:
        raise SystemExit(f"public/private join missing {len(missing)} rows: {missing[:5]}")
    if len(output) != len(public_rows):
        raise SystemExit(
            f"oracle rows {len(output)} != public rows {len(public_rows)}"
        )
    output.sort(key=lambda row: row["release_task_id"])

    oracle_dir = args.out / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = oracle_dir / "core_oracle.jsonl"
    with oracle_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = _sha256(oracle_path)
    (oracle_dir / "core_oracle.jsonl.sha256").write_text(
        f"{digest}  core_oracle.jsonl\n", encoding="utf-8"
    )
    print(f"oracle: {oracle_path}")
    print(f"rows: {len(output)}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())