#!/usr/bin/env python3
"""Assemble, check, or verify the Core-track benchmark release.

Examples:
    python src/scripts/assemble_core_release.py --check-pack pilot_worlds/world_L04/pack
    python src/scripts/assemble_core_release.py --pack pilot_worlds/world_L04/pack --out release_core
    python src/scripts/assemble_core_release.py --verify --out release_core

The release is a benchmark-only packaging of validated world packs (public task
records + public materials + manifests). It is not the full 19,947-task
Core-track release; see release_core/CORE_RELEASE_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_release.assemble import assemble_release  # noqa: E402
from scripts.core_release.errors import PackContractError, ReleaseError  # noqa: E402
from scripts.core_release.validate import collect_violations  # noqa: E402
from scripts.core_release.verify import verify_release  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        metavar="DIR",
        help="input world pack (repeatable; one per world)",
    )
    parser.add_argument("--out", default="release_core", metavar="DIR", help="release root")
    parser.add_argument(
        "--force", action="store_true", help="replace a non-empty --out directory"
    )
    parser.add_argument(
        "--check-pack",
        action="append",
        default=[],
        metavar="DIR",
        help="validate a pack without writing anything (repeatable)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="verify an existing release tree"
    )
    return parser


def _check_packs(packs: list[str]) -> int:
    status = 0
    for raw in packs:
        violations, world = collect_violations(Path(raw), REPO_ROOT)
        if violations or world is None:
            print(f"FAIL {raw}")
            for violation in violations:
                print(f"  {violation.describe()}")
            status = 1
            continue
        splits = ", ".join(f"{split}={count}" for split, count in world.per_split.items())
        print(f"PASS {raw}: {world.world_id} total={len(world.tasks)} {splits}")
    return status


def _verify(out: str) -> int:
    problems = verify_release(Path(out))
    if problems:
        print(f"FAIL {out}: {len(problems)} integrity problem(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"PASS {out}: manifest, hashes, counts, and no-oracle rule verified")
    return 0


def _assemble(packs: list[str], out: str, force: bool) -> int:
    if not packs:
        print(
            "error: at least one --pack is required (or use --verify/--check-pack)",
            file=sys.stderr,
        )
        return 2
    manifest = assemble_release(
        [Path(raw) for raw in packs], Path(out), REPO_ROOT, force=force
    )
    totals = manifest["totals"]
    print(f"assembled {totals['worlds']} world(s), {totals['tasks']} task(s) -> {out}")
    print(f"root_sha256 {manifest['root_sha256']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_pack:
            return _check_packs(args.check_pack)
        if args.verify:
            return _verify(args.out)
        return _assemble(args.pack, args.out, args.force)
    except (PackContractError, ReleaseError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())