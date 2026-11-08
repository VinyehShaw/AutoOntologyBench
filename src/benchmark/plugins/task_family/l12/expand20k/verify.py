"""L12 20K expansion: post-build collision & integrity verification.

Independent re-check (does not trust the build-time gates):
1. id / public-content / signature / content-key / answer-key intersections
   vs the legacy 8K lane must all be empty;
2. in-build uniqueness of task ids, instance ids, public hashes, signatures,
   content keys, answer keys;
3. shard manifests re-verified (sha256 over the exact bytes);
4. oracle files present, private (mode 0600), one per qualified record;
5. per-task check statuses tallied (no fabricated PASS: human_review NOT_RUN);
6. legacy staging untouched (counts still 350/355, no new files).
Writes ``verification_report.json`` into the expansion report dir.
"""
from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.expand20k.compute import answer_key
from benchmark.plugins.task_family.l12.expand20k.registry import (
    DATASET_ID,
    EXPANSION_DIR,
    STAGING_20K,
)
from benchmark.plugins.task_family.l12.pipeline import ShardWriter  # noqa: F401 (convention check)
from benchmark.plugins.task_family.l12.topup import variant_signature

LEGACY_STAGING = STAGING_20K.parent.parent / "staging" / "l12"


def _content_key(ast: dict[str, Any]) -> tuple[Any, ...]:
    kind = ast["goal_kind"]
    if kind == "repair_trace":
        return ("repair", ast.get("net_rel") or ast["log_rel"],
                tuple(ast["corrupted_trace"]))
    if kind == "repair_model":
        return ("relabel", ast["log_rel"], ast["net_rel"],
                tuple(sorted(ast["corrupted_tids"].items())))
    return ("exact", ast["operation"], answer_key(ast["reference_answer"]))


def _load_jsonl(dirpath: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(dirpath.glob("shard-*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def run_verification() -> dict[str, Any]:
    report: dict[str, Any] = {
        "verification": "l12-expand20k-post-build/1",
        "ran_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }

    new_q = _load_jsonl(STAGING_20K / "qualified")
    new_c = _load_jsonl(STAGING_20K / "candidates")
    leg_q = _load_jsonl(LEGACY_STAGING / "qualified")
    leg_c = _load_jsonl(LEGACY_STAGING / "candidates")

    def idx(records: list[dict[str, Any]], oracle_dir: Path | None):
        out: dict[str, Any] = {
            "task_ids": set(), "instance_ids": set(), "public_hashes": set(),
        }
        for r in records:
            out["task_ids"].add(r["task_id"])
            out["instance_ids"].add(r["instance_id"])
            out["public_hashes"].add(engine.sha256_json(r.get("public", {})))
        if oracle_dir is not None:
            sigs, ckeys, akeys = set(), set(), set()
            for p in sorted(oracle_dir.glob("*.json")):
                ast = json.loads(p.read_text(encoding="utf-8"))["ast"]
                sigs.add(variant_signature(ast))
                ckeys.add(_content_key(ast))
                if ast["goal_kind"] == "exact_artifact":
                    akeys.add(answer_key(ast["reference_answer"]))
            out["signatures"] = sigs
            out["content_keys"] = ckeys
            out["answer_keys"] = akeys
        return out

    new = idx(new_q, STAGING_20K / "oracle")
    leg = idx(leg_q + leg_c, LEGACY_STAGING / "oracle")

    collisions = {
        "task_ids": sorted(new["task_ids"] & leg["task_ids"]),
        "instance_ids": sorted(new["instance_ids"] & leg["instance_ids"]),
        "public_hashes": len(new["public_hashes"] & leg["public_hashes"]),
        "signatures": len(new["signatures"] & leg["signatures"]),
        "content_keys": len(new["content_keys"] & leg["content_keys"]),
        "answer_keys": len(new["answer_keys"] & leg["answer_keys"]),
    }
    report["legacy_collisions"] = collisions
    report["legacy_collisions_clean"] = not any(collisions.values())

    uniqueness = {
        "records": len(new_q),
        "task_ids": len(new["task_ids"]),
        "instance_ids": len(new["instance_ids"]),
        "public_hashes": len(new["public_hashes"]),
        "signatures": len(new["signatures"]),
        "content_keys": len(new["content_keys"]),
        "answer_keys": len(new["answer_keys"]),
        "base_goals": len({r["base_goal_id"] for r in new_q}),
        "families": len({r["family_id"] for r in new_q}),
    }
    report["in_build_uniqueness"] = uniqueness
    report["in_build_unique"] = (
        uniqueness["records"] == uniqueness["task_ids"] ==
        uniqueness["instance_ids"] == uniqueness["public_hashes"] ==
        uniqueness["signatures"] == uniqueness["content_keys"])

    manifest_ok, manifest_bad = 0, []
    for sub in ("qualified", "candidates"):
        for mp in sorted((STAGING_20K / sub).glob("*.manifest.json")):
            man = json.loads(mp.read_text(encoding="utf-8"))
            shard = Path(man["path"])
            if not shard.is_file():
                manifest_bad.append(str(mp))
                continue
            blob = shard.read_text(encoding="utf-8")
            if engine.sha256_json({"blob": blob}) != man["sha256"]:
                manifest_bad.append(str(mp))
                continue
            n_rows = sum(1 for line in blob.splitlines() if line.strip())
            if n_rows != man["records"]:
                manifest_bad.append(str(mp))
                continue
            manifest_ok += 1
    report["shard_manifests"] = {"verified": manifest_ok, "bad": manifest_bad}

    oracle_dir = STAGING_20K / "oracle"
    oracle_files = sorted(oracle_dir.glob("*.json")) if oracle_dir.is_dir() else []
    modes = {stat.S_IMODE(p.stat().st_mode) for p in oracle_files}
    handles = {r["oracle_handle"] for r in new_q}
    files_match = {p.stem for p in oracle_files} == handles
    report["oracles"] = {
        "files": len(oracle_files),
        "handles_match": files_match,
        "modes": sorted(modes),
        "private_0600": modes == {0o600},
    }

    status_tally: dict[str, dict[str, int]] = {}
    for r in new_q:
        for k, v in r["checks"].items():
            status_tally.setdefault(k, {})
            status_tally[k][v] = status_tally[k].get(v, 0) + 1
    report["check_status_tally"] = status_tally
    report["human_review_not_run"] = status_tally.get("human_review_status") == {"NOT_RUN": len(new_q)}
    report["all_qualified_native"] = all(
        r["status"] == "QUALIFIED_NATIVE" for r in new_q)
    report["dataset_id_uniform"] = {r["dataset_id"] for r in new_q} == {DATASET_ID}
    report["seed_uniform"] = {r["generation"]["seed"] for r in new_q} == {20261101}

    report["legacy_untouched"] = {
        "legacy_qualified": len(leg_q),
        "legacy_candidates": len(leg_c),
        "legacy_oracles": len(list((LEGACY_STAGING / "oracle").glob("*.json"))),
        "legacy_build_report_qualified": json.loads(
            (LEGACY_STAGING / "build_report.json").read_text(encoding="utf-8")
        )["qualified"],
    }
    report["legacy_untouched_ok"] = (
        len(leg_q) == 350 and len(leg_c) == 355
        and report["legacy_untouched"]["legacy_build_report_qualified"] == 350)

    report["candidates_total"] = len(new_c)
    report["ok"] = (
        report["legacy_collisions_clean"]
        and report["in_build_unique"]
        and not manifest_bad
        and report["oracles"]["files"] == len(new_q)
        and report["oracles"]["private_0600"]
        and report["oracles"]["handles_match"]
        and report["human_review_not_run"]
        and report["all_qualified_native"]
        and report["dataset_id_uniform"]
        and report["legacy_untouched_ok"])

    out = EXPANSION_DIR / "verification_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run_verification()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("check_status_tally",)}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
