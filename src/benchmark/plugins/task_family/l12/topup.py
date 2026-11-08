"""L12 targeted top-up: extend the lane along its declared state axis.

The lane's declared state axis is the per-base-goal instance index: the base
build materialises ``instances_per_goal=5`` deterministic variants per base goal
(``propose.propose`` -> ``range(per_goal)``).  This module adds the *next*
deterministic variants (instance index >= 5) for a minimal, deterministic sweep
of existing base goals, until the requested number of genuinely new variants is
qualified.  No new base goals, families, or paraphrases are introduced, and a
variant whose semantic signature (reference answer + underlying corruption,
with cosmetic synthetic placeholder labels normalised) already exists in the
lane is skipped.

Everything is qualified through the exact same native path as the base build
(``pipeline._qualify_one``: prove_exists -> oracle -> verify_terminal ->
semantic witness -> leak screen).  New records are appended as a fresh shard so
existing shards stay immutable; manifests are recomputed and build_report.json
is refreshed from disk.

Usage:
    PYTHONPATH=. .venv/bin/python -m benchmark.plugins.task_family.l12.topup \
        --target 5 --max-inst 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST
from benchmark.plugins.task_family.l12 import engine, pipeline
from benchmark.plugins.task_family.l12.families import FAMILIES
from benchmark.plugins.task_family.l12.propose import (
    Pm4pyTaskFamilies,
    _GENERATORS,
    _ids,
    _source_units,
)
from benchmark.plugins.verifier.l12.pm4py_verifier import Pm4pyNativeVerifier

STAGING = pipeline.STAGING
NEXT_SHARD = 2  # shard-0000 + shard-0001 already hold the 345/350 base build


def _normalise(value: Any) -> Any:
    """Replace cosmetic synthetic placeholder labels so they do not mask dupes."""
    if isinstance(value, str) and value.startswith("zzz_"):
        return "<synthetic-placeholder>"
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def variant_signature(ast: dict[str, Any]) -> str:
    """Semantic identity of a variant, independent of placeholder wording."""
    kind = ast["goal_kind"]
    if kind == "repair_trace":
        corrupted = _normalise(ast["corrupted_trace"])
        corruption = _normalise(ast["corruption"])
        return engine.sha256_json([
            kind, ast["log_rel"], ast["trace_idx"],
            ast["reference_trace"], corrupted, corruption,
        ])
    if kind == "repair_model":
        return engine.sha256_json([
            kind, ast["log_rel"],
            sorted(ast["reference_mapping"].items()),
        ])
    return engine.sha256_json([
        kind, ast["log_rel"], ast["projection"]["first_n"],
        ast["reference_answer"],
    ])


def _existing_signatures(records: list[dict[str, Any]], oracle_dir: Path) -> set[str]:
    sigs: set[str] = set()
    for record in records:
        path = oracle_dir / ("%s.json" % record["oracle_handle"])
        oracle = json.loads(path.read_text(encoding="utf-8"))
        sigs.add(variant_signature(oracle["ast"]))
    return sigs


def _schema_record_counts() -> dict[str, Any]:
    """Recompute build_report counts from the shards actually on disk."""
    manifest_glob = "*.manifest.json"
    qman = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((STAGING / "qualified").glob(manifest_glob))]
    cman = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((STAGING / "candidates").glob(manifest_glob))]
    qrec = pipeline._iter_qualified()
    crec: list[dict[str, Any]] = []
    for p in sorted((STAGING / "candidates").glob("shard-*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    crec.append(json.loads(line))
    rej_path = STAGING / "rejections" / "rejections.jsonl"
    rej = []
    if rej_path.is_file():
        for line in rej_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rej.append(json.loads(line))
    fams: dict[str, dict[str, int]] = {}
    for r in crec:
        f = fams.setdefault(r["family_id"], {"candidates": 0, "qualified": 0, "rejected": 0})
        f["candidates"] += 1
    for r in qrec:
        fams.setdefault(r["family_id"], {"candidates": 0, "qualified": 0, "rejected": 0})
        fams[r["family_id"]]["qualified"] += 1
    for r in rej:
        fams.setdefault(r["family_id"], {"candidates": 0, "qualified": 0, "rejected": 0})
        fams[r["family_id"]]["rejected"] += 1
    rej_counts: dict[str, int] = {}
    for r in rej:
        rej_counts[r["code"]] = rej_counts.get(r["code"], 0) + 1
    return {
        "candidates": len(crec),
        "qualified": len(qrec),
        "distinct_instances": len({r["instance_id"] for r in qrec}),
        "base_goals": len({r["base_goal_id"] for r in qrec}),
        "families": fams,
        "rejections": rej_counts,
        "qualified_manifests": qman,
        "candidate_manifests": cman,
    }


def run_topup(target: int = 5, max_inst: int = 30) -> dict[str, Any]:
    plugin = Pm4pyTaskFamilies()
    verifier = Pm4pyNativeVerifier()
    oracle_dir = STAGING / "oracle"

    existing = pipeline._iter_qualified()
    signatures = _existing_signatures(existing, oracle_dir)
    public_seen = {json.dumps(r["public"], sort_keys=True) for r in existing}
    instance_seen = {r["instance_id"] for r in existing}

    qwriter = pipeline.ShardWriter(STAGING / "qualified")
    qwriter.shard_idx = NEXT_SHARD
    cwriter = pipeline.ShardWriter(STAGING / "candidates")
    cwriter.shard_idx = NEXT_SHARD
    rejection_path = STAGING / "rejections" / "rejections.jsonl"

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"rejections": {}}
    skipped = 0
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
        for inst in range(5, max_inst):
            if len(accepted) >= target:
                break
            for spec in FAMILIES:
                if len(accepted) >= target:
                    break
                gen = _GENERATORS[spec.family_id]
                for base in spec.bases:
                    if len(accepted) >= target:
                        break
                    payload = gen(spec, base, inst)
                    if payload is None:
                        continue
                    base_goal_id, instance_id = _ids(spec.family_id, dict(base), inst)
                    if instance_id in instance_seen:
                        continue
                    ast = {
                        "goal_kind": spec.goal_kind,
                        "operation": spec.operation,
                        "regime": spec.regime,
                        "generator": "l12-propose/topup",
                        **payload,
                    }
                    sig = variant_signature(ast)
                    if sig in signatures:
                        skipped += 1
                        continue
                    rels = [payload["log_rel"]]
                    if payload.get("net_rel"):
                        rels.append(payload["net_rel"])
                    candidate = CandidateTaskAST(
                        lane_id="L12",
                        family_id=spec.family_id,
                        base_goal_id=base_goal_id,
                        instance_id=instance_id,
                        regime=spec.regime,
                        semantic_profile="pm4py-conformance-offline/1",
                        ast=ast,
                        source_unit_ids=_source_units(rels),
                    )
                    grounded = plugin.ground(candidate)
                    pub = json.dumps(grounded.public, sort_keys=True)
                    if pub in public_seen:
                        skipped += 1
                        continue
                    record, rejection = pipeline._qualify_one(
                        plugin, verifier, candidate
                    )
                    cwriter.add(record)
                    if rejection is not None:
                        rejected.append(rejection)
                        stats["rejections"][rejection["code"]] = (
                            stats["rejections"].get(rejection["code"], 0) + 1)
                        rej_fh.write(json.dumps({
                            "task_id": record["task_id"],
                            "family_id": record["family_id"],
                            "base_goal_id": record["base_goal_id"],
                            **rejection,
                        }, sort_keys=True) + "\n")
                        # an oracle may have been written before the later gates failed
                        opath = oracle_dir / ("%s.json" % record["oracle_handle"])
                        if opath.is_file():
                            opath.unlink()
                        continue
                    signatures.add(sig)
                    public_seen.add(pub)
                    instance_seen.add(instance_id)
                    accepted.append(record)
                    qwriter.add(record)
    qwriter.close()
    cwriter.close()

    counts = _schema_record_counts()
    report_path = STAGING / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.update(counts)
    report["lane"] = "L12"
    report["top_up"] = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": "close the 5-instance quota shortfall left by base-build TRIVIAL rejections",
        "state_axis": "per-base-goal instance index (deterministic variants inst>=5)",
        "target": target,
        "added_qualified": len(accepted),
        "added_rejected": len(rejected),
        "skipped_duplicate_variants": skipped,
        "new_task_ids": [r["task_id"] for r in accepted],
        "wall_seconds": round(time.time() - t0, 2),
    }
    pipeline._write_json_atomic(report_path, report)
    return {
        "added_qualified": len(accepted),
        "added_rejected": len(rejected),
        "skipped_duplicate_variants": skipped,
        "new_task_ids": [r["task_id"] for r in accepted],
        "counts": {k: counts[k] for k in ("candidates", "qualified", "base_goals",
                                          "distinct_instances")},
        "wall_seconds": report["top_up"]["wall_seconds"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L12 state-axis top-up")
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--max-inst", type=int, default=30)
    args = ap.parse_args()
    print(json.dumps(run_topup(target=args.target, max_inst=args.max_inst),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())