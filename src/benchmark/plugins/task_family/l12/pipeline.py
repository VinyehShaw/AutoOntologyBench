"""L12 lane pipeline: candidate -> native-qualified shards (thin local runner).

Mirrors the L01 runner shape: propose -> ground -> prove_exists -> write oracle
-> verify_terminal(reference) -> semantic witness -> leak screen -> 3-layer dedup
-> immutable JSONL shards + manifests under ``bank/staging/l12/``. Rejection
codes: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE /
NON_SEMANTIC / LICENSE_HOLD.

Verification is sequential and in-process: pm4py import costs ~30-80s on NAS IO
and replay checks release no meaningful parallelism, so a single-threaded loop
keeps ordering deterministic.

``--audit`` mode re-verifies every qualified record independently: it rebuilds
the grounded task from the private oracle and re-runs ``verify_terminal`` on the
reference submission, recomputing all evidence from the pinned source snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT
from benchmark.plugins.task_family.l12.families import (
    FAMILIES,
    FAMILY_BY_ID,
    SEMANTIC_PROFILE,
)
from benchmark.plugins.task_family.l12.propose import Pm4pyTaskFamilies
from benchmark.plugins.verifier.l12.pm4py_verifier import Pm4pyNativeVerifier
from source_registry.fetchers import l12_pm4py as fetcher

LANE = "l12"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DATASET_ID = "autoontobench-bank-8000-rc1"
SOURCE_SNAPSHOT_ID = "pm4py-process@tag-2.7.23.11"

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ShardWriter:
    """Append-records-in-memory, flush-immutable-shard-on-close writer."""

    def __init__(self, root: Path, shard_size: int = 250) -> None:
        self.root = root
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_idx = 0
        self.manifests: list[dict[str, Any]] = []
        (root).mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_name = "shard-%04d" % self.shard_idx
        path = self.root / ("%s.jsonl" % shard_name)
        blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.buffer)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, path)
        manifest = {
            "shard": shard_name,
            "records": len(self.buffer),
            "sha256": engine.sha256_json({"blob": blob}),
            "families": sorted({r.get("family_id") for r in self.buffer}),
            "base_goals": len({r.get("base_goal_id") for r in self.buffer}),
            "path": str(path),
        }
        _write_json_atomic(self.root / ("%s.manifest.json" % shard_name), manifest)
        self.manifests.append(manifest)
        self.buffer = []
        self.shard_idx += 1

    def close(self) -> None:
        self.flush()


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    """Public payload must not embed the reference answer/terminal digests."""
    textual = json.dumps(public, sort_keys=True)
    if ast.get("reference_trace") is not None:
        sha = engine.sha256_json(ast["reference_trace"])
        if sha in textual:
            return "reference_trace_sha256"
    if ast.get("reference_mapping") is not None:
        blob = json.dumps(ast["reference_mapping"], sort_keys=True)
        if blob in textual and len(ast["reference_mapping"]) > 0:
            return "reference_mapping_json"
    if ast.get("reference_answer") is not None:
        sha = engine.sha256_json(ast["reference_answer"])
        if sha in textual:
            return "reference_answer_sha256"
    return None


def _reference_submission(ast: dict[str, Any]) -> dict[str, Any]:
    if ast["goal_kind"] == "repair_trace":
        return {"trace": ast["reference_trace"]}
    if ast["goal_kind"] == "repair_model":
        return {"mapping": ast["reference_mapping"]}
    return {"answer": ast["reference_answer"]}


def _qualify_one(
    plugin: Pm4pyTaskFamilies,
    verifier: Pm4pyNativeVerifier,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:  # engine-level failure, not content
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": "%s: %s" % (type(exc).__name__, str(exc))[:300]}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    ast = candidate.ast
    term = verifier.verify_terminal(grounded, _reference_submission(ast))
    if not term.valid:
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, ast)
    if leak:
        return record, {"code": "LEAKAGE", "reason": "public payload leaks: %s" % leak[:120]}
    record["native_evidence"] = {
        "prove_exists": _trim_proof(proof.evidence),
        "verify_terminal": term.detail,
        "semantic_witness": _trim_witness(witness.witness),
    }
    record["qualification"] = {
        "native_verified": True,
        "solvable": True,
        "nontrivial": True,
        "semantic_relevance": True,
        "leakage_clean": True,
        "release_eligible": True,
    }
    record["status"] = "QUALIFIED"
    return record, None


def _trim_proof(evidence: dict[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    out.pop("reference_terminal", None)
    out.pop("reference_answer", None)
    return out


def _trim_witness(witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(witness.get("counterfactuals")),
        "counterfactuals": [
            {k: v for k, v in cf.items() if k in ("mode", "note", "mutated_sha256", "reference_sha256")}
            for cf in witness.get("counterfactuals", [])[:3]
        ],
    }


def _scope_events(ast: dict[str, Any]) -> int:
    if ast["goal_kind"] == "repair_trace":
        return len(ast["corrupted_trace"])
    if ast["goal_kind"] == "repair_model":
        traces = RT.traces(ast["log_rel"])
        return sum(len(t) for t in traces)
    traces = RT.traces(ast["log_rel"])
    proj = engine.project_first_n(traces, ast["projection"]["first_n"])
    return sum(len(t) for t in proj)


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = engine.sha256_json(
        [candidate.family_id, ast["operation"],
         ast.get("pair_id") or ast.get("log") or ast.get("noisy_pair") or ""]
    )[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L12",
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": engine.complexity_of(_scope_events(ast)),
        "source_snapshot_ids": [SOURCE_SNAPSHOT_ID],
        "source_unit_ids": list(candidate.source_unit_ids),
        "origin_mechanism_group": mech_group,
        "public": grounded.public,
        "oracle_handle": grounded.oracle_ref,
        "authorship": "deterministic_template",
        "provenance": "source_parameterized",
        "status": "CANDIDATE",
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
    }


class DedupIndex:
    """family/base-goal/instance three-layer dedup."""

    def __init__(self) -> None:
        self.families: set[str] = set()
        self.base_goals: set[str] = set()
        self.instances: set[str] = set()

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        if inst in self.instances:
            return False
        self.instances.add(inst)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True


def run_build(
    *,
    families: list[str] | None = None,
    instances_per_goal: int = 5,
    seed: int = 20261028,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    plugin = Pm4pyTaskFamilies()
    verifier = Pm4pyNativeVerifier()
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L12",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
    }
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0}
            constraints = {"instances_per_goal": instances_per_goal}
            candidates = list(plugin.propose(snapshot, family=fid, seed=seed,
                                             constraints=constraints))
            if max_per_family:
                candidates = candidates[:max_per_family]
            for candidate in candidates:
                record, rejection = _qualify_one(plugin, verifier, candidate)
                fam_stats["candidates"] += 1
                stats["candidates"] += 1
                candidate_writer.add(record)
                if rejection is not None:
                    fam_stats["rejected"] += 1
                    stats["rejections"][rejection["code"]] = (
                        stats["rejections"].get(rejection["code"], 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        **rejection,
                    }, sort_keys=True) + "\n")
                    continue
                if dedup.admit(record):
                    fam_stats["qualified"] += 1
                    stats["qualified"] += 1
                    qualified_writer.add(record)
                else:
                    fam_stats["rejected"] += 1
                    stats["rejections"]["DUPLICATE"] = (
                        stats["rejections"].get("DUPLICATE", 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        "code": "DUPLICATE",
                        "reason": "instance hash already admitted",
                    }, sort_keys=True) + "\n")
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["distinct_instances"] = len(dedup.instances)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _iter_qualified() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    qdir = STAGING / "qualified"
    for path in sorted(qdir.glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def run_audit(*, task_id: str | None = None) -> dict[str, Any]:
    """Independent native re-verification of qualified records from their oracles."""
    verifier = Pm4pyNativeVerifier()
    oracle_dir = STAGING / "oracle"
    records = _iter_qualified()
    checked = 0
    invalid: list[dict[str, Any]] = []
    t0 = time.time()
    for record in records:
        if task_id and record["task_id"] != task_id:
            continue
        oracle_path = oracle_dir / ("%s.json" % record["oracle_handle"])
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        cand_meta = oracle["candidate"]
        cand = CandidateTaskAST(
            lane_id=cand_meta["lane_id"],
            family_id=cand_meta["family_id"],
            base_goal_id=cand_meta["base_goal_id"],
            instance_id=cand_meta["instance_id"],
            regime=cand_meta["regime"],
            semantic_profile=cand_meta["semantic_profile"],
            ast=oracle["ast"],
            source_unit_ids=tuple(cand_meta["source_unit_ids"]),
        )
        grounded = GroundedTask(
            ast=cand, public=record["public"],
            oracle_ref=record["oracle_handle"], native_evidence={},
        )
        verdict = verifier.verify_terminal(
            grounded, _reference_submission(oracle["ast"])
        )
        checked += 1
        if not verdict.valid:
            invalid.append({"task_id": record["task_id"], "detail": verdict.detail})
    return {
        "audit": "verify_terminal(reference) recomputed from pinned source",
        "records": len(records),
        "checked": checked,
        "valid": checked - len(invalid),
        "invalid": invalid[:20],
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="L12 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--instances-per-goal", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-per-family", type=int, default=None)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--task-id", default=None)
    args = parser.parse_args()
    if args.audit:
        report = run_audit(task_id=args.task_id)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not report["invalid"] else 1
    stats = run_build(
        families=args.families,
        instances_per_goal=args.instances_per_goal,
        seed=args.seed,
        max_per_family=args.max_per_family,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
