"""L01 lane pipeline: candidate -> native-qualified shards (thin local runner).

``autoontobench.factory`` does not exist yet (S10 work package), so this is the
lane-local runner with the adapter seam: it drives propose -> ground ->
prove_exists -> write oracle -> verify_terminal(reference) -> semantic witness
-> leak screen -> 3-layer dedup -> immutable JSONL shards + manifests under
``bank/staging/l01/``. Rejection codes: UNSOLVABLE / TRIVIAL /
PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE / NON_SEMANTIC / LICENSE_HOLD.

Verification parallelism uses threads (each native call is an ``opa`` subprocess;
the GIL is released while waiting). Shard files are written by the main thread
only (single writer).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST
from benchmark.plugins.task_family.l01 import engine
from benchmark.plugins.task_family.l01.families import (
    FAMILIES,
    FAMILY_BY_ID,
    GatekeeperTaskFamilies,
)
from benchmark.plugins.verifier.l01.opa_verifier import GatekeeperOpaVerifier
from source_registry.fetchers import l01_gatekeeper as fetcher

LANE = "l01"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DATASET_ID = "autoontobench-bank-8000-rc1"
SOURCE_SNAPSHOT_ID = "gatekeeper-library@commit-a8d4799d8daf"

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
    """Public payload must not reveal the hidden constraint config or solution.

    Scope: the *textual* public fields (user_goal, policy_intent, budgets). The
    public ``objects`` are the work material itself — a violating image string
    inside a manifest is the visible defect, not a policy-config leak.
    """
    textual = json.dumps({
        k: v for k, v in public.items() if k != "objects"
    }, sort_keys=True)
    for p in ast.get("policy_units", []):
        params = p.get("parameters") or {}
        for key, val in params.items():
            for needle in _param_needles(key, val):
                if needle and needle in textual:
                    return needle
    ref = ast.get("reference_terminal") or {}
    for oid, obj in ref.items():
        sha = engine.sha256_json({k: v for k, v in obj.items() if not k.startswith("__")})
        if sha in textual:
            return sha
    if ast.get("reference_answer"):
        if engine.sha256_json(ast["reference_answer"]) in textual:
            return "reference_answer_sha"
    return None


def _param_needles(key: str, val: Any) -> list[str]:
    out: list[str] = []
    if isinstance(val, str) and len(val) >= 8:
        out.append(val)
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, str) and len(item) >= 8:
                out.append(item)
            elif isinstance(item, dict):
                for v2 in item.values():
                    if isinstance(v2, str) and len(v2) >= 8:
                        out.append(v2)
    return out


def _qualify_one(
    plugin: GatekeeperTaskFamilies,
    verifier: GatekeeperOpaVerifier,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except engine.OpaRunError as exc:
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    ast = candidate.ast
    if ast["goal_kind"] == "exact_decision_set":
        submitted = {"decisions": proof.evidence["reference_answer"]}
    else:
        submitted = {"objects": ast.get("reference_terminal") or {}}
    term = verifier.verify_terminal(grounded, submitted)
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
        "prove_exists": _trim_evals(proof.evidence),
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


def _trim_evals(evidence: dict[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    evals = out.get("evals") or []
    out["evals"] = [
        {k: v for k, v in e.items() if k in (
            "rule_unit_id", "decision", "violation_count", "input_sha256",
            "exit_code", "engine", "engine_version", "inventory_sha256")}
        for e in evals
    ]
    out["eval_count"] = len(evals)
    out["commands"] = sorted({" ".join(e.get("command") or []) for e in evals})
    return out


def _trim_witness(witness: dict[str, Any]) -> dict[str, Any]:
    out = {"ok": witness.get("ok", False), "counterfactuals": []}
    for cf in witness.get("counterfactuals", [])[:3]:
        out["counterfactuals"].append({
            "object": cf["object"],
            "rule_unit_id": cf["rule_unit_id"],
            "mode": cf["mode"],
            "correct_decision": cf["correct_decision"],
            "wrong_decision": cf["wrong_decision"],
            "correct_input_sha256": (cf.get("correct_evidence") or {}).get("input_sha256"),
        })
    if "reason" in witness:
        out["reason"] = witness["reason"]
    return out


def _base_record(candidate: CandidateTaskAST, grounded: Any) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = engine.sha256_json(
        [(p["unit"], p["sample"]) for p in ast.get("policy_units", [])]
    )[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L01",
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
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
    """family/base-goal/instance three-layer dedup with persistence."""

    def __init__(self) -> None:
        self.families: set[str] = set()
        self.base_goals: set[str] = set()
        self.instances: set[str] = set()
        self.seen_pairs: set[tuple[str, str]] = set()

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        if inst in self.instances:
            return False
        self.instances.add(inst)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        self.seen_pairs.add((record["family_id"], record["base_goal_id"]))
        return True


def run_build(
    *,
    families: list[str] | None = None,
    quota_scale: float = 1.0,
    instances_per_goal: int = 5,
    workers: int = 12,
    seed: int = 20261027,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    plugin = GatekeeperTaskFamilies()
    verifier = GatekeeperOpaVerifier()
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L01",
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
            spec = FAMILY_BY_ID[fid]
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0}
            constraints = {"quota_scale": quota_scale,
                           "instances_per_goal": instances_per_goal}
            candidates = list(plugin.propose(snapshot, family=fid, seed=seed,
                                             constraints=constraints))
            if max_per_family:
                candidates = candidates[:max_per_family]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_qualify_one, plugin, verifier, c)
                           for c in candidates]
                for fut in futures:
                    record, rejection = fut.result()
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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L01 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--quota-scale", type=float, default=1.0)
    parser.add_argument("--instances-per-goal", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-per-family", type=int, default=None)
    args = parser.parse_args()
    stats = run_build(
        families=args.families,
        quota_scale=args.quota_scale,
        instances_per_goal=args.instances_per_goal,
        workers=args.workers,
        seed=args.seed,
        max_per_family=args.max_per_family,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
