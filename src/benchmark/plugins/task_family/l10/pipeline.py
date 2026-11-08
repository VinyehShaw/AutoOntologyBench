"""L10 lane pipeline: candidate -> native-qualified shards (thin local runner).

Mirrors the L12/L13 runner shape: propose -> ground -> prove_exists -> write
oracle -> verify_terminal(reference) -> semantic witness -> leak screen ->
3-layer dedup -> immutable JSONL shards + manifests under
``bank/staging/l10/``. Rejection codes: UNSOLVABLE / TRIVIAL /
PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE / NON_SEMANTIC / LICENSE_HOLD.

Verification is in-process and deterministic: every verdict is recomputed from
the pinned BEHAVIOR-1K snapshot bytes by two independent legs (the lane's own
parser/evaluator and the vendored upstream ``bddl`` package). ``--recheck``
re-runs the full qualification gate over every qualified record and fails on
any drift; ``--recheck-task <id>`` is the native_evidence command.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from benchmark.plugins.task_family.l10.engine import sha256_json
from benchmark.plugins.task_family.l10.families import (
    FAMILIES,
    FAMILY_BY_ID,
    INSTANCES_PER_GOAL,
)
from benchmark.plugins.task_family.l10.propose import BehaviorBddlTaskFamilies
from benchmark.plugins.verifier.l10.bddl_verifier import (
    ENGINE,
    SOURCE_SNAPSHOT_ID,
    BddlNativeVerifier,
)
from source_registry.fetchers import l10_behavior_bddl as fetcher

LANE = "l10"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DATASET_ID = "autoontobench-bank-8000-rc1"

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

    def __init__(self, root: Path, shard_size: int = 250) -> None:
        self.root = root
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_idx = 0
        self.manifests: list[dict[str, Any]] = []
        root.mkdir(parents=True, exist_ok=True)

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
        import hashlib
        manifest = {
            "shard": shard_name,
            "records": len(self.buffer),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
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


_REPAIR_KINDS = {
    "repair_predicate", "repair_arity", "repair_binding",
    "repair_init_predicate", "repair_quantifier_category",
}


def _gold_expression(ast: dict[str, Any]) -> str | None:
    """Serialized gold expression at the defect position (repair kinds only).

    Single-token fragments are public vocabulary (the domain predicate table
    is public knowledge); the leak screen instead targets the full gold
    expression, which must never appear verbatim in the public payload.
    """
    from benchmark.plugins.task_family.l10 import textgen
    from benchmark.plugins.task_family.l10.public_render import load_problem
    kind = ast["goal_kind"]
    if kind not in _REPAIR_KINDS:
        return None
    problem, _text = load_problem(ast["problem_rel"])
    params = ast["params"]
    if kind in ("repair_predicate", "repair_arity", "repair_binding"):
        leaves = textgen.goal_leaf_positions(tuple(problem.goal))
        _path, leaf = leaves[params["leaf_index"] % len(leaves)]
        return textgen.sexpr(leaf)
    if kind == "repair_init_predicate":
        inits = textgen.positive_init_atoms(problem)
        return textgen.sexpr(inits[params["init_index"] % len(inits)])
    quants = textgen.quantifier_positions(tuple(problem.goal))
    _p, node = quants[params["quant_index"] % len(quants)]
    head = node[0]
    iter_idx = 2 if head in ("forn", "fornpairs") else 1
    return textgen.sexpr(node[iter_idx])


def _leak_screen(public: dict[str, Any], ast: dict[str, Any],
                 reference: dict[str, Any]) -> str | None:
    textual = json.dumps(public, sort_keys=True)
    ref_blob = json.dumps(reference, sort_keys=True)
    if len(ref_blob) > 60 and ref_blob in textual:
        return "reference_submission_json"
    gold_expr = _gold_expression(ast)
    if gold_expr is not None and gold_expr in textual:
        return "gold_expression_verbatim"
    return None


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = sha256_json(
        [candidate.family_id, ast["goal_kind"], ast["problem_rel"],
         ast.get("variant_key", "")])[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L10",
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


def _qualify_one(
    plugin: BehaviorBddlTaskFamilies,
    verifier: BddlNativeVerifier,
    candidate: CandidateTaskAST,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "%s: %s" % (type(exc).__name__, str(exc))[:300]}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    try:
        reference = verifier.reference_submission(candidate.ast)
    except Exception as exc:
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "reference: %s: %s" % (
                            type(exc).__name__, str(exc))[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    term = verifier.verify_terminal(grounded, reference)
    if not term.valid:
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, candidate.ast, reference)
    if leak:
        return record, {"code": "LEAKAGE",
                        "reason": "public payload leaks: %s" % leak[:120]}
    record["native_evidence"] = {
        "engine": ENGINE,
        "source_snapshot_id": SOURCE_SNAPSHOT_ID,
        "commands": [
            ".venv/bin/python -m benchmark.plugins.task_family.l10.pipeline "
            "--recheck-task %s" % record["task_id"],
        ],
        "expected": {"ok": True, "proof": True, "terminal": True, "witness": True},
        "prove_exists": _trim(proof.evidence),
        "verify_terminal": term.detail,
        "semantic_witness": _trim_witness(witness.witness),
    }
    record["qualification"] = {
        "native_verified": True, "solvable": True, "nontrivial": True,
        "semantic_relevance": True, "leakage_clean": True,
        "release_eligible": True,
    }
    record["status"] = "QUALIFIED"
    return record, None


def _trim(evidence: dict[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    out.pop("reference_terminal", None)
    return out


def _trim_witness(witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": witness.get("wrong_verdict_valid") is False,
        "counterfactual_kind": witness.get("counterfactual_kind"),
        "wrong_submission": witness.get("wrong_submission"),
    }


class DedupIndex:

    def __init__(self) -> None:
        self.families: set[str] = set()
        self.base_goals: set[str] = set()
        self.instances: set[str] = set()
        self.content: set[str] = set()

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        content = sha256_json([
            record["family_id"], record["public"].get("task_material", {})])
        if inst in self.instances or content in self.content:
            return False
        self.instances.add(inst)
        self.content.add(content)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True


def run_build(
    *,
    families: list[str] | None = None,
    seed: int = 20261027,
    base_goals: dict[str, int] | None = None,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    plugin = BehaviorBddlTaskFamilies()
    verifier = BddlNativeVerifier(STAGING / "oracle")
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L10",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
    }
    t0 = time.time()
    with rejection_path.open("w", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            spec = FAMILY_BY_ID[fid]
            quota = (base_goals or {}).get(fid, spec.base_goals)
            constraints = {
                "instances_per_goal": INSTANCES_PER_GOAL,
                "base_goals": quota,
            }
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "base_goals": 0}
            fam_base_goals: set[str] = set()
            candidates = list(plugin.propose(
                snapshot, family=fid, seed=seed, constraints=constraints))
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
                    fam_base_goals.add(record["base_goal_id"])
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
                        "reason": "instance or content hash already admitted",
                    }, sort_keys=True) + "\n")
            fam_stats["base_goals"] = len(fam_base_goals)
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["task_families"] = len(stats["families"])
    stats["distinct_instances"] = len(dedup.instances)
    stats["dataset_id"] = DATASET_ID
    stats["source_snapshot_ids"] = [SOURCE_SNAPSHOT_ID]
    stats["seed"] = seed
    stats["instances_per_goal"] = INSTANCES_PER_GOAL
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _iter_qualified() -> Iterator[dict[str, Any]]:
    for path in sorted((STAGING / "qualified").glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def run_recheck(*, task_id: str | None = None) -> dict[str, Any]:
    verifier = BddlNativeVerifier(STAGING / "oracle")
    checked = 0
    mismatches: list[dict[str, Any]] = []
    t0 = time.time()
    for record in _iter_qualified():
        if task_id and record["task_id"] != task_id:
            continue
        oracle = verifier.load_oracle(record["oracle_handle"])
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
            oracle_ref=record["oracle_handle"], native_evidence={})
        proof = verifier.prove_exists(grounded)
        reference = verifier.reference_submission(oracle["ast"])
        term = verifier.verify_terminal(grounded, reference)
        witness = verifier.certify_semantic_relevance(grounded)
        res = {
            "task_id": record["task_id"], "family_id": record["family_id"],
            "proof": proof.exists, "terminal": term.valid,
            "witness": witness.ok,
            "ok": proof.exists and term.valid and witness.ok,
        }
        checked += 1
        if not res["ok"]:
            res["proof_reason"] = proof.reason
            res["terminal_detail"] = term.detail
            res["witness_detail"] = witness.witness
            mismatches.append(res)
        if task_id:
            return res
    return {
        "recheck": "full qualification gate recomputed from pinned snapshot",
        "checked": checked,
        "mismatches": mismatches[:20],
        "ok": not mismatches,
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="L10 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-per-family", type=int, default=None)
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--recheck-task", default=None)
    args = parser.parse_args()
    if args.recheck_task:
        result = run_recheck(task_id=args.recheck_task)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("ok") else 1
    if args.recheck:
        report = run_recheck()
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["ok"] else 1
    stats = run_build(
        families=args.families,
        seed=args.seed,
        max_per_family=args.max_per_family,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
