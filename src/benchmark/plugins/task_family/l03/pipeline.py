"""L03 lane pipeline: candidates -> native-qualified shards (local runner).

Drives propose -> ground -> prove_exists -> write oracle -> verify_terminal
(reference replay) -> semantic witness -> leak screen -> instance dedup ->
immutable JSONL shards + sha256 manifests under ``bank/staging/l03/``.
Rejection codes: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE /
DUPLICATE / NON_SEMANTIC / LICENSE_HOLD.

Verification is in-process but never trusts stored state: every verdict is
recomputed from the pinned Camunda XML by ``L03NativeVerifier`` (digest
re-checked per decision). ``--recheck`` re-runs the full qualification gate
over every qualified record; ``--recheck-task`` recomputes one record and is
the native-evidence command embedded in each task record.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.task_family.l03.families import FAMILIES, FAMILY_BY_ID
from benchmark.plugins.task_family.l03.plugin import (
    DATASET_ID,
    FAMILY_QUOTA,
    L03TaskFamilies,
)
from benchmark.plugins.verifier.l03.plugin import (
    ENGINE,
    L03NativeVerifier,
    SOURCE_SNAPSHOT_IDS,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from benchmark.plugins.verifier.l03.dmn_model import sha256_json

LANE = "l03"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

_SUBMIT_KEY = {
    "exact_output": "output",
    "rule_audit": "matched_rule_ids",
    "rule_repair": "table",
    "hit_policy_repair": "table",
    "chain_eval": "output",
    "binding_repair": "tables",
    "collect_result": "output",
    "input_synthesis": "input",
}


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
        manifest = {
            "shard": shard_name,
            "records": len(self.buffer),
            "sha256": sha256_json({"blob": blob}),
            "file_sha256": _sha256_file(path),
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


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference_submission(ast: dict[str, Any]) -> dict[str, Any]:
    return {_SUBMIT_KEY[ast["goal_kind"]]: ast["reference"]}


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    """Structured references must not appear verbatim in the public payload.

    Scalar answers (an output value, a policy name) are enumerable from the
    public table itself by design - the task is the evaluation - so the check
    targets structured references whose exact serialization cannot
    legitimately appear in public (golden repair payloads, reference input
    assignments, matched-rule-id lists).
    """
    reference = ast.get("reference")
    if isinstance(reference, (dict, list)) and len(reference) >= 2:
        frag = json.dumps(reference, sort_keys=True)
        if frag in json.dumps(public, sort_keys=True):
            return "reference_fragment"
    return None


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = sha256_json(
        [ast["goal_kind"], ast["unit_id"], json.dumps(ast["coordinate"], sort_keys=True)]
    )[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L03",
        "family_id": candidate.family_id,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
        "source_snapshot_ids": ["camunda-dmn-examples@%s" % ast["release_id"]],
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


def _qualify_one(verifier: L03NativeVerifier, candidate: CandidateTaskAST,
                 plugin: L03TaskFamilies) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "%s: %s" % (type(exc).__name__, str(exc)[:250])}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    try:
        term = verifier.verify_terminal(grounded, _reference_submission(candidate.ast))
    except Exception as exc:
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference replay raised %s: %s"
                                  % (type(exc).__name__, str(exc)[:250])}
    if not term.valid:
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    try:
        witness = verifier.certify_semantic_relevance(grounded)
    except Exception as exc:
        return record, {"code": "NON_SEMANTIC",
                        "reason": "witness raised %s: %s"
                                  % (type(exc).__name__, str(exc)[:250])}
    if not witness.ok:
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, candidate.ast)
    if leak:
        return record, {"code": "LEAKAGE",
                        "reason": "public payload leaks: %s" % leak[:120]}
    record["native_evidence"] = {
        "engine": ENGINE,
        "source_snapshot_ids": record["source_snapshot_ids"],
        "commands": [
            ".venv/bin/python -m benchmark.plugins.task_family.l03.pipeline "
            "--recheck-task %s" % record["task_id"],
        ],
        "expected": {"ok": True, "proof": True, "terminal": True, "witness": True},
        "prove_exists": proof.evidence,
        "verify_terminal": term.detail,
        "semantic_witness": witness.witness,
    }
    record["qualification"] = {
        "native_verified": True, "solvable": True, "nontrivial": True,
        "semantic_relevance": True, "leakage_clean": True,
        "release_eligible": True,
    }
    record["status"] = "QUALIFIED"
    return record, None


class DedupIndex:
    """Instance-level dedup; base goals and families counted from admits."""

    def __init__(self) -> None:
        self.instances: set[str] = set()
        self.base_goals: set[str] = set()
        self.families: set[str] = set()

    def admit(self, record: dict[str, Any]) -> bool:
        inst = record["instance_id"]
        if inst in self.instances:
            return False
        self.instances.add(inst)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True


def run_build(*, families: list[str] | None = None, seed: int = 20261027) -> dict[str, Any]:
    plugin = L03TaskFamilies()
    verifier = L03NativeVerifier(STAGING / "oracle")
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L03",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
        "seed": seed,
        "dataset_id": DATASET_ID,
        "source_snapshot_ids": SOURCE_SNAPSHOT_IDS,
        "semantic_profile": "camunda-dmn-1.1-table-static/1",
        "engine": ENGINE,
    }
    t0 = time.time()
    with rejection_path.open("w", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            goals_target, instances_target = FAMILY_QUOTA[fid]
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "goals_target": goals_target,
                         "instances_target": instances_target}
            candidates = plugin.propose(None, family=fid, seed=seed, constraints={})
            for candidate in candidates:
                record, rejection = _qualify_one(verifier, candidate, plugin)
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
    stats["task_families"] = len(stats["families"])
    stats["distinct_instances"] = len(dedup.instances)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _iter_qualified() -> Iterator[dict[str, Any]]:
    for shard in sorted((STAGING / "qualified").glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _recheck_record(verifier: L03NativeVerifier,
                    record: dict[str, Any]) -> dict[str, Any]:
    """Recompute one record's full qualification from the pinned XML."""
    oracle = verifier.load_oracle(record["oracle_handle"])
    ast_like = {
        "goal_kind": oracle["goal_kind"],
        "unit_id": oracle["unit_id"],
        "decision_id": oracle.get("decision_id"),
        "table_digest": oracle.get("table_digest"),
        "source_path": oracle["source_path"],
        "coordinate": oracle.get("coordinate"),
        "universe": oracle.get("universe"),
        "table": oracle.get("table"),
        "broken_table": oracle.get("broken_table"),
        "chain": oracle.get("chain"),
        "broken_chain": oracle.get("broken_chain"),
        "mutation": oracle.get("mutation"),
        "suite": oracle.get("suite"),
        "input_case": oracle.get("input_case"),
        "target_output": oracle.get("target_output"),
        "reference": oracle.get("reference"),
    }
    candidate = CandidateTaskAST(
        lane_id="L03", family_id=oracle["family_id"],
        base_goal_id=record["base_goal_id"], instance_id=record["instance_id"],
        regime=record["task_regime"], semantic_profile=record["semantic_profile"],
        ast=ast_like, source_unit_ids=tuple(record["source_unit_ids"]))
    grounded = GroundedTask(ast=candidate, public=record["public"],
                            oracle_ref=record["oracle_handle"], native_evidence={})
    proof = verifier.prove_exists(grounded)
    term = verifier.verify_terminal(grounded, _reference_submission(ast_like))
    witness = verifier.certify_semantic_relevance(grounded)
    leak = _leak_screen(record["public"], ast_like)
    return {"task_id": record["task_id"], "family_id": record["family_id"],
            "proof": proof.exists, "terminal": term.valid, "witness": witness.ok,
            "leakage_clean": leak is None,
            "ok": proof.exists and term.valid and witness.ok and leak is None}


def run_recheck() -> dict[str, Any]:
    """Recompute every qualified record's qualification from the pinned XML."""
    verifier = L03NativeVerifier(STAGING / "oracle")
    checked, mismatches = 0, []
    for record in _iter_qualified():
        res = _recheck_record(verifier, record)
        checked += 1
        if not res["ok"]:
            mismatches.append(res)
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def run_recheck_task(task_id: str) -> dict[str, Any]:
    """Recompute a single qualified record (native_evidence command)."""
    verifier = L03NativeVerifier(STAGING / "oracle")
    for record in _iter_qualified():
        if record["task_id"] == task_id:
            return _recheck_record(verifier, record)
    return {"task_id": task_id, "ok": False, "error": "task_id not found in qualified shards"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L03 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--recheck-task", default=None,
                        help="recompute one qualified task_id (native_evidence command)")
    args = parser.parse_args()
    if args.recheck_task:
        result = run_recheck_task(args.recheck_task)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.recheck:
        print(json.dumps(run_recheck(), indent=2, sort_keys=True))
        return 0
    stats = run_build(families=args.families, seed=args.seed)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
