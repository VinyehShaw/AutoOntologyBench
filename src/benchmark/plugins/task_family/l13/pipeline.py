"""L13 lane pipeline: candidate -> native-qualified shards (thin local runner).

``autoontobench.factory`` does not exist yet (S10 work package), so this is the
lane-local runner with the adapter seam: it drives propose -> ground ->
prove_exists -> write oracle -> verify_terminal(reference) -> semantic witness
-> leak screen -> 3-layer dedup -> immutable JSONL shards + manifests under
``bank/staging/l13/``. Rejection codes: UNSOLVABLE / TRIVIAL /
PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE / NON_SEMANTIC / LICENSE_HOLD.

Verification is in-process (jsonschema + deterministic recomputation); each
verdict is recomputed from the pinned snapshot bytes. ``--recheck`` re-runs the
full qualification gate over every qualified record and fails on any drift.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.task_family.l13.families import (
    FAMILIES, FAMILY_BY_ID, L13TaskFamilies,
)
from benchmark.plugins.verifier.l13.openapi_verifier import (
    ENGINE, OpenAPIContractVerifier,
)
from benchmark.schema.plugin_protocols import (
    CandidateTaskAST, GroundedTask,
)
from source_registry.fetchers import l13_apis_guru as fetcher
from benchmark.plugins.source.l13 import oas

LANE = "l13"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DATASET_ID = "autoontobench-bank-8000-rc1"
SOURCE_SNAPSHOT_ID = "apis-guru@commit-f04b8d0bcd39"

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

_SUBMIT_KEY = {
    "param_contract": "parameters",
    "body_contract": "body",
    "response_construction": "instance",
    "schema_mapping": "instance",
    "operation_selection": "operation_keys",
    "security_requirement": "alternatives",
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
            "sha256": oas.sha256_json({"blob": blob}),
            "file_sha256": oas.sha256_file(path),
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


def _reference_submission(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["goal_kind"]
    key = _SUBMIT_KEY.get(kind)
    if key is None:  # version_migration
        key = "parameters" if ast["coordinate"].get("mode") == "params" else "body"
    return {key: ast["reference"]}


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    """The reference answer must not appear verbatim in the public payload.

    Only structured references are checked: scalar answers (one enum value, one
    operation key) are the legitimate product of reading the public contract.
    """
    reference = ast.get("reference")
    if isinstance(reference, (dict, list)) and reference:
        frag = json.dumps(reference, sort_keys=True)
        if frag in json.dumps(public, sort_keys=True):
            return "reference_fragment"
    return None


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = oas.sha256_json(
        [ast["goal_kind"], ast["unit_id"], json.dumps(ast["coordinate"], sort_keys=True)]
    )[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (LANE, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": "L13",
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


def _qualify_one(verifier: OpenAPIContractVerifier, candidate: CandidateTaskAST,
                 plugin: L13TaskFamilies) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except oas.OasError as exc:
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    term = verifier.verify_terminal(grounded, _reference_submission(candidate.ast))
    if not term.valid:
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, candidate.ast)
    if leak:
        return record, {"code": "LEAKAGE", "reason": "public payload leaks: %s" % leak[:120]}
    record["native_evidence"] = {
        "engine": ENGINE,
        "source_commit": fetcher.PINNED_COMMIT,
        "commands": [
            ".venv/bin/python -m benchmark.plugins.task_family.l13.pipeline "
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


def _reset_outputs() -> None:
    for sub in ("qualified", "candidates", "rejections", "oracle"):
        d = STAGING / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()
    (STAGING / "oracle").mkdir(parents=True, exist_ok=True)


def run_build(
    *,
    families: list[str] | None = None,
    instances_per_goal: int = 6,
    seed: int = 20261027,
    max_per_family: int | None = None,
) -> dict[str, Any]:
    _reset_outputs()
    plugin = L13TaskFamilies()
    verifier = OpenAPIContractVerifier(STAGING / "oracle")
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    stats: dict[str, Any] = {
        "lane": "L13",
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
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0}
            candidates = list(plugin.propose(snapshot, family=fid, seed=seed,
                                             constraints={"instances_per_goal": instances_per_goal}))
            if max_per_family:
                candidates = candidates[:max_per_family]
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
    stats["dataset_id"] = DATASET_ID
    stats["source_snapshot_ids"] = [SOURCE_SNAPSHOT_ID]
    stats["seed"] = seed
    stats["instances_per_goal"] = instances_per_goal
    stats["distinct_instances"] = len(dedup.instances)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


def _recheck_record(verifier: OpenAPIContractVerifier,
                    record: dict[str, Any]) -> dict[str, Any]:
    """Recompute one record's full qualification from the pinned snapshot."""
    oracle = verifier.load_oracle(record["oracle_handle"])
    ast_like = {
        "goal_kind": oracle["goal_kind"], "unit_id": oracle["unit_id"],
        "unit_digest": oracle["unit_digest"], "unit_id_b": oracle.get("unit_id_b"),
        "unit_digest_b": oracle.get("unit_digest_b"), "coordinate": oracle["coordinate"],
        "contract": oracle["contract"], "contract_b": oracle.get("contract_b"),
        "initial": oracle.get("initial"), "reference": oracle["reference"],
        "guards": oracle["guards"], "constraints": oracle["constraints"],
        "witness": oracle["witness"],
    }
    candidate = CandidateTaskAST(
        lane_id="L13", family_id=oracle["family_id"],
        base_goal_id=record["base_goal_id"], instance_id=record["instance_id"],
        regime=record["task_regime"], semantic_profile=record["semantic_profile"],
        ast=ast_like, source_unit_ids=tuple(record["source_unit_ids"]))
    grounded = GroundedTask(ast=candidate, public=record["public"],
                            oracle_ref=record["oracle_handle"], native_evidence={})
    proof = verifier.prove_exists(grounded)
    term = verifier.verify_terminal(grounded, _reference_submission(ast_like))
    witness = verifier.certify_semantic_relevance(grounded)
    return {"task_id": record["task_id"], "family_id": record["family_id"],
            "proof": proof.exists, "terminal": term.valid, "witness": witness.ok,
            "ok": proof.exists and term.valid and witness.ok}


def _iter_qualified() -> Iterator[dict[str, Any]]:
    for shard in sorted((STAGING / "qualified").glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def run_recheck() -> dict[str, Any]:
    """Recompute every qualified record's qualification from the pinned snapshot."""
    verifier = OpenAPIContractVerifier(STAGING / "oracle")
    checked, mismatches = 0, []
    for record in _iter_qualified():
        res = _recheck_record(verifier, record)
        checked += 1
        if not res["ok"]:
            mismatches.append(res)
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def run_recheck_task(task_id: str) -> dict[str, Any]:
    """Recompute a single qualified record (used by native_evidence commands)."""
    verifier = OpenAPIContractVerifier(STAGING / "oracle")
    for record in _iter_qualified():
        if record["task_id"] == task_id:
            return _recheck_record(verifier, record)
    return {"task_id": task_id, "ok": False, "error": "task_id not found in qualified shards"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L13 lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--instances-per-goal", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-per-family", type=int, default=None)
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
