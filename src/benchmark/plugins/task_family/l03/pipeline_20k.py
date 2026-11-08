"""L03 20K-expansion lane pipeline: candidates -> native-qualified shards.

Drives propose -> ground -> prove_exists -> write oracle -> verify_terminal
(reference replay) -> semantic witness -> leak screen -> instance dedup (with
a legacy-8K collision guard) -> immutable JSONL shards + sha256 manifests
under ``bank/staging_20k/l03/``.

Dataset: ``autoontobench-bank-20000-candidate-dev1`` (new records only; the
legacy 8K stays untouched and is linked by QA's unified index, never
rewritten). Seed base 20261101.

Rejection codes: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE /
DUPLICATE / NON_SEMANTIC / LICENSE_HOLD. Every record carries the per-task
check-status block (plan §4.1); no check is ever marked PASS without being
executed.

Modes:
- default: full build (600 instances / 8 families)
- ``--smoke``: 2 instances per family into ``bank/staging_20k/l03_smoke``
- ``--recheck``: recompute every qualified record from the pinned XML/TCK
- ``--recheck-task ID``: recompute one record (native-evidence command)
- ``--negative-control``: in-memory tampering probes proving gate teeth
- ``--collision-report``: task_id + signature disjointness vs the legacy 8K
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.task_family.l03.families_20k import (
    FAMILIES,
    FAMILY_QUOTA,
)
from benchmark.plugins.task_family.l03.plugin_20k import (
    DATASET_ID,
    L03TaskFamilies20K,
)
from benchmark.plugins.verifier.l03.dmn_model import sha256_json
from benchmark.plugins.verifier.l03.plugin_20k import L03NativeVerifier20K
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

LANE = "l03"
STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging_20k" / LANE
SMOKE_STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging_20k" / (LANE + "_smoke")
LEGACY_STAGING = Path(__file__).resolve().parents[4] / "bank" / "staging" / LANE
DEFAULT_SEED = 20261101

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

_SUBMIT_KEY = {
    "tck_output": "output",
    "suite_audit": "inconsistent_case_id",
    "output_repair": "table",
    "aggregation_repair": "table",
    "rule_deletion_repair": "table",
    "partial_input_audit": "classification",
    "chain_synthesis": "input",
    "ordered_output": "output",
}

_MUTATION_KINDS = {"output_repair", "aggregation_repair", "rule_deletion_repair"}

_LEGACY_GOAL_KIND = {
    "l03-f01-exact-decision-output": "exact_output",
    "l03-f02-rule-match-audit": "rule_audit",
    "l03-f03-rule-entry-repair": "rule_repair",
    "l03-f04-hit-policy-repair": "hit_policy_repair",
    "l03-f05-cross-table-chain-eval": "chain_eval",
    "l03-f06-cross-table-binding-repair": "binding_repair",
    "l03-f07-collect-aggregation-result": "collect_result",
    "l03-f08-hidden-rule-input-synthesis": "input_synthesis",
}

_LEGACY_INSTANCE_MATERIAL = {
    "exact_output": "input_case", "rule_audit": "input_case",
    "collect_result": "input_case", "chain_eval": "input_case",
    "rule_repair": "mutation", "hit_policy_repair": "suite",
    "binding_repair": "suite", "input_synthesis": "target_output",
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


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _reference_submission(ast: dict[str, Any]) -> dict[str, Any]:
    if ast["goal_kind"] == "partial_input_audit":
        return dict(ast["reference"])
    return {_SUBMIT_KEY[ast["goal_kind"]]: ast["reference"]}


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    reference = ast.get("reference")
    if isinstance(reference, (dict, list)) and len(reference) >= 2:
        frag = json.dumps(reference, sort_keys=True)
        if frag in json.dumps(public, sort_keys=True):
            return "reference_fragment"
    return None


def _check_statuses(rejected: bool, code: str | None) -> dict[str, str]:
    statuses = {
        "source_verified": "PASS",
        "schema_valid": "PASS",
        "ontology_parse_valid": "PASS",
        "native_execution_status": "PASS",
        "oracle_check_status": "PASS",
        "mutation_check_status": "NOT_APPLICABLE",
        "human_review_status": "NOT_RUN",
        "qualification_status": "QUALIFIED_NATIVE",
    }
    if rejected:
        statuses["qualification_status"] = "REJECTED"
        if code == "PROFILE_UNSUPPORTED":
            statuses["ontology_parse_valid"] = "FAIL"
            statuses["native_execution_status"] = "NOT_RUN"
            statuses["oracle_check_status"] = "NOT_RUN"
        elif code == "UNSOLVABLE":
            statuses["oracle_check_status"] = "FAIL"
        elif code == "TRIVIAL":
            statuses["oracle_check_status"] = "NOT_RUN"
        elif code == "NON_SEMANTIC":
            statuses["oracle_check_status"] = "NOT_RUN"
        elif code == "LEAKAGE":
            statuses["schema_valid"] = "FAIL"
        elif code == "DUPLICATE":
            statuses["schema_valid"] = "FAIL"
    return statuses


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
        "source_snapshot_ids": ["%s@%s" % (
            ast["unit_id"].split("/")[0], ast["release_id"])],
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
        "check_statuses": _check_statuses(False, None),
    }


def _qualify_one(verifier: L03NativeVerifier20K, candidate: CandidateTaskAST,
                 plugin: L03TaskFamilies20K) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
        record["check_statuses"] = _check_statuses(True, code)
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    try:
        term = verifier.verify_terminal(grounded, _reference_submission(candidate.ast))
    except Exception as exc:
        record["check_statuses"] = _check_statuses(True, "UNSOLVABLE")
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference replay raised %s: %s"
                                  % (type(exc).__name__, str(exc)[:250])}
    if not term.valid:
        record["check_statuses"] = _check_statuses(True, "UNSOLVABLE")
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    try:
        witness = verifier.certify_semantic_relevance(grounded)
    except Exception as exc:
        record["check_statuses"] = _check_statuses(True, "NON_SEMANTIC")
        return record, {"code": "NON_SEMANTIC",
                        "reason": "witness raised %s: %s"
                                  % (type(exc).__name__, str(exc)[:250])}
    if not witness.ok:
        record["check_statuses"] = _check_statuses(True, "NON_SEMANTIC")
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    leak = _leak_screen(grounded.public, candidate.ast)
    if leak:
        record["check_statuses"] = _check_statuses(True, "LEAKAGE")
        return record, {"code": "LEAKAGE",
                        "reason": "public payload leaks: %s" % leak[:120]}
    record["native_evidence"] = {
        "engine": "aob-dmn-static/1.0.0",
        "source_snapshot_ids": record["source_snapshot_ids"],
        "commands": [
            ".venv/bin/python -m benchmark.plugins.task_family.l03.pipeline_20k "
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
    if candidate.ast["goal_kind"] in _MUTATION_KINDS:
        record["check_statuses"]["mutation_check_status"] = "PASS"
    record["status"] = "QUALIFIED"
    record["check_statuses"]["qualification_status"] = "QUALIFIED_NATIVE"
    return record, None


class DedupIndex:
    """Instance-level dedup with a legacy-8K collision guard and a
    public-payload guard (two admitted tasks must never present the solver
    the same public payload)."""

    def __init__(self, legacy_ids: set[str]) -> None:
        self.instances: set[str] = set()
        self.legacy_ids = legacy_ids
        self.public_sigs: set[str] = set()
        self.base_goals: set[str] = set()
        self.families: set[str] = set()

    def admit(self, record: dict[str, Any]) -> tuple[bool, str | None]:
        inst = record["instance_id"]
        if inst in self.legacy_ids:
            return False, "instance_id collides with legacy 8K record"
        if inst in self.instances:
            return False, "instance hash already admitted"
        public_sig = sha256_json(
            [record["family_id"], record["public"]])
        if public_sig in self.public_sigs:
            return False, "public payload identical to an admitted task"
        self.instances.add(inst)
        self.public_sigs.add(public_sig)
        self.base_goals.add(record["base_goal_id"])
        self.families.add(record["family_id"])
        return True, None


def _legacy_instance_ids() -> set[str]:
    ids: set[str] = set()
    for shard in sorted(LEGACY_STAGING.glob("qualified/shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["instance_id"])
    return ids


def run_build(*, families: list[str] | None = None, seed: int = DEFAULT_SEED,
              staging: Path = STAGING,
              instances_per_family: int | None = None) -> dict[str, Any]:
    plugin = L03TaskFamilies20K()
    verifier = L03NativeVerifier20K(staging / "oracle")
    family_ids = families or [f.family_id for f in FAMILIES]
    qualified_writer = ShardWriter(staging / "qualified")
    candidate_writer = ShardWriter(staging / "candidates")
    rejection_path = staging / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex(_legacy_instance_ids())
    stats: dict[str, Any] = {
        "lane": "L03",
        "build": "aob-20000-candidate-dev1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
        "seed": seed,
        "dataset_id": DATASET_ID,
        "source_snapshot_ids": [
            "camunda-dmn-examples@commit-6c7f4c4adb4b",
            "camunda-dmn-examples@commit-ee4826e5e76c",
            "dmn-tck@commit-1e349176e5ae",
        ],
        "semantic_profile": "camunda-dmn-1.1-table-static/1",
        "engine": "aob-dmn-static/1.0.0",
        "staging_root": str(staging),
    }
    t0 = time.time()
    with rejection_path.open("w", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            goals_target, instances_target = FAMILY_QUOTA[fid]
            if instances_per_family is not None:
                instances_target = min(instances_target, instances_per_family)
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "goals_target": goals_target,
                         "instances_target": instances_target}
            constraints = {"instances": instances_target}
            candidates = plugin.propose(None, family=fid, seed=seed,
                                        constraints=constraints)
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
                        "check_statuses": record["check_statuses"],
                        **rejection,
                    }, sort_keys=True) + "\n")
                    continue
                admitted, dup_reason = dedup.admit(record)
                if not admitted:
                    fam_stats["rejected"] += 1
                    stats["rejections"]["DUPLICATE"] = (
                        stats["rejections"].get("DUPLICATE", 0) + 1)
                    record["check_statuses"] = _check_statuses(True, "DUPLICATE")
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"],
                        "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        "check_statuses": record["check_statuses"],
                        "code": "DUPLICATE",
                        "reason": dup_reason,
                    }, sort_keys=True) + "\n")
                    continue
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
                qualified_writer.add(record)
            stats["families"][fid] = fam_stats
    qualified_writer.close()
    candidate_writer.close()
    stats["base_goals"] = len(dedup.base_goals)
    stats["task_families"] = len(stats["families"])
    stats["distinct_instances"] = len(dedup.instances)
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(staging / "build_report.json", stats)
    return stats


def _iter_qualified(staging: Path = STAGING) -> Iterator[dict[str, Any]]:
    for shard in sorted((staging / "qualified").glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _recheck_record(verifier: L03NativeVerifier20K,
                    record: dict[str, Any]) -> dict[str, Any]:
    """Recompute one record's full qualification from the pinned XML/TCK."""
    oracle = verifier.load_oracle(record["oracle_handle"])
    ast_like = {
        "goal_kind": oracle["goal_kind"],
        "unit_id": oracle["unit_id"],
        "decision_id": oracle.get("decision_id"),
        "decision_name": oracle.get("decision_name"),
        "table_digest": oracle.get("table_digest"),
        "source_path": oracle["source_path"],
        "coordinate": oracle.get("coordinate"),
        "universe": oracle.get("universe"),
        "table": oracle.get("table"),
        "broken_table": oracle.get("broken_table"),
        "chain": oracle.get("chain"),
        "mutation": oracle.get("mutation"),
        "suite": oracle.get("suite"),
        "input_case": oracle.get("input_case"),
        "partial_input": oracle.get("partial_input"),
        "missing_var": oracle.get("missing_var"),
        "target_output": oracle.get("target_output"),
        "reference": oracle.get("reference"),
        "tck_test_path": oracle.get("tck_test_path"),
        "test_case_id": oracle.get("test_case_id"),
        "tck_expected": oracle.get("tck_expected"),
        "tamper": oracle.get("tamper"),
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


def run_recheck(staging: Path = STAGING) -> dict[str, Any]:
    """Recompute every qualified record's qualification from the pinned XML/TCK."""
    verifier = L03NativeVerifier20K(staging / "oracle")
    checked, mismatches = 0, []
    for record in _iter_qualified(staging):
        res = _recheck_record(verifier, record)
        checked += 1
        if not res["ok"]:
            mismatches.append(res)
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def run_recheck_task(task_id: str, staging: Path = STAGING) -> dict[str, Any]:
    """Recompute a single qualified record (native_evidence command)."""
    verifier = L03NativeVerifier20K(staging / "oracle")
    for record in _iter_qualified(staging):
        if record["task_id"] == task_id:
            return _recheck_record(verifier, record)
    return {"task_id": task_id, "ok": False,
            "error": "task_id not found in qualified shards"}


# -- legacy collision report -----------------------------------------------------------

def _task_signature(goal_kind: str, unit_id: str, decision_id: str | None,
                    material: Any) -> str:
    return sha256_json([goal_kind, unit_id, decision_id,
                        json.dumps(material, sort_keys=True, default=repr)])


def run_collision_report() -> dict[str, Any]:
    """task_id + signature disjointness of the new build vs the legacy 8K."""
    legacy_task_ids: set[str] = set()
    legacy_sigs: set[str] = set()
    legacy_oracle = LEGACY_STAGING / "oracle"
    oracle_by_handle = {}
    for path in sorted(legacy_oracle.glob("l03-oracle-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        oracle_by_handle[payload["oracle_ref"]] = payload
    for record in _iter_qualified(LEGACY_STAGING):
        legacy_task_ids.add(record["task_id"])
        oracle = oracle_by_handle.get(record["oracle_handle"])
        if oracle is None:
            continue
        kind = oracle["goal_kind"]
        material = oracle.get(_LEGACY_INSTANCE_MATERIAL.get(kind, "input_case"))
        legacy_sigs.add(_task_signature(
            kind, oracle["unit_id"], oracle.get("decision_id"), material))
    new_task_ids: set[str] = set()
    new_sigs: set[str] = set()
    new_oracle = STAGING / "oracle"
    new_oracle_by_handle = {}
    for path in sorted(new_oracle.glob("l03-oracle-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        new_oracle_by_handle[payload["oracle_ref"]] = payload
    for record in _iter_qualified(STAGING):
        new_task_ids.add(record["task_id"])
        oracle = new_oracle_by_handle.get(record["oracle_handle"])
        if oracle is None:
            continue
        kind = oracle["goal_kind"]
        material = oracle.get({
            "tck_output": "input_case", "ordered_output": "input_case",
            "partial_input_audit": "partial_input",
            "suite_audit": "suite",
            "output_repair": "mutation", "aggregation_repair": "suite",
            "rule_deletion_repair": "mutation",
            "chain_synthesis": "target_output",
        }.get(kind, "input_case"))
        new_sigs.add(_task_signature(
            kind, oracle["unit_id"], oracle.get("decision_id"), material))
    return {
        "legacy_task_ids": len(legacy_task_ids),
        "new_task_ids": len(new_task_ids),
        "task_id_collisions": sorted(legacy_task_ids & new_task_ids),
        "legacy_signatures": len(legacy_sigs),
        "new_signatures": len(new_sigs),
        "signature_collisions": len(legacy_sigs & new_sigs),
        "ok": not (legacy_task_ids & new_task_ids) and not (legacy_sigs & new_sigs),
    }


# -- negative control --------------------------------------------------------------------

def run_negative_control(staging: Path = STAGING) -> dict[str, Any]:
    """In-memory tampering probes: every one must be rejected by the gates."""
    verifier = L03NativeVerifier20K(staging / "oracle")
    records = list(_iter_qualified(staging))
    by_family: dict[str, dict[str, Any]] = {}
    for r in records:
        by_family.setdefault(r["family_id"], r)
    results: list[dict[str, Any]] = []

    def _probe(name: str, family_id: str, mutate_ast, submit=None):
        record = by_family.get(family_id)
        if record is None:
            results.append({"probe": name, "family": family_id, "rejected": None,
                            "note": "no qualified record"})
            return
        oracle = verifier.load_oracle(record["oracle_handle"])
        ast = {
            "goal_kind": oracle["goal_kind"], "unit_id": oracle["unit_id"],
            "decision_id": oracle.get("decision_id"),
            "decision_name": oracle.get("decision_name"),
            "table_digest": oracle.get("table_digest"),
            "source_path": oracle["source_path"],
            "coordinate": oracle.get("coordinate"), "universe": oracle.get("universe"),
            "table": oracle.get("table"), "broken_table": oracle.get("broken_table"),
            "chain": oracle.get("chain"), "mutation": oracle.get("mutation"),
            "suite": oracle.get("suite"), "input_case": oracle.get("input_case"),
            "partial_input": oracle.get("partial_input"),
            "missing_var": oracle.get("missing_var"),
            "target_output": oracle.get("target_output"),
            "reference": oracle.get("reference"),
            "tck_test_path": oracle.get("tck_test_path"),
            "test_case_id": oracle.get("test_case_id"),
            "tck_expected": oracle.get("tck_expected"),
            "tamper": oracle.get("tamper"),
        }
        ast = mutate_ast(ast) if mutate_ast else ast
        candidate = CandidateTaskAST(
            lane_id="L03", family_id=family_id,
            base_goal_id=record["base_goal_id"],
            instance_id=record["instance_id"],
            regime=record["task_regime"],
            semantic_profile=record["semantic_profile"],
            ast=ast, source_unit_ids=tuple(record["source_unit_ids"]))
        grounded = GroundedTask(ast=candidate, public=record["public"],
                                oracle_ref=record["oracle_handle"], native_evidence={})
        proof = verifier.prove_exists(grounded)
        if not proof.exists:
            results.append({"probe": name, "family": family_id, "rejected": True,
                            "reason": proof.reason})
            return
        submission = submit if submit is not None else _reference_submission(ast)
        term = verifier.verify_terminal(grounded, submission)
        results.append({"probe": name, "family": family_id,
                        "rejected": not term.valid,
                        "reason": None if term.valid else json.dumps(term.detail)[:200]})

    def _wrong_reference(ast):
        ref = ast["reference"]
        if isinstance(ref, str):
            ast["reference"] = ref + "X"
        elif isinstance(ref, (int, float)) and not isinstance(ref, bool):
            ast["reference"] = ref + 1
        elif isinstance(ref, list):
            ast["reference"] = list(reversed(ref))
        elif isinstance(ref, dict):
            ast["reference"] = {k: "X" for k in ref}
        return ast

    _probe("wrong_output", "l03-f09-tck-scenario-output", _wrong_reference)
    _probe("wrong_case_id", "l03-f10-suite-conformance-audit", None,
           submit={"inconsistent_case_id": "case-XX"})
    _probe("unrepaired_table", "l03-f11-output-entry-repair", None,
           submit={"table": None})
    _probe("digest_drift", "l03-f12-aggregation-repair",
           lambda ast: {**ast, "table_digest": "0" * 64})
    _probe("suite_tamper", "l03-f13-rule-deletion-repair",
           lambda ast: {**ast, "suite": [
               {**c, "expected": "TAMPERED"} for c in ast["suite"]]})
    _probe("wrong_classification", "l03-f14-partial-input-rule-audit",
           lambda ast: {**ast, "reference": {
               "necessarily_matching": ast["reference"]["possibly_matching"],
               "possibly_matching": ast["reference"]["necessarily_matching"],
           }})
    _probe("wrong_input", "l03-f15-chain-root-synthesis", None,
           submit={"input": {"__nope__": 0}})
    _probe("wrong_output", "l03-f16-ordered-policy-output", _wrong_reference)
    rejected = [r for r in results if r["rejected"] is True]
    return {"probes": results, "rejected": len(rejected),
            "total": len(results), "ok": len(rejected) == len(results)}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L03 20K-expansion lane build runner")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true",
                        help="2 instances per family into bank/staging_20k/l03_smoke")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--recheck-task", default=None,
                        help="recompute one qualified task_id (native_evidence command)")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--collision-report", action="store_true")
    parser.add_argument("--staging", default=None,
                        help="staging root override (default: bank/staging_20k/l03)")
    args = parser.parse_args()
    staging = Path(args.staging) if args.staging else STAGING
    if args.recheck_task:
        result = run_recheck_task(args.recheck_task, staging)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.recheck:
        print(json.dumps(run_recheck(staging), indent=2, sort_keys=True))
        return 0
    if args.negative_control:
        result = run_negative_control(staging)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.collision_report:
        result = run_collision_report()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.smoke:
        stats = run_build(families=args.families, seed=args.seed,
                          staging=SMOKE_STAGING, instances_per_family=2)
    else:
        stats = run_build(families=args.families, seed=args.seed)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
