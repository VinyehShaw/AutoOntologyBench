"""L12 20K expansion pipeline: candidate -> native-qualified shards.

Writes ONLY to ``bank/staging_20k/l12/`` (candidates/qualified/oracle/
rejections/build_report.json + per-shard manifests). The legacy 8K staging
``bank/staging/l12/`` is opened read-only for collision checks and never
written.

Funnel per candidate (same discipline as the legacy lane, plus explicit
per-task check statuses and legacy-collision gates):
  pre-grounding gates (instance/task id, semantic signature, content key,
  answer key — vs legacy 8K AND vs this build) -> ground -> schema validate ->
  prove_exists -> write oracle -> verify_terminal(reference) -> semantic
  witness (counterfactual) -> leak screen -> qualified shard.

Rejection codes: UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE /
DUPLICATE / NON_SEMANTIC / LICENSE_HOLD (legacy vocabulary).

``--audit`` re-verifies qualified records independently from their persisted
oracles (rebuild candidate -> verify_terminal(reference submission) with the
pinned pm4py engine), recording per-task verdicts in ``audit_report.json``.

Usage:
    PYTHONPATH=. .venv/bin/python -m benchmark.plugins.task_family.l12.expand20k.pipeline
    PYTHONPATH=. .venv/bin/python -m benchmark.plugins.task_family.l12.expand20k.pipeline --audit
    PYTHONPATH=. .venv/bin/python -m benchmark.plugins.task_family.l12.expand20k.pipeline --audit --task-id l12-in-...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT
from benchmark.plugins.task_family.l12.pipeline import (
    ShardWriter,
    _reference_submission,
    _write_json_atomic,
)
from benchmark.plugins.task_family.l12.topup import variant_signature
from benchmark.plugins.task_family.l12.expand20k.compute import answer_key
from benchmark.plugins.task_family.l12.expand20k.families import (
    FAMILIES_20K,
    FAMILY_BY_ID_20K,
    INSTANCES_PER_GOAL,
)
from benchmark.plugins.task_family.l12.expand20k.generators import Pm4pyTaskFamilies20k
from benchmark.plugins.task_family.l12.expand20k.registry import (
    DATASET_ID,
    EXPANSION_DIR,
    FP_CONF_PAIRS,
    GENERATOR_VERSION,
    NEW_GOLD_PAIRS,
    NEW_LOGS,
    NEW_NOISY_PAIRS,
    SEED_BASE,
    SEMANTIC_PROFILE,
    SOURCE_SNAPSHOT_ID,
    STAGING_20K,
)
from benchmark.plugins.task_family.l12.expand20k.verifier import Pm4pyVerifier20k
from source_registry.fetchers import l12_pm4py as fetcher

LANE = "l12"
LEGACY_STAGING = STAGING_20K.parent.parent / "staging" / "l12"
PINNED_MANIFEST = (
    engine.PINNED_ROOT.parent / "tag-2.7.23.11.files.sha256.json"
)

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

CHECK_KEYS = (
    "source_verified", "schema_valid", "ontology_parse_valid",
    "native_execution_status", "oracle_check_status", "mutation_check_status",
    "human_review_status", "qualification_status",
)


# ---------------------------------------------------------------------------
# source verification (pinned manifest)
# ---------------------------------------------------------------------------

def _verify_source_units() -> dict[str, Any]:
    """sha256-verify every registry unit against the pinned fetch manifest."""
    rels: set[str] = set()
    for pair in NEW_GOLD_PAIRS.values():
        rels.update(pair.values())
    rels.update(NEW_LOGS.values())
    for pair in NEW_NOISY_PAIRS.values():
        rels.update(pair.values())
    for pair in FP_CONF_PAIRS.values():
        rels.update(pair.values())
    manifest = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"units": len(rels), "verified": 0, "mismatches": []}
    for rel in sorted(rels):
        path = engine.PINNED_ROOT / rel
        if not path.is_file():
            out["mismatches"].append({"rel": rel, "error": "missing"})
            continue
        if rel not in manifest:
            out["mismatches"].append({"rel": rel, "error": "not_in_manifest"})
            continue
        if engine.sha256_file(path) != manifest[rel]:
            out["mismatches"].append({"rel": rel, "error": "sha256_mismatch"})
            continue
        out["verified"] += 1
    out["ok"] = not out["mismatches"]
    return out


# ---------------------------------------------------------------------------
# legacy collision index (read-only over bank/staging/l12)
# ---------------------------------------------------------------------------

def _content_key(ast: dict[str, Any]) -> tuple[Any, ...]:
    kind = ast["goal_kind"]
    if kind == "repair_trace":
        return ("repair", ast.get("net_rel") or ast["log_rel"],
                tuple(ast["corrupted_trace"]))
    if kind == "repair_model":
        return ("relabel", ast["log_rel"], ast["net_rel"],
                tuple(sorted(ast["corrupted_tids"].items())))
    return ("exact", ast["operation"], answer_key(ast["reference_answer"]))


def _legacy_collision_index() -> dict[str, Any]:
    """Read-only scan of the legacy 8K lane: ids, signatures, content keys."""
    idx: dict[str, Any] = {
        "task_ids": set(), "instance_ids": set(), "public_hashes": set(),
        "signatures": set(), "content_keys": set(), "answer_keys": set(),
        "legacy_records_scanned": 0, "legacy_oracles_scanned": 0,
    }
    for sub in ("qualified", "candidates"):
        d = LEGACY_STAGING / sub
        if not d.is_dir():
            continue
        for path in sorted(d.glob("shard-*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    idx["legacy_records_scanned"] += 1
                    idx["task_ids"].add(rec["task_id"])
                    idx["instance_ids"].add(rec["instance_id"])
                    idx["public_hashes"].add(
                        engine.sha256_json(rec.get("public", {})))
    odir = LEGACY_STAGING / "oracle"
    if odir.is_dir():
        for path in sorted(odir.glob("*.json")):
            oracle = json.loads(path.read_text(encoding="utf-8"))
            ast = oracle["ast"]
            idx["legacy_oracles_scanned"] += 1
            idx["signatures"].add(variant_signature(ast))
            idx["content_keys"].add(_content_key(ast))
            if ast["goal_kind"] == "exact_artifact":
                idx["answer_keys"].add(answer_key(ast["reference_answer"]))
    return idx


# ---------------------------------------------------------------------------
# record construction + schema validation
# ---------------------------------------------------------------------------

def _scope_events(ast: dict[str, Any]) -> int:
    if ast["goal_kind"] == "repair_trace":
        return len(ast["corrupted_trace"])
    if ast["goal_kind"] == "repair_model":
        traces = RT.traces(ast["log_rel"])
        return sum(len(t) for t in traces)
    traces = RT.traces(ast["log_rel"])
    proj = engine.project_first_n(traces, ast["projection"]["first_n"])
    return sum(len(t) for t in proj)


def _mech_key(ast: dict[str, Any]) -> str:
    return (
        ast.get("pair_id") or ast.get("noisy_pair") or ast.get("log_rel") or ""
    )


def _base_record(
    candidate: CandidateTaskAST, grounded: GroundedTask, source_ok: bool
) -> dict[str, Any]:
    ast = candidate.ast
    mech_group = engine.sha256_json(
        [candidate.family_id, ast["operation"], _mech_key(ast)]
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
        "generation": {"seed": SEED_BASE, "generator": GENERATOR_VERSION},
        "status": "CANDIDATE",
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
        "checks": {
            "source_verified": "PASS" if source_ok else "FAIL",
            "schema_valid": "NOT_RUN",
            "ontology_parse_valid": "NOT_RUN",
            "native_execution_status": "NOT_RUN",
            "oracle_check_status": "NOT_RUN",
            "mutation_check_status": "NOT_RUN",
            "human_review_status": "NOT_RUN",
            "qualification_status": "CANDIDATE",
        },
    }


_REGIMES = {"repair", "application", "construction", "organization"}
_COMPLEXITIES = {"low", "medium", "high"}


def _validate_record(record: dict[str, Any], *, full: bool = False) -> list[str]:
    """aob-task/5.0 field-contract validation; returns error list (empty=PASS)."""
    errs: list[str] = []
    r = record
    if r.get("schema_version") != "aob-task/5.0":
        errs.append("schema_version")
    for k in ("task_id", "dataset_id", "lane_id", "family_id", "base_goal_id",
              "instance_id", "task_regime", "semantic_profile", "complexity",
              "oracle_handle", "status"):
        if not isinstance(r.get(k), str) or not r[k]:
            errs.append(k)
    if r.get("lane_id") != "L12":
        errs.append("lane_id_value")
    if not r.get("task_id", "").startswith("l12-in-"):
        errs.append("task_id_shape")
    if r.get("task_regime") not in _REGIMES:
        errs.append("task_regime_value")
    if r.get("complexity") not in _COMPLEXITIES:
        errs.append("complexity_value")
    if not r.get("base_goal_id", "").startswith("bg-"):
        errs.append("base_goal_id_shape")
    if not r.get("instance_id", "").startswith("in-"):
        errs.append("instance_id_shape")
    for k in ("source_snapshot_ids", "source_unit_ids"):
        if not isinstance(r.get(k), list) or not r[k]:
            errs.append(k)
    pub = r.get("public")
    if not isinstance(pub, dict):
        errs.append("public")
    else:
        if not isinstance(pub.get("user_goal"), str) or not pub["user_goal"]:
            errs.append("public.user_goal")
        if not isinstance(pub.get("task_material"), dict):
            errs.append("public.task_material")
        if not isinstance(pub.get("allowed_tools"), list) or not pub["allowed_tools"]:
            errs.append("public.allowed_tools")
        if not isinstance(pub.get("budget"), dict):
            errs.append("public.budget")
        if not isinstance(pub.get("answer_contract"), dict):
            errs.append("public.answer_contract")
        if not isinstance(pub.get("observation_refs"), list) or not pub["observation_refs"]:
            errs.append("public.observation_refs")
    qual = r.get("qualification")
    if not isinstance(qual, dict) or set(qual) != {
            "native_verified", "solvable", "nontrivial", "semantic_relevance",
            "leakage_clean", "release_eligible"}:
        errs.append("qualification")
    checks = r.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECK_KEYS):
        errs.append("checks")
    if full and not isinstance(r.get("native_evidence"), dict):
        errs.append("native_evidence")
    if full and isinstance(r.get("native_evidence"), dict):
        ne = r["native_evidence"]
        for k in ("prove_exists", "verify_terminal", "semantic_witness"):
            if k not in ne:
                errs.append("native_evidence.%s" % k)
    return errs


# ---------------------------------------------------------------------------
# qualification (one candidate)
# ---------------------------------------------------------------------------

def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
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


def _trim_proof(evidence: dict[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    out.pop("reference_terminal", None)
    out.pop("reference_answer", None)
    return out


def _trim_witness(witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(witness.get("counterfactuals")),
        "counterfactuals": [
            {k: v for k, v in cf.items()
             if k in ("mode", "note", "mutated_sha256", "reference_sha256")}
            for cf in witness.get("counterfactuals", [])[:3]
        ],
    }


def _qualify_one(
    plugin: Pm4pyTaskFamilies20k,
    verifier: Pm4pyVerifier20k,
    candidate: CandidateTaskAST,
    source_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (record, rejection|None). Never raises for content reasons."""
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded, source_ok)
    checks = record["checks"]
    schema_errors = _validate_record(record)
    checks["schema_valid"] = "PASS" if not schema_errors else "FAIL"
    if schema_errors:
        return record, {"code": "NON_SEMANTIC",
                        "reason": "schema invalid: %s" % ",".join(schema_errors)}
    try:
        proof = verifier.prove_exists(grounded)
    except Exception as exc:  # engine-level failure, not content
        checks["ontology_parse_valid"] = "FAIL"
        checks["native_execution_status"] = "FAIL"
        return record, {"code": "PROFILE_UNSUPPORTED",
                        "reason": "%s: %s" % (type(exc).__name__, str(exc))[:300]}
    checks["ontology_parse_valid"] = "PASS"
    if not proof.exists:
        checks["native_execution_status"] = "FAIL"
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    checks["native_execution_status"] = "PASS"
    verifier.write_oracle(grounded, proof.evidence)
    ast = candidate.ast
    term = verifier.verify_terminal(grounded, _reference_submission(ast))
    if not term.valid:
        checks["oracle_check_status"] = "FAIL"
        return record, {"code": "UNSOLVABLE",
                        "reason": "reference solution failed terminal replay: %s"
                                  % json.dumps(term.detail)[:300]}
    checks["oracle_check_status"] = "PASS"
    witness = verifier.certify_semantic_relevance(grounded)
    if not witness.ok:
        checks["mutation_check_status"] = "FAIL"
        return record, {"code": "NON_SEMANTIC",
                        "reason": json.dumps(witness.witness)[:300]}
    checks["mutation_check_status"] = "PASS"
    leak = _leak_screen(grounded.public, ast)
    if leak:
        return record, {"code": "LEAKAGE",
                        "reason": "public payload leaks: %s" % leak[:120]}
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
    record["status"] = "QUALIFIED_NATIVE"
    record["checks"]["qualification_status"] = "QUALIFIED_NATIVE"
    return record, None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

class _Gate:
    """In-build + legacy collision gates."""

    def __init__(self, legacy: dict[str, Any]) -> None:
        self.legacy = legacy
        self.instance_ids: set[str] = set()
        self.task_ids: set[str] = set()
        self.signatures: set[str] = set()
        self.content_keys: set[tuple[Any, ...]] = set()
        self.answer_keys: set[str] = set()
        self.public_hashes: set[str] = set()
        self.skipped: dict[str, int] = {}

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def pre_grounding(self, candidate: CandidateTaskAST) -> str | None:
        ast = candidate.ast
        if candidate.instance_id in self.legacy["instance_ids"]:
            return "legacy_instance_id_collision"
        if "l12-%s" % candidate.instance_id in self.legacy["task_ids"]:
            return "legacy_task_id_collision"
        if candidate.instance_id in self.instance_ids:
            return "duplicate_instance_id"
        sig = variant_signature(ast)
        if sig in self.legacy["signatures"]:
            return "legacy_signature_collision"
        if sig in self.signatures:
            return "duplicate_signature"
        key = _content_key(ast)
        if key in self.legacy["content_keys"]:
            return "legacy_content_collision"
        if key in self.content_keys:
            return "duplicate_content"
        if ast["goal_kind"] == "exact_artifact":
            akey = answer_key(ast["reference_answer"])
            if akey in self.legacy["answer_keys"]:
                return "legacy_answer_collision"
            if akey in self.answer_keys:
                return "duplicate_answer"
        return None

    def post_grounding(self, record: dict[str, Any]) -> str | None:
        ph = engine.sha256_json(record["public"])
        if ph in self.legacy["public_hashes"]:
            return "legacy_public_collision"
        if ph in self.public_hashes:
            return "duplicate_public"
        return None

    def admit(self, candidate: CandidateTaskAST, record: dict[str, Any]) -> None:
        self.instance_ids.add(candidate.instance_id)
        self.task_ids.add(record["task_id"])
        self.signatures.add(variant_signature(candidate.ast))
        self.content_keys.add(_content_key(candidate.ast))
        if candidate.ast["goal_kind"] == "exact_artifact":
            self.answer_keys.add(answer_key(candidate.ast["reference_answer"]))
        self.public_hashes.add(engine.sha256_json(record["public"]))


def _propose_all(plugin: Pm4pyTaskFamilies20k, snapshot, family_ids, seed, per_goal):
    for fid in family_ids:
        for candidate in plugin.propose(
            snapshot, family=fid, seed=seed,
            constraints={"instances_per_goal": per_goal},
        ):
            yield fid, candidate


def run_build(
    *,
    families: list[str] | None = None,
    instances_per_goal: int = INSTANCES_PER_GOAL,
    seed: int = SEED_BASE,
) -> dict[str, Any]:
    plugin = Pm4pyTaskFamilies20k()
    verifier = Pm4pyVerifier20k()
    snapshot = fetcher.load_snapshot()
    family_ids = families or [f.family_id for f in FAMILIES_20K]

    source_check = _verify_source_units()
    legacy = _legacy_collision_index()
    gate = _Gate(legacy)

    qualified_writer = ShardWriter(STAGING_20K / "qualified")
    candidate_writer = ShardWriter(STAGING_20K / "candidates")
    rejection_path = STAGING_20K / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "lane": "L12",
        "build": "20k-expansion",
        "dataset_id": DATASET_ID,
        "seed": seed,
        "generator": GENERATOR_VERSION,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "base_goals": 0,
        "source_units": source_check,
        "legacy_collision_scan": {
            "legacy_records_scanned": legacy["legacy_records_scanned"],
            "legacy_oracles_scanned": legacy["legacy_oracles_scanned"],
            "legacy_task_ids": len(legacy["task_ids"]),
        },
    }
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
        for fid, candidate in _propose_all(
            plugin, snapshot, family_ids, seed, instances_per_goal
        ):
            fam = stats["families"].setdefault(
                fid, {"candidates": 0, "qualified": 0, "rejected": 0, "skipped": 0})
            skip = gate.pre_grounding(candidate)
            if skip is not None:
                gate._skip(skip)
                fam["skipped"] += 1  # honest funnel count (not persisted as rows)
                continue
            record, rejection = _qualify_one(
                plugin, verifier, candidate, source_check["ok"])
            pub_skip = gate.post_grounding(record)
            if pub_skip is not None:
                gate._skip(pub_skip)
                rejection = {"code": "DUPLICATE",
                             "reason": "public payload collision: %s" % pub_skip}
            fam["candidates"] += 1
            stats["candidates"] += 1
            candidate_writer.add(record)
            if rejection is not None:
                fam["rejected"] += 1
                stats["rejections"][rejection["code"]] = (
                    stats["rejections"].get(rejection["code"], 0) + 1)
                record["status"] = "REJECTED"
                record["checks"]["qualification_status"] = "REJECTED"
                record["qualification"] = {
                    "native_verified": False, "solvable": False, "nontrivial": False,
                    "semantic_relevance": False, "leakage_clean": False,
                    "release_eligible": False,
                }
                rej_fh.write(json.dumps({
                    "task_id": record["task_id"],
                    "family_id": record["family_id"],
                    "base_goal_id": record["base_goal_id"],
                    **rejection,
                }, sort_keys=True) + "\n")
                # an oracle may exist before a later gate failed; remove it
                opath = (STAGING_20K / "oracle" / ("%s.json" % record["oracle_handle"]))
                if opath.is_file():
                    opath.unlink()
                continue
            gate.admit(candidate, record)
            fam["qualified"] += 1
            stats["qualified"] += 1
            qualified_writer.add(record)
    qualified_writer.close()
    candidate_writer.close()

    counts = _recount_from_disk()
    for k in ("complexity", "regimes"):
        stats[k] = counts[k]
    stats["base_goals"] = counts["base_goals"]
    stats["distinct_instances"] = counts["distinct_instances"]
    stats["skipped_pre_grounding"] = gate.skipped
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    _write_json_atomic(STAGING_20K / "build_report.json", stats)
    return stats


# ---------------------------------------------------------------------------
# top-up (state-axis variants, legacy precedent) — closes any shortfall
# ---------------------------------------------------------------------------

def run_topup(target_total: int = 800, max_inst: int = 12) -> dict[str, Any]:
    from benchmark.plugins.task_family.l12.expand20k.generators import _GENERATORS_20K, _ids

    plugin = Pm4pyTaskFamilies20k()
    verifier = Pm4pyVerifier20k()
    snapshot = fetcher.load_snapshot()
    source_check = _verify_source_units()
    legacy = _legacy_collision_index()

    existing = _iter_qualified_20k()
    gate = _Gate(legacy)
    for rec in existing:
        gate.instance_ids.add(rec["instance_id"])
        gate.task_ids.add(rec["task_id"])
        gate.public_hashes.add(engine.sha256_json(rec["public"]))
    oracle_dir = STAGING_20K / "oracle"
    for rec in existing:
        opath = oracle_dir / ("%s.json" % rec["oracle_handle"])
        oracle = json.loads(opath.read_text(encoding="utf-8"))
        ast = oracle["ast"]
        gate.signatures.add(variant_signature(ast))
        gate.content_keys.add(_content_key(ast))
        if ast["goal_kind"] == "exact_artifact":
            gate.answer_keys.add(answer_key(ast["reference_answer"]))

    next_shard = _next_shard_idx(STAGING_20K / "qualified")
    qwriter = ShardWriter(STAGING_20K / "qualified")
    qwriter.shard_idx = next_shard
    cwriter = ShardWriter(STAGING_20K / "candidates")
    cwriter.shard_idx = _next_shard_idx(STAGING_20K / "candidates")
    rejection_path = STAGING_20K / "rejections" / "rejections.jsonl"

    accepted: list[dict[str, Any]] = []
    t0 = time.time()
    with rejection_path.open("a", encoding="utf-8") as rej_fh:
        outer_break = False
        for inst in range(INSTANCES_PER_GOAL, max_inst):
            if outer_break:
                break
            for spec in FAMILIES_20K:
                if outer_break:
                    break
                gen = _GENERATORS_20K[spec.family_id]
                for base in spec.bases:
                    if len(existing) + len(accepted) >= target_total:
                        outer_break = True
                        break
                    payload = gen(spec, base, inst)
                    if payload is None:
                        continue
                    base_goal_id, instance_id = _ids(spec.family_id, dict(base), inst)
                    ast = {
                        "goal_kind": spec.goal_kind,
                        "operation": spec.operation,
                        "regime": spec.regime,
                        "generator": GENERATOR_VERSION,
                        "seed": SEED_BASE,
                        **payload,
                    }
                    rels = [payload["log_rel"]]
                    if payload.get("net_rel"):
                        rels.append(payload["net_rel"])
                    candidate = CandidateTaskAST(
                        lane_id="L12",
                        family_id=spec.family_id,
                        base_goal_id=base_goal_id,
                        instance_id=instance_id,
                        regime=spec.regime,
                        semantic_profile=SEMANTIC_PROFILE,
                        ast=ast,
                        source_unit_ids=tuple(engine.unit_id_for(r) for r in rels),
                    )
                    skip = gate.pre_grounding(candidate)
                    if skip is not None:
                        gate._skip(skip)
                        continue
                    record, rejection = _qualify_one(
                        plugin, verifier, candidate, source_check["ok"])
                    pub_skip = gate.post_grounding(record)
                    if pub_skip is not None:
                        gate._skip(pub_skip)
                        rejection = {"code": "DUPLICATE",
                                     "reason": "public payload collision: %s" % pub_skip}
                    cwriter.add(record)
                    if rejection is not None:
                        record["status"] = "REJECTED"
                        record["checks"]["qualification_status"] = "REJECTED"
                        record["qualification"] = {
                            "native_verified": False, "solvable": False,
                            "nontrivial": False, "semantic_relevance": False,
                            "leakage_clean": False, "release_eligible": False,
                        }
                        rej_fh.write(json.dumps({
                            "task_id": record["task_id"],
                            "family_id": record["family_id"],
                            "base_goal_id": record["base_goal_id"],
                            **rejection,
                        }, sort_keys=True) + "\n")
                        opath = (STAGING_20K / "oracle" /
                                 ("%s.json" % record["oracle_handle"]))
                        if opath.is_file():
                            opath.unlink()
                        continue
                    gate.admit(candidate, record)
                    accepted.append(record)
                    qwriter.add(record)
    qwriter.close()
    cwriter.close()

    counts = _recount_from_disk()
    report_path = STAGING_20K / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) \
        if report_path.is_file() else {}
    report.update(counts)
    report["top_up"] = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": "close the gap to the 800-new-task lane target",
        "state_axis": "per-base-goal instance index (deterministic variants inst>=5)",
        "target_total": target_total,
        "added_qualified": len(accepted),
        "skipped_duplicate_variants": gate.skipped,
        "new_task_ids": [r["task_id"] for r in accepted],
        "wall_seconds": round(time.time() - t0, 2),
    }
    _write_json_atomic(report_path, report)
    return report["top_up"]


def _next_shard_idx(root: Path) -> int:
    if not root.is_dir():
        return 0
    idxs = []
    for p in root.glob("shard-*.jsonl"):
        try:
            idxs.append(int(p.stem.split("-")[1]))
        except (ValueError, IndexError):
            continue
    return (max(idxs) + 1) if idxs else 0


def _iter_qualified_20k() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    qdir = STAGING_20K / "qualified"
    for path in sorted(qdir.glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _recount_from_disk() -> dict[str, Any]:
    qrec = _iter_qualified_20k()
    crec: list[dict[str, Any]] = []
    for p in sorted((STAGING_20K / "candidates").glob("shard-*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    crec.append(json.loads(line))
    rej_path = STAGING_20K / "rejections" / "rejections.jsonl"
    rej: list[dict[str, Any]] = []
    if rej_path.is_file():
        for line in rej_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rej.append(json.loads(line))
    fams: dict[str, dict[str, int]] = {}
    for r in crec:
        f = fams.setdefault(r["family_id"],
                            {"candidates": 0, "qualified": 0, "rejected": 0})
        f["candidates"] += 1
    for r in qrec:
        fams.setdefault(r["family_id"],
                        {"candidates": 0, "qualified": 0, "rejected": 0})
        fams[r["family_id"]]["qualified"] += 1
    for r in rej:
        fams.setdefault(r["family_id"],
                        {"candidates": 0, "qualified": 0, "rejected": 0})
        fams[r["family_id"]]["rejected"] += 1
    rej_counts: dict[str, int] = {}
    for r in rej:
        rej_counts[r["code"]] = rej_counts.get(r["code"], 0) + 1
    qman = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((STAGING_20K / "qualified").glob("*.manifest.json"))]
    cman = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((STAGING_20K / "candidates").glob("*.manifest.json"))]
    complexity: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    regimes: dict[str, int] = {}
    for r in qrec:
        complexity[r["complexity"]] = complexity.get(r["complexity"], 0) + 1
        regimes[r["task_regime"]] = regimes.get(r["task_regime"], 0) + 1
    return {
        "candidates": len(crec),
        "qualified": len(qrec),
        "distinct_instances": len({r["instance_id"] for r in qrec}),
        "base_goals": len({r["base_goal_id"] for r in qrec}),
        "families": fams,
        "rejections": rej_counts,
        "complexity": complexity,
        "regimes": regimes,
        "qualified_manifests": qman,
        "candidate_manifests": cman,
    }


# ---------------------------------------------------------------------------
# audit (independent re-verification from persisted oracles)
# ---------------------------------------------------------------------------

def run_audit(*, task_id: str | None = None) -> dict[str, Any]:
    verifier = Pm4pyVerifier20k()
    oracle_dir = STAGING_20K / "oracle"
    records = _iter_qualified_20k()
    if task_id:
        records = [r for r in records if r["task_id"] == task_id]
    per_task: list[dict[str, Any]] = []
    invalid = 0
    t0 = time.time()
    for record in records:
        t_task = time.time()
        oracle_path = oracle_dir / ("%s.json" % record["oracle_handle"])
        if not oracle_path.is_file():
            per_task.append({
                "task_id": record["task_id"], "family_id": record["family_id"],
                "valid": False, "schema_valid": None,
                "reason": "oracle_missing", "wall_ms": 0,
            })
            invalid += 1
            continue
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
            grounded, _reference_submission(oracle["ast"]))
        schema_errors = _validate_record(record, full=True)
        row = {
            "task_id": record["task_id"],
            "family_id": record["family_id"],
            "complexity": record["complexity"],
            "valid": bool(verdict.valid),
            "schema_valid": not schema_errors,
            "schema_errors": schema_errors[:5],
            "wall_ms": round((time.time() - t_task) * 1000, 1),
        }
        if not verdict.valid:
            row["detail"] = verdict.detail
            invalid += 1
        per_task.append(row)
    report = {
        "audit": "verify_terminal(reference) recomputed from pinned source + persisted oracle",
        "dataset_id": DATASET_ID,
        "records": len(records),
        "checked": len(per_task),
        "valid": len(per_task) - invalid,
        "invalid_count": invalid,
        "wall_seconds": round(time.time() - t0, 2),
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not task_id:
        out = STAGING_20K / "audit_report.json"
        _write_json_atomic(out, {"summary": report, "per_task": per_task})
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="L12 20K expansion build runner")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--instances-per-goal", type=int, default=INSTANCES_PER_GOAL)
    ap.add_argument("--seed", type=int, default=SEED_BASE)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--topup", action="store_true",
                    help="close any shortfall to the 800-new-task target")
    ap.add_argument("--topup-target", type=int, default=800)
    args = ap.parse_args()
    if args.audit:
        report = run_audit(task_id=args.task_id)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not report["invalid_count"] else 1
    if args.topup:
        top = run_topup(target_total=args.topup_target)
        print(json.dumps(top, indent=2, sort_keys=True))
        return 0
    stats = run_build(
        families=args.families,
        instances_per_goal=args.instances_per_goal,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("qualified_manifests", "candidate_manifests")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
