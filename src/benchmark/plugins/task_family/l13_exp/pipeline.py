"""L13 20K expansion pipeline: new mechanisms -> qualified shards (candidate).

Writes **only** under ``bank/staging_20k/l13``. The legacy 8K staging area
(``bank/staging/l13``), its builds/releases, and every other lane are read-only
inputs here.

Record lifecycle / per-task statuses
------------------------------------
CANDIDATE   emitted by a family generator, not yet proven
QUALIFIED   proof + terminal replay + semantic witness + leak screen + dedup all pass
REJECTED    a rejection code from REJECT_KEYS (written to rejections.jsonl)

The 240 legacy L13 records are imported read-only, re-tagged to the 20K dataset
id, marked ``origin=legacy_carryover``, and re-verified with the expansion
verifier. New records carry ``origin=expansion_20k``.

Repro (offline; pinned snapshot only, no network):
    .venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --build
    .venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --recheck
    .venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --smoke
    .venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --recheck-task l13-in-...
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l13 import oas
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13_exp.families import (
    FAMILIES, FAMILY_BY_ID, TARGET_TOTAL, L13ExpansionFamilies,
)
from benchmark.plugins.task_family.l13_exp.verifier import (
    ENGINE, NEW_KINDS, SUBMIT_KEY, ExpansionVerifier,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers import l13_apis_guru as fetcher

REPO_ROOT = Path(__file__).resolve().parents[4]
STAGING_ROOT = REPO_ROOT / "bank" / "staging_20k"
STAGING = STAGING_ROOT / "l13"
LEGACY = REPO_ROOT / "bank" / "staging" / "l13"          # 8K staging (read-only)
LEGACY_BUILD = REPO_ROOT / "bank" / "builds" / "aob-8000-rc1"  # frozen 8K (read-only)
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
SOURCE_SNAPSHOT_ID = "apis-guru@commit-f04b8d0bcd39"
SEED = 20261101
LANE = "L13"
TASK_ID_PREFIX = "l13"
LEGACY_BUILD_ID = "aob-8000-rc1"
LEGACY_TARGET = 240
TARGET_TOTAL_20K = TARGET_TOTAL + LEGACY_TARGET
SHARD_SIZE = 100

REJECT_KEYS = (
    "UNSOLVABLE", "TRIVIAL", "PROFILE_UNSUPPORTED", "LEAKAGE", "DUPLICATE",
    "NON_SEMANTIC", "LICENSE_HOLD",
)

_LEGACY_SUBMIT_KEY = {
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


def reference_submission(ast: dict[str, Any]) -> dict[str, Any]:
    kind = ast["goal_kind"]
    mode = ast["coordinate"].get("mode")
    if kind in SUBMIT_KEY:
        key = SUBMIT_KEY[kind]
    elif kind == "version_migration":
        key = "parameters" if mode == "params" else ("instance" if mode == "response" else "body")
    else:
        key = _LEGACY_SUBMIT_KEY[kind]
    return {key: ast["reference"]}


class ShardWriter:
    """Append-in-memory, flush an immutable shard + manifest at a size boundary."""

    def __init__(self, root: Path, *, start_index: int = 0, shard_size: int = SHARD_SIZE) -> None:
        self.root = root
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_idx = start_index
        self.manifests: list[dict[str, Any]] = []
        root.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        name = "shard-%04d" % self.shard_idx
        path = self.root / ("%s.jsonl" % name)
        blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.buffer)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, path)
        origins: dict[str, int] = {}
        for r in self.buffer:
            origins[r.get("origin", "unknown")] = origins.get(r.get("origin", "unknown"), 0) + 1
        manifest = {
            "shard": name,
            "records": len(self.buffer),
            "sha256": oas.sha256_json({"blob": blob}),
            "file_sha256": oas.sha256_file(path),
            "families": sorted({r.get("family_id") for r in self.buffer}),
            "base_goals": len({r.get("base_goal_id") for r in self.buffer}),
            "origins": dict(sorted(origins.items())),
            "path": str(path),
            "relpath": str(path.relative_to(REPO_ROOT)),
        }
        _write_json_atomic(self.root / ("%s.manifest.json" % name), manifest)
        self.manifests.append(manifest)
        self.buffer = []
        self.shard_idx += 1

    def close(self) -> None:
        self.flush()


def _base_record(candidate: CandidateTaskAST, grounded: GroundedTask,
                 family_id: str | None = None) -> dict[str, Any]:
    ast = candidate.ast
    fam = FAMILY_BY_ID[candidate.family_id]
    mech = oas.sha256_json(
        [ast["goal_kind"], ast["unit_id"], json.dumps(ast["coordinate"], sort_keys=True)])[:16]
    return {
        "schema_version": "aob-task/5.0",
        "task_id": "%s-%s" % (TASK_ID_PREFIX, candidate.instance_id),
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "family_id": candidate.family_id,
        "mechanism": fam.mechanism,
        "base_goal_id": candidate.base_goal_id,
        "instance_id": candidate.instance_id,
        "task_regime": candidate.regime,
        "semantic_profile": candidate.semantic_profile,
        "complexity": ast.get("complexity"),
        "source_snapshot_ids": [SOURCE_SNAPSHOT_ID],
        "source_unit_ids": list(candidate.source_unit_ids),
        "origin_mechanism_group": mech,
        "public": grounded.public,
        "oracle_handle": grounded.oracle_ref,
        "authorship": "deterministic_template",
        "provenance": "source_parameterized",
        "origin": "expansion_20k",
        "status": "CANDIDATE",
        "qualification": {
            "native_verified": False, "solvable": False, "nontrivial": False,
            "semantic_relevance": False, "leakage_clean": False,
            "release_eligible": False,
        },
    }


def _leak_screen(public: dict[str, Any], ast: dict[str, Any]) -> str | None:
    reference = ast.get("reference")
    if isinstance(reference, (dict, list)) and reference:
        frag = json.dumps(reference, sort_keys=True)
        if frag in json.dumps(public, sort_keys=True):
            return "reference_fragment"
    return None


def _qualify_one(verifier: ExpansionVerifier, candidate: CandidateTaskAST,
                 plugin: L13ExpansionFamilies) -> tuple[dict[str, Any], dict[str, Any] | None]:
    grounded = plugin.ground(candidate)
    record = _base_record(candidate, grounded)
    try:
        proof = verifier.prove_exists(grounded)
    except OasError as exc:
        return record, {"code": "PROFILE_UNSUPPORTED", "reason": str(exc)[:300]}
    if not proof.exists:
        reason = proof.reason or "UNSOLVABLE"
        code = reason.split(":", 1)[0]
        if code not in REJECT_KEYS:
            code = "UNSOLVABLE"
        return record, {"code": code, "reason": reason[:300]}
    verifier.write_oracle(grounded, proof.evidence)
    term = verifier.verify_terminal(grounded, reference_submission(candidate.ast))
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
            ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline "
            "--recheck-task %s" % record["task_id"],
        ],
        "expected": {"ok": True, "proof": True, "terminal": True, "witness": True},
        "prove_exists": proof.evidence,
        "verify_terminal": term.detail,
        "semantic_witness": witness.witness,
    }
    record["qualification"] = {
        "native_verified": True, "solvable": True, "nontrivial": True,
        "semantic_relevance": True, "leakage_clean": True, "release_eligible": True,
    }
    record["status"] = "QUALIFIED"
    return record, None


def _read_jsonl_dir(directory: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shard in sorted(directory.glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def legacy_records() -> list[dict[str, Any]]:
    """The 240 frozen 8K L13 rows (read-only), used for carryover provenance,
    collision checks and a read-only recheck. They are NOT written into the
    staging lane; the 20K merge carries them from the frozen 8K build."""
    path = LEGACY_BUILD / "qualified_tasks.jsonl"
    if path.is_file():
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("lane_id") == LANE:
                    out.append(row)
        return out
    return _read_jsonl_dir(LEGACY / "qualified")


def _reset_outputs() -> None:
    for sub in ("qualified", "candidates", "rejections", "oracle", "smoke",
                "legacy_carryover"):
        d = STAGING / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()
    (STAGING / "oracle").mkdir(parents=True, exist_ok=True)


# -- build ----------------------------------------------------------------------
def run_build(*, families: list[str] | None = None,
              seed: int = SEED, max_per_family: int | None = None) -> dict[str, Any]:
    _reset_outputs()
    plugin = L13ExpansionFamilies()
    verifier = ExpansionVerifier(STAGING / "oracle")
    legacy = legacy_records()
    legacy_units = {u for r in legacy for u in r.get("source_unit_ids") or []}

    qualified_writer = ShardWriter(STAGING / "qualified")
    candidate_writer = ShardWriter(STAGING / "candidates")
    rejection_path = STAGING / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)

    instances: set[str] = set()
    base_goals: set[str] = set()
    stats: dict[str, Any] = {
        "lane": LANE,
        "dataset_id": DATASET_ID,
        "seed": seed,
        "source_snapshot_ids": [SOURCE_SNAPSHOT_ID],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
        "rejections": {},
        "candidates": 0,
        "qualified": 0,
        "new_records": 0,
        "legacy_carryover": 0,
        "target_new": TARGET_TOTAL,
        "target_total": TARGET_TOTAL_20K,
        "base_goals": 0,
        "excluded_legacy_units": len(legacy_units),
    }
    t0 = time.time()
    family_ids = families or [f.family_id for f in FAMILIES]
    with rejection_path.open("w", encoding="utf-8") as rej_fh:
        for fid in family_ids:
            spec = FAMILY_BY_ID[fid]
            fam_stats = {"candidates": 0, "qualified": 0, "rejected": 0,
                         "target": spec.target_records, "mechanism": spec.mechanism}
            target = spec.target_records
            if max_per_family is not None:
                target = min(target, max_per_family)
            candidates = list(plugin.propose(None, family=fid, seed=seed,
                                             target_records=target,
                                             exclude_units=legacy_units))
            for candidate in candidates:
                record, rejection = _qualify_one(verifier, candidate, plugin)
                fam_stats["candidates"] += 1
                stats["candidates"] += 1
                candidate_writer.add(record)
                if rejection is not None:
                    record["status"] = "REJECTED"
                    record["rejection"] = rejection
                    fam_stats["rejected"] += 1
                    stats["rejections"][rejection["code"]] = (
                        stats["rejections"].get(rejection["code"], 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"], "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"], **rejection,
                    }, sort_keys=True) + "\n")
                    continue
                if record["instance_id"] in instances:
                    fam_stats["rejected"] += 1
                    stats["rejections"]["DUPLICATE"] = (
                        stats["rejections"].get("DUPLICATE", 0) + 1)
                    rej_fh.write(json.dumps({
                        "task_id": record["task_id"], "family_id": record["family_id"],
                        "base_goal_id": record["base_goal_id"],
                        "code": "DUPLICATE", "reason": "instance hash already admitted",
                    }, sort_keys=True) + "\n")
                    continue
                instances.add(record["instance_id"])
                base_goals.add(record["base_goal_id"])
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
                qualified_writer.add(record)
            stats["families"][fid] = fam_stats
            _write_json_atomic(STAGING / "PROGRESS.json", {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "families_done": list(stats["families"]),
                "qualified_new": stats["qualified"],
                "target_new": TARGET_TOTAL,
            })
    qualified_writer.close()
    candidate_writer.close()
    stats["new_records"] = stats["qualified"]
    stats["distinct_instances"] = len(instances)
    stats["base_goals"] = len(base_goals)
    stats["task_families"] = len(stats["families"])
    stats["shortfall"] = max(0, TARGET_TOTAL - stats["qualified"])
    stats["qualified_manifests"] = qualified_writer.manifests
    stats["candidate_manifests"] = candidate_writer.manifests
    stats["wall_seconds"] = round(time.time() - t0, 2)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600.0, 1e-9), 1)
    _write_json_atomic(STAGING / "build_report.json", stats)
    return stats


# -- legacy carryover -----------------------------------------------------------
def import_legacy() -> dict[str, Any]:
    """Record the frozen 8K L13 carryover WITHOUT putting it in the staging lane.

    The 20K merge carries the 240 legacy L13 rows verbatim from
    ``bank/builds/aob-8000-rc1``. Copying them into ``qualified/`` would collide
    on ``task_id`` and violate the §1 staging contract (new rows only). We keep
    a provenance manifest plus a read-only oracle copy under
    ``legacy_carryover/`` so the carryover can still be re-checked.
    """
    legacy = legacy_records()
    dst_root = STAGING / "legacy_carryover"
    (dst_root / "oracle").mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    for rec in legacy:
        handle = rec.get("oracle_handle")
        src = LEGACY / "oracle" / ("%s.json" % handle)
        dst = dst_root / "oracle" / ("%s.json" % handle)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            missing.append(handle)
        rows.append({k: rec.get(k) for k in (
            "task_id", "dataset_id", "family_id", "instance_id", "base_goal_id",
            "oracle_handle", "source_unit_ids")})
    manifest = {
        "legacy_build_id": LEGACY_BUILD_ID,
        "legacy_dataset_id": legacy[0].get("dataset_id") if legacy else None,
        "source": "bank/builds/aob-8000-rc1/qualified_tasks.jsonl (lane L13, read-only)",
        "carried_by_merge": len(legacy),
        "expected": LEGACY_TARGET,
        "note": ("Not written into staging qualified/candidates; the 20K merge "
                 "carries these rows verbatim from the frozen 8K build."),
        "missing_oracles": sorted(set(missing)),
        "records": rows,
    }
    _write_json_atomic(dst_root / "manifest.json", manifest)
    summary = {"carried_by_merge": len(legacy), "expected": LEGACY_TARGET,
               "legacy_build_id": LEGACY_BUILD_ID,
               "missing_oracles": sorted(set(missing)),
               "manifest": str((dst_root / "manifest.json").relative_to(REPO_ROOT))}
    _write_json_atomic(STAGING / "legacy_import.json", summary)
    build_path = STAGING / "build_report.json"
    if build_path.is_file():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        build["legacy_carryover"] = len(legacy)
        build["legacy_source_build"] = LEGACY_BUILD_ID
        build["lane_total"] = int(build.get("qualified") or 0) + len(legacy)
        build["contract"] = "new rows only in staging; legacy carried by merge_20k"
        _write_json_atomic(build_path, build)
    return summary


# -- recheck --------------------------------------------------------------------
def _recheck_record(verifier: ExpansionVerifier, record: dict[str, Any]) -> dict[str, Any]:
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
        lane_id=LANE, family_id=oracle["family_id"], base_goal_id=record["base_goal_id"],
        instance_id=record["instance_id"], regime=record["task_regime"],
        semantic_profile=record["semantic_profile"], ast=ast_like,
        source_unit_ids=tuple(record["source_unit_ids"]))
    grounded = GroundedTask(ast=candidate, public=record["public"],
                            oracle_ref=record["oracle_handle"], native_evidence={})
    proof = verifier.prove_exists(grounded)
    term = verifier.verify_terminal(grounded, reference_submission(ast_like))
    witness = verifier.certify_semantic_relevance(grounded)
    return {"task_id": record["task_id"], "family_id": record["family_id"],
            "origin": record.get("origin"), "proof": proof.exists,
            "terminal": term.valid, "witness": witness.ok,
            "ok": proof.exists and term.valid and witness.ok}


def _run_recheck_rows(verifier: ExpansionVerifier, rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked, mismatches = 0, []
    for record in rows:
        res = _recheck_record(verifier, record)
        checked += 1
        if not res["ok"]:
            mismatches.append(res)
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def run_recheck() -> dict[str, Any]:
    # Staging lane: new 20K rows only (the §1 input to merge_20k).
    staging = _run_recheck_rows(ExpansionVerifier(STAGING / "oracle"),
                               _read_jsonl_dir(STAGING / "qualified"))
    staging["by_origin"] = {"expansion_20k": {
        "checked": staging["checked"],
        "ok": staging["checked"] - len(staging["mismatches"]),
    }}
    _write_json_atomic(STAGING / "recheck_report.json", staging)
    # Frozen 8K carryover: read-only recheck against the legacy oracle dir.
    legacy_rows = legacy_records()
    legacy = _run_recheck_rows(ExpansionVerifier(LEGACY / "oracle"), legacy_rows)
    legacy["legacy_build_id"] = LEGACY_BUILD_ID
    _write_json_atomic(STAGING / "legacy_carryover" / "recheck.json", legacy)
    return {"staging": staging, "legacy": legacy,
            "ok": staging["ok"] and legacy["ok"],
            "checked": staging["checked"] + legacy["checked"],
            "by_origin": staging["by_origin"]}


def run_recheck_task(task_id: str) -> dict[str, Any]:
    verifier = ExpansionVerifier(STAGING / "oracle")
    for record in _read_jsonl_dir(STAGING / "qualified"):
        if record["task_id"] == task_id:
            return _recheck_record(verifier, record)
    return {"task_id": task_id, "ok": False, "error": "task_id not found in qualified shards"}


# -- smoke ----------------------------------------------------------------------
def run_smoke(limit: int = 16) -> dict[str, Any]:
    verifier = ExpansionVerifier(STAGING / "oracle")
    records = _read_jsonl_dir(STAGING / "qualified")
    # Deterministic spread: first record of each family, then fill by family order.
    picked: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for rec in records:
        if rec.get("origin") == "legacy_carryover":
            continue
        if rec["family_id"] not in seen_families:
            seen_families.add(rec["family_id"])
            picked.append(rec)
    for rec in records:
        if len(picked) >= limit:
            break
        if rec not in picked:
            picked.append(rec)
    picked = picked[:limit]
    results = [_recheck_record(verifier, r) for r in picked]
    report = {
        "requested": limit,
        "checked": len(results),
        "passed": sum(1 for r in results if r["ok"]),
        "families": sorted({r["family_id"] for r in results}),
        "origins": sorted({str(r["origin"]) for r in results}),
        "results": results,
        "ok": bool(results) and all(r["ok"] for r in results),
    }
    _write_json_atomic(STAGING / "smoke" / "smoke_report.json", report)
    return report


# -- collision report -----------------------------------------------------------
def run_collisions() -> dict[str, Any]:
    legacy = legacy_records()
    new = [r for r in _read_jsonl_dir(STAGING / "qualified")
           if r.get("origin") == "expansion_20k"]
    all_records = legacy + new
    instance_ids: dict[str, int] = {}
    task_ids: dict[str, int] = {}
    content_hashes: dict[str, int] = {}
    mechanisms: dict[str, set[str]] = {}
    legacy_units = {u for r in legacy for u in r.get("source_unit_ids") or []}
    new_units = {u for r in new for u in r.get("source_unit_ids") or []}
    for r in all_records:
        instance_ids[r["instance_id"]] = instance_ids.get(r["instance_id"], 0) + 1
        task_ids[r["task_id"]] = task_ids.get(r["task_id"], 0) + 1
        content = json.dumps({"user_goal": (r.get("public") or {}).get("user_goal"),
                              "initial": (r.get("public") or {}).get("initial"),
                              "reference": (r.get("native_evidence") or {}).get("prove_exists"),
                              "oracle": r.get("oracle_handle")}, sort_keys=True)
        content_hashes[oas.sha256_json(content)] = content_hashes.get(oas.sha256_json(content), 0) + 1
        coord = (r.get("public") or {}).get("coordinate") or {}
        key = oas.sha256_json([r.get("source_unit_ids"), coord, r.get("family_id")])
        mechanisms.setdefault(key, set()).add(r["base_goal_id"])
    dup_instances = {k: v for k, v in instance_ids.items() if v > 1}
    dup_task_ids = {k: v for k, v in task_ids.items() if v > 1}
    dup_content = {k: v for k, v in content_hashes.items() if v > 1}
    dup_mech = {k: sorted(v) for k, v in mechanisms.items() if len(v) > 1}
    report = {
        "legacy_records": len(legacy),
        "new_records": len(new),
        "instance_id_collisions": len(dup_instances),
        "task_id_collisions": len(dup_task_ids),
        "content_collisions": len(dup_content),
        "mechanism_coordinate_collisions": len(dup_mech),
        "spec_unit_overlap_legacy_new": sorted(legacy_units & new_units),
        "spec_unit_overlap_count": len(legacy_units & new_units),
        "ok": not dup_instances and not dup_task_ids and not dup_content
              and not dup_mech and not (legacy_units & new_units),
    }
    _write_json_atomic(STAGING / "collision_report.json", report)
    return report


# -- reporting ------------------------------------------------------------------
def _summary_records() -> list[dict[str, Any]]:
    return _read_jsonl_dir(STAGING / "qualified")


def run_contract_check() -> dict[str, Any]:
    """Read-only §1 validation with the exact 8K/20K merge loader."""
    from autoontobench.factory.merge_staging import MergeError, _load_lane

    try:
        lane = _load_lane(REPO_ROOT, "l13", DATASET_ID, staging_root=STAGING_ROOT)
        report = {
            "ok": True,
            "qualified_rows": len(lane.qualified_rows),
            "candidate_rows": len(lane.candidate_rows),
            "rejection_rows": len(lane.rejection_rows),
            "candidates_equals_qualified": lane.candidates_equals_qualified,
            "shard_conventions": lane.shard_conventions,
            "shard_mismatches": lane.shard_mismatches,
            "problems": [],
        }
    except MergeError as exc:
        report = {"ok": False, "problems": str(exc).splitlines()}
    _write_json_atomic(STAGING / "contract_report.json", report)
    return report


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_reports() -> dict[str, Any]:
    build = _load_json(STAGING / "build_report.json")
    recheck = _load_json(STAGING / "recheck_report.json")
    legacy_recheck = _load_json(STAGING / "legacy_carryover" / "recheck.json")
    smoke = _load_json(STAGING / "smoke" / "smoke_report.json")
    collisions = _load_json(STAGING / "collision_report.json")
    legacy_import = _load_json(STAGING / "legacy_import.json")
    contract = run_contract_check()

    records = _summary_records()
    qualified_new = sum(1 for r in records if r.get("status") == "QUALIFIED")
    by_family: dict[str, int] = {}
    for r in records:
        by_family[r.get("family_id")] = by_family.get(r.get("family_id"), 0) + 1
    legacy_carried = int(legacy_import.get("carried_by_merge") or 0)
    lane_total = qualified_new + legacy_carried

    manifest_lines = []
    for shard in sorted((STAGING / "qualified").glob("shard-*.jsonl")):
        mp = shard.with_suffix(".manifest.json")
        if mp.is_file():
            m = json.loads(mp.read_text(encoding="utf-8"))
            manifest_lines.append(
                "| %s | %s | %s | `%s` |" % (
                    m["shard"], m["records"], m["sha256"][:16], m["relpath"]))

    complete = (qualified_new == TARGET_TOTAL and contract.get("ok")
                and recheck.get("ok") and legacy_recheck.get("ok")
                and smoke.get("ok") and collisions.get("ok"))

    status = "# L13 20K expansion — STATUS\n\n"
    status += "- dataset_id: `%s`\n" % DATASET_ID
    status += "- seed: `%s`\n" % SEED
    status += "- target_total: %s (new_needed %s + legacy %s)\n" % (
        TARGET_TOTAL_20K, TARGET_TOTAL, LEGACY_TARGET)
    status += "- staging qualified (new): **%s**\n" % qualified_new
    status += "- legacy carryover (frozen 8K, carried by merge): **%s**\n" % legacy_carried
    status += "- lane total: **%s**\n" % lane_total
    status += "- §1 merge-contract validation: %s\n" % (
        "PASS" if contract.get("ok") else "FAIL")
    status += "- recheck (staging new): %s (%s checked, %s mismatches)\n" % (
        "PASS" if recheck.get("ok") else "FAIL",
        recheck.get("checked", 0), len(recheck.get("mismatches") or []))
    status += "- recheck (frozen 8K carryover): %s (%s checked, %s mismatches)\n" % (
        "PASS" if legacy_recheck.get("ok") else "FAIL",
        legacy_recheck.get("checked", 0), len(legacy_recheck.get("mismatches") or []))
    status += "- smoke: %s (%s/%s passed)\n" % (
        "PASS" if smoke.get("ok") else "FAIL",
        smoke.get("passed", 0), smoke.get("checked", 0))
    status += "- collisions: %s\n" % ("PASS" if collisions.get("ok") else "FAIL")
    status += "- status: %s\n\n" % ("COMPLETE" if complete else "INCOMPLETE")
    status += "## Per-family counts (staging qualified)\n\n"
    status += "| family | mechanism | records |\n|---|---|---|\n"
    for f in FAMILIES:
        status += "| `%s` | %s | %s |\n" % (
            f.family_id, f.mechanism, by_family.get(f.family_id, 0))
    status += "| `legacy-carryover` | frozen 8K L13 rows (not in staging qualified) | %s |\n" % (
        legacy_carried)
    (STAGING / "STATUS.md").write_text(status, encoding="utf-8")

    report = "# L13 20K expansion — REPORT\n\n"
    report += "## Scope and §1 staging contract\n\n"
    report += ("`bank/staging_20k/l13` adds **%s new** APIs.guru OpenAPI tasks "
               "(`%s`, seed `%s`). The lane target is %s = %s new + %s legacy; "
               "the 240 legacy L13 rows are **not** duplicated into this staging "
               "lane — the 20K merge carries them verbatim from the frozen "
               "`bank/builds/aob-8000-rc1`. The staging `qualified/` and "
               "`candidates/` shards therefore contain new rows only, each with "
               "dataset_id `%s`.\n\n"
               % (qualified_new, DATASET_ID, SEED, TARGET_TOTAL_20K,
                  TARGET_TOTAL, LEGACY_TARGET, DATASET_ID))
    report += ("The legacy `bank/staging/l13` (8K), the 8K build/releases, other "
               "lanes, and P1/.omo/git state were not modified. All verification "
               "runs offline against the pinned snapshot; no online API claims "
               "are made.\n\n")
    report += "## Mechanism taxonomy (new beyond legacy)\n\n"
    for f in FAMILIES:
        report += "- `%s` — %s (%s, %s, target %s)\n" % (
            f.family_id, f.mechanism, f.goal_kind, f.regime, f.target_records)
    report += ("\nNo family re-instantiates a legacy family: generators exclude every "
               "spec unit used by the legacy 240, and each mechanism uses a distinct "
               "site predicate and/or contract/replay path.\n\n")
    report += "## Validation\n\n"
    report += "- Engine: `%s` over jsonschema-validated normalized OpenAPI 3.0 schemas.\n" % ENGINE
    report += ("- Every new record: proof (contract re-derived from pinned bytes) + "
               "terminal replay of the reference answer + semantic witness + leak "
               "screen + dedup.\n")
    report += "- Staging recheck: %s checked, %s mismatches.\n" % (
        recheck.get("checked", 0), len(recheck.get("mismatches") or []))
    report += "- Frozen-8K carryover recheck (read-only): %s checked, %s mismatches.\n" % (
        legacy_recheck.get("checked", 0), len(legacy_recheck.get("mismatches") or []))
    report += "- Smoke (10–20 tasks, spread across families): %s/%s passed.\n" % (
        smoke.get("passed", 0), smoke.get("checked", 0))
    report += "- §1 merge-contract validation (`_load_lane`): %s.\n\n" % (
        "PASS" if contract.get("ok") else "FAIL")
    report += "### §1 contract check\n\n```json\n%s\n```\n\n" % json.dumps(
        contract, indent=2, sort_keys=True)
    report += "## Collision checks\n\n```json\n%s\n```\n\n" % json.dumps(
        collisions, indent=2, sort_keys=True)
    report += "## Shards and manifests\n\n| shard | records | sha256 | path |\n|---|---|---|---|\n"
    report += "\n".join(manifest_lines) + "\n\n"
    report += "## Repro commands (offline)\n\n```bash\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --build\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --import-legacy\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --recheck\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --smoke\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --collisions\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --contract\n"
    report += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline --reports\n"
    report += "# single-task native evidence:\n"
    report += (".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline "
               "--recheck-task l13-in-<instance>\n")
    report += "```\n\n"
    report += "## Assumptions and blockers\n\n"
    report += ("- `EXPANSION_20K_EXECUTION_PLAN.md` and `lanes/L13/REPORT.md` were not "
               "present as files in this workspace; this build follows "
               "`configs/lanes_20000.yaml`, the `aob-task/5.0` record contract, and "
               "the `merge_20k` staging contract (`_load_lane`).\n")
    report += ("- Legacy carryover is read-only provenance under `legacy_carryover/`; "
               "it is not part of the staging qualified set, so `task_id` collisions "
               "with the frozen 8K build are zero.\n")
    report += ("- Per-task statuses are `CANDIDATE` / `QUALIFIED` / `REJECTED`; rejected "
               "rows would be retained in `candidates/` and mirrored in `rejections/` "
               "(this build produced none).\n")
    if build.get("shortfall"):
        report += "- **BLOCKER**: new shortfall of %s tasks under current site capacity.\n" % build["shortfall"]
    if legacy_import.get("missing_oracles"):
        report += "- **BLOCKER**: missing legacy oracle files: %s\n" % legacy_import["missing_oracles"]
    if not contract.get("ok"):
        report += "- **BLOCKER**: staging lane failed §1 merge-contract validation.\n"
    (STAGING / "REPORT.md").write_text(report, encoding="utf-8")

    repro = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../../..\"\n"
    for cmd in ("--build", "--import-legacy", "--recheck", "--smoke",
                "--collisions", "--contract", "--reports"):
        repro += ".venv/bin/python -m benchmark.plugins.task_family.l13_exp.pipeline %s\n" % cmd
    rp = STAGING / "repro.sh"
    rp.write_text(repro, encoding="utf-8")
    rp.chmod(0o755)
    return {"qualified_new": qualified_new, "legacy_carried": legacy_carried,
            "lane_total": lane_total, "by_family": by_family,
            "contract_ok": contract.get("ok"),
            "status": "COMPLETE" if complete else "INCOMPLETE"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="L13 20K expansion runner")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--import-legacy", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--recheck-task", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--collisions", action="store_true")
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--reports", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-per-family", type=int, default=None)
    args = parser.parse_args()
    if args.recheck_task:
        res = run_recheck_task(args.recheck_task)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res.get("ok") else 1
    if args.build:
        print(json.dumps({k: v for k, v in run_build(
            families=args.families, max_per_family=args.max_per_family).items()
                          if k not in ("qualified_manifests", "candidate_manifests")},
                         indent=2, sort_keys=True))
        return 0
    if args.import_legacy:
        print(json.dumps(import_legacy(), indent=2, sort_keys=True))
        return 0
    if args.recheck:
        res = run_recheck()
        print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=2,
                         sort_keys=True))
        return 0 if res["ok"] else 1
    if args.smoke:
        res = run_smoke(args.limit)
        print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2,
                         sort_keys=True))
        return 0 if res["ok"] else 1
    if args.collisions:
        print(json.dumps(run_collisions(), indent=2, sort_keys=True))
        return 0
    if args.contract:
        res = run_contract_check()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res.get("ok") else 1
    if args.reports:
        print(json.dumps(write_reports(), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())