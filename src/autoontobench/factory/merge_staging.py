"""Single-writer staging -> bank/builds aggregator (owner: R1).

RESUME_EXECUTION_PLAN_20261029 §4 R1: read ``bank/staging/<lane>`` for the
lanes listed on the command line, validate every row against the §1 unified
output contract (aob-task/5.0), deduplicate by task/instance ids
(first-writer-wins in the given lane order), and write the canonical build
artifacts under ``bank/builds/<build_id>/``:

    candidates.jsonl            every candidate row (verbatim + 2 derived index fields)
    qualified_tasks.jsonl       every QUALIFIED row (verbatim + 2 derived index fields)
    task_manifest.jsonl         per-task index: signatures, lineage group, split
    native_evidence_index.jsonl per-task evidence pointers for the S11 report gates
    rejections.jsonl            lane rejections (lane_id added by the merge)
    dedup/dedup_decisions.jsonl S11 SignatureIndex verdicts (instance/base_goal level)
    dedup/lineage_splits.jsonl  task -> lineage group -> 70/15/15 whole-group split
    dedup/DEDUP_STATS.json      per-lane + TOTAL duplicate-rate statistics
    BUILD_MANIFEST.json         per-file sha256 + row counts (idempotency proof)

plus a per-lane merge report at ``bank/reports/<build_id>/MERGE_REPORT.json``.

Honesty rules baked in:
- L01/L02 staging candidates are byte-identical to qualified (accepted-only,
  no rejected rows were recorded at generation time). The merge reports this
  fact; it never fabricates rejection rows.
- The merge deduplicates by task_id / (lane_id, instance_id) only. Structural
  (AST-signature) duplication is measured by the S11 dedup pass and reported
  as statistics; it does not silently drop tasks.
- Two derived index fields are added to the build copies of lane records and
  are documented in MERGE_REPORT.json: ``difficulty`` (1:1 mapping of the
  lane-authored ``complexity`` field onto the spec's structural tier names)
  and ``source_refs`` (derived from ``source_snapshot_ids`` x
  ``source_unit_ids``). All lane-authored fields are preserved unchanged.
- Every output byte is a deterministic function of the staging inputs: no
  timestamps, no randomness other than the fixed dedup seed. Re-running the
  same command yields byte-identical artifacts (verified via
  BUILD_MANIFEST.json sha256s).

Idempotent re-runs: to add lanes later (e.g. new-schema L05, L09, or the
currently running lanes), re-run with the full lane list::

    .venv/bin/python -m autoontobench.factory.merge_staging \
        --repo-root . --build-id aob-8000-rc1 \
        --lanes l01,l02,l12,l05,l09 --resume

``--resume`` is required whenever the build directory already contains
output files; it acknowledges the intentional overwrite of a previous merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoontobench.factory.gates.canonicalize import (
    base_goal_signature,
    family_signature,
    instance_signature,
    source_lineage_group,
)
from autoontobench.factory.gates.dedup import (
    DEFAULT_SPLIT_RATIOS,
    DedupRecord,
    run_dedup,
)

SCHEMA_VERSION = "aob-task/5.0"
DEFAULT_BUILD_ID = "aob-8000-rc1"
DEFAULT_SEED = 20261027
LANE_NAME_RE = re.compile(r"^l\d{2}$")
COMPLEXITY_TO_TIER = {"low": "structural-low", "medium": "structural-medium",
                      "high": "structural-high"}
QUALIFICATION_FLAGS = ("native_verified", "solvable", "nontrivial",
                       "semantic_relevance", "leakage_clean", "release_eligible")
GOAL_LEVEL_PUBLIC_KEYS = ("allowed_tools", "budget", "answer_contract",
                          "policy_intent")

OUTPUT_FILES = (
    "candidates.jsonl", "qualified_tasks.jsonl", "task_manifest.jsonl",
    "native_evidence_index.jsonl", "rejections.jsonl",
    "dedup/dedup_decisions.jsonl", "dedup/lineage_splits.jsonl",
    "dedup/DEDUP_STATS.json", "BUILD_MANIFEST.json",
)


# --------------------------------------------------------------------- helpers

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via temp file + rename so readers never see partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MergeError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise MergeError(f"{path}:{lineno}: non-object JSONL row")
        rows.append(obj)
    return rows


class MergeError(RuntimeError):
    """Fatal merge input problem; aborts before any output is written."""


# Shard-manifest verification conventions observed in the landed lanes
# (both wrap the shard text as {"blob": <text>} before hashing):
#   l01/l02 ShardWriter: json.dumps(..., ensure_ascii=False, sort_keys=True,
#                                   separators=(",", ":"))
#   l12 engine:          json.dumps(..., sort_keys=True)   (default separators)
# A shard is verified iff its manifest sha256 matches ANY convention; the
# matched convention is recorded in the merge report.
SHARD_HASH_CONVENTIONS: tuple[tuple[str, Any], ...] = (
    ("wrapped-compact-utf8",
     lambda blob: _sha256_bytes(json.dumps({"blob": blob}, ensure_ascii=False,
                                           sort_keys=True, separators=(",", ":")).encode("utf-8"))),
    ("wrapped-default-utf8",
     lambda blob: _sha256_bytes(json.dumps({"blob": blob}, ensure_ascii=False,
                                           sort_keys=True).encode("utf-8"))),
    ("wrapped-default-ascii",
     lambda blob: _sha256_bytes(json.dumps({"blob": blob}, sort_keys=True).encode("utf-8"))),
    ("raw-file-bytes",
     lambda blob: None),  # handled separately against the raw bytes
)


@dataclass
class LaneInput:
    lane: str                      # staging dir name, e.g. "l01"
    lane_id: str                   # record lane_id, e.g. "L01"
    staging_dir: Path
    build_report: dict[str, Any] | None
    candidate_rows: list[dict[str, Any]]
    qualified_rows: list[dict[str, Any]]
    rejection_rows: list[dict[str, Any]]
    shard_checksums: dict[str, str] = field(default_factory=dict)
    shard_mismatches: list[str] = field(default_factory=list)
    shard_conventions: dict[str, str] = field(default_factory=dict)
    field_presence: dict[str, int] = field(default_factory=dict)
    candidates_equals_qualified: bool = True


# ------------------------------------------------------------------ validation

def _validate_identity(row: dict[str, Any], lane: str, kind: str,
                       qualified_statuses: tuple[str, ...] = ("QUALIFIED",),
                       candidate_statuses: tuple[str, ...] = ("QUALIFIED", "CANDIDATE"),
                       ) -> list[str]:
    """Identity-level checks shared by candidate and qualified rows.

    ``qualified_statuses`` defaults to the 8K contract (exactly ``QUALIFIED``).
    The 20K expansion merge widens it to also accept ``QUALIFIED_NATIVE`` /
    ``RELEASE_ELIGIBLE`` (EXPANSION_20K_EXECUTION_PLAN §2 status coverage);
    every deviation is recorded in the merge report, never silently normalized.
    """
    problems: list[str] = []
    tid = row.get("task_id")
    if not isinstance(tid, str) or not tid:
        problems.append("task_id missing/empty")
    for key in ("instance_id", "family_id", "base_goal_id"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"{key} missing/empty")
    rec_lane = row.get("lane_id")
    if not isinstance(rec_lane, str) or rec_lane.lower() != lane:
        problems.append(f"lane_id {rec_lane!r} does not match staging lane {lane!r}")
    if row.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {row.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    dataset_id = row.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        problems.append("dataset_id missing/empty")
    status = row.get("status")
    if kind == "qualified" and status not in qualified_statuses:
        problems.append(f"status {status!r} not in {'/'.join(qualified_statuses)}")
    if kind == "candidate" and status not in candidate_statuses:
        problems.append(f"status {status!r} not in {'/'.join(candidate_statuses)}")
    return [f"{kind}:{tid or '<missing>'}: {p}" for p in problems]


def _validate_qualified(row: dict[str, Any], lane: str,
                        expected_dataset_id: str,
                        qualified_statuses: tuple[str, ...] = ("QUALIFIED",)) -> list[str]:
    """Full §1 contract checks for a QUALIFIED row."""
    problems = _validate_identity(row, lane, "qualified", qualified_statuses)
    tid = row.get("task_id", "<missing>")
    if row.get("dataset_id") != expected_dataset_id:
        problems.append(f"qualified:{tid}: dataset_id {row.get('dataset_id')!r} "
                        f"!= {expected_dataset_id!r}")
    for key in ("task_regime", "complexity", "semantic_profile", "oracle_handle"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"qualified:{tid}: {key} missing/empty")
    public = row.get("public")
    if not isinstance(public, dict):
        problems.append(f"qualified:{tid}: public is not an object")
    else:
        user_goal = public.get("user_goal")
        if not isinstance(user_goal, str) or not user_goal.strip():
            problems.append(f"qualified:{tid}: public.user_goal missing/empty")
        tools = public.get("allowed_tools")
        if not isinstance(tools, list) or not tools:
            problems.append(f"qualified:{tid}: public.allowed_tools missing/empty")
        budget = public.get("budget")
        if not isinstance(budget, dict):
            problems.append(f"qualified:{tid}: public.budget missing/not an object")
        else:
            for bk in ("actions", "probes"):
                bv = budget.get(bk)
                if not isinstance(bv, int) or isinstance(bv, bool) or bv < 0:
                    problems.append(f"qualified:{tid}: public.budget.{bk} invalid ({bv!r})")
        # §1 names observation_refs; lanes carry it under lane-specific public
        # keys. Accept any non-meta key and record the convention (no fabrication).
        meta_public = {"user_goal", "allowed_tools", "budget", "complexity",
                       "semantic_profile"}
        observation_keys = sorted(k for k in public if k not in meta_public)
        if not observation_keys:
            problems.append(f"qualified:{tid}: no observation material "
                            "(public payload carries no observation key)")
    qualification = row.get("qualification")
    if not isinstance(qualification, dict):
        problems.append(f"qualified:{tid}: qualification missing/not an object")
    else:
        for flag in QUALIFICATION_FLAGS:
            if qualification.get(flag) is not True:
                problems.append(f"qualified:{tid}: qualification.{flag} is not true")
    evidence = row.get("native_evidence")
    if not isinstance(evidence, dict) or not evidence:
        problems.append(f"qualified:{tid}: native_evidence missing/empty")
    for key in ("source_snapshot_ids", "source_unit_ids"):
        value = row.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            problems.append(f"qualified:{tid}: {key} missing/empty/not a string list")
    return problems


# ------------------------------------------------------------- derived fields

def _derived_difficulty(row: dict[str, Any]) -> str:
    complexity = row.get("complexity")
    if isinstance(complexity, str):
        return COMPLEXITY_TO_TIER.get(complexity, "unknown")
    return "unknown"


def _derived_source_refs(row: dict[str, Any]) -> list[dict[str, str]]:
    """(source_id, source_rule_unit_id) refs derived from the record's own ids.

    Each source_unit_id embeds its source prefix (e.g.
    "gatekeeper-library/library/general/allowedrepos/src.rego"); the prefix is
    cross-checked against source_snapshot_ids and falls back to the first
    snapshot's source id.
    """
    snapshots = row.get("source_snapshot_ids") or []
    snap_sources = {s.split("@")[0] for s in snapshots if isinstance(s, str) and "@" in s}
    fallback = next(iter(sorted(snap_sources)), "")
    refs: list[dict[str, str]] = []
    for unit in row.get("source_unit_ids") or []:
        if not isinstance(unit, str) or not unit:
            continue
        source_id = unit.split("/", 1)[0]
        if snap_sources and source_id not in snap_sources:
            source_id = fallback
        refs.append({"source_id": source_id, "source_rule_unit_id": unit})
    refs.sort(key=lambda r: (r["source_id"], r["source_rule_unit_id"]))
    return refs


def _task_ast(row: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize-compatible AST projection of a task record (documented).

    Goal-level content (what the agent must achieve and with which tools /
    budget / contract) lives in ``goal``; instance-varying content (the
    concrete objects / task material and the wording) lives under the keys
    canonicalize strips for base_goal_signature ("entities", "wording").
    Lane taxonomy fields (family_id, semantic_profile) are deliberately NOT
    part of the AST so signatures remain comparable across lanes.
    """
    public = row.get("public") or {}
    goal = {k: public[k] for k in GOAL_LEVEL_PUBLIC_KEYS if k in public}
    entities = {k: v for k, v in public.items()
                if k not in GOAL_LEVEL_PUBLIC_KEYS and k != "user_goal"}
    return {
        "op": "task",
        "schema_version": row.get("schema_version"),
        "regime": row.get("task_regime"),
        "goal": goal,
        "wording": {"user_goal": public.get("user_goal")},
        "entities": entities,
    }


def _evidence_index_row(row: dict[str, Any]) -> dict[str, Any]:
    """Evidence pointer row for the S11 report gates; flags derived from the
    record's own qualification flags and native_evidence content."""
    qualification = row.get("qualification") or {}
    evidence = row.get("native_evidence") or {}
    prove = evidence.get("prove_exists") or {}
    witness = evidence.get("semantic_witness") or {}
    evals = prove.get("evals") or []
    engine = prove.get("engine")
    if not isinstance(engine, str) and evals and isinstance(evals[0], dict):
        engine = evals[0].get("engine")
    return {
        "task_id": row.get("task_id"),
        "lane_id": row.get("lane_id"),
        "ok": bool(qualification.get("native_verified") is True and evidence),
        "solvable": qualification.get("solvable") is True,
        "nontrivial": qualification.get("nontrivial") is True,
        "semantic_relevance": qualification.get("semantic_relevance") is True,
        "leakage_clean": qualification.get("leakage_clean") is True,
        "semantic_witness": {"ok": witness.get("ok") is True},
        "proof": {"exists": bool(prove)},
        "engine": engine if isinstance(engine, str) else None,
        "evidence_ref": _sha256_bytes(_dumps(evidence).encode("utf-8")),
    }


# ------------------------------------------------------------------- lane load

def _load_lane(repo_root: Path, lane: str, expected_dataset_id: str, *,
               staging_root: Path | None = None,
               qualified_statuses: tuple[str, ...] = ("QUALIFIED",),
               candidate_statuses: tuple[str, ...] = ("QUALIFIED", "CANDIDATE"),
               ) -> LaneInput:
    """Load + §1-validate one staging lane.

    ``staging_root`` defaults to ``<repo_root>/bank/staging`` (the frozen 8K
    layout). The 20K expansion merge passes ``bank/staging_20k`` instead; the
    default keeps every existing 8K caller byte-for-byte unchanged.

    ``qualified_statuses`` defaults to the 8K contract; see ``_validate_identity``.
    """
    root = staging_root if staging_root is not None else repo_root / "bank" / "staging"
    staging = root / lane
    if not staging.is_dir():
        raise MergeError(f"staging dir not found: {staging}")
    report_path = staging / "build_report.json"
    build_report = None
    if report_path.exists():
        try:
            build_report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MergeError(f"{report_path}: malformed JSON ({exc})") from exc

    problems: list[str] = []
    candidate_rows: list[dict[str, Any]] = []
    qualified_rows: list[dict[str, Any]] = []
    shard_checksums: dict[str, str] = {}
    shard_mismatches: list[str] = []
    shard_conventions: dict[str, str] = {}
    shard_notes: list[str] = []

    def _verify_shard(kind: str, shard: Path, data: bytes) -> None:
        name = f"{kind}/{shard.name}"
        text = data.decode("utf-8")
        raw_sha = _sha256_bytes(data)
        shard_checksums[name] = raw_sha
        manifest = shard.with_name(shard.stem + ".manifest.json")
        if not manifest.exists():
            shard_notes.append(f"{name}: per-file manifest absent (lane-specific layout)")
            return
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            shard_mismatches.append(f"{name}: manifest malformed")
            return
        expected = meta.get("sha256")
        matched: str | None = None
        if isinstance(expected, str):
            if expected == raw_sha:
                matched = "raw-file-bytes"
            else:
                for label, fn in SHARD_HASH_CONVENTIONS:
                    digest = fn(text)
                    if digest is not None and digest == expected:
                        matched = label
                        break
        if matched is None and isinstance(expected, str):
            shard_mismatches.append(f"{name}: manifest sha256 mismatch "
                                    f"(expected {expected[:16]}…, file {raw_sha[:16]}…)")
        elif matched is not None:
            shard_conventions[name] = matched
        expected_records = meta.get("records")
        if isinstance(expected_records, int):
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) != expected_records:
                shard_mismatches.append(
                    f"{name}: manifest records {expected_records} != on-disk {len(lines)}")

    def _kind_shards(kind: str) -> list[Path]:
        kind_dir = staging / kind
        if kind_dir.is_dir():
            return sorted(p for p in kind_dir.rglob("*.jsonl"))
        return sorted(staging.glob(f"shard_*/{kind}.jsonl"))

    def _read_kind(kind: str) -> tuple[list[Path], list[dict[str, Any]]]:
        files = _kind_shards(kind)
        rows: list[dict[str, Any]] = []
        for shard in files:
            data = shard.read_bytes()
            _verify_shard(kind, shard, data)
            rows.extend(_read_jsonl(shard))
        return files, rows

    q_files, q_rows = _read_kind("qualified")
    if not q_files:
        raise MergeError(f"{staging}/qualified: no *.jsonl shards (empty lane)")
    for row in q_rows:
        problems.extend(_validate_qualified(row, lane, expected_dataset_id,
                                            qualified_statuses))
    qualified_rows = q_rows

    c_files, c_rows = _read_kind("candidates")
    candidate_ast_only = bool(c_rows) and not c_rows[0].get("task_id")
    if not c_files or candidate_ast_only:
        candidate_rows = [dict(r) for r in qualified_rows]
        shard_notes.append(
            "candidates: AST-only or absent -> synthesized from qualified rows (accepted-only)")
    else:
        for row in c_rows:
            problems.extend(_validate_identity(row, lane, "candidate",
                                               qualified_statuses,
                                               candidate_statuses))
        candidate_rows = c_rows

    if shard_mismatches:
        problems.extend(shard_mismatches)
    if problems:
        preview = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise MergeError(f"lane {lane}: {len(problems)} validation problem(s):\n  {preview}{more}")
    if not qualified_rows:
        raise MergeError(f"lane {lane}: 0 qualified rows; refusing to merge an empty lane "
                         "(legacy/incomplete staging must not enter the bank)")

    rejection_rows: list[dict[str, Any]] = []
    rej_files: list[Path] = []
    rej_dir = staging / "rejections"
    if rej_dir.is_dir():
        rej_files.extend(sorted(rej_dir.rglob("*.jsonl")))
    rej_files.extend(sorted(staging.glob("shard_*/rejections*.jsonl")))
    for path in rej_files:
        rejection_rows.extend(_read_jsonl(path))
    without_id = [r for r in rejection_rows if not r.get("task_id")]
    if without_id:
        shard_notes.append(
            f"rejections: {len(without_id)} candidate-stage rows without task_id (counts only)")
        rejection_rows = [r for r in rejection_rows if r.get("task_id")]

    # accepted-only detection: candidates byte-identical to qualified?
    cand_blob = b"".join(_dumps(r).encode("utf-8") + b"\n" for r in candidate_rows)
    qual_blob = b"".join(_dumps(r).encode("utf-8") + b"\n" for r in qualified_rows)
    accepted_only = cand_blob == qual_blob and not rejection_rows

    field_presence: dict[str, int] = {}
    for row in qualified_rows:
        for key in row:
            field_presence[key] = field_presence.get(key, 0) + 1
        for key in (row.get("public") or {}):
            field_presence[f"public.{key}"] = field_presence.get(f"public.{key}", 0) + 1

    lane_id = qualified_rows[0].get("lane_id", lane.upper())
    return LaneInput(
        lane=lane, lane_id=lane_id, staging_dir=staging, build_report=build_report,
        candidate_rows=candidate_rows, qualified_rows=qualified_rows,
        rejection_rows=rejection_rows, shard_checksums=shard_checksums,
        shard_mismatches=shard_mismatches, shard_conventions=shard_conventions,
        field_presence=field_presence,
        candidates_equals_qualified=accepted_only,
    )


# ---------------------------------------------------------------------- merge

@dataclass
class MergeResult:
    build_dir: Path
    report_path: Path
    files: dict[str, dict[str, Any]]
    totals: dict[str, Any]


def run_merge(
    repo_root: Path,
    *,
    build_id: str = DEFAULT_BUILD_ID,
    lanes: list[str],
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> MergeResult:
    """Merge the given staging lanes into bank/builds/<build_id> (single writer)."""
    if not lanes:
        raise MergeError("no lanes given; pass --lanes l01,l02,...")
    normalized: list[str] = []
    for raw in lanes:
        lane = raw.strip().lower()
        if not LANE_NAME_RE.match(lane):
            raise MergeError(f"invalid lane name {raw!r} (expected l01..l99)")
        if lane in normalized:
            raise MergeError(f"duplicate lane {lane} in --lanes")
        normalized.append(lane)

    expected_dataset_id = "autoontobench-bank-" + build_id.removeprefix("aob-")
    build_dir = repo_root / "bank" / "builds" / build_id
    report_dir = repo_root / "bank" / "reports" / build_id
    report_path = report_dir / "MERGE_REPORT.json"

    existing = [f for f in OUTPUT_FILES if (build_dir / f).exists()]
    if existing and not resume:
        raise MergeError(
            f"build dir already contains {len(existing)} output file(s) "
            f"(e.g. {existing[0]}); pass --resume to intentionally re-merge")

    lane_inputs = [_load_lane(repo_root, lane, expected_dataset_id) for lane in normalized]

    # ---- merge with id-level dedup (first-writer-wins in lane order) -------
    merged_candidates: dict[str, dict[str, Any]] = {}
    merged_qualified: dict[str, dict[str, Any]] = {}
    candidate_dedup_events: list[dict[str, str]] = []
    qualified_dedup_events: list[dict[str, str]] = []
    instance_id_owners: dict[str, str] = {}
    instance_id_collisions: list[dict[str, str]] = []
    for li in lane_inputs:
        for row in li.candidate_rows:
            tid = row["task_id"]
            if tid in merged_candidates:
                candidate_dedup_events.append(
                    {"task_id": tid, "kept_from": merged_candidates[tid]["lane_id"],
                     "dropped_from": li.lane_id})
            else:
                merged_candidates[tid] = row
        for row in li.qualified_rows:
            tid = row["task_id"]
            if tid in merged_qualified:
                qualified_dedup_events.append(
                    {"task_id": tid, "kept_from": merged_qualified[tid]["lane_id"],
                     "dropped_from": li.lane_id})
                continue
            merged_qualified[tid] = row
            iid = row.get("instance_id")
            if isinstance(iid, str) and iid:
                owner = instance_id_owners.get(iid)
                if owner is not None and owner != tid:
                    instance_id_collisions.append(
                        {"instance_id": iid, "task_ids": sorted((owner, tid))})
                else:
                    instance_id_owners[iid] = tid

    merged_rejections: list[dict[str, Any]] = []
    for li in lane_inputs:
        for row in li.rejection_rows:
            out = dict(row)
            out.setdefault("lane_id", li.lane_id)
            merged_rejections.append(out)
    merged_rejections.sort(key=lambda r: (str(r.get("lane_id")), str(r.get("task_id"))))

    qualified_rows = [merged_qualified[tid] for tid in sorted(merged_qualified)]
    candidate_rows = [merged_candidates[tid] for tid in sorted(merged_candidates)]

    # ---- S11 dedup over the merged qualified set ---------------------------
    dedup_records: list[DedupRecord] = []
    sig_by_task: dict[str, dict[str, str]] = {}
    for row in qualified_rows:
        ast = _task_ast(row)
        refs = [(ref["source_id"], ref["source_rule_unit_id"])
                for ref in _derived_source_refs(row)]
        instance_sig = instance_signature(ast)
        goal_sig = base_goal_signature(ast)
        fam_sig = family_signature(ast)
        sig_by_task[row["task_id"]] = {
            "instance_signature": instance_sig,
            "base_goal_signature": goal_sig,
            "family_signature": fam_sig,
        }
        dedup_records.append(DedupRecord(
            task_id=row["task_id"], lane_id=str(row.get("lane_id", "")),
            family_id=str(row.get("family_id", "")),
            base_goal_id=str(row.get("base_goal_id", "")),
            instance_sig=instance_sig, base_goal_sig=goal_sig, family_sig=fam_sig,
            lineage_group=source_lineage_group(refs),
            mechanism_group_id=str(row.get("origin_mechanism_group") or ""),
        ))
    decisions, groups, splits, stats = run_dedup(dedup_records, seed=seed)

    split_sizes: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    for split in splits.values():
        split_sizes[split] += 1
    lineage_group_sizes: dict[str, int] = {}
    for group in groups.values():
        lineage_group_sizes[group] = lineage_group_sizes.get(group, 0) + 1

    # ---- serialize outputs (deterministic bytes) ---------------------------
    def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
        return b"".join(_dumps(r).encode("utf-8") + b"\n" for r in rows)

    build_candidates = []
    stripped_private_keys: dict[str, int] = {}
    for row in candidate_rows:
        out = {k: v for k, v in row.items() if not k.startswith("private_")}
        for k in row:
            if k.startswith("private_"):
                stripped_private_keys[k] = stripped_private_keys.get(k, 0) + 1
        out["difficulty"] = _derived_difficulty(row)
        out["source_refs"] = _derived_source_refs(row)
        build_candidates.append(out)

    build_qualified = []
    for row in qualified_rows:
        out = {k: v for k, v in row.items() if not k.startswith("private_")}
        for k in row:
            if k.startswith("private_"):
                stripped_private_keys[k] = stripped_private_keys.get(k, 0) + 1
        out["difficulty"] = _derived_difficulty(row)
        out["source_refs"] = _derived_source_refs(row)
        build_qualified.append(out)

    manifest_rows = []
    for row in qualified_rows:
        tid = row["task_id"]
        decision = decisions[tid]
        manifest_rows.append({
            "task_id": tid,
            "instance_id": row.get("instance_id"),
            "lane_id": row.get("lane_id"),
            "family_id": row.get("family_id"),
            "base_goal_id": row.get("base_goal_id"),
            "task_regime": row.get("task_regime"),
            "complexity": row.get("complexity"),
            "difficulty": _derived_difficulty(row),
            "semantic_profile": row.get("semantic_profile"),
            "mechanism_group_id": row.get("origin_mechanism_group"),
            "instance_signature": sig_by_task[tid]["instance_signature"],
            "base_goal_signature": sig_by_task[tid]["base_goal_signature"],
            "family_signature": sig_by_task[tid]["family_signature"],
            "lineage_group": groups[tid],
            "split": splits[tid],
            "is_duplicate": decision.is_duplicate,
            "duplicate_of": decision.duplicate_of,
            "duplicate_level": decision.level,
            "source_snapshot_ids": row.get("source_snapshot_ids"),
            "source_unit_ids": row.get("source_unit_ids"),
            "source_refs": _derived_source_refs(row),
            "oracle_handle": row.get("oracle_handle"),
            "status": row.get("status"),
            "dataset_id": row.get("dataset_id"),
            "schema_version": row.get("schema_version"),
            "native_evidence_sha256": _sha256_bytes(
                _dumps(row.get("native_evidence") or {}).encode("utf-8")),
        })

    evidence_rows = [_evidence_index_row(row) for row in qualified_rows]

    decision_rows = []
    lane_by_task = {row["task_id"]: row.get("lane_id") for row in qualified_rows}
    for tid in sorted(decisions):
        decision = decisions[tid]
        decision_rows.append({
            "task_id": tid,
            "lane_id": lane_by_task.get(tid),
            "is_duplicate": decision.is_duplicate,
            "duplicate_of": decision.duplicate_of,
            "level": decision.level,
        })

    lineage_rows = [{
        "task_id": row["task_id"],
        "lane_id": row.get("lane_id"),
        "family_id": row.get("family_id"),
        "base_goal_id": row.get("base_goal_id"),
        "lineage_group": groups[row["task_id"]],
        "split": splits[row["task_id"]],
    } for row in qualified_rows]

    stats_payload = {
        "seed": seed,
        "split_ratios": DEFAULT_SPLIT_RATIOS,
        "split_sizes": split_sizes,
        "lineage_groups": len(lineage_group_sizes),
        "largest_lineage_groups": sorted(lineage_group_sizes.items(),
                                         key=lambda kv: (-kv[1], kv[0]))[:10],
        "stats": [{
            "lane_id": s.lane_id, "total": s.total,
            "unique_instances": s.unique_instances,
            "unique_base_goals": s.unique_base_goals,
            "families": s.families,
            "mechanism_groups": s.mechanism_groups,
            "lineage_groups": s.lineage_groups,
            "instance_duplicate_rate": round(s.instance_duplicate_rate, 6),
            "base_goal_duplicate_rate": round(s.base_goal_duplicate_rate, 6),
            "max_family_share": round(s.max_family_share, 6),
        } for s in stats],
        "semantics": {
            "instance_level": "true content duplicates: identical public task "
                              "content under different task ids; merge keeps them "
                              "(id-level dedup only) and flags them here",
            "base_goal_level": "template density, NOT bank rejections: additional "
                               "instances of a structurally identical goal-level "
                               "signature (the intended instances-per-goal "
                               "parameterization). Lane base_goal_id counts are "
                               "reported alongside in MERGE_REPORT.json.",
            "split": "whole-group assignment: a lineage group never crosses a "
                     "split; actual sizes deviate from 70/15/15 by group "
                     "granularity and smooth out as more lanes land",
        },
    }

    outputs: dict[str, bytes] = {
        "candidates.jsonl": jsonl_bytes(build_candidates),
        "qualified_tasks.jsonl": jsonl_bytes(build_qualified),
        "task_manifest.jsonl": jsonl_bytes(manifest_rows),
        "native_evidence_index.jsonl": jsonl_bytes(evidence_rows),
        "rejections.jsonl": jsonl_bytes(merged_rejections),
        "dedup/dedup_decisions.jsonl": jsonl_bytes(decision_rows),
        "dedup/lineage_splits.jsonl": jsonl_bytes(lineage_rows),
        "dedup/DEDUP_STATS.json": (_dumps(stats_payload) + "\n").encode("utf-8"),
    }

    files_manifest: dict[str, dict[str, Any]] = {}
    for name, data in outputs.items():
        files_manifest[name] = {
            "rows": data.count(b"\n"),
            "sha256": _sha256_bytes(data),
        }
    build_manifest = {
        "build_id": build_id,
        "dataset_id": expected_dataset_id,
        "lane_order": normalized,
        "dedup_seed": seed,
        "totals": {
            "lanes": len(lane_inputs),
            "candidates": len(candidate_rows),
            "qualified": len(qualified_rows),
            "rejections": len(merged_rejections),
            "distinct_base_goal_ids": len({r.get("base_goal_id") for r in qualified_rows}),
            "distinct_families": len({r.get("family_id") for r in qualified_rows}),
            "lineage_groups": len(lineage_group_sizes),
        },
        "files": files_manifest,
    }
    outputs["BUILD_MANIFEST.json"] = (_dumps(build_manifest) + "\n").encode("utf-8")

    # ---- per-lane merge report (deterministic; no timestamps) --------------
    lane_reports = []
    for li in lane_inputs:
        br = li.build_report or {}
        counted_qualified = len(li.qualified_rows)
        counted_candidates = len(li.candidate_rows)
        report_qualified = br.get("qualified")
        report_candidates = br.get("candidates")
        lane_reports.append({
            "lane": li.lane,
            "lane_id": li.lane_id,
            "staging_dir": str(li.staging_dir.relative_to(repo_root)),
            "shards": {name: checksum for name, checksum in sorted(li.shard_checksums.items())},
            "shard_manifest_mismatches": li.shard_mismatches,
            "shard_manifest_conventions": dict(sorted(li.shard_conventions.items())),
            "counts": {
                "candidates_on_disk": counted_candidates,
                "qualified_on_disk": counted_qualified,
                "rejections_on_disk": len(li.rejection_rows),
                "build_report_candidates": report_candidates,
                "build_report_qualified": report_qualified,
                "build_report_matches_disk":
                    report_candidates == counted_candidates and report_qualified == counted_qualified,
                "distinct_base_goal_ids": len({r.get("base_goal_id") for r in li.qualified_rows}),
                "distinct_families": len({r.get("family_id") for r in li.qualified_rows}),
                "distinct_mechanism_groups": len(
                    {r.get("origin_mechanism_group") for r in li.qualified_rows}),
            },
            "accepted_only_candidates": li.candidates_equals_qualified,
            "accepted_only_note": (
                "candidates.jsonl is byte-identical to qualified.jsonl and "
                "rejections/ is empty: the lane generator recorded no rejected "
                "rows at generation time. Documented as accepted-only; no "
                "rejection rows were fabricated by the merge."
            ) if li.candidates_equals_qualified else None,
            "rejection_reasons": dict(br.get("rejections") or {}),
            "field_presence_qualified": dict(sorted(li.field_presence.items())),
            "contract_deviations": [
                ("public.observation_refs (§1) absent; observation material "
                 "carried under public.* keys: "
                 + "/".join(sorted({
                     k.split(".", 1)[1] for k in li.field_presence
                     if k.startswith("public.") and k.split(".", 1)[1] not in (
                         "user_goal", "allowed_tools", "budget", "complexity",
                         "semantic_profile")})))
                if "public.observation_refs" not in li.field_presence else None,
            ],
        })
    for lr in lane_reports:
        lr["contract_deviations"] = [d for d in lr["contract_deviations"] if d]

    merge_report = {
        "build_id": build_id,
        "dataset_id": expected_dataset_id,
        "merge_command": "python -m autoontobench.factory.merge_staging "
                         f"--repo-root . --build-id {build_id} "
                         f"--lanes {','.join(normalized)} --resume",
        "lane_order": normalized,
        "dedup_seed": seed,
        "derived_fields_note": (
            "Build copies of lane records add exactly two derived index fields: "
            "difficulty (1:1 mapping of the lane-authored complexity field onto "
            "structural-low/medium/high) and source_refs (derived from "
            "source_snapshot_ids x source_unit_ids). Lane-authored private_* "
            "reference fields are stripped from public build outputs (they stay "
            "in lane staging/oracle); all other fields are preserved unchanged."
        ),
        "lanes": lane_reports,
        "private_fields_stripped": dict(sorted(stripped_private_keys.items())),
        "totals": build_manifest["totals"],
        "id_level_dedup": {
            "candidate_task_id_duplicates": candidate_dedup_events,
            "qualified_task_id_duplicates": qualified_dedup_events,
            "cross_lane_instance_id_collisions": instance_id_collisions,
        },
        "dedup_summary": {
            "instance_level_duplicates_flagged": sum(
                1 for d in decisions.values() if d.is_duplicate and d.level == "instance"),
            "base_goal_level_duplicates_flagged": sum(
                1 for d in decisions.values() if d.is_duplicate and d.level == "base_goal"),
            "unique_instance_signatures": stats[-1].unique_instances if stats else 0,
            "unique_base_goal_signatures": stats[-1].unique_base_goals if stats else 0,
            "instance_duplicate_rate": round(stats[-1].instance_duplicate_rate, 6) if stats else 0.0,
            "base_goal_signature_duplicate_rate": round(
                stats[-1].base_goal_duplicate_rate, 6) if stats else 0.0,
            "lineage_groups": len(lineage_group_sizes),
            "split_sizes": split_sizes,
        },
        "outputs": {name: {"rows": meta["rows"], "sha256": meta["sha256"]}
                    for name, meta in files_manifest.items()},
        "re_run": {
            "command": "python -m autoontobench.factory.merge_staging "
                       f"--repo-root . --build-id {build_id} "
                       f"--lanes {','.join(normalized)} --resume",
            "add_lanes": "append lane ids to --lanes (full list each run); the "
                         "merge is a pure function of the listed lanes' staging "
                         "state, so re-runs are byte-identical until staging "
                         "changes",
            "reports": "python -m autoontobench.factory.reports --repo-root . --strict",
        },
    }

    # ---- atomic writes ------------------------------------------------------
    for name, data in outputs.items():
        _atomic_write(build_dir / name, data)
    _atomic_write(report_path, (_dumps(merge_report) + "\n").encode("utf-8"))

    return MergeResult(
        build_dir=build_dir, report_path=report_path, files=files_manifest,
        totals=build_manifest["totals"],
    )


# ------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m autoontobench.factory.merge_staging",
        description="Single-writer staging -> bank/builds aggregator with §1 "
                    "validation, id-level dedup, S11 dedup stats and per-lane "
                    "merge report (owner: R1)",
    )
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    ap.add_argument("--lanes", required=True,
                    help="comma-separated staging lane ids in merge priority "
                         "order, e.g. l01,l02,l12")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"dedup/split seed (default {DEFAULT_SEED})")
    ap.add_argument("--resume", action="store_true",
                    help="required to overwrite an existing build; re-merging "
                         "is byte-identical for unchanged staging")
    args = ap.parse_args(argv)
    lanes = [chunk for chunk in args.lanes.replace(" ", "").split(",") if chunk]
    try:
        result = run_merge(args.repo_root, build_id=args.build_id, lanes=lanes,
                           seed=args.seed, resume=args.resume)
    except MergeError as exc:
        print(f"merge_staging: ERROR: {exc}", file=sys.stderr)
        return 2
    totals = result.totals
    print(f"merged lanes={totals['lanes']} candidates={totals['candidates']} "
          f"qualified={totals['qualified']} rejections={totals['rejections']} "
          f"base_goals={totals['distinct_base_goal_ids']} "
          f"families={totals['distinct_families']} "
          f"lineage_groups={totals['lineage_groups']}")
    for name in sorted(result.files):
        meta = result.files[name]
        print(f"  {name}: rows={meta['rows']} sha256={meta['sha256'][:16]}…")
    print(f"merge report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
