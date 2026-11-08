"""20K expansion merge: frozen legacy 8K build + ``bank/staging_20k`` (owner: QA).

EXPANSION_20K_EXECUTION_PLAN §0/§6/§7. This is the single writer for
``bank/builds/aob-20000-candidate-dev1/`` and ``bank/reports/aob-20000-candidate-dev1/``.

Inputs
------
1. **Legacy 8K (frozen, read-only)** — ``bank/builds/aob-8000-rc1/``. Read via
   its own ``BUILD_MANIFEST.json``: every listed file's sha256 is re-verified
   against disk before a single row is trusted, and every row is re-serialized
   and compared to its frozen source line. Legacy task records are then
   re-emitted **verbatim** (no derived fields recomputed, no keys added or
   removed), so the 20K build's legacy rows are byte-identical to the 8K build's.
   Nothing under ``bank/staging/``, ``bank/builds/aob-8000-rc1/`` or
   ``bank/releases/`` is ever opened for writing.
2. **New 20K lanes** — ``bank/staging_20k/<lane>/{candidates,qualified,
   rejections}/`` + ``build_report.json``, loaded and §1-validated by the same
   ``merge_staging._load_lane`` the 8K merge uses (so the contract checks cannot
   drift between builds), pointed at ``bank/staging_20k`` via ``staging_root``.

Mixed dataset ids are expected and required: legacy rows keep
``autoontobench-bank-8000-rc1``; new rows carry
``autoontobench-bank-20000-candidate-dev1``. Each row is validated against the
dataset id of **its own origin**, never against a single build-wide value.

Merge rules
-----------
- ``task_id`` is the unique key. **Legacy wins on collision**; the dropped new
  row is recorded in ``dedup/task_id_collisions.jsonl`` and counted in
  ``MERGE_REPORT.json`` — collisions are reported, never silently absorbed.
- New-vs-new collisions resolve first-writer-wins in ``--lanes`` order.
- Canonical (AST-signature) dedup, lineage grouping and whole-group 70/15/15
  splits are recomputed over the **union** by S11 ``run_dedup``. Splits are a
  build-level property: the 20K build has its own seed (default 20261101, the
  plan §2 new seed base) and its own lineage groups; the frozen 8K splits stay
  in the 8K build. Per-legacy-task split churn vs 8K is reported, not hidden.

Outputs (``bank/builds/aob-20000-candidate-dev1/``)
---------------------------------------------------
    candidates.jsonl               legacy verbatim + new (2 derived index fields)
    qualified_tasks.jsonl          same, QUALIFIED-only
    task_manifest.jsonl            per-task index + origin/dataset_id provenance
    native_evidence_index.jsonl    per-task evidence pointers for report gates
    rejections.jsonl               legacy + new lane rejections
    dedup/dedup_decisions.jsonl    S11 SignatureIndex verdicts
    dedup/lineage_splits.jsonl     task -> lineage group -> split
    dedup/DEDUP_STATS.json         per-lane + TOTAL duplicate-rate statistics
    dedup/task_id_collisions.jsonl legacy-wins / first-writer-wins resolutions
    LEGACY_INTEGRITY.json          8K manifest sha256 proof + byte-identity result
    BUILD_MANIFEST.json            per-file sha256 + row counts + §6 counts

plus ``bank/reports/aob-20000-candidate-dev1/``: ``MERGE_REPORT.json`` and the
five §6 global reports (see ``reports_20k.py``).

Honesty rules
-------------
- No count is invented. ``native_checked`` / ``native_pass`` are derived only
  from verdict tokens the row itself records; an absent verdict yields
  ``NOT_RECORDED`` / ``NOT_RUN``, never ``PASS`` (plan §2).
- Lanes that have not landed are skipped and listed with the reason; the merge
  never waits for them and never pads toward 20,000. The gap to target is
  reported as a gap.
- Every output byte is a deterministic function of the inputs: no timestamps, no
  randomness beyond the fixed dedup seed. Re-running the same command over
  unchanged inputs yields byte-identical artifacts (prove it with
  ``BUILD_MANIFEST.json``).

Re-runnable / idempotent
------------------------
``--resume`` is required whenever the build dir already holds outputs; it
acknowledges an intentional re-merge. Lanes may be added incrementally — pass
the full lane list each time, or ``--lanes auto`` to take whatever has landed::

    .venv/bin/python -m autoontobench.factory.merge_20k --repo-root . --resume

``--dry-run`` computes everything (including all counts and the integrity
proof) and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
# Shared §1-contract helpers are imported from the 8K merge on purpose: the
# serializer, the derived-field rules and the contract checks must be literally
# the same code in both builds, otherwise legacy bytes could drift. They are
# module-private by 8K convention, not by intent to fork them here.
from autoontobench.factory.merge_staging import (
    LANE_NAME_RE,
    MergeError,
    _atomic_write,
    _derived_difficulty,
    _derived_source_refs,
    _dumps,
    _evidence_index_row,
    _load_lane,
    _sha256_bytes,
    _task_ast,
    _validate_qualified,
)

DEFAULT_BUILD_ID = "aob-20000-candidate-dev1"
DEFAULT_LEGACY_BUILD_ID = "aob-8000-rc1"
DEFAULT_SEED = 20261101          # plan §2 new seed base
PLAN_DATE = "2026-10-31"         # fixed string: outputs carry no wall clock
TARGET_TOTAL = 20000             # plan §1 totals.target_total (includes legacy)

STAGING_20K_RELPATH = ("bank", "staging_20k")
LANES_YAML_RELPATH = ("configs", "lanes_20000.yaml")

ORIGIN_LEGACY = "legacy-8k"
ORIGIN_NEW = "staging-20k"

# Plan §2 status coverage: new lanes may record the native-qualification status
# under any of these tokens. Legacy 8K is exactly "QUALIFIED" and is validated
# against the frozen 8K contract, never against this widened set.
NEW_LANE_QUALIFIED_STATUSES = ("QUALIFIED", "QUALIFIED_NATIVE", "RELEASE_ELIGIBLE")
QUALIFIED_STATUSES = frozenset(NEW_LANE_QUALIFIED_STATUSES)

# The candidate pool is the whole pipeline, so it legitimately carries rows that
# never qualified. The 8K lanes were accepted-only (candidates == qualified) and
# so only ever showed QUALIFIED/CANDIDATE; the 20K lanes use the full plan §2
# status vocabulary.
NEW_LANE_CANDIDATE_STATUSES = (*NEW_LANE_QUALIFIED_STATUSES, "CANDIDATE",
                               "PARTIALLY_CHECKED", "REJECTED")

# Read-only paths that this tool must never write into (plan §0).
PROTECTED_WRITE_RELPATHS = (
    ("bank", "staging"),
    ("bank", "builds", DEFAULT_LEGACY_BUILD_ID),
    ("bank", "releases"),
    ("bank", "reports", DEFAULT_LEGACY_BUILD_ID),
)

OUTPUT_FILES = (
    "candidates.jsonl", "qualified_tasks.jsonl", "task_manifest.jsonl",
    "native_evidence_index.jsonl", "rejections.jsonl",
    "dedup/dedup_decisions.jsonl", "dedup/lineage_splits.jsonl",
    "dedup/DEDUP_STATS.json", "dedup/task_id_collisions.jsonl",
    "LEGACY_INTEGRITY.json", "BUILD_MANIFEST.json",
)


# --------------------------------------------------------------------- helpers

def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_dumps(r).encode("utf-8") + b"\n" for r in rows)


def _read_jsonl_verifying(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL and prove each row re-serializes to its exact frozen line.

    Returns (rows, byte_identity_mismatches). Used for the legacy build only:
    it is the mechanism that makes "legacy rows are byte-identical" a checked
    property rather than a claim.
    """
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    if not path.exists():
        return rows, mismatches
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise MergeError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise MergeError(f"{path}:{lineno}: non-object JSONL row")
        if _dumps(obj) != stripped:
            mismatches.append(f"{path.name}:{lineno}: re-serialization differs "
                              "from the frozen bytes")
        rows.append(obj)
    return rows, mismatches


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------- native verdicts

def _native_check_state(row: dict[str, Any]) -> tuple[bool, str]:
    """``(native_checked, verdict_token)`` from the row's own recorded fields.

    Plan §2: never write PASS for a check that was not executed. Precedence is
    the explicit per-task status coverage field, then a recorded verdict token,
    then the ``qualification.native_verified`` flag. Anything less is reported
    as ``NOT_RECORDED`` (evidence present, no verdict) or ``NOT_RUN``.
    """
    checks = row.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    explicit = checks.get("native_execution_status") or row.get("native_execution_status")
    if isinstance(explicit, str) and explicit.strip():
        return True, explicit.strip().upper()

    evidence = row.get("native_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    tokens: list[str] = []
    candidates: list[Any] = [evidence.get("verdict")]
    prove = evidence.get("prove_exists")
    if isinstance(prove, dict):
        candidates.append(prove.get("verdict"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            tokens.append(value.strip().upper())
    if tokens:
        return True, "PASS" if all(t == "PASS" for t in tokens) else tokens[0]

    qualification = row.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    if qualification.get("native_verified") is True:
        return True, "PASS"
    return (False, "NOT_RECORDED") if evidence else (False, "NOT_RUN")


def _is_qualified_native(row: dict[str, Any]) -> bool:
    """QUALIFIED_NATIVE ladder rung: qualified status AND native_verified true."""
    qualification = row.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    return (str(row.get("status", "")).upper() in QUALIFIED_STATUSES
            and qualification.get("native_verified") is True)


# --------------------------------------------------------------- legacy input

@dataclass
class LegacyInput:
    """The frozen 8K build, integrity-checked before any row is trusted."""

    build_id: str
    build_dir: Path
    dataset_id: str
    candidate_rows: list[dict[str, Any]]
    qualified_rows: list[dict[str, Any]]
    rejection_rows: list[dict[str, Any]]
    source_sha256: dict[str, str] = field(default_factory=dict)
    manifest_sha256: dict[str, str] = field(default_factory=dict)
    byte_identity_mismatches: list[str] = field(default_factory=list)
    unique_task_ids: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    lane_counts: dict[str, int] = field(default_factory=dict)
    legacy_splits: dict[str, str] = field(default_factory=dict)
    legacy_lineage_groups: dict[str, str] = field(default_factory=dict)


def _load_legacy_build(repo_root: Path, legacy_build_id: str) -> LegacyInput:
    """Load + integrity-verify the frozen 8K build (never written to)."""
    build_dir = repo_root / "bank" / "builds" / legacy_build_id
    if not build_dir.is_dir():
        raise MergeError(f"legacy build dir not found: {build_dir}")
    manifest_path = build_dir / "BUILD_MANIFEST.json"
    if not manifest_path.exists():
        raise MergeError(f"legacy BUILD_MANIFEST.json absent: {manifest_path}; "
                         "cannot prove legacy integrity, refusing to merge")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MergeError(f"{manifest_path}: malformed JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise MergeError(f"{manifest_path}: not a JSON object")

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise MergeError(f"{manifest_path}: dataset_id missing/empty")

    # --- integrity gate: every manifest-listed file must match its sha256 ----
    source_sha: dict[str, str] = {}
    manifest_sha: dict[str, str] = {}
    problems: list[str] = []
    files = manifest.get("files")
    files = files if isinstance(files, dict) else {}
    for name in sorted(files):
        meta = files[name] or {}
        expected = meta.get("sha256") if isinstance(meta, dict) else None
        path = build_dir / name
        if not path.exists():
            problems.append(f"{name}: in BUILD_MANIFEST.json but absent on disk")
            continue
        data = path.read_bytes()
        actual = _sha256_bytes(data)
        source_sha[name] = actual
        if not isinstance(expected, str):
            problems.append(f"{name}: BUILD_MANIFEST.json records no sha256")
            continue
        manifest_sha[name] = expected
        if expected != actual:
            problems.append(f"{name}: sha256 {actual[:16]}… != manifest {expected[:16]}…")
        expected_rows = meta.get("rows") if isinstance(meta, dict) else None
        if isinstance(expected_rows, int):
            actual_rows = data.count(b"\n")
            if actual_rows != expected_rows:
                problems.append(f"{name}: rows {actual_rows} != manifest {expected_rows}")
    for required in ("qualified_tasks.jsonl", "candidates.jsonl"):
        if required not in source_sha:
            problems.append(f"{required}: not covered by BUILD_MANIFEST.json")
    if problems:
        raise MergeError("legacy 8K integrity check FAILED (refusing to merge):\n  "
                         + "\n  ".join(problems[:20]))

    # --- rows + byte-identity proof -----------------------------------------
    qualified_rows, q_bad = _read_jsonl_verifying(build_dir / "qualified_tasks.jsonl")
    candidate_rows, c_bad = _read_jsonl_verifying(build_dir / "candidates.jsonl")
    rejection_rows, r_bad = _read_jsonl_verifying(build_dir / "rejections.jsonl")
    byte_mismatches = q_bad + c_bad + r_bad
    if byte_mismatches:
        raise MergeError(
            f"legacy byte-identity proof FAILED for {len(byte_mismatches)} row(s); "
            "the 20K build could not carry legacy rows verbatim:\n  "
            + "\n  ".join(byte_mismatches[:20]))

    # --- §1 contract re-validation against the LEGACY dataset id ------------
    contract: list[str] = []
    for row in qualified_rows:
        lane = str(row.get("lane_id", "")).lower()
        contract.extend(_validate_qualified(row, lane, dataset_id))
    if contract:
        preview = "\n  ".join(contract[:20])
        more = f"\n  ... and {len(contract) - 20} more" if len(contract) > 20 else ""
        raise MergeError(f"legacy build: {len(contract)} §1 validation problem(s):\n"
                         f"  {preview}{more}")

    if not qualified_rows:
        raise MergeError(f"legacy build {legacy_build_id}: 0 qualified rows")

    legacy_splits: dict[str, str] = {}
    legacy_groups: dict[str, str] = {}
    for row in _read_jsonl_verifying(build_dir / "dedup" / "lineage_splits.jsonl")[0]:
        tid = row.get("task_id")
        if isinstance(tid, str):
            if isinstance(row.get("split"), str):
                legacy_splits[tid] = row["split"]
            if isinstance(row.get("lineage_group"), str):
                legacy_groups[tid] = row["lineage_group"]

    return LegacyInput(
        build_id=legacy_build_id, build_dir=build_dir, dataset_id=dataset_id,
        candidate_rows=candidate_rows, qualified_rows=qualified_rows,
        rejection_rows=rejection_rows, source_sha256=source_sha,
        manifest_sha256=manifest_sha, byte_identity_mismatches=byte_mismatches,
        unique_task_ids=len({r.get("task_id") for r in qualified_rows}),
        status_counts=_counts(r.get("status") for r in qualified_rows),
        lane_counts=_counts(r.get("lane_id") for r in qualified_rows),
        legacy_splits=legacy_splits, legacy_lineage_groups=legacy_groups,
    )


# ------------------------------------------------------------- lane discovery

def _lane_qualified_shards(lane_dir: Path) -> list[Path]:
    """Mirror ``merge_staging._load_lane``'s shard layout resolution."""
    qualified = lane_dir / "qualified"
    if qualified.is_dir():
        return sorted(p for p in qualified.rglob("*.jsonl"))
    return sorted(lane_dir.glob("shard_*/qualified.jsonl"))


def discover_lanes(staging_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Split ``bank/staging_20k/*`` into landed lanes and skipped/pending dirs.

    The merge never waits for lanes: a dir that cannot contribute qualified rows
    is skipped with the reason recorded, so the tool is re-runnable as lanes
    finish. Only dirs matching the lane-name contract ``^l\\d{2}$`` are eligible
    — ``merge_staging._load_lane`` requires ``lane_id.lower() == <dir name>``,
    and lane ids are ``L01..L16``, so any other name (``_smoke``, ``l03_smoke``,
    scratch or partial dirs written by a concurrent worker) can never validate
    and must not be mistaken for a lane.
    """
    landed: list[str] = []
    pending: list[dict[str, str]] = []
    if not staging_root.is_dir():
        return landed, [{"lane": "<none>", "kind": "absent_root",
                         "reason": f"staging root absent: {staging_root}"}]
    for entry in sorted((p for p in staging_root.iterdir() if p.is_dir()),
                        key=lambda p: p.name):
        if not LANE_NAME_RE.match(entry.name):
            pending.append({"lane": entry.name, "kind": "scratch",
                            "reason": "not a lane name (expected l01..l99): scratch "
                                      "or partial dir, skipped"})
            continue
        shards = _lane_qualified_shards(entry)
        rows = 0
        for shard in shards:
            rows += sum(1 for ln in shard.read_text(encoding="utf-8",
                                                    errors="replace").splitlines()
                        if ln.strip())
        if shards and rows:
            landed.append(entry.name)
        else:
            pending.append({"lane": entry.name, "kind": "not_landed",
                            "reason": "no qualified/*.jsonl rows yet (lane not landed)"})
    return landed, pending


def _load_lane_targets(lanes_yaml: Path) -> dict[str, dict[str, Any]]:
    """``configs/lanes_20000.yaml`` -> lane_id -> {target_total, legacy_count, new_needed}."""
    if not lanes_yaml.exists():
        return {}
    doc = yaml.safe_load(lanes_yaml.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for entry in doc.get("lanes") or []:
        if isinstance(entry, dict) and isinstance(entry.get("lane_id"), str):
            out[entry["lane_id"]] = entry
    return out


# ---------------------------------------------------------------- merge result

@dataclass
class MergeResult:
    build_dir: Path
    report_dir: Path
    report_path: Path
    files: dict[str, dict[str, Any]]
    totals: dict[str, Any]
    counts: dict[str, Any]
    merge_report: dict[str, Any]
    dry_run: bool
    written: list[str] = field(default_factory=list)
    report_files: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------- the merge

def run_merge_20k(
    repo_root: Path,
    *,
    build_id: str = DEFAULT_BUILD_ID,
    legacy_build_id: str = DEFAULT_LEGACY_BUILD_ID,
    lanes: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
    dry_run: bool = False,
    write_reports: bool = True,
) -> MergeResult:
    """Merge the frozen legacy 8K build with landed ``bank/staging_20k`` lanes."""
    repo_root = Path(repo_root)
    if build_id == legacy_build_id:
        raise MergeError(f"--build-id must differ from the frozen legacy build "
                         f"{legacy_build_id!r} (plan §0: legacy is read-only)")

    build_dir = repo_root / "bank" / "builds" / build_id
    report_dir = repo_root / "bank" / "reports" / build_id
    report_path = report_dir / "MERGE_REPORT.json"
    staging_root = repo_root / Path(*STAGING_20K_RELPATH)
    lanes_yaml = repo_root / Path(*LANES_YAML_RELPATH)
    new_dataset_id = "autoontobench-bank-" + build_id.removeprefix("aob-")

    _assert_writable(build_dir, repo_root)
    _assert_writable(report_dir, repo_root)

    existing = [f for f in OUTPUT_FILES if (build_dir / f).exists()]
    if existing and not resume and not dry_run:
        raise MergeError(
            f"build dir already contains {len(existing)} output file(s) "
            f"(e.g. {existing[0]}); pass --resume to intentionally re-merge")

    # ---- inputs -------------------------------------------------------------
    legacy = _load_legacy_build(repo_root, legacy_build_id)
    landed, pending_lanes = discover_lanes(staging_root)

    lane_selection = "auto"
    skipped_lanes = pending_lanes
    selected = landed
    if lanes is not None:
        normalized: list[str] = []
        for raw in lanes:
            lane = raw.strip().lower()
            if not lane:
                continue
            if lane in ("auto", "all"):
                normalized = list(landed)
                break
            lane_selection = "explicit"
            if lane not in landed:
                state = ("dir exists but has no qualified/*.jsonl rows yet "
                         "(worker still writing?)"
                         if (staging_root / lane).is_dir()
                         else "no such dir")
                raise MergeError(
                    f"lane {lane!r} has not landed under {staging_root}: {state}; "
                    f"landed now: {','.join(landed) or '<none>'}")
            if lane in normalized:
                raise MergeError(f"duplicate lane {lane} in --lanes")
            normalized.append(lane)
        selected = normalized
        skipped_lanes = [] if lane_selection == "explicit" else pending_lanes

    lane_inputs = [
        _load_lane(repo_root, lane, new_dataset_id, staging_root=staging_root,
                   qualified_statuses=NEW_LANE_QUALIFIED_STATUSES,
                   candidate_statuses=NEW_LANE_CANDIDATE_STATUSES)
        for lane in selected
    ]
    lane_targets = _load_lane_targets(lanes_yaml)

    # ---- origin-tagged merge; legacy wins on task_id collision --------------
    merged_qualified: dict[str, dict[str, Any]] = {}
    merged_candidates: dict[str, dict[str, Any]] = {}
    qual_origin: dict[str, str] = {}
    cand_origin: dict[str, str] = {}
    collisions: list[dict[str, Any]] = []
    instance_owners: dict[str, str] = {}
    instance_collisions: list[dict[str, Any]] = []

    ordered: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]],
                        list[dict[str, Any]]]] = [
        (ORIGIN_LEGACY, f"legacy:{legacy.build_id}", legacy.qualified_rows,
         legacy.candidate_rows, legacy.rejection_rows),
    ]
    ordered.extend((ORIGIN_NEW, li.lane_id, li.qualified_rows, li.candidate_rows,
                    li.rejection_rows) for li in lane_inputs)

    for origin, label, q_rows, c_rows, _r_rows in ordered:
        for row in q_rows:
            tid = row.get("task_id")
            if not isinstance(tid, str) or not tid:
                raise MergeError(f"{label}: qualified row without task_id")
            if tid in merged_qualified:
                kept = merged_qualified[tid]
                kept_origin = qual_origin[tid]
                collisions.append({
                    "task_id": tid,
                    "pool": "qualified",
                    "kept_origin": kept_origin,
                    "kept_from": kept.get("lane_id"),
                    "kept_dataset_id": kept.get("dataset_id"),
                    "dropped_origin": origin,
                    "dropped_from": label,
                    "dropped_dataset_id": row.get("dataset_id"),
                    "resolution": ("legacy-8k wins (plan §6: no overwrite of 8K)"
                                   if kept_origin == ORIGIN_LEGACY
                                   else "first-writer-wins in --lanes order"),
                })
                continue
            merged_qualified[tid] = row
            qual_origin[tid] = origin
            iid = row.get("instance_id")
            if isinstance(iid, str) and iid:
                owner = instance_owners.get(iid)
                if owner is not None and owner != tid:
                    instance_collisions.append(
                        {"instance_id": iid, "task_ids": sorted((owner, tid))})
                else:
                    instance_owners[iid] = tid
        for row in c_rows:
            tid = row.get("task_id")
            if not isinstance(tid, str) or not tid:
                continue  # AST-only candidate rows carry no id (8K convention)
            if tid in merged_candidates:
                collisions.append({
                    "task_id": tid,
                    "pool": "candidate",
                    "kept_origin": cand_origin[tid],
                    "kept_from": merged_candidates[tid].get("lane_id"),
                    "kept_dataset_id": merged_candidates[tid].get("dataset_id"),
                    "dropped_origin": origin,
                    "dropped_from": label,
                    "dropped_dataset_id": row.get("dataset_id"),
                    "resolution": ("legacy-8k wins (plan §6: no overwrite of 8K)"
                                   if cand_origin[tid] == ORIGIN_LEGACY
                                   else "first-writer-wins in --lanes order"),
                })
                continue
            merged_candidates[tid] = row
            cand_origin[tid] = origin

    tagged_rejections: list[tuple[str, dict[str, Any]]] = []
    for origin, _label, _q, _c, r_rows in ordered:
        tagged_rejections.extend((origin, row) for row in r_rows)
    tagged_rejections.sort(key=lambda t: (t[0], str(t[1].get("lane_id")),
                                          str(t[1].get("task_id"))))
    merged_rejections = [row for _origin, row in tagged_rejections]
    new_rejections = sum(1 for origin, _row in tagged_rejections if origin == ORIGIN_NEW)

    qualified_rows = [merged_qualified[t] for t in sorted(merged_qualified)]
    candidate_rows = [merged_candidates[t] for t in sorted(merged_candidates)]
    cand_origin = {t: cand_origin.get(t, qual_origin.get(t, ORIGIN_NEW))
                   for t in merged_candidates}

    # ---- S11 canonical dedup over the union --------------------------------
    dedup_records: list[DedupRecord] = []
    sig_by_task: dict[str, dict[str, str]] = {}
    for row in qualified_rows:
        ast = _task_ast(row)
        refs = [(r["source_id"], r["source_rule_unit_id"])
                for r in _derived_source_refs(row)]
        instance_sig = instance_signature(ast)
        goal_sig = base_goal_signature(ast)
        fam_sig = family_signature(ast)
        tid = row["task_id"]
        sig_by_task[tid] = {"instance_signature": instance_sig,
                            "base_goal_signature": goal_sig,
                            "family_signature": fam_sig}
        dedup_records.append(DedupRecord(
            task_id=tid, lane_id=str(row.get("lane_id", "")),
            family_id=str(row.get("family_id", "")),
            base_goal_id=str(row.get("base_goal_id", "")),
            instance_sig=instance_sig, base_goal_sig=goal_sig, family_sig=fam_sig,
            lineage_group=source_lineage_group(refs),
            mechanism_group_id=str(row.get("origin_mechanism_group") or ""),
        ))
    decisions, groups, splits, stats = run_dedup(dedup_records, seed=seed)

    split_sizes = {"train": 0, "dev": 0, "test": 0}
    for split in splits.values():
        split_sizes[split] += 1
    lineage_sizes: dict[str, int] = {}
    for group in groups.values():
        lineage_sizes[group] = lineage_sizes.get(group, 0) + 1

    split_churn = sum(1 for tid, split in splits.items()
                      if legacy.legacy_splits.get(tid) not in (None, split))
    lineage_churn = sum(1 for tid, group in groups.items()
                        if legacy.legacy_lineage_groups.get(tid) not in (None, group))

    # ---- serialize ----------------------------------------------------------
    stripped_private: dict[str, int] = {}

    def _build_row(row: dict[str, Any], origin: str) -> dict[str, Any]:
        if origin == ORIGIN_LEGACY:
            # Frozen: re-emit exactly what the 8K build froze. Derived fields are
            # already present and were proven idempotent; recomputing them here
            # would risk a byte difference for zero information gain.
            return dict(row)
        out = {k: v for k, v in row.items() if not k.startswith("private_")}
        for key in row:
            if key.startswith("private_"):
                stripped_private[key] = stripped_private.get(key, 0) + 1
        out["difficulty"] = _derived_difficulty(row)
        out["source_refs"] = _derived_source_refs(row)
        return out

    build_qualified = [_build_row(r, qual_origin[r["task_id"]]) for r in qualified_rows]
    build_candidates = [_build_row(r, cand_origin[r["task_id"]]) for r in candidate_rows]

    frozen_legacy_lines = {_dumps(r) for r in legacy.qualified_rows}
    emitted_legacy_lines = {_dumps(row) for row, src in zip(build_qualified, qualified_rows)
                            if qual_origin[src["task_id"]] == ORIGIN_LEGACY}
    missing_legacy_lines = sorted(frozen_legacy_lines - emitted_legacy_lines)
    if missing_legacy_lines:
        raise MergeError(
            f"legacy byte-identity FAILED before write: {len(missing_legacy_lines)} "
            "frozen 8K row(s) would not be carried verbatim into the 20K build")

    manifest_rows: list[dict[str, Any]] = []
    for row in qualified_rows:
        tid = row["task_id"]
        decision = decisions[tid]
        origin = qual_origin[tid]
        checked, verdict = _native_check_state(row)
        manifest_rows.append({
            "task_id": tid,
            "instance_id": row.get("instance_id"),
            "lane_id": row.get("lane_id"),
            "origin": origin,
            "source_build": legacy.build_id if origin == ORIGIN_LEGACY else build_id,
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
            "legacy_split_8k": legacy.legacy_splits.get(tid),
            "legacy_lineage_group_8k": legacy.legacy_lineage_groups.get(tid),
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
            "native_checked": checked,
            "native_verdict": verdict,
            "qualified_native": _is_qualified_native(row),
            "native_evidence_sha256": _sha256_bytes(
                _dumps(row.get("native_evidence") or {}).encode("utf-8")),
        })

    evidence_rows = [_evidence_index_row(row) for row in qualified_rows]

    lane_by_task = {r["task_id"]: r.get("lane_id") for r in qualified_rows}
    decision_rows = [{
        "task_id": tid,
        "lane_id": lane_by_task.get(tid),
        "origin": qual_origin.get(tid),
        "is_duplicate": decisions[tid].is_duplicate,
        "duplicate_of": decisions[tid].duplicate_of,
        "level": decisions[tid].level,
    } for tid in sorted(decisions)]

    lineage_rows = [{
        "task_id": row["task_id"],
        "lane_id": row.get("lane_id"),
        "origin": qual_origin[row["task_id"]],
        "family_id": row.get("family_id"),
        "base_goal_id": row.get("base_goal_id"),
        "lineage_group": groups[row["task_id"]],
        "split": splits[row["task_id"]],
        "legacy_split_8k": legacy.legacy_splits.get(row["task_id"]),
    } for row in qualified_rows]

    collisions.sort(key=lambda c: (c["pool"], c["task_id"]))

    # ---- §6 counts (separately reported; nothing fabricated) ----------------
    new_qualified = [r for r in qualified_rows if qual_origin[r["task_id"]] == ORIGIN_NEW]
    legacy_qualified = [r for r in qualified_rows if qual_origin[r["task_id"]] == ORIGIN_LEGACY]
    new_candidates = [r for r in candidate_rows if cand_origin[r["task_id"]] == ORIGIN_NEW]
    new_checked = [r for r in new_qualified if _native_check_state(r)[0]]
    new_pass = [r for r in new_qualified if _native_check_state(r)[1] == "PASS"]

    per_lane: dict[str, dict[str, Any]] = {}
    for row in qualified_rows:
        lane = str(row.get("lane_id", "<unknown>"))
        origin = qual_origin[row["task_id"]]
        bucket = per_lane.setdefault(lane, {
            "lane_id": lane,
            # plural: legacy 8K and new 20K rows share lane ids, so one lane can
            # carry both origins
            "origins": [],
            "target_total": (lane_targets.get(lane) or {}).get("target_total"),
            "legacy_count": (lane_targets.get(lane) or {}).get("legacy_count"),
            "new_needed": (lane_targets.get(lane) or {}).get("new_needed"),
            "qualified": 0, "legacy_qualified": 0, "new_qualified": 0,
            "native_checked": 0, "native_pass": 0, "qualified_native": 0,
        })
        if origin not in bucket["origins"]:
            bucket["origins"].append(origin)
        checked, verdict = _native_check_state(row)
        bucket["qualified"] += 1
        bucket["legacy_qualified" if origin == ORIGIN_LEGACY else "new_qualified"] += 1
        bucket["native_checked"] += 1 if checked else 0
        bucket["native_pass"] += 1 if verdict == "PASS" else 0
        bucket["qualified_native"] += 1 if _is_qualified_native(row) else 0
    for bucket in per_lane.values():
        bucket["origins"] = sorted(bucket["origins"])

    counts = {
        "legacy_8k_qualified": len(legacy_qualified),
        "legacy_8k_candidates": len(legacy.candidate_rows),
        "legacy_8k_rejections": len(legacy.rejection_rows),
        "legacy_8k_unique_task_ids": legacy.unique_task_ids,
        "new_candidates": len(new_candidates),
        "new_candidate_status_counts": _counts(r.get("status") for r in new_candidates),
        "new_qualified": len(new_qualified),
        "new_qualified_status_counts": _counts(r.get("status") for r in new_qualified),
        "new_native_checked": len(new_checked),
        "new_native_pass": len(new_pass),
        "new_rejections": new_rejections,
        "legacy_rejections_carried": len(merged_rejections) - new_rejections,
        "unique_task_ids_20k": len(merged_qualified),
        "unique_candidate_task_ids": len(merged_candidates),
        "qualified_native": sum(1 for r in qualified_rows if _is_qualified_native(r)),
        "task_id_collisions": len({c["task_id"] for c in collisions}),
        "task_id_collision_events": len(collisions),
        "task_id_collisions_legacy_wins": len(
            {c["task_id"] for c in collisions if c["kept_origin"] == ORIGIN_LEGACY}),
        "task_id_collisions_by_dropped_lane": _counts(
            c["dropped_from"] for c in collisions if c["pool"] == "qualified"),
        "cross_origin_instance_id_collisions": len(instance_collisions),
        "target_total": TARGET_TOTAL,
        "gap_to_target": TARGET_TOTAL - len(merged_qualified),
        "lanes_landed": len(lane_inputs),
        "lanes_pending": len([p for p in skipped_lanes
                              if p.get("kind") == "not_landed"]),
        "scratch_dirs_skipped": len([p for p in skipped_lanes
                                     if p.get("kind") == "scratch"]),
    }

    totals = {
        "origins": 2 if lane_inputs else 1,
        "lanes": len(lane_inputs),
        "candidates": len(candidate_rows),
        "qualified": len(qualified_rows),
        "rejections": len(merged_rejections),
        "distinct_base_goal_ids": len({r.get("base_goal_id") for r in qualified_rows}),
        "distinct_families": len({r.get("family_id") for r in qualified_rows}),
        "lineage_groups": len(lineage_sizes),
    }

    stats_payload = {
        "seed": seed,
        "split_ratios": DEFAULT_SPLIT_RATIOS,
        "split_sizes": split_sizes,
        "lineage_groups": len(lineage_sizes),
        "largest_lineage_groups": sorted(lineage_sizes.items(),
                                         key=lambda kv: (-kv[1], kv[0]))[:10],
        "legacy_split_churn_vs_8k": split_churn,
        "legacy_lineage_churn_vs_8k": lineage_churn,
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
            "instance_level": "true content duplicates: identical public task content "
                              "under different task ids; the merge keeps them "
                              "(id-level dedup only) and flags them here",
            "base_goal_level": "template density, NOT bank rejections: additional "
                               "instances of a structurally identical goal-level "
                               "signature (the intended instances-per-goal "
                               "parameterization)",
            "split": "whole-group assignment: a lineage group never crosses a split; "
                     "recomputed for the 20K build over the legacy+new union, so "
                     "legacy tasks may move split vs the frozen 8K build "
                     "(legacy_split_churn_vs_8k counts them)",
        },
    }

    legacy_integrity = {
        "legacy_build_id": legacy.build_id,
        "legacy_build_dir": str(legacy.build_dir.relative_to(repo_root)),
        "legacy_dataset_id": legacy.dataset_id,
        "read_only": True,
        "manifest_files_verified": len(legacy.manifest_sha256),
        "manifest_sha256_mismatches": [],
        "files": {name: {"sha256": legacy.source_sha256[name],
                         "manifest_sha256": legacy.manifest_sha256.get(name),
                         "match": legacy.source_sha256[name]
                         == legacy.manifest_sha256.get(name)}
                  for name in sorted(legacy.source_sha256)},
        "legacy_rows_byte_identical": not legacy.byte_identity_mismatches,
        "legacy_row_count": len(legacy.qualified_rows),
        "legacy_unique_task_ids": legacy.unique_task_ids,
        "legacy_status_counts": legacy.status_counts,
        "legacy_lane_counts": legacy.lane_counts,
        "carried_verbatim": True,
        "pre_write_legacy_rows_verified": len(emitted_legacy_lines & frozen_legacy_lines),
        "pre_write_legacy_rows_missing": len(missing_legacy_lines),
        "pre_write_legacy_byte_identical": not missing_legacy_lines,
        "note": "Legacy task records are re-emitted byte-for-byte from "
                f"bank/builds/{legacy.build_id}/; derived fields are NOT "
                "recomputed and no key is added or removed. Index files "
                "(task_manifest, native_evidence_index, dedup/*) ARE recomputed "
                "for the 20K build because lineage groups and splits are "
                "build-level properties of the legacy+new union.",
    }

    outputs: dict[str, bytes] = {
        "candidates.jsonl": _jsonl_bytes(build_candidates),
        "qualified_tasks.jsonl": _jsonl_bytes(build_qualified),
        "task_manifest.jsonl": _jsonl_bytes(manifest_rows),
        "native_evidence_index.jsonl": _jsonl_bytes(evidence_rows),
        "rejections.jsonl": _jsonl_bytes(merged_rejections),
        "dedup/dedup_decisions.jsonl": _jsonl_bytes(decision_rows),
        "dedup/lineage_splits.jsonl": _jsonl_bytes(lineage_rows),
        "dedup/DEDUP_STATS.json": (_dumps(stats_payload) + "\n").encode("utf-8"),
        "dedup/task_id_collisions.jsonl": _jsonl_bytes(collisions),
        "LEGACY_INTEGRITY.json": (_dumps(legacy_integrity) + "\n").encode("utf-8"),
    }

    files_manifest: dict[str, dict[str, Any]] = {
        name: {"rows": data.count(b"\n"), "sha256": _sha256_bytes(data)}
        for name, data in outputs.items()
    }
    build_manifest = {
        "build_id": build_id,
        "dataset_id": new_dataset_id,
        "dataset_ids": {
            ORIGIN_LEGACY: legacy.dataset_id,
            ORIGIN_NEW: new_dataset_id,
        },
        "legacy_build_id": legacy.build_id,
        "lane_selection": lane_selection,
        "lane_order": selected,
        "lanes_pending": skipped_lanes,
        "dedup_seed": seed,
        "plan": {"source": "EXPANSION_20K_EXECUTION_PLAN.md", "date": PLAN_DATE,
                 "target_total": TARGET_TOTAL},
        "totals": totals,
        "counts": counts,
        "per_lane": [per_lane[k] for k in sorted(per_lane)],
        "files": files_manifest,
    }
    outputs["BUILD_MANIFEST.json"] = (_dumps(build_manifest) + "\n").encode("utf-8")
    files_manifest["BUILD_MANIFEST.json"] = {
        "rows": outputs["BUILD_MANIFEST.json"].count(b"\n"),
        "sha256": _sha256_bytes(outputs["BUILD_MANIFEST.json"]),
    }

    lane_arg = "auto" if lane_selection == "auto" else (",".join(selected) or "none")
    merge_command = (".venv/bin/python -m autoontobench.factory.merge_20k "
                     f"--repo-root . --build-id {build_id} "
                     f"--legacy-build-id {legacy_build_id} "
                     f"--lanes {lane_arg} --resume")

    merge_report = {
        "build_id": build_id,
        "dataset_id": new_dataset_id,
        "dataset_ids": build_manifest["dataset_ids"],
        "plan_date": PLAN_DATE,
        "merge_command": merge_command,
        "lane_selection": lane_selection,
        "lane_order": selected,
        "lanes_pending": skipped_lanes,
        "dedup_seed": seed,
        "legacy": {
            "build_id": legacy.build_id,
            "dataset_id": legacy.dataset_id,
            "read_only": True,
            "integrity": legacy_integrity,
        },
        "counts": counts,
        "counts_semantics": {
            "legacy_8k_qualified": "rows carried verbatim from the frozen 8K build",
            "new_candidates": "candidate rows from bank/staging_20k lanes (post id-dedup); "
                              "the candidate pool is the whole pipeline, so it "
                              "legitimately includes REJECTED / PARTIALLY_CHECKED rows "
                              "(see new_candidate_status_counts)",
            "new_qualified": "QUALIFIED rows from bank/staging_20k lanes (post id-dedup)",
            "new_native_checked": "new qualified rows recording an executed native check "
                                  "(explicit status field, a verdict token, or "
                                  "qualification.native_verified=true)",
            "new_native_pass": "new qualified rows whose derived native verdict is PASS",
            "unique_task_ids_20k": "distinct task_id in the merged qualified set",
            "qualified_native": "merged rows with a qualified status AND "
                                "qualification.native_verified=true",
            "gap_to_target": f"{TARGET_TOTAL} - unique_task_ids_20k; reported as an "
                             "honest gap, never padded (plan §1)",
            "task_id_collisions": "distinct task_ids that collided across origins or "
                                  "lanes; legacy 8K always wins (plan §6)",
            "task_id_collision_events": "rows in dedup/task_id_collisions.jsonl; one "
                                        "colliding task_id can resolve in both the "
                                        "qualified and the candidate pool, so this is "
                                        ">= task_id_collisions",
        },
        "per_lane": [per_lane[k] for k in sorted(per_lane)],
        "totals": totals,
        "id_level_dedup": {
            "task_id_collisions": collisions[:200],
            "task_id_collisions_truncated": max(0, len(collisions) - 200),
            "task_id_collisions_file": "dedup/task_id_collisions.jsonl",
            "cross_origin_instance_id_collisions": instance_collisions[:200],
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
            "lineage_groups": len(lineage_sizes),
            "split_sizes": split_sizes,
            "legacy_split_churn_vs_8k": split_churn,
            "legacy_lineage_churn_vs_8k": lineage_churn,
        },
        "private_fields_stripped": dict(sorted(stripped_private.items())),
        "outputs": {name: {"rows": meta["rows"], "sha256": meta["sha256"]}
                    for name, meta in sorted(files_manifest.items())},
        "re_run": {
            "command": merge_command,
            "add_lanes": "re-run with --lanes auto (or the full explicit list) once more "
                         "lanes land; the merge is a pure function of the legacy build "
                         "plus the listed lanes' staging state, so re-runs are "
                         "byte-identical until an input changes",
            "dry_run": "add --dry-run to compute every count and the integrity proof "
                       "without writing a byte",
            "reports": "regenerated with the merge; or pass --reports-only",
        },
    }

    # ---- write (atomic, single writer) --------------------------------------
    written: list[str] = []
    if not dry_run:
        for name, data in outputs.items():
            _atomic_write(build_dir / name, data)
            written.append(str((build_dir / name).relative_to(repo_root)))
        _atomic_write(report_path, (_dumps(merge_report) + "\n").encode("utf-8"))
        written.append(str(report_path.relative_to(repo_root)))
        _verify_written_legacy_bytes(build_dir, legacy)

    result = MergeResult(
        build_dir=build_dir, report_dir=report_dir, report_path=report_path,
        files=files_manifest, totals=totals, counts=counts,
        merge_report=merge_report, dry_run=dry_run, written=written,
    )

    if write_reports and not dry_run:
        from autoontobench.factory.reports_20k import write_global_reports
        result.report_files = {
            name: str(path.relative_to(repo_root))
            for name, path in write_global_reports(
                repo_root, build_id=build_id, build_dir=build_dir,
                report_dir=report_dir, merge_report=merge_report,
                counts=counts, per_lane=build_manifest["per_lane"],
                dedup_stats=stats_payload, legacy_integrity=legacy_integrity,
                lanes_pending=skipped_lanes, lane_targets=lane_targets,
                seed=seed, merge_command=merge_command,
            ).items()
        }
    return result


def _assert_writable(target: Path, repo_root: Path) -> None:
    """Refuse to write anywhere the plan §0 marks read-only."""
    resolved = target.resolve()
    root = repo_root.resolve()
    for rel in PROTECTED_WRITE_RELPATHS:
        protected = (root / Path(*rel)).resolve()
        if resolved == protected or protected in resolved.parents:
            raise MergeError(f"refusing to write into protected path {protected} "
                             f"(plan §0: legacy 8K is read-only)")


def _verify_written_legacy_bytes(build_dir: Path, legacy: LegacyInput) -> None:
    """Post-write guard: re-read disk and abort if the legacy bytes are not there.

    Records nothing. ``LEGACY_INTEGRITY.json`` carries only the pre-write proof,
    which is computed before serialization so ``BUILD_MANIFEST.json`` can hash
    its final bytes; rewriting it here would leave the manifest stale.
    """
    frozen = {_dumps(r) for r in legacy.qualified_rows}
    emitted: set[str] = set()
    path = build_dir / "qualified_tasks.jsonl"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if row.get("dataset_id") == legacy.dataset_id:
            emitted.add(stripped)
    missing = frozen - emitted
    if missing:
        raise MergeError(
            f"post-write legacy byte-identity FAILED: {len(missing)} frozen 8K row(s) "
            "are not present verbatim in the 20K build")


# ------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m autoontobench.factory.merge_20k",
        description="Merge the frozen legacy 8K build with bank/staging_20k lanes "
                    "into bank/builds/aob-20000-candidate-dev1 (owner: QA; "
                    "EXPANSION_20K_EXECUTION_PLAN §6)",
    )
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    ap.add_argument("--legacy-build-id", default=DEFAULT_LEGACY_BUILD_ID)
    ap.add_argument("--lanes", default="auto",
                    help="'auto' (default: take every landed lane under "
                         "bank/staging_20k), 'none' (legacy 8K only — the "
                         "skeleton/integrity baseline), or a comma-separated "
                         "lane list in merge priority order, e.g. l14,l15,l16")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"dedup/split seed (default {DEFAULT_SEED}, plan §2 seed base)")
    ap.add_argument("--resume", action="store_true",
                    help="required to overwrite an existing build; re-merging is "
                         "byte-identical for unchanged inputs")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute all counts + the legacy integrity proof, write nothing")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip the five §6 global report templates")
    ap.add_argument("--reports-only", action="store_true",
                    help="regenerate reports from an existing build, do not re-merge")
    args = ap.parse_args(argv)

    if args.reports_only:
        from autoontobench.factory.reports_20k import regenerate_from_build
        try:
            paths = regenerate_from_build(args.repo_root, build_id=args.build_id)
        except MergeError as exc:
            print(f"merge_20k: ERROR: {exc}", file=sys.stderr)
            return 2
        for name, path in sorted(paths.items()):
            print(f"{name}: {path}")
        return 0

    lanes: list[str] | None
    if args.lanes.strip().lower() in ("auto", "all", ""):
        lanes = None
    elif args.lanes.strip().lower() in ("none", "legacy", "legacy-only"):
        lanes = []          # legacy 8K only: the skeleton/integrity baseline
    else:
        lanes = [c for c in args.lanes.replace(" ", "").split(",") if c]

    try:
        result = run_merge_20k(
            args.repo_root, build_id=args.build_id,
            legacy_build_id=args.legacy_build_id, lanes=lanes, seed=args.seed,
            resume=args.resume, dry_run=args.dry_run,
            write_reports=not args.no_reports,
        )
    except MergeError as exc:
        print(f"merge_20k: ERROR: {exc}", file=sys.stderr)
        return 2

    c = result.counts
    prefix = "[dry-run] " if result.dry_run else ""
    print(f"{prefix}build={args.build_id} legacy={args.legacy_build_id} "
          f"lanes_landed={c['lanes_landed']} lanes_pending={c['lanes_pending']}")
    print(f"{prefix}legacy_8k_qualified={c['legacy_8k_qualified']} "
          f"new_candidates={c['new_candidates']} new_qualified={c['new_qualified']}")
    print(f"{prefix}new_native_checked={c['new_native_checked']} "
          f"new_native_pass={c['new_native_pass']} "
          f"qualified_native={c['qualified_native']}")
    print(f"{prefix}unique_task_ids_20k={c['unique_task_ids_20k']} "
          f"target={c['target_total']} gap_to_target={c['gap_to_target']}")
    print(f"{prefix}task_id_collisions={c['task_id_collisions']} "
          f"(legacy_wins={c['task_id_collisions_legacy_wins']}) "
          f"instance_id_collisions={c['cross_origin_instance_id_collisions']}")
    print(f"{prefix}lineage_groups={result.totals['lineage_groups']} "
          f"base_goals={result.totals['distinct_base_goal_ids']} "
          f"families={result.totals['distinct_families']}")
    for name in sorted(result.files):
        meta = result.files[name]
        print(f"  {name}: rows={meta['rows']} sha256={meta['sha256'][:16]}…")
    for name, path in sorted(result.report_files.items()):
        print(f"  report {name}: {path}")
    if result.dry_run:
        print("[dry-run] nothing written")
    else:
        print(f"merge report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
