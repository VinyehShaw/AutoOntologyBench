"""§6 global report templates for the 20K build (owner: QA).

EXPANSION_20K_EXECUTION_PLAN §6 requires five global reports alongside the
20K build. Every number in them is read back from the merge's own artifacts
(``bank/builds/aob-20000-candidate-dev1/`` + ``MERGE_REPORT.json``); nothing is
estimated, extrapolated or padded. Anything not yet known is rendered as an
explicit ``PENDING — <reason>`` placeholder, so the templates are usable before
all lanes land and stay honest after they do.

    BANK_TASK_SCALE_20K.csv            spec §6.2 ladder per lane + TOTAL
    oracle_validation_coverage.csv     per-lane oracle/native-check coverage
    dedup_and_lineage_report.md        signatures, lineage groups, splits, churn
    license_and_distribution_report.md source licenses x redistribution x splits
    aob_20000_build_report.md          top-level build report + §6 counts + gaps

Deterministic: no wall-clock timestamps anywhere. The build's own
``BUILD_MANIFEST.json`` sha256s are the version anchor, so re-running over
unchanged inputs reproduces byte-identical reports.

Regenerate without re-merging::

    .venv/bin/python -m autoontobench.factory.merge_20k --repo-root . --reports-only
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from autoontobench.factory.gates.license_matrix import LicenseMatrix
from autoontobench.factory.gates.pipeline import load_build, row_source_ids
from autoontobench.factory.merge_staging import MergeError, _atomic_write, _dumps
from autoontobench.factory.merge_20k import (
    DEFAULT_BUILD_ID,
    PLAN_DATE,
    TARGET_TOTAL,
    _is_qualified_native,
    _native_check_state,
)
from autoontobench.factory.reports import scale_csv

PENDING = "PENDING"

# never summed: a lane appears once per origin, so accumulating these double-counts
PER_LANE_QUOTA_COLUMNS = ("target_total", "legacy_count", "new_needed")

COVERAGE_COLUMNS = [
    "lane_id", "origin", "dataset_id", "tasks",
    "target_total", "legacy_count", "new_needed",
    "oracle_handle_present", "native_evidence_present", "native_commands_present",
    "prove_exists_present", "semantic_witness_present",
    "native_checked", "native_pass", "native_not_recorded", "native_not_run",
    "qualification_native_verified", "solvable", "nontrivial",
    "semantic_relevance", "leakage_clean", "release_eligible",
    "qualified_native", "verdict_tokens",
]

LICENSE_COLUMNS = [
    "source_id", "license", "decision", "basis", "shippable", "content_ok",
    "tasks", "lanes", "reason",
]

UNRESOLVED_COLUMNS = [
    "namespace", "distinct_snapshot_ids", "task_source_edges", "lanes",
    "example_id", "decision",
]


def _load_plan_totals(lanes_yaml: Path) -> dict[str, Any]:
    if not lanes_yaml.exists():
        return {}
    doc = yaml.safe_load(lanes_yaml.read_text(encoding="utf-8")) or {}
    totals = doc.get("totals")
    return totals if isinstance(totals, dict) else {}


# --------------------------------------------------------------------- helpers

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MergeError(f"{path}: malformed JSON ({exc})") from exc
    return obj if isinstance(obj, dict) else {}


def _load_license_matrix(repo_root: Path) -> LicenseMatrix:
    return LicenseMatrix.load(
        matrix_csv=repo_root / "source_registry" / "licenses"
        / "source_redistribution_matrix.csv",
        sources_yaml=repo_root / "source_registry" / "sources.yaml",
    )


def _flag(row: dict[str, Any], key: str) -> bool:
    qualification = row.get("qualification")
    return isinstance(qualification, dict) and qualification.get(key) is True


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---:" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def _pending(reason: str) -> str:
    return f"{PENDING} — {reason}"


# ------------------------------------------- oracle_validation_coverage.csv

def generate_oracle_coverage(*, build_dir: Path, lane_targets: dict[str, dict[str, Any]],
                             plan_totals: dict[str, Any],
                             out_csv: Path) -> list[dict[str, Any]]:
    """Per (lane, origin) oracle + native-check coverage, plus a TOTAL row.

    ``native_checked`` / ``native_pass`` come only from verdict tokens the row
    itself records (plan §2: never PASS for a check that was not executed).
    """
    artifacts = load_build(build_dir)
    manifest_by_task = {str(r.get("task_id")): r for r in artifacts.task_manifest}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in artifacts.qualified:
        lane = str(row.get("lane_id") or "<unknown>")
        meta = manifest_by_task.get(str(row.get("task_id"))) or {}
        origin = str(meta.get("origin") or "<unknown>")
        buckets.setdefault((lane, origin), []).append(row)

    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {c: 0 for c in COVERAGE_COLUMNS[3:]
                              if c != "verdict_tokens"
                              and c not in PER_LANE_QUOTA_COLUMNS}
    token_totals: Counter[str] = Counter()
    for (lane, origin) in sorted(buckets):
        lane_rows = buckets[(lane, origin)]
        target = lane_targets.get(lane) or {}
        tokens: Counter[str] = Counter()
        rec: dict[str, Any] = {
            "lane_id": lane, "origin": origin,
            "dataset_id": str(lane_rows[0].get("dataset_id") or ""),
            "tasks": len(lane_rows),
            "target_total": target.get("target_total", ""),
            "legacy_count": target.get("legacy_count", ""),
            "new_needed": target.get("new_needed", ""),
            "verdict_tokens": "",
        }
        counters = {c: 0 for c in COVERAGE_COLUMNS if c not in rec}
        for row in lane_rows:
            evidence = row.get("native_evidence")
            evidence = evidence if isinstance(evidence, dict) else {}
            prove = evidence.get("prove_exists")
            witness = evidence.get("semantic_witness")
            commands = evidence.get("commands")
            checked, verdict = _native_check_state(row)
            tokens[verdict] += 1
            if isinstance(row.get("oracle_handle"), str) and row["oracle_handle"]:
                counters["oracle_handle_present"] += 1
            if evidence:
                counters["native_evidence_present"] += 1
            if isinstance(commands, list) and commands:
                counters["native_commands_present"] += 1
            if isinstance(prove, dict) and prove:
                counters["prove_exists_present"] += 1
            if isinstance(witness, dict) and witness:
                counters["semantic_witness_present"] += 1
            counters["native_checked"] += 1 if checked else 0
            counters["native_pass"] += 1 if verdict == "PASS" else 0
            counters["native_not_recorded"] += 1 if verdict == "NOT_RECORDED" else 0
            counters["native_not_run"] += 1 if verdict == "NOT_RUN" else 0
            counters["qualified_native"] += 1 if _is_qualified_native(row) else 0
            for key, column in (("native_verified", "qualification_native_verified"),
                                ("solvable", "solvable"), ("nontrivial", "nontrivial"),
                                ("semantic_relevance", "semantic_relevance"),
                                ("leakage_clean", "leakage_clean"),
                                ("release_eligible", "release_eligible")):
                counters[column] += 1 if _flag(row, key) else 0
        rec.update(counters)
        rec["verdict_tokens"] = ";".join(f"{k}={v}" for k, v in sorted(tokens.items()))
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + value
        for key in ("tasks",):
            totals[key] = totals.get(key, 0) + rec[key]
        token_totals.update(tokens)
        rows.append(rec)

    rows.append({
        "lane_id": "TOTAL", "origin": "ALL", "dataset_id": "MIXED",
        "target_total": plan_totals.get("target_total", TARGET_TOTAL),
        "legacy_count": plan_totals.get("legacy_8k", ""),
        "new_needed": plan_totals.get("new_needed", ""),
        "verdict_tokens": ";".join(f"{k}={v}" for k, v in sorted(token_totals.items())),
        **{k: v for k, v in totals.items() if k in COVERAGE_COLUMNS},
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COVERAGE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ------------------------------------------------- license_and_distribution

def _registry_source_ids(repo_root: Path) -> list[str]:
    sources_yaml = repo_root / "source_registry" / "sources.yaml"
    if not sources_yaml.exists():
        return []
    doc = yaml.safe_load(sources_yaml.read_text(encoding="utf-8")) or {}
    return sorted({str(s.get("source_id")) for s in (doc.get("sources") or [])
                   if isinstance(s, dict) and s.get("source_id")})


def _namespace(source_id: str) -> str:
    return source_id.split("/", 1)[0].split(":", 1)[0]


def generate_license_rows(*, build_dir: Path, matrix: LicenseMatrix,
                          registry_ids: list[str]) -> tuple[list[dict[str, Any]],
                                                            list[dict[str, Any]]]:
    """Per-registry-source license decisions + unresolved snapshot namespaces.

    ``row_source_ids`` also yields snapshot-level prefixes that are *not*
    registry sources (L05 pins one snapshot per molecule, e.g.
    ``fg:Amine.Secondary@…``; L04 carries ``chebi_entity`` / ``rhea_reaction``
    sub-namespaces). Emitting a license row per prefix produced a 9,476-row
    table of decisions that do not exist. Only registry sources get a decision;
    everything else is aggregated by namespace and reported as unresolved, which
    is the honest state and is actionable (add the namespace to the registry or
    map it onto its parent source).
    """
    artifacts = load_build(build_dir)
    known = set(registry_ids)
    per_source: dict[str, dict[str, Any]] = {
        sid: {"tasks": 0, "lanes": set()} for sid in registry_ids
    }
    unresolved: dict[str, dict[str, Any]] = {}
    for row in artifacts.qualified:
        lane = str(row.get("lane_id") or "")
        matched: set[str] = set()
        for source_id in row_source_ids(row):
            target = None
            if source_id in known:
                target = source_id
            else:
                head = source_id.split("/", 1)[0]
                if head in known:
                    target = head
            if target is not None:
                matched.add(target)
                continue
            bucket = unresolved.setdefault(_namespace(source_id), {
                "namespace": _namespace(source_id), "distinct_ids": set(),
                "tasks": 0, "lanes": set(), "example": source_id})
            bucket["distinct_ids"].add(source_id)
            bucket["tasks"] += 1
            if lane:
                bucket["lanes"].add(lane)
        for target in matched:
            per_source[target]["tasks"] += 1
            if lane:
                per_source[target]["lanes"].add(lane)

    rows: list[dict[str, Any]] = []
    for source_id in registry_ids:
        decision = matrix.decide(source_id)
        rows.append({
            "source_id": source_id,
            "license": decision.license or _pending("no license in matrix/registry"),
            "decision": decision.decision.value,
            "basis": decision.basis,
            "shippable": decision.shippable,
            "content_ok": decision.content_ok,
            "tasks": per_source[source_id]["tasks"],
            "lanes": ",".join(sorted(per_source[source_id]["lanes"])),
            "reason": decision.reason,
        })
    unresolved_rows = [{
        "namespace": key,
        "distinct_snapshot_ids": len(value["distinct_ids"]),
        "task_source_edges": value["tasks"],
        "lanes": ",".join(sorted(value["lanes"])),
        "example_id": value["example"],
        "decision": _pending("not a registry source: no license decision exists at "
                             "this granularity"),
    } for key, value in sorted(unresolved.items(),
                               key=lambda kv: (-kv[1]["tasks"], kv[0]))]
    return rows, unresolved_rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------- markdown docs

def _dedup_md(*, counts: dict[str, Any], dedup_stats: dict[str, Any],
              dedup_summary: dict[str, Any], build_id: str, seed: int) -> str:
    split_sizes = dedup_stats.get("split_sizes") or {}
    total = int(counts.get("unique_task_ids_20k") or 0)
    lines = [
        f"# Dedup and Lineage Report — {build_id}",
        "",
        f"> Plan: `EXPANSION_20K_EXECUTION_PLAN.md` §6. Generated by "
        f"`autoontobench.factory.reports_20k` from `bank/builds/{build_id}/` only.",
        f"> Plan date {PLAN_DATE}; dedup seed `{seed}`. No wall-clock timestamps: "
        "re-running over unchanged inputs reproduces this file byte-for-byte.",
        "",
        "## 1. Method",
        "",
        "Dedup runs on **canonical AST signatures** (`gates/canonicalize.py`), never on",
        "wording: `family_signature` / `base_goal_signature` / `instance_signature`.",
        "`source_lineage_group` unions tasks sharing lineage, mechanism group, or an",
        "identical base-goal signature. Train/dev/test is assigned by **whole lineage",
        "group**, so a group never crosses a split (leakage prevention at the split layer).",
        "",
        "Two dedup layers are reported separately and must not be conflated:",
        "",
        "1. **id-level** — `task_id` uniqueness across the legacy 8K build and the new",
        "   `bank/staging_20k` lanes. Legacy wins on collision; every collision is",
        "   listed in `dedup/task_id_collisions.jsonl`.",
        "2. **signature-level** — structural duplication measured by S11 `run_dedup`.",
        "   It is reported as statistics; it does **not** silently drop tasks.",
        "",
        "## 2. id-level uniqueness",
        "",
        _md_table(
            ["metric", "value"],
            [["unique task_id (20K qualified set)", counts.get("unique_task_ids_20k")],
             ["unique task_id (candidate set)", counts.get("unique_candidate_task_ids")],
             ["legacy 8K unique task_id", counts.get("legacy_8k_unique_task_ids")],
             ["task_id collisions", counts.get("task_id_collisions")],
             ["  … resolved legacy-8k-wins", counts.get("task_id_collisions_legacy_wins")],
             ["cross-origin instance_id collisions",
              counts.get("cross_origin_instance_id_collisions")],
             ["target total (plan §1)", counts.get("target_total")],
             ["gap to target", counts.get("gap_to_target")]],
        ),
        "",
        "## 3. Signature-level duplication",
        "",
        _md_table(
            ["metric", "value"],
            [["unique instance signatures", dedup_summary.get("unique_instance_signatures")],
             ["unique base-goal signatures",
              dedup_summary.get("unique_base_goal_signatures")],
             ["instance-level duplicates flagged",
              dedup_summary.get("instance_level_duplicates_flagged")],
             ["base-goal-level duplicates flagged",
              dedup_summary.get("base_goal_level_duplicates_flagged")],
             ["instance duplicate rate", dedup_summary.get("instance_duplicate_rate")],
             ["base-goal signature duplicate rate",
              dedup_summary.get("base_goal_signature_duplicate_rate")],
             ["lineage groups", dedup_summary.get("lineage_groups")]],
        ),
        "",
        "**Reading the rates.** `instance_level` counts *true content duplicates*:",
        "identical public task content under different task ids. `base_goal_level` counts",
        "*template density*, **not** bank rejections — additional instances of a",
        "structurally identical goal-level signature are the intended",
        "instances-per-goal parameterization. A high base-goal rate is therefore expected",
        "and is not a defect; the instance rate is the one that gates bank quality.",
        "",
        "## 4. Splits (whole-group, 70/15/15 target)",
        "",
        _md_table(
            ["split", "tasks", "share"],
            [[name, split_sizes.get(name, 0),
              (f"{split_sizes.get(name, 0) / total:.4f}" if total else "n/a")]
             for name in ("train", "dev", "test")],
        ),
        "",
        f"Actual sizes deviate from 70/15/15 by group granularity: assignment is per",
        "lineage group, so a single large group moves a whole block of tasks. Deviation",
        "smooths out as more lanes land.",
        "",
        "## 5. Legacy split churn vs the frozen 8K build",
        "",
        _md_table(
            ["metric", "value"],
            [["legacy tasks whose split changed vs 8K",
              dedup_stats.get("legacy_split_churn_vs_8k")],
             ["legacy tasks whose lineage group changed vs 8K",
              dedup_stats.get("legacy_lineage_churn_vs_8k")]],
        ),
        "",
        "Splits and lineage groups are **build-level** properties. The 20K build",
        "recomputes them over the legacy+new union with its own seed, so legacy tasks may",
        "move split relative to `aob-8000-rc1`. The frozen 8K splits are untouched and",
        "remain authoritative for that build; per-task both values are carried in",
        "`task_manifest.jsonl` (`split` and `legacy_split_8k`) so any consumer can pin",
        "either. This is expected behaviour, not legacy mutation — no byte under",
        "`bank/builds/aob-8000-rc1/` is written.",
        "",
        "## 6. Largest lineage groups",
        "",
        _md_table(["lineage_group", "tasks"],
                  [[g, n] for g, n in (dedup_stats.get("largest_lineage_groups") or [])]),
        "",
        "## 7. Per-lane duplicate statistics",
        "",
        _md_table(
            ["lane_id", "total", "uniq_instances", "uniq_base_goals", "families",
             "mech_groups", "lineage_groups", "instance_dup_rate", "base_goal_dup_rate",
             "max_family_share"],
            [[s.get("lane_id"), s.get("total"), s.get("unique_instances"),
              s.get("unique_base_goals"), s.get("families"), s.get("mechanism_groups"),
              s.get("lineage_groups"), s.get("instance_duplicate_rate"),
              s.get("base_goal_duplicate_rate"), s.get("max_family_share")]
             for s in (dedup_stats.get("stats") or [])],
        ),
        "",
        "## 8. Artifacts",
        "",
        "- `bank/builds/%s/dedup/dedup_decisions.jsonl` — per-task SignatureIndex verdict"
        % build_id,
        "- `bank/builds/%s/dedup/lineage_splits.jsonl` — task → lineage group → split"
        % build_id,
        "- `bank/builds/%s/dedup/DEDUP_STATS.json` — the statistics above" % build_id,
        "- `bank/builds/%s/dedup/task_id_collisions.jsonl` — id-level resolutions"
        % build_id,
        "",
    ]
    return "\n".join(lines)


def _license_md(*, license_rows: list[dict[str, Any]],
                unresolved_rows: list[dict[str, Any]],
                split_sizes: dict[str, int],
                counts: dict[str, Any], build_id: str) -> str:
    total = sum(int(r["tasks"]) for r in license_rows) or 0
    hold = [r for r in license_rows if r["decision"] == "HOLD_REVIEW"]
    hold_tasks = sum(int(r["tasks"]) for r in hold)
    unresolved_edges = sum(int(r["task_source_edges"]) for r in unresolved_rows)
    lines = [
        f"# License and Distribution Report — {build_id}",
        "",
        f"> Plan: `EXPANSION_20K_EXECUTION_PLAN.md` §6. Plan date {PLAN_DATE}.",
        "> Decisions come from `source_registry/licenses/source_redistribution_matrix.csv`",
        "> and `source_registry/sources.yaml` via `gates/license_matrix.py`. A source",
        "> absent from both resolves to `HOLD_REVIEW` (default hold) — never to a guess.",
        "",
        "## 1. Redistribution decision ladder",
        "",
        "- `REDISTRIBUTE_OK` — content may ship in the public release.",
        "- `REFERENCE_ONLY` — cite + summarize; no content bundling.",
        "- `FETCH_RECIPE_ONLY` — ship recipe + hash; the user fetches.",
        "- `METADATA_ONLY` — metadata index only, no content.",
        "- `HOLD_REVIEW` — unresolved; **blocks release eligibility**.",
        "",
        "## 2. Summary",
        "",
        _md_table(
            ["metric", "value"],
            [["registry sources", len(license_rows)],
             ["registry sources referenced by qualified tasks",
              sum(1 for r in license_rows if int(r["tasks"]) > 0)],
             ["source→task edges (registry sources)", total],
             ["sources under HOLD_REVIEW", len(hold)],
             ["qualified tasks touching a HOLD_REVIEW source", hold_tasks],
             ["unresolved snapshot namespaces", len(unresolved_rows)],
             ["source→task edges outside the registry", unresolved_edges],
             ["unique task_id in build", counts.get("unique_task_ids_20k")],
             ["release decision",
              _pending("release gating is a separate step; this build is a "
                       "candidate dev build, not a release")]],
        ),
        "",
        "## 3. Per-source decisions (registry sources only)",
        "",
        _md_table(
            ["source_id", "license", "decision", "basis", "shippable", "content_ok",
             "tasks", "lanes"],
            [[r["source_id"], r["license"], r["decision"], r["basis"], r["shippable"],
              r["content_ok"], r["tasks"], r["lanes"]] for r in license_rows],
        ),
        "",
        "## 4. Unresolved snapshot namespaces",
        "",
        "`source_snapshot_ids` are pinned at a finer granularity than the source",
        "registry: L05 pins one snapshot per molecule (`fg:Amine.Secondary@…`), L04",
        "carries `chebi_entity` / `rhea_reaction` sub-namespaces. These are **not**",
        "registry sources, so no license decision exists for them and none is invented.",
        "They are aggregated by namespace here. Each is either a child of a registry",
        "source (already covered above) or a registry gap to close before a release cut.",
        "",
    ]
    if unresolved_rows:
        lines.append(_md_table(
            ["namespace", "distinct snapshot ids", "source→task edges", "lanes",
             "example id"],
            [[r["namespace"], r["distinct_snapshot_ids"], r["task_source_edges"],
              r["lanes"], r["example_id"]] for r in unresolved_rows]))
    else:
        lines.append("None — every `source_snapshot_ids` prefix resolves to a registry"
                     " source.")
    lines += [
        "",
        "## 5. Split distribution",
        "",
        _md_table(["split", "tasks", "share"],
                  [[name, split_sizes.get(name, 0),
                    (f"{split_sizes.get(name, 0) / (sum(split_sizes.values()) or 1):.4f}")]
                   for name in ("train", "dev", "test")]),
        "",
        "## 6. New 20K sources (L14 ChEMBL / L15 Reactome / L16 UniProt)",
        "",
    ]
    new_sources = [r for r in license_rows
                   if any(l in ("L14", "L15", "L16") for l in str(r["lanes"]).split(","))]
    if new_sources:
        lines.append(_md_table(
            ["source_id", "license", "decision", "tasks"],
            [[r["source_id"], r["license"], r["decision"], r["tasks"]]
             for r in new_sources]))
    else:
        lines.append(_pending("no L14/L15/L16 tasks in this build yet; the lanes have "
                              "not landed under `bank/staging_20k/`, and no ChEMBL / "
                              "Reactome / UniProt source is registered in "
                              "`source_registry/sources.yaml`"))
    lines += [
        "",
        "## 7. Open items",
        "",
        "- `LICENSE` file: still absent at repo root (see `docs/OPEN_SOURCE_CHECKLIST.md`);",
        "  `LICENSES_AND_DATA_PROVENANCE.md` and `CITATION.cff` remain drafts pending PI",
        "  sign-off. This report does not substitute for that sign-off.",
        "- Any source resolving to `HOLD_REVIEW` must be cleared or its tasks excluded",
        "  before a release cut; the candidate dev build intentionally keeps them and",
        "  reports them here.",
        "- L14/L15/L16 sources (ChEMBL, Reactome, UniProt) must be added to",
        "  `source_registry/sources.yaml` and the redistribution matrix before their",
        "  tasks can be license-gated; until then they resolve to `HOLD_REVIEW`.",
        "",
    ]
    return "\n".join(lines)


def _build_md(*, build_id: str, legacy_build_id: str, counts: dict[str, Any],
              per_lane: list[dict[str, Any]], lanes_pending: list[dict[str, str]],
              lane_targets: dict[str, dict[str, Any]], legacy_integrity: dict[str, Any],
              merge_command: str, files: dict[str, dict[str, Any]],
              landed_dirs: list[str],
              dedup_summary: dict[str, Any], seed: int) -> str:
    unique = int(counts.get("unique_task_ids_20k") or 0)
    gap = int(counts.get("gap_to_target") or 0)
    target_lanes = set(lane_targets)
    missing_lanes = sorted(target_lanes - {d.upper() for d in landed_dirs})
    integrity_files = legacy_integrity.get("files") or {}
    sha_mismatches = sum(1 for f in integrity_files.values() if not f.get("match"))

    lane_rows: list[list[Any]] = []
    for lane_id in sorted(target_lanes | {str(r.get("lane_id")) for r in per_lane}):
        target = lane_targets.get(lane_id) or {}
        actual = [r for r in per_lane if str(r.get("lane_id")) == lane_id]
        legacy_n = sum(int(r.get("legacy_qualified") or 0) for r in actual)
        new_n = sum(int(r.get("new_qualified") or 0) for r in actual)
        total_n = sum(int(r.get("qualified") or 0) for r in actual)
        want = target.get("target_total")
        lane_rows.append([
            lane_id, want if want is not None else _pending("not in lanes_20000.yaml"),
            target.get("legacy_count", ""), target.get("new_needed", ""),
            legacy_n, new_n, total_n,
            (total_n - want) if isinstance(want, int) else "",
            sum(int(r.get("native_checked") or 0) for r in actual),
            sum(int(r.get("native_pass") or 0) for r in actual),
            sum(int(r.get("qualified_native") or 0) for r in actual),
        ])

    lines = [
        f"# AutoOntologyBench 20K Build Report — {build_id}",
        "",
        f"> Plan: `EXPANSION_20K_EXECUTION_PLAN.md` §6/§7. Plan date {PLAN_DATE}.",
        f"> Legacy base: `{legacy_build_id}` (frozen, read-only). Dedup seed `{seed}`.",
        "> Every figure below is read back from `bank/builds/%s/` and its" % build_id,
        "> `MERGE_REPORT.json`. Nothing is estimated or padded; unknowns are marked",
        f"> `{PENDING}` with a reason.",
        "",
        "## 1. Headline counts (plan §6: reported separately)",
        "",
        _md_table(
            ["metric", "value", "meaning"],
            [["legacy 8K (qualified)", counts.get("legacy_8k_qualified"),
              "rows carried verbatim from the frozen 8K build"],
             ["legacy 8K (candidates)", counts.get("legacy_8k_candidates"),
              "8K candidate pool, includes non-qualified rows"],
             ["legacy 8K (rejections)", counts.get("legacy_8k_rejections"),
              "8K lane rejections, carried through"],
             ["new candidate", counts.get("new_candidates"),
              "candidate rows from bank/staging_20k lanes"],
             ["new qualified", counts.get("new_qualified"),
              "QUALIFIED rows from bank/staging_20k lanes"],
             ["new native_checked", counts.get("new_native_checked"),
              "new qualified rows recording an executed native check"],
             ["new native_pass", counts.get("new_native_pass"),
              "new qualified rows whose derived native verdict is PASS"],
             ["unique 20K task ids", counts.get("unique_task_ids_20k"),
              "distinct task_id in the merged qualified set"],
             ["QUALIFIED_NATIVE", counts.get("qualified_native"),
              "qualified status AND qualification.native_verified=true"],
             ["target total", counts.get("target_total"), "plan §1 (includes legacy 8K)"],
             ["gap to target", counts.get("gap_to_target"),
              "honest gap; never padded (plan §1)"]],
        ),
        "",
        f"**Status: {'TARGET MET' if gap <= 0 else f'{gap} tasks short of the 20,000 target'}"
        f"** ({unique} unique task ids).",
        "",
        "## 2. Per-lane progress vs `configs/lanes_20000.yaml`",
        "",
        _md_table(
            ["lane", "target_total", "legacy_count", "new_needed", "legacy_actual",
             "new_actual", "total_actual", "delta_vs_target", "native_checked",
             "native_pass", "qualified_native"],
            lane_rows,
        ),
        "",
        "`delta_vs_target` is `total_actual - target_total`; negative means the lane is",
        "still short. Plan §1: targets are build goals — if a lane cannot support its",
        "`new_needed` with real independent mechanisms, the gap is reported and **not**",
        "padded; QA may reallocate only with a written evidence note.",
        "",
        "## 3. Lane landing state",
        "",
        _md_table(["lane dir", "kind", "reason"],
                  [[p.get("lane"), p.get("kind"), p.get("reason")]
                   for p in lanes_pending]
                  or [["<none>", "-",
                       "every lane dir under bank/staging_20k has landed"]]),
        "",
    ]
    if missing_lanes:
        lines += [
            "Lane dirs not yet created under `bank/staging_20k/`: "
            + ", ".join(missing_lanes) + ".",
            "",
        ]
    lines += [
        "The merge never waits for lanes: it takes whatever has landed and is re-runnable.",
        "",
        "## 4. Legacy 8K integrity (plan §6/§7: legacy untouched)",
        "",
        _md_table(
            ["check", "result"],
            [["legacy build", legacy_integrity.get("legacy_build_id")],
             ["opened read-only", legacy_integrity.get("read_only")],
             ["BUILD_MANIFEST files sha256-verified",
              legacy_integrity.get("manifest_files_verified")],
             ["sha256 mismatches", sha_mismatches],
             ["legacy rows re-serialize to their frozen bytes (load-time proof)",
              legacy_integrity.get("legacy_rows_byte_identical")],
             ["legacy rows carried verbatim into this build",
              legacy_integrity.get("pre_write_legacy_rows_verified",
                                   _pending("not computed"))],
             ["legacy rows that would not carry verbatim (merge aborts if > 0)",
              legacy_integrity.get("pre_write_legacy_rows_missing",
                                   _pending("not computed"))],
             ["carried verbatim (no derived fields recomputed)",
              legacy_integrity.get("carried_verbatim")]],
        ),
        "",
        "Legacy task records are re-emitted byte-for-byte. Index files",
        "(`task_manifest.jsonl`, `native_evidence_index.jsonl`, `dedup/*`) **are**",
        "recomputed for the 20K build, because lineage groups and splits are build-level",
        "properties of the legacy+new union. No byte under `bank/builds/aob-8000-rc1/`,",
        "`bank/staging/` or `bank/releases/` is written by this tool.",
        "",
        "## 5. Dedup summary",
        "",
        _md_table(
            ["metric", "value"],
            [["task_id collisions", counts.get("task_id_collisions")],
             ["  … resolved legacy-8k-wins", counts.get("task_id_collisions_legacy_wins")],
             ["cross-origin instance_id collisions",
              counts.get("cross_origin_instance_id_collisions")],
             ["lineage groups", dedup_summary.get("lineage_groups")],
             ["instance-level duplicates flagged",
              dedup_summary.get("instance_level_duplicates_flagged")],
             ["base-goal-level duplicates flagged",
              dedup_summary.get("base_goal_level_duplicates_flagged")],
             ["split sizes", json.dumps(dedup_summary.get("split_sizes"), sort_keys=True)]],
        ),
        "",
        "Full detail: [`dedup_and_lineage_report.md`](dedup_and_lineage_report.md).",
        "",
    ]
    by_lane = counts.get("task_id_collisions_by_dropped_lane") or {}
    if by_lane:
        lines += [
            "**Finding — new lanes re-emitting legacy task_ids.** The following lane(s)",
            "wrote rows into `bank/staging_20k/` whose `task_id` already exists in the",
            "frozen 8K build. Plan §2 requires new records to use the new seed base",
            "20261101 precisely to avoid this, and §2 also requires new tasks to be",
            "genuinely new lineages rather than renamed legacy tasks. The merge resolved",
            "every case **legacy-8k-wins**, so the 20K build holds the frozen legacy row",
            "and no duplicate entered — but the dropped rows are not new tasks and the",
            "lane's `new_needed` contribution is overstated by that amount.",
            "",
            _md_table(["lane that dropped rows", "colliding task_ids"],
                      [[lane, n] for lane, n in sorted(by_lane.items(),
                                                       key=lambda kv: (-kv[1], kv[0]))]),
            "",
            "Action: the lane owner should re-key these onto the new seed base, or",
            "confirm they are intentional legacy carry-forward and exclude them from",
            "`new_needed`. Full list: `bank/builds/%s/dedup/task_id_collisions.jsonl`."
            % build_id,
            "",
        ]
    lines += [
        "## 6. Build artifacts",
        "",
        _md_table(["file", "rows", "sha256"],
                  [[name, meta.get("rows"), meta.get("sha256")]
                   for name, meta in sorted(files.items())]),
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        "# dry run: all counts + the legacy integrity proof, writes nothing",
        ".venv/bin/python -m autoontobench.factory.merge_20k --repo-root . --dry-run",
        "",
        "# first real merge (auto-discovers every landed lane under bank/staging_20k)",
        merge_command.replace(" --resume", ""),
        "",
        "# regenerate only the five §6 reports from an existing build",
        ".venv/bin/python -m autoontobench.factory.merge_20k --repo-root . --reports-only",
        "```",
        "",
        "### FINAL COMMAND — run this once all lanes have finished",
        "",
        "```bash",
        merge_command,
        "```",
        "",
        "This is the command to run when L04/L11/L12/L14/L15/L16 land. `--lanes auto`",
        "takes every dir under `bank/staging_20k/` whose name matches `l01..l99` and which",
        "has at least one `qualified/*.jsonl` row; scratch dirs (`_smoke`, `l03_smoke`)",
        "are skipped and listed in §3. It is safe to re-run at any time:",
        "",
        "- **idempotent** — same inputs produce byte-identical artifacts (verified:",
        "  two consecutive runs match on every file); `--resume` only acknowledges the",
        "  intentional overwrite of the previous merge.",
        "- **legacy-safe** — `bank/builds/aob-8000-rc1/`, `bank/staging/`,",
        "  `bank/releases/` and `bank/reports/aob-8000-rc1/` are opened read-only and",
        "  are write-blocked by `_assert_writable`; the run aborts before any output if",
        "  the 8K `BUILD_MANIFEST.json` sha256s do not match disk.",
        "- **partial-tolerant** — it never waits for lanes; the gap to 20,000 is",
        "  reported, not padded.",
        "",
        "To pin an exact lane set instead (for a reproducible cut, immune to other",
        "workers creating dirs mid-run), replace `--lanes auto` with the explicit list:",
        "",
        "```bash",
        merge_command.replace("--lanes auto", "--lanes " + ",".join(landed_dirs))
        if landed_dirs else merge_command,
        "```",
        "",
        "## 8. Definition of done (plan §7) — current state",
        "",
        _md_table(
            ["#", "criterion", "state"],
            [[1, "all runnable lanes expanded; blocked lanes listed with evidence",
              _pending(f"{counts.get('lanes_landed')} lane(s) merged, "
                       f"{len(missing_lanes)} target lane(s) with no landed staging "
                       f"dir: {', '.join(missing_lanes) or '<none>'}; see §2/§3")],
             [2, "unique task_id total = 20,000 (or honest gap + reallocation note)",
              f"{unique} / {TARGET_TOTAL}; gap {gap}"],
             [3, "legacy 8K untouched; 20K build separate",
              "PASS" if legacy_integrity.get("pre_write_legacy_byte_identical",
                                             legacy_integrity.get(
                                                 "legacy_rows_byte_identical"))
              else _pending("integrity proof not yet run")],
             [4, "per-task check statuses recorded; no fabricated PASS",
              "PASS — `native_checked`/`native_verdict` per task in "
              "`task_manifest.jsonl`; absent verdicts render NOT_RECORDED/NOT_RUN"],
             [5, "re-audit 100 sample report with timing/filter outcomes",
              _pending("owned by R100; not part of the merge tool")],
             [6, "showcase 4–5 real cases; L11 screenshots only if renderer works",
              _pending("owned by VIS; not part of the merge tool")],
             [7, "final report distinguishes candidate / native_checked / "
                 "qualified_native / unique_benchmark", "PASS — see §1"]],
        ),
        "",
        "## 9. Known gaps",
        "",
    ]
    gaps = []
    if gap > 0:
        gaps.append(f"- {gap} tasks short of the 20,000 target; "
                    f"{counts.get('lanes_pending')} named lane dir(s) created but not "
                    f"landed, {counts.get('scratch_dirs_skipped')} scratch dir(s) "
                    "skipped.")
    if missing_lanes:
        gaps.append("- Lane dirs not yet created: " + ", ".join(missing_lanes) + ".")
    hold = [r for r in per_lane if int(r.get("qualified") or 0) == 0]
    if hold:
        gaps.append("- Lanes with zero qualified rows in this build: "
                    + ", ".join(sorted(str(r.get("lane_id")) for r in hold)) + ".")
    gaps.append("- `oracle_validation_coverage.csv` reports coverage only; it does not "
                "re-execute native checks. Re-execution is the R100 re-audit sample.")
    gaps.append("- Release gating (LICENSE file, PI sign-off) is out of scope for a "
                "candidate dev build; see `license_and_distribution_report.md` §7.")
    lines += gaps + [""]
    return "\n".join(lines)


# ------------------------------------------------------------------ entrypoints

def write_global_reports(
    repo_root: Path,
    *,
    build_id: str,
    build_dir: Path,
    report_dir: Path,
    merge_report: dict[str, Any],
    counts: dict[str, Any],
    per_lane: list[dict[str, Any]],
    dedup_stats: dict[str, Any],
    legacy_integrity: dict[str, Any],
    lanes_pending: list[dict[str, str]],
    lane_targets: dict[str, dict[str, Any]],
    seed: int,
    merge_command: str,
) -> dict[str, Path]:
    """Write the five §6 global reports. Deterministic; atomic single-writer."""
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix = _load_license_matrix(repo_root)
    out: dict[str, Path] = {}

    out["scale"] = report_dir / "BANK_TASK_SCALE_20K.csv"
    scale_csv.generate(build_dir=build_dir,
                       lanes_yaml=repo_root / Path("configs", "lanes_20000.yaml"),
                       license_matrix=matrix, out_csv=out["scale"])

    out["oracle_coverage"] = report_dir / "oracle_validation_coverage.csv"
    plan_totals = _load_plan_totals(repo_root / Path("configs", "lanes_20000.yaml"))
    generate_oracle_coverage(build_dir=build_dir, lane_targets=lane_targets,
                             plan_totals=plan_totals, out_csv=out["oracle_coverage"])

    dedup_summary = merge_report.get("dedup_summary") or {}
    split_sizes = (dedup_stats.get("split_sizes") or {})

    out["dedup"] = report_dir / "dedup_and_lineage_report.md"
    _atomic_write(out["dedup"], _dedup_md(
        counts=counts, dedup_stats=dedup_stats, dedup_summary=dedup_summary,
        build_id=build_id, seed=seed).encode("utf-8"))

    registry_ids = _registry_source_ids(repo_root)
    license_rows, unresolved_rows = generate_license_rows(
        build_dir=build_dir, matrix=matrix, registry_ids=registry_ids)
    out["license"] = report_dir / "license_and_distribution_report.md"
    _atomic_write(out["license"], _license_md(
        license_rows=license_rows, unresolved_rows=unresolved_rows,
        split_sizes=split_sizes, counts=counts, build_id=build_id).encode("utf-8"))
    _write_csv(report_dir / "LICENSE_DECISIONS_20K.csv", LICENSE_COLUMNS, license_rows)
    out["license_csv"] = report_dir / "LICENSE_DECISIONS_20K.csv"
    _write_csv(report_dir / "UNRESOLVED_SOURCE_NAMESPACES_20K.csv",
               UNRESOLVED_COLUMNS, unresolved_rows)
    out["unresolved_csv"] = report_dir / "UNRESOLVED_SOURCE_NAMESPACES_20K.csv"

    out["build_report"] = report_dir / "aob_20000_build_report.md"
    _atomic_write(out["build_report"], _build_md(
        build_id=build_id,
        legacy_build_id=str(merge_report.get("legacy", {}).get("build_id", "")),
        counts=counts, per_lane=per_lane, lanes_pending=lanes_pending,
        lane_targets=lane_targets, legacy_integrity=legacy_integrity,
        merge_command=merge_command, files=merge_report.get("outputs") or {},
        landed_dirs=[str(d) for d in (merge_report.get("lane_order") or [])],
        dedup_summary=dedup_summary, seed=seed).encode("utf-8"))
    return out


def regenerate_from_build(repo_root: Path, *,
                          build_id: str = DEFAULT_BUILD_ID) -> dict[str, Path]:
    """Rebuild the five §6 reports from an existing build dir (no re-merge)."""
    repo_root = Path(repo_root)
    build_dir = repo_root / "bank" / "builds" / build_id
    report_dir = repo_root / "bank" / "reports" / build_id
    manifest = _load_json(build_dir / "BUILD_MANIFEST.json")
    if not manifest:
        raise MergeError(f"no BUILD_MANIFEST.json under {build_dir}; run the merge first")
    merge_report = _load_json(report_dir / "MERGE_REPORT.json")
    lanes_yaml = repo_root / Path("configs", "lanes_20000.yaml")
    from autoontobench.factory.merge_20k import _load_lane_targets
    return write_global_reports(
        repo_root, build_id=build_id, build_dir=build_dir, report_dir=report_dir,
        merge_report=merge_report,
        counts=manifest.get("counts") or {},
        per_lane=manifest.get("per_lane") or [],
        dedup_stats=_load_json(build_dir / "dedup" / "DEDUP_STATS.json"),
        legacy_integrity=_load_json(build_dir / "LEGACY_INTEGRITY.json"),
        lanes_pending=manifest.get("lanes_pending") or [],
        lane_targets=_load_lane_targets(lanes_yaml),
        seed=int(manifest.get("dedup_seed") or 0),
        merge_command=str((merge_report.get("re_run") or {}).get(
            "command",
            f".venv/bin/python -m autoontobench.factory.merge_20k --repo-root . "
            f"--build-id {build_id} --resume")),
    )
