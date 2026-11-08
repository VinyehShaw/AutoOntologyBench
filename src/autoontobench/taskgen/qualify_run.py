"""W8 M3 qualification pipeline: bind T4 wording candidates onto W7-verified
blueprints, run the T5 validator, and emit the qualified-task record
(`benchmarks/qualified/tasks.jsonl`), the dev split (`benchmarks/splits/dev.jsonl`)
and a digest manifest. Deterministic; no network, no model calls (the wording
readability critique stays NOT_RUN at M3 by design).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import (
    Authorship, TaskSpecV5, TaskStatus, WordingCandidate,
)
from autoontobench.taskgen.bind import bind_blueprint
from autoontobench.taskgen.checks_public import spec_goal_payload
from autoontobench.taskgen.dedupe import goal_sha
from autoontobench.taskgen.split import lineage_key
from autoontobench.taskgen.validate import QualificationReport, TaskQualificationValidator

LADDER_RANK = {"REJECTED": -1, "CANDIDATE": 0, "NATIVE_VERIFIED": 1, "TASK_FEASIBLE": 2,
               "SEMANTICALLY_GROUNDED": 3, "LEAK_FREE": 4, "QUALIFIED": 5}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class QualifiedTaskRow:
    spec: TaskSpecV5
    wording: WordingCandidate | None
    base_goal_id: str
    canonical_sha: str
    report: QualificationReport


@dataclass(frozen=True)
class QualificationRun:
    rows: list[QualifiedTaskRow] = field(default_factory=list)
    blueprint_paths: dict[str, str] = field(default_factory=dict)
    blueprint_shas: dict[str, str] = field(default_factory=dict)
    wordings_path: str | None = None
    n_wording_rows: int = 0

    @property
    def unique_goals(self) -> dict[str, list[QualifiedTaskRow]]:
        goals: dict[str, list[QualifiedTaskRow]] = {}
        for row in self.rows:
            goals.setdefault(row.base_goal_id, []).append(row)
        return goals


def row_to_json(row: QualifiedTaskRow, *, log_root: Path, registry_dir: str,
                acceptance_ref: str = "audit/review5/P2_ACCEPTANCE.md (command 8)"
                ) -> dict[str, Any]:
    report = row.report
    return {
        "schema_version": "aob-qualified/1.0",
        "task_id": row.spec.task_id,
        "base_goal_id": row.base_goal_id,
        "canonical_sha": row.canonical_sha,
        "spec": json.loads(row.spec.model_dump_json()),
        "wording": json.loads(row.wording.model_dump_json()) if row.wording else None,
        "qualification": {
            "decision": report.decision,
            "ladder_status": report.ladder_status,
            "qualification_scope": row.spec.ontology_contract,
            "checks": [{"check": r.check, "status": r.status.value, "reason": r.reason,
                        "dep": r.dep, "evidence": dict(r.evidence)}
                       for r in report.results],
            "rejection_reasons": report.rejection_reasons,
            "pending_deps": report.pending_deps,
            "not_applicable": report.not_applicable_checks,
            "wording_model_critique": "NOT_RUN",
            "evidence_refs": {
                "native_log_root": str(log_root),
                "oracle_ledger": str(log_root / "oracle_ledger.jsonl"),
                "w7_verify_acceptance": acceptance_ref,
                "registry_dir": registry_dir,
            },
        },
    }


def load_wordings(path: Path | None) -> dict[str, list[WordingCandidate]]:
    """Load wording candidates from a JSONL file (T4 output) or from a directory of
    per-task ``*.wording.json`` files (W15 deterministic v2 wordings)."""
    by_id: dict[str, list[WordingCandidate]] = {}
    if path is None:
        return by_id
    if path.is_dir():
        for file in sorted(path.glob("*.wording.json")):
            cand = WordingCandidate.model_validate_json(file.read_text(encoding="utf-8"))
            by_id.setdefault(cand.task_id, []).append(cand)
    elif path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cand = WordingCandidate.model_validate_json(line)
                by_id.setdefault(cand.task_id, []).append(cand)
    for cands in by_id.values():
        cands.sort(key=lambda c: c.seed)
    return by_id


def run_qualification(
    *,
    blueprints_dir: Path,
    wordings_path: Path | None,
    validator: TaskQualificationValidator,
    limit: int | None = None,
    pattern: str = "gk_blueprint_*.json",
) -> QualificationRun:
    blueprint_files = sorted(Path(blueprints_dir).glob(pattern))
    if limit:
        blueprint_files = blueprint_files[:limit]
    wordings = load_wordings(wordings_path)
    run = QualificationRun(wordings_path=str(wordings_path) if wordings_path else None,
                           n_wording_rows=sum(len(v) for v in wordings.values()))
    rows: list[QualifiedTaskRow] = []
    for path in blueprint_files:
        blueprint = json.loads(path.read_text(encoding="utf-8"))
        bid = str(blueprint["blueprint_id"])
        run.blueprint_paths[bid] = str(path)
        run.blueprint_shas[bid] = sha256_file(path)
        base_spec = bind_blueprint(blueprint["spec"],
                                   seed=int(blueprint["spec"].get("seed", 0) or 0))
        native = validator.native_evidence(base_spec, blueprint)
        candidates: list[WordingCandidate | None] = wordings.get(bid) or [None]
        for wording in candidates:
            if wording is None:
                spec, task_id = base_spec, bid
            else:
                task_id = f"{bid}__w{wording.seed}"
                authorship = (Authorship.deterministic_template
                              if wording.author_model.startswith("deterministic-template")
                              else Authorship.dashscope_paraphrase)
                spec = base_spec.model_copy(update={
                    "task_id": task_id, "user_goal": wording.user_goal_en,
                    "task_authorship": authorship})
            report = validator.validate(spec, blueprint=blueprint, wording=wording,
                                        native=native)
            if report.decision != "REJECTED":
                validator.seen_goal_shas.setdefault(
                    goal_sha(spec_goal_payload(spec, blueprint)), bid)
            status = report.ladder_status
            spec = spec.model_copy(update={"status": TaskStatus(status)})
            rows.append(QualifiedTaskRow(
                spec=spec, wording=wording, base_goal_id=bid,
                canonical_sha=goal_sha(spec_goal_payload(spec, blueprint)), report=report))
    run.rows.extend(rows)
    return run


def dev_split_rows(run: QualificationRun) -> list[dict[str, Any]]:
    """One row per unique canonical goal; paraphrase variants aggregated (T6)."""
    out: list[dict[str, Any]] = []
    for base_id, members in sorted(run.unique_goals.items()):
        best = max(members, key=lambda m: LADDER_RANK[m.report.ladder_status])
        spec = best.spec
        out.append({
            "task_id": base_id,
            "split": spec.split,
            "status": best.report.ladder_status,
            "variant_task_ids": sorted(m.spec.task_id for m in members),
            "canonical_sha": best.canonical_sha,
            "lineage_key": lineage_key(spec),
            "source_group_id": spec.source_group_id,
            "mechanism_group_id": spec.mechanism_group_id,
            "combination_signature": spec.combination_signature,
            "source_rule_unit_ids": sorted(r.source_rule_unit_id for r in spec.source_refs),
            "domain": spec.domain,
            "track": spec.track.value,
            "min_action_count": next(
                (r.evidence.get("min_action_count") for r in best.report.results
                 if r.check == "action_length_calibrated"), None),
        })
    return out


def build_manifest(run: QualificationRun, *, tasks_path: Path, split_path: Path,
                   registry_shards: tuple[str, ...], registry_broken: tuple[str, ...],
                   wording_model_critique: str = "NOT_RUN") -> dict[str, Any]:
    checks_summary: dict[str, dict[str, int]] = {}
    ladder_dist: dict[str, int] = {}
    for row in run.rows:
        ladder_dist[row.report.ladder_status] = ladder_dist.get(row.report.ladder_status, 0) + 1
        for r in row.report.results:
            bucket = checks_summary.setdefault(
                r.check, {"PASS": 0, "FAIL": 0, "NOT_IMPLEMENTED_DEP": 0, "NOT_APPLICABLE": 0})
            bucket[r.status.value] += 1
    goal_status: dict[str, int] = {}
    for split_row in dev_split_rows(run):
        goal_status[split_row["status"]] = goal_status.get(split_row["status"], 0) + 1
    n_qualified_goals = goal_status.get("QUALIFIED", 0)
    multi_action = sum(1 for s in dev_split_rows(run)
                       if isinstance(s["min_action_count"], int) and s["min_action_count"] >= 2)
    wordings_sha = None
    if run.wordings_path and Path(run.wordings_path).is_file():
        wordings_sha = sha256_file(Path(run.wordings_path))
    return {
        "schema": "aob-qualified-manifest/1.0",
        "generated_by": "scripts/qualify_tasks.py (W8 M3)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "blueprint_files": {bid: {"path": run.blueprint_paths[bid],
                                      "sha256": run.blueprint_shas[bid]}
                                for bid in sorted(run.blueprint_paths)},
            "wordings": {"path": run.wordings_path, "rows": run.n_wording_rows,
                         "sha256": wordings_sha},
            "registry": {"shards_loaded": list(registry_shards),
                         "broken_shards": list(registry_broken)},
        },
        "outputs": {
            "tasks_jsonl": {"path": str(tasks_path), "sha256": sha256_file(tasks_path),
                            "rows": len(run.rows)},
            "dev_split": {"path": str(split_path), "sha256": sha256_file(split_path),
                          "rows": len(run.unique_goals)},
        },
        "counts": {"task_rows": len(run.rows), "unique_goals": len(run.unique_goals),
                   "blueprints": len(run.blueprint_paths)},
        "ladder_distribution_rows": ladder_dist,
        "ladder_distribution_unique_goals": goal_status,
        "checks_summary": checks_summary,
        "wording_model_critique": wording_model_critique,
        "g1": {
            "gate": "G1 / T7: >=10 independently verified real user goals in one source",
            "source": "gatekeeper",
            "qualified_unique_goals": n_qualified_goals,
            "qualified_task_rows_incl_paraphrases": ladder_dist.get("QUALIFIED", 0),
            "multi_action_goals_min_action_count_ge2": multi_action,
            "met": n_qualified_goals >= 10,
            "ceiling": ("QUALIFIED scope = ontology_contract native-only/0: T5.7/T5.14 "
                        "NOT_APPLICABLE. Attaching an O profile (P3/W6) re-activates both "
                        "as REQUIRED and would cap every task below QUALIFIED today."),
            "evidence": [str(tasks_path), str(split_path)],
        },
    }
