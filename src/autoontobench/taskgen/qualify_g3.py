"""G3 qualified-manifest builder (W14): the linked-data/1 O-carrying cohort
qualifies WITH T5.7/T5.14 actually run, so the gate block records the
Phi-causality evidence instead of the G1 ceiling note. Split from
`qualify_run.py` to keep both modules under the review ceiling.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoontobench.taskgen.qualify_run import (
    QualificationRun, dev_split_rows, sha256_file,
)


def build_manifest_g3(run: QualificationRun, *, tasks_path: Path, split_path: Path,
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
    phi_rows = [
        r for row in run.rows for r in row.report.results
        if r.check in ("controlled_o_content_sensitivity",
                       "operator_profile_consumed_as_authored")
    ]
    phi_pass = sum(1 for r in phi_rows if r.status.value == "PASS")
    return {
        "schema": "aob-qualified-manifest/1.0",
        "generated_by": "scripts/qualify_tasks.py (W14, Rhea+ChEBI analytical track)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "blueprint_files": {bid: {"path": run.blueprint_paths[bid],
                                      "sha256": run.blueprint_shas[bid]}
                                for bid in sorted(run.blueprint_paths)},
            "wordings": {"path": run.wordings_path, "rows": run.n_wording_rows,
                         "sha256": sha256_file(Path(run.wordings_path)) if run.wordings_path else None},
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
        "g3": {
            "gate": "G3: second source domain with runnable tasks + cross-source "
                    "Organization regime (linked-data/1)",
            "source": "rhea+chebi",
            "ontology_contract": "linked-data/1",
            "qualified_unique_goals": n_qualified_goals,
            "qualified_task_rows_incl_paraphrases": ladder_dist.get("QUALIFIED", 0),
            "query_families": sorted({
                json.loads((Path(bp).read_text()))["goal"]["query"]["type"]
                for bp in run.blueprint_paths.values()}),
            "met": n_qualified_goals >= 8,
            "phi_causality": {
                "checks_run": ["controlled_o_content_sensitivity",
                               "operator_profile_consumed_as_authored"],
                "rows_pass": phi_pass, "rows_total": len(phi_rows),
                "note": "Phi_correct==certified truth, Phi_empty/Phi_swapped change the "
                        "governed answer, non-effect control unchanged; entry-level "
                        "drop-own/drop-unrelated authorship proof per task",
            },
            "not_certified": [
                "no second independent multi-source pair (Rhea+ChEBI is ONE pair)",
                "no agent/runner executions (W6 arms not run on these tasks)",
                "dev split only; hidden_* splits open (T6/preregistration)",
                "T5 human audit sample not done (P6 step; DashScope critique is no substitute)",
                "wording readability critique NOT_RUN",
            ],
            "evidence": ["benchmarks/qualified/tasks_rhea.jsonl",
                          "benchmarks/splits/dev_rhea.jsonl",
                          "data/erule/review5/scientific/logs/rhea_task_build.json",
                          "audit/review5/G3_ACCEPTANCE.md"],
        },
    }
