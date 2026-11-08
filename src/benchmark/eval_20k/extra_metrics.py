"""Offline extra-metric audit and report builder for the 20K v3 benchmark.

No model endpoint is contacted, no inference is rerun, and no existing
task/prediction/report/oracle file is written.  The script reads the public
release, the private build manifest, the mounted oracle payloads and existing
``runs/eval_20k_full2-*`` score rows, then writes only::

    reports/extra_metrics/decision_ood.json
    reports/extra_metrics/decision_ood.md

Status vocabulary is fixed: ``MEASURED`` (faithful to the paper definition),
``PROXY`` (different unit/estimand, explicitly labelled) and ``BLOCKED`` (the
data do not exist; no value is invented).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .extra_metrics_audit import (
    oracle_audit,
    prediction_audit,
    rows_audit,
    split_audit,
    task_schema_audit,
)
from .extra_metrics_data import (
    BUILD_MANIFEST,
    OUT_DIR,
    OUT_JSON,
    OUT_MD,
    PAPER_SPLITS,
    REPO_ROOT,
    TASKS,
    TASK_MANIFEST,
    manifest_rows,
)
from .extra_metrics_markdown import render_markdown
from .extra_metrics_proxy import decision_proxy, ood_proxy

REQUIRED_DECISION_FIELDS = [
    "per-step ground-truth consumer legality label (legal vs illegal)",
    "per-step consumer decision (accept/block) or executor trace",
]


def build_report() -> dict[str, object]:
    tasks = manifest_rows(TASK_MANIFEST)
    return {
        "schema": "aob-extra-metrics/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": {
            "model_api_calls": 0,
            "inference_reruns": 0,
            "network": False,
            "existing_artifacts_only": True,
            "wrote_existing_reports_predictions_oracles_tasks": False,
        },
        "inputs": {
            "public_tasks": str(TASKS.relative_to(REPO_ROOT)),
            "public_task_manifest": str(TASK_MANIFEST.relative_to(REPO_ROOT)),
            "build_task_manifest": str(BUILD_MANIFEST.relative_to(REPO_ROOT)),
            "paper_split_dir": str(PAPER_SPLITS.relative_to(REPO_ROOT)),
            "runs_glob": "runs/eval_20k_full2-g*/",
        },
        "metrics": {
            "paper_frr": {
                "status": "BLOCKED",
                "definition": (
                    "ground-truth-legal consumer decision steps blocked / "
                    "ground-truth-legal consumer decision steps"),
                "reason": (
                    "The 20K harness records answer-only submissions; no "
                    "per-step consumer legal/illegal labels and no consumer "
                    "decision trace exist."),
                "required_data": REQUIRED_DECISION_FIELDS,
            },
            "paper_mrr": {
                "status": "BLOCKED",
                "definition": (
                    "ground-truth-illegal consumer decision steps waved through / "
                    "ground-truth-illegal consumer decision steps"),
                "reason": (
                    "Same missing per-step consumer legality labels and consumer "
                    "decision trace; task success is not a substitute."),
                "required_data": REQUIRED_DECISION_FIELDS,
            },
            "answer_decision_proxy": decision_proxy(
                manifest_rows(BUILD_MANIFEST)),
            "paper_ood": {
                "status": "BLOCKED",
                "definition": (
                    "mean step-accuracy drop of comp and transfer splits vs iid"),
                "reason": (
                    "The 20K release has split=test for all 20,000 rows and no "
                    "dev/iid/comp/transfer labels. The internal build split is "
                    "train/dev/test only; the paper dev/iid/comp/transfer files "
                    "belong to a different (world_0001_v4) task set with zero "
                    "task-id overlap."),
                "required_data": [
                    "official per-task dev/iid/comp/transfer labels for the 20K set",
                    "step-level success or decision outcomes for each split",
                ],
            },
            "family_heldout_proxy": ood_proxy(tasks),
        },
        "audits": {
            "decision": {
                "harness_contract": {
                    "submission_schema": "benchmark/eval_20k/submission.py",
                    "prediction_generator": "benchmark/eval_20k/model_runner.py",
                    "scoring": "benchmark/eval_20k/scoring.py",
                    "note": "No run exercised the actions path.",
                },
                "task_schema": task_schema_audit(),
                "predictions": prediction_audit(),
                "rows": rows_audit(),
                "oracle_payloads": oracle_audit(),
            },
            "split": split_audit(tasks),
        },
    }


def main() -> int:
    report = build_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())