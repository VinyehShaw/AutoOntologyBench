"""Markdown renderer for the offline extra-metric audit report."""
from __future__ import annotations

from collections.abc import Mapping


def _pct(value: object) -> str:
    return "n/a" if value is None else "%.2f%%" % (100.0 * float(value))


def _num(value: object) -> str:
    return "n/a" if value is None else "%.4f" % float(value)


def _status_table(report: Mapping[str, object]) -> list[str]:
    metrics = report["metrics"]
    rows = [
        ("paper FRR", metrics["paper_frr"]["status"], "no consumer decision trace"),
        ("paper MRR", metrics["paper_mrr"]["status"], "no consumer decision trace"),
        ("answer-decision proxy (L02)", metrics["answer_decision_proxy"]["status"],
         "task-answer ALLOW/DENY cells, not consumer steps"),
        ("paper OOD", metrics["paper_ood"]["status"], "no dev/iid/comp/transfer split"),
        ("family-held-out proxy", metrics["family_heldout_proxy"]["status"],
         "leave-one-family-out over answer-only success"),
    ]
    lines = ["| metric | status | scope |", "|---|---|---|"]
    lines.extend("| %s | **%s** | %s |" % row for row in rows)
    return lines


def _decision_section(report: Mapping[str, object]) -> list[str]:
    proxy = report["metrics"]["answer_decision_proxy"]
    lines = [
        "## Decision-level metrics",
        "",
        "Paper FRR and MRR are **BLOCKED**: the 20K submissions are answer-only "
        "(no per-step consumer legality labels, no consumer decision trace). The table "
        "below is a clearly labelled **PROXY** over the only per-decision "
        "labels in the 20K release, the L02 `exact_decision_set` family "
        "(50 tasks, 480 `(object, policy)` cells). `ALLOW` is treated as the "
        "ground-truth-legal decision and `DENY` as the ground-truth-illegal "
        "decision. This is not the paper metric.",
        "",
        "| model | tasks labeled | truth cells | known legal | known illegal | FRR known | MRR known | unknown legal | unknown illegal | known coverage | FRR unknown=blocked | MRR unknown=passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, block in sorted(proxy["models"].items()):
        m = block["l02_exact_decision_set"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (model, m.get("tasks_labeled", 0), m.get("truth_cells", 0),
               m.get("known_legal_cells", 0), m.get("known_illegal_cells", 0),
               _pct(m.get("frr_known")), _pct(m.get("mrr_known")),
               m.get("unknown_legal_cells", 0), m.get("unknown_illegal_cells", 0),
               _pct(m.get("known_coverage")),
               _pct(m.get("frr_unknown_as_blocked")),
               _pct(m.get("mrr_unknown_as_passed"))))
    lines.extend([
        "",
        "L01 `exact_decision_set` is excluded because its oracle reference keys "
        "are policy-unit ids that the public task does not expose, so prediction "
        "cells cannot be aligned:",
        "",
        "| model | L01 tasks | truth cells | predicted cells | matched key cells | exact key tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model, block in sorted(proxy["models"].items()):
        m = block["l01_exact_decision_set_key_alignment"]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            model, m.get("tasks_labeled", 0), m.get("truth_cells", 0),
            m.get("predicted_cells", 0), m.get("matched_key_cells", 0),
            m.get("exact_key_tasks", 0)))
    return lines


def _ood_section(report: Mapping[str, object]) -> list[str]:
    split = report["audits"]["split"]
    proxy = report["metrics"]["family_heldout_proxy"]
    lines = [
        "## OOD",
        "",
        "Paper OOD is **BLOCKED**. Public 20K split counts: `%s`. Internal "
        "build split counts: `%s`. `release_20k/README.md` states all 20,000 "
        "tasks are the official TEST split and there is no train/dev split. "
        "The paper `dev/iid/comp/transfer` files "
        "exist only for `world_0001_v4` and have zero task-id overlap with the "
        "20K set:"
        % (split["public_20k_split_counts"], split["internal_build_split_counts"]),
        "",
        "| paper split file | rows | overlap with 20K task ids |",
        "|---|---:|---:|",
    ]
    for name, info in sorted(split["paper_world_split_files"].items()):
        lines.append("| %s | %d | %d |" % (
            info["path"], info["rows"], info["overlap_with_20k_task_ids"]))
    lines.extend([
        "",
        "Proxy: leave-one-family-out, stratified within lane, using existing "
        "score rows only. The held-out unit is defined solely by `lane_id` + "
        "`family_id` (or `lineage_group`); no outcome field enters assignment. "
        "`drop = reference_rate - heldout_rate`; positive means the held-out "
        "family did worse. Models were not trained on this set, so this is "
        "family-difficulty heterogeneity, not paper OOD.",
        "",
        "| model | rows | scored | TSR | family units evaluated | task-weighted mean drop | unweighted mean drop | macro lane mean drop | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, block in sorted(proxy["models"].items()):
        fam = block["family_leave_one_out"]
        lines.append(
            "| %s | %d | %d | %s | %d | %s | %s | %s | %s | %s |"
            % (model, block["unique_tasks"], block["scored"], _pct(block["tsr"]),
               fam["units_evaluated"], _num(fam["task_weighted_mean_drop"]),
               _num(fam["unweighted_mean_drop"]),
               _num(fam["macro_lane_mean_drop"]), _num(fam["min_drop"]),
               _num(fam["max_drop"])))
    lines.extend([
        "",
        "Lineage-group variant (release contamination unit) task-weighted mean "
        "drop:",
        "",
        "| model | units evaluated | task-weighted mean drop | unweighted mean drop | macro lane mean drop |",
        "|---|---:|---:|---:|---:|",
    ])
    for model, block in sorted(proxy["models"].items()):
        lg = block["lineage_group_leave_one_out"]
        lines.append("| %s | %d | %s | %s | %s |" % (
            model, lg["units_evaluated"], _num(lg["task_weighted_mean_drop"]),
            _num(lg["unweighted_mean_drop"]), _num(lg["macro_lane_mean_drop"])))
    lines.extend([
        "",
        "Run metadata scan: %d `report.json` files, %d with a `split` key."
        % (split["report_files_scanned"], split["report_files_with_split_key"]),
    ])
    return lines


def _evidence_section(report: Mapping[str, object]) -> list[str]:
    decision = report["audits"]["decision"]
    schema = decision["task_schema"]
    pred = decision["predictions"]
    rows = decision["rows"]
    oracle = decision["oracle_payloads"]
    flags = oracle["flags"]
    harness = decision["harness_contract"]
    lines = [
        "## Exact data evidence",
        "",
        "Harness contract: `%s` accepts optional `actions`; `%s` writes only "
        "`answer`; `%s` marks rows `unsafe_unchecked` when actions are "
        "present. %s"
        % (harness["submission_schema"], harness["prediction_generator"],
           harness["scoring"], harness["note"]),
        "",
        "Public task records scanned: **%d**. Top-level keys: `%s`. No "
        "top-level `legal`/`illegal`/`consumer`/`ground_truth` key exists. "
        "Decision-like key paths found: `%s` (DMN decision ids and ordinary "
        "API fields such as `legal_name`/`rejected_at`, not per-step "
        "legality labels)."
        % (schema["tasks_scanned"], schema["top_level_keys"],
           schema["decision_like_key_paths"]),
        "",
        "Prediction records scanned: **%d files / %d data rows**. Row keys: "
        "`%s`. Rows with non-empty `actions`: **%d**. Rows with row-level "
        "`decisions`/`trace`: **%d**. Rows with a `split` field: **%d**. Rows "
        "whose answer payload carries `decisions`: **%d**."
        % (pred["files"], pred["data_rows"], sorted(pred["row_keys"]),
           pred["rows_with_actions"], pred["rows_with_row_level_decisions"],
           pred["rows_with_split_field"], pred["rows_with_answer_decisions"]),
        "",
        "Score rows scanned: **%d files / %d rows**. Statuses: `%s`. Rows with "
        "`unsafe_unchecked`: **%d**. `verdict_detail` keys are task-level "
        "verifier detail (`%s`), not a per-step consumer legality/decision trace."
        % (rows["files"], rows["rows"], rows["statuses"],
           rows["rows_with_unsafe_unchecked"], rows["verdict_detail_keys"]),
        "",
        "Oracle payloads scanned: **%d** (errors: %d). Flags: `%s`. No "
        "top-level `legal`/`illegal`/`consumer`/`ground_truth` key exists; "
        "`decision_id` is a DMN decision identifier, and `suite` / "
        "`reference_terminal` / `step_modes` are task-answer or trace material, "
        "not consumer decisions."
        % (oracle["files_scanned"], oracle["errors"], flags),
        "",
        "Required for paper FRR/MRR (absent here): per-step ground-truth consumer "
        "legality labels and per-step consumer accept/block decisions or an "
        "executor trace. `evaluation/dev_feedback.py` shows the expected "
        "record shape (`actions`, `outcomes`, `expected_ok`, per-step "
        "`predictions`).",
    ]
    return lines


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# 20K v3 offline extra metrics: FRR / MRR / OOD feasibility",
        "",
        "Generated: `%s`" % report["generated_at"],
        "",
        "Policy: no model API call, no inference rerun, no network, no edits to "
        "existing tasks, predictions, oracles, or reports. Only existing "
        "artifacts were read.",
        "",
        "## Status",
        "",
    ]
    lines.extend(_status_table(report))
    lines.extend([""])
    lines.extend(_decision_section(report))
    lines.extend([""])
    lines.extend(_ood_section(report))
    lines.extend([""])
    lines.extend(_evidence_section(report))
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python -m benchmark.eval_20k.extra_metrics",
        ".venv/bin/python -m pytest benchmark/eval_20k/tests/test_metrics_decision.py "
        "benchmark/eval_20k/tests/test_metrics_ood.py -q",
        "```",
        "",
        "## Verdict",
        "",
        "- Paper FRR: **BLOCKED**; no value is reported.",
        "- Paper MRR: **BLOCKED**; no value is reported.",
        "- Paper OOD: **BLOCKED**; no value is reported.",
        "- The L02 answer-decision proxy and the family-held-out proxy are "
        "**PROXY** values. They must never be cited as paper FRR/MRR/OOD.",
        "",
    ])
    return "\n".join(lines)