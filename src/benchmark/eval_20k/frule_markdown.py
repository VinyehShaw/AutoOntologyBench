"""Markdown/report writer for :mod:`benchmark.eval_20k.metrics_frule`."""
from __future__ import annotations

from typing import Any

from .frule import lane_status


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else ("%." + str(digits) + "f") % value


def _counts(row: dict[str, Any]) -> str:
    return "%d | %d | %d | %d" % (
        row["tp"], row["fp"], row["tn"], row["fn"])


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# F_rule (balanced rule fidelity) - 20K v3", "",
        "Offline aggregation of the completed `runs/eval_20k_full2-*` "
        "artifacts. No model was called and inference was not rerun.", "",
        "## Definition (fixed before scoring)", "",
        "* Rule unit: `(lane, family)`; macro = unweighted mean of per-family "
        "balanced F_rule.",
        "* Positive witness: the oracle reference submission.",
        "* Negative witness: either the oracle's stored `mutation_submission` "
        "(L15) or the lane's own build-time negative-submission helper "
        "re-run on the same AST, admitted only when the oracle `self_check` "
        "records positive=pass and negative=rejected (L05/L06/L07/L08).",
        "* One prediction is compared against both witnesses: matching the "
        "positive answer gives TP (positive witness) and TN (negative "
        "witness); matching the negative answer gives FN and FP; any other "
        "answer is UNKNOWN. With one prediction per task this yields TP=TN "
        "and FN=FP, so TPR=TNR: F_rule is the accuracy of choosing the "
        "oracle's positive witness over its rejected negative witness among "
        "known selections.",
        "* `balanced F_rule = (TPR + TNR) / 2`; UNKNOWN is excluded from "
        "TP/FN/TN/FP and from TPR/TNR, and is reflected in coverage. A "
        "missing witness or zero known classifications yields N/A, never 0.",
        "",
        "Coverage is not a quality adjustment: it is the fraction of "
        "witness-eligible tasks whose answer matched one of the two "
        "witnesses. Read F_rule together with coverage.", "",
        "## Lane eligibility", "",
        "| Lane | Status | Unique 20K tasks | Reason |",
        "|---|---|---:|---|",
    ]
    for lane, item in report["lanes"].items():
        lines.append("| %s | %s | %d | %s |" % (
            lane, item["status"], item["tasks_unique"],
            item["reason"].replace("|", "\\|")))
    lines += ["", "## Per model", "",
              "| Model | Runs | Tasks | Eligible | Known | Coverage | "
              "TP | FP | TN | FN | TPR | TNR | Balanced F_rule | Macro F_rule |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
              "---:|---:|---:|"]
    for model, block in report["models"].items():
        row = block["overall"]
        lines.append("| %s | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
            model, len(block["runs"]), row["tasks"], row["eligible"],
            row["known"], _fmt(row["coverage"]), _counts(row),
            _fmt(row["tpr"]), _fmt(row["tnr"]),
            _fmt(row["balanced_accuracy"]),
            _fmt(row["macro_balanced_frule"])))
    lines += ["", "## Per lane", "",
              "| Model | Lane | Status | Tasks | Eligible | Known | "
              "Coverage | TP | FP | TN | FN | TPR | TNR | Balanced | Macro |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
              "---:|---:|---:|---:|"]
    for model, block in report["models"].items():
        for lane, lane_block in block["per_lane"].items():
            row = lane_block["overall"]
            status, _reason = lane_status(lane)
            lines.append("| %s | %s | %s | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                model, lane, status, row["tasks"], row["eligible"],
                row["known"], _fmt(row["coverage"]), _counts(row),
                _fmt(row["tpr"]), _fmt(row["tnr"]),
                _fmt(row["balanced_accuracy"]),
                _fmt(row["macro_balanced_frule"])))
    lines += ["", "## Per lane / family", "",
              "| Model | Lane | Family | Tasks | Eligible | Known | "
              "Coverage | TP | FP | TN | FN | TPR | TNR | Balanced | Blocked |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
              "---:|---:|---:|---|"]
    for model, block in report["models"].items():
        for lane, lane_block in block["per_lane"].items():
            for family, row in lane_block["per_family"].items():
                blocked = ", ".join("%s=%d" % item
                                    for item in row["blocked"].items()) or "-"
                lines.append("| %s | %s | `%s` | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                    model, lane, family.split("/", 1)[1], row["tasks"],
                    row["eligible"], row["known"], _fmt(row["coverage"]),
                    _counts(row), _fmt(row["tpr"]), _fmt(row["tnr"]),
                    _fmt(row["balanced_accuracy"]), blocked))
    lines += ["", "## Blocked / N/A lanes", "",
              "| Lane | Status | Reason |", "|---|---|---|"]
    for lane, item in report["lanes"].items():
        if item["status"] != "ELIGIBLE":
            lines.append("| %s | %s | %s |" % (
                lane, item["status"], item["reason"].replace("|", "\\|")))
    lines += ["", "A blocked lane/family is absent from TPR/TNR; it is not "
              "scored as zero.", "", "## Sources", ""]
    lines += ["- `%s` (%s, %s)" % (run["path"], run["group"], run["model"])
              for run in report["generated_from"]["runs"]]
    lines.append("")
    return "\n".join(lines)


