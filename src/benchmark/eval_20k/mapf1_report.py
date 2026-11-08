"""Markdown rendering for the offline 20K MapF1 report."""
from __future__ import annotations

from typing import Any


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else ("%." + str(digits) + "f") % value


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MapF1 (mapping micro/macro-F1) - 20K v3", "",
        "Offline aggregation of the completed `runs/eval_20k_full2-*` "
        "artifacts. No model was called and inference was not rerun.", "",
        "A mapping is a set of `(key, value)` pairs. A task is scored only "
        "when both the prediction and the oracle reference decode to a real "
        "mapping; everything else is excluded with a reason and reflected in "
        "coverage, never zero-filled. Macro-F1 is the unweighted mean of "
        "per-family micro-F1 (unit: family).", "",
        "Coverage is not a quality adjustment: excluded rows (API errors, "
        "malformed answers, or flat sets that are not mappings) are absent "
        "from TP/FP/FN and may correlate with model failure, so read F1 "
        "together with coverage.", "",
        "## Applicable families", "",
        "| Lane | Family | Encoding | Gold field |", "|---|---|---|---|",
    ]
    for spec in report["applicable_families"]:
        encoding = ("pairs `KEY%sVALUE`" % spec["pair_separator"]
                    if spec["pair_separator"]
                    else "object `%s`" % "/".join(spec["answer_keys"]))
        lines.append("| %s | `%s` | %s | `%s` |" % (
            spec["lane_id"], spec["family_id"], _escape(encoding),
            ".".join(spec["gold_path"])))
    lines += ["", "## Per model", "",
              "| Model | Runs | Tasks | Gold mappings | Scored | Coverage | "
              "TP | FP | FN | Micro P | Micro R | Micro F1 | Macro F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, block in report["models"].items():
        overall = block["overall"]
        lines.append(
            "| %s | %d | %d | %d | %d | %s | %d | %d | %d | %s | %s | %s | %s |" % (
                model, len(block["runs"]), overall["family_tasks"],
                overall["oracle_mapping_tasks"], overall["scored_tasks"],
                _fmt(overall["coverage"]), overall["tp"], overall["fp"],
                overall["fn"], _fmt(overall["precision"]),
                _fmt(overall["recall"]), _fmt(overall["f1"]),
                _fmt(overall["macro_f1"])))
    lines += ["", "## Per lane", "",
              "| Model | Lane | Tasks | Gold | Scored | Coverage | TP | FP | "
              "FN | Micro F1 | Macro F1 (family) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, block in report["models"].items():
        for lane_id, lane in block["per_lane"].items():
            row = lane["overall"]
            lines.append(
                "| %s | %s | %d | %d | %d | %s | %d | %d | %d | %s | %s |" % (
                    model, lane_id, row["family_tasks"],
                    row["oracle_mapping_tasks"], row["scored_tasks"],
                    _fmt(row["coverage"]), row["tp"], row["fp"], row["fn"],
                    _fmt(row["f1"]), _fmt(row["macro_f1"])))
    lines += ["", "## Per lane / family", "",
              "| Model | Lane | Family | Tasks | Gold | Scored | Coverage | "
              "TP | FP | FN | Micro F1 | Excluded |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model, block in report["models"].items():
        for lane_id, lane in block["per_lane"].items():
            for family, row in lane["per_family"].items():
                excluded = ", ".join("%s=%d" % item
                                     for item in row["excluded"].items()) or "-"
                lines.append(
                    "| %s | %s | `%s` | %d | %d | %d | %s | %d | %d | %d | %s | %s |" % (
                        model, lane_id, family.split("/", 1)[1],
                        row["family_tasks"], row["oracle_mapping_tasks"],
                        row["scored_tasks"], _fmt(row["coverage"]), row["tp"],
                        row["fp"], row["fn"], _fmt(row["f1"]), excluded))
    lines += ["", "## Not applicable (investigated, no both-sides mapping)", "",
              "| Lane | Family | Reason |", "|---|---|---|"]
    for item in report["not_applicable_families"]:
        lines.append("| %s | `%s` | %s |" % (
            item["lane_id"], item["family_id"], _escape(item["reason"])))
    lines += ["", "All other 20K families do not expose a key/value mapping on "
              "both sides and are excluded; they are not scored as zero.", "",
              "## Sources", ""]
    lines += ["- `%s` (%s, %s)" % (run["path"], run["group"], run["model"])
              for run in report["generated_from"]["runs"]]
    lines.append("")
    return "\n".join(lines)