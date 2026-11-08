"""Offline mapping micro/macro-F1 for the completed 20K v3 runs.

Reads already-written ``predictions.jsonl`` / ``rows.jsonl`` artifacts and the
mounted oracle payloads.  It never calls a model and never reruns a verifier,
and only scores families where both sides expose a real mapping (see
:mod:`benchmark.eval_20k.mapf1` for the family table and decoders).

Micro-F1 aggregates TP/FP/FN over all scored pairs.  Macro-F1 is the
unweighted mean of per-family micro-F1 over families with at least one scored
task (unit: family).  Everything else is excluded with a reason and counted
in coverage, never zero-filled.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .mapf1 import (
    APPLICABLE_FAMILIES,
    DEFINITIONS,
    NOT_APPLICABLE_FAMILIES,
    SPECS,
    TaskMapping,
    add_task,
    decode_mapping,
    finalize,
    new_bucket,
)
from .mapf1_report import render_markdown

SCHEMA = "aob-mapping-f1/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR_RE = re.compile(
    r"^eval_20k_full2-(?P<group>g[0-9]+)-(?P<model>.+?)_(?P<suffix>[A-Za-z0-9]+)$")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _oracle_json(row: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(row.get("oracle_path") or "")
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _dig(payload: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def discover_runs(root: Path) -> list[dict[str, str]]:
    """Pick completed runs from a run directory or a legacy run root."""
    direct_report = root / "report.json"
    if direct_report.is_file() and (root / "rows.jsonl").is_file() and (
            root / "predictions.jsonl").is_file():
        report = json.loads(direct_report.read_text(encoding="utf-8"))
        model = str((report.get("predictions") or {}).get("model") or "unknown")
        return [{"group": "local", "model": model, "path": str(root)}]
    chosen: dict[tuple[str, str], tuple[int, Path]] = {}
    for path in sorted(root.glob("eval_20k_full2-*")):
        match = RUN_DIR_RE.match(path.name)
        if not match or not (path / "rows.jsonl").is_file():
            continue
        if not (path / "predictions.jsonl").is_file():
            continue
        key = (match.group("group"), match.group("model"))
        size = sum(1 for _ in (path / "rows.jsonl").open(encoding="utf-8"))
        if key not in chosen or size > chosen[key][0]:
            chosen[key] = (size, path)
    return [{"group": key[0], "model": key[1], "path": str(path)}
            for key, (_, path) in sorted(chosen.items())]


def _answer_reason(prediction: dict[str, Any] | None) -> tuple[Any, str]:
    if prediction is None:
        return None, "prediction_missing"
    answer = prediction.get("answer")
    if isinstance(answer, dict) and "_error" in answer:
        return None, "prediction_error"
    return answer, ""


def _score_run(run: dict[str, str], per_model: dict[str, Any]) -> None:
    block = per_model.setdefault(run["model"], {
        "runs": [], "overall": new_bucket(), "per_lane": {}, "per_family": {}})
    block["runs"].append({"run": Path(run["path"]).name,
                          "group": run["group"], "path": run["path"]})
    preds = {p["task_id"]: p for p in _load_jsonl(
        Path(run["path"]) / "predictions.jsonl") if "task_id" in p}
    for row in _load_jsonl(Path(run["path"]) / "rows.jsonl"):
        spec = SPECS.get((row.get("lane_id"), row.get("family_id")))
        if spec is None:
            continue
        payload = _oracle_json(row)
        gold_node = _dig(payload, spec.gold_path)
        gold_raw = (gold_node if spec.pair_separator
                    else {spec.answer_keys[0]: gold_node})
        gold, gold_reason = decode_mapping(gold_raw, spec)
        if gold is not None and not gold:
            gold, gold_reason = None, "oracle_mapping_empty"
        answer, answer_reason = _answer_reason(preds.get(row.get("task_id")))
        pred, pred_reason = decode_mapping(answer, spec)
        if gold is None and payload is None:
            gold_reason = "oracle_missing"
        reason = gold_reason or answer_reason or pred_reason
        family_key = "%s/%s" % (spec.lane_id, spec.family_id)
        lane = block["per_lane"].setdefault(spec.lane_id, {
            "overall": new_bucket(), "families": {}})
        for bucket in (block["overall"], lane["overall"],
                       lane["families"].setdefault(family_key, new_bucket()),
                       block["per_family"].setdefault(family_key, new_bucket())):
            add_task(bucket, TaskMapping(gold, pred, reason))


def _finalize_model(block: dict[str, Any]) -> dict[str, Any]:
    families = {name: finalize(bucket)
                for name, bucket in sorted(block["per_family"].items())}
    f1s = [f["f1"] for f in families.values()
           if f["scored_tasks"] and f["f1"] is not None]
    lanes = {}
    for lane_id, lane in sorted(block["per_lane"].items()):
        lane_families = {name: finalize(bucket) for name, bucket
                         in sorted(lane["families"].items())}
        lane_overall = finalize(lane["overall"])
        lane_f1s = [f["f1"] for f in lane_families.values()
                    if f["scored_tasks"] and f["f1"] is not None]
        lane_overall["macro_f1"] = (sum(lane_f1s) / len(lane_f1s)
                                    if lane_f1s else None)
        lane_overall["macro_f1_unit"] = "family"
        lane_overall["scored_families"] = len(lane_f1s)
        lanes[lane_id] = {"overall": lane_overall,
                          "per_family": lane_families}
    overall = finalize(block["overall"])
    overall["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
    overall["macro_f1_unit"] = "family"
    overall["scored_families"] = len(f1s)
    return {"runs": sorted(block["runs"], key=lambda r: r["run"]),
            "overall": overall, "per_lane": lanes, "per_family": families}


def build_report(run_dirs: Iterable[dict[str, str]],
                 run_root: str = "runs") -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    runs = list(run_dirs)
    for run in runs:
        _score_run(run, per_model)
    return {
        "schema": SCHEMA,
        "generated_from": {"run_root": run_root, "run_count": len(runs),
                           "runs": runs, "model_calls": 0,
                           "inference_reruns": 0},
        "definitions": DEFINITIONS,
        "applicable_families": [
            {"lane_id": s.lane_id, "family_id": s.family_id,
             "answer_keys": list(s.answer_keys), "gold_path": list(s.gold_path),
             "pair_separator": s.pair_separator or None, "note": s.note}
            for s in APPLICABLE_FAMILIES],
        "not_applicable_families": [
            {"lane_id": lane, "family_id": family, "reason": reason}
            for lane, family, reason in NOT_APPLICABLE_FAMILIES],
        "models": {model: _finalize_model(block)
                   for model, block in sorted(per_model.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--out-json", default=str(
        REPO_ROOT / "reports" / "extra_metrics" / "mapf1.json"))
    parser.add_argument("--out-md", default=str(
        REPO_ROOT / "reports" / "extra_metrics" / "MAPF1.md"))
    args = parser.parse_args(argv)
    run_root = Path(args.run_root)
    runs = discover_runs(run_root)
    report = build_report(runs, run_root=str(run_root))
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print("wrote %s (%d runs, %d models)" % (
        out_json, len(runs), len(report["models"])))
    print("wrote %s" % out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())