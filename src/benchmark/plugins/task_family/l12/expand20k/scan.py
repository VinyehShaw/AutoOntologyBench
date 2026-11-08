"""L12/20K source scan: establish new-content facts from the pinned snapshot.

Deterministically answers, from the pinned pm4py 2.7.23.11 tree only:
1. which candidate NEW logs parse, and their trace/event/alphabet stats;
2. which candidate NEW log+net pairs are GOLD (all traces fit strict TBR);
3. which candidate NEW noisy pairs are mixed-fit (some but not all unfit);
4. trace-overlap facts vs the legacy logs (roadtraffic100/10000 vs 50,
   fairness logs pairwise) so the build can exclude repeated traces;
5. which log units the legacy 8K build actually used (read-only scan of
   ``bank/staging/l12``) so "new" means genuinely unused;
6. a probe of the pm4py footprints discovery/conformance API used by the
   new footprints families.

Output: JSON report (default ``data/erule/review5/build8000/expansion/lanes/L12/scan_report.json``).
Exit 0 on success. Read-only everywhere; writes only the report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPANSION_DIR = REPO_ROOT / "data" / "erule" / "review5" / "build8000" / "expansion" / "lanes" / "L12"
LEGACY_STAGING = REPO_ROOT / "bank" / "staging" / "l12"

#: candidate NEW logs for the audit families (never used by legacy families).
NEW_LOG_CANDIDATES: dict[str, str] = {
    "bpic2012": "tests/input_data/bpic2012.xes.gz",
    "helpdesk": "tests/input_data/helpdesk.xes.gz",
    "hiring": "tests/input_data/fairness/hiring_log_high.xes.gz",
    "hospital_fairness": "tests/input_data/fairness/hospital_log_high.xes.gz",
    "lending": "tests/input_data/fairness/lending_log_high.xes.gz",
    "renting": "tests/input_data/fairness/renting_log_high.xes.gz",
    "cyber_incident_response": "tests/input_data/inh_res_nets/logs/cyber_incident_response.xes",
    "emergency_evacuation": "tests/input_data/inh_res_nets/logs/emergency_evacuation.xes",
    "loan_underwriting": "tests/input_data/inh_res_nets/logs/loan_underwriting.xes",
    "manufacturing_batch": "tests/input_data/inh_res_nets/logs/manufacturing_batch.xes",
    "interval_event_log": "tests/input_data/interval_event_log.xes",
    "partial_order_example": "tests/input_data/partial order example.xes",
    "partial_order_example_3": "tests/input_data/partial order example 3.xes",
    "roadtraffic100": "tests/input_data/roadtraffic100traces.xes",
    "roadtraffic10000": "tests/input_data/roadtraffic_10000.xes.gz",
    "xes_20": "tests/input_data/xes_20.xes",
    # registered in legacy engine.LOGS but never used by any legacy family
    # (verified below); still new *task content*:
    "hospital_discharge": "tests/input_data/inh_res_nets/logs/hospital_discharge.xes",
    "insurance_claim": "tests/input_data/inh_res_nets/logs/insurance_claim.xes",
    "order_fulfillment": "tests/input_data/inh_res_nets/logs/order_fulfillment.xes",
}

#: candidate NEW gold pairs (log, net) for the repair families.
GOLD_PAIR_CANDIDATES: dict[str, dict[str, str]] = {
    "cyber_incident_response": {
        "log": "tests/input_data/inh_res_nets/logs/cyber_incident_response.xes",
        "net": "tests/input_data/inh_res_nets/cyber_incident_response.pnml",
    },
    "emergency_evacuation": {
        "log": "tests/input_data/inh_res_nets/logs/emergency_evacuation.xes",
        "net": "tests/input_data/inh_res_nets/emergency_evacuation.pnml",
    },
    "loan_underwriting": {
        "log": "tests/input_data/inh_res_nets/logs/loan_underwriting.xes",
        "net": "tests/input_data/inh_res_nets/loan_underwriting.pnml",
    },
    "manufacturing_batch": {
        "log": "tests/input_data/inh_res_nets/logs/manufacturing_batch.xes",
        "net": "tests/input_data/inh_res_nets/manufacturing_batch.pnml",
    },
    "hospital_discharge": {
        "log": "tests/input_data/inh_res_nets/logs/hospital_discharge.xes",
        "net": "tests/input_data/inh_res_nets/hospital_discharge.pnml",
    },
    "insurance_claim": {
        "log": "tests/input_data/inh_res_nets/logs/insurance_claim.xes",
        "net": "tests/input_data/inh_res_nets/insurance_claim.pnml",
    },
    "order_fulfillment": {
        "log": "tests/input_data/inh_res_nets/logs/order_fulfillment.xes",
        "net": "tests/input_data/inh_res_nets/order_fulfillment.pnml",
    },
    "roadtraffic100": {
        "log": "tests/input_data/roadtraffic100traces.xes",
        "net": "tests/input_data/roadtraffic.pnml",
    },
    "roadtraffic10000": {
        "log": "tests/input_data/roadtraffic_10000.xes.gz",
        "net": "tests/input_data/roadtraffic.pnml",
    },
    "stochastic_running_example": {
        "log": "tests/input_data/running-example.xes",
        "net": "tests/input_data/stochastic_running_example.pnml",
    },
}

#: candidate NEW noisy pairs (mixed strict-TBR fitness expected).
NOISY_PAIR_CANDIDATES: dict[str, dict[str, str]] = {
    "a12_n20": {
        "log": "tests/input_data/synthetic_logs/a12/a12f0n20.xes.gz",
        "net": "tests/input_data/synthetic_logs/a12/a12.pnml",
    },
    "a12_n50": {
        "log": "tests/input_data/synthetic_logs/a12/a12f0n50.xes.gz",
        "net": "tests/input_data/synthetic_logs/a12/a12.pnml",
    },
    "a22_n20": {
        "log": "tests/input_data/synthetic_logs/a22/a22f0n20.xes.gz",
        "net": "tests/input_data/synthetic_logs/a22/a22.pnml",
    },
    "a22_n50": {
        "log": "tests/input_data/synthetic_logs/a22/a22f0n50.xes.gz",
        "net": "tests/input_data/synthetic_logs/a22/a22.pnml",
    },
    "a32_n20": {
        "log": "tests/input_data/synthetic_logs/a32/a32f0n20.xes.gz",
        "net": "tests/input_data/synthetic_logs/a32/a32.pnml",
    },
    "a32_n50": {
        "log": "tests/input_data/synthetic_logs/a32/a32f0n50.xes.gz",
        "net": "tests/input_data/synthetic_logs/a32/a32.pnml",
    },
    "a42_n00": {
        "log": "tests/input_data/synthetic_logs/a42/a42f0n00.xes.gz",
        "net": "tests/input_data/synthetic_logs/a42/a42.pnml",
    },
    "a42_n05": {
        "log": "tests/input_data/synthetic_logs/a42/a42f0n05.xes.gz",
        "net": "tests/input_data/synthetic_logs/a42/a42.pnml",
    },
    "a42_n10": {
        "log": "tests/input_data/synthetic_logs/a42/a42f0n10.xes.gz",
        "net": "tests/input_data/synthetic_logs/a42/a42.pnml",
    },
    "a42_n20": {
        "log": "tests/input_data/synthetic_logs/a42/a42f0n20.xes.gz",
        "net": "tests/input_data/synthetic_logs/a42/a42.pnml",
    },
    "a42_n50": {
        "log": "tests/input_data/synthetic_logs/a42/a42f0n50.xes.gz",
        "net": "tests/input_data/synthetic_logs/a42/a42.pnml",
    },
}

LEGACY_LOGS = {
    "running_example": "tests/input_data/running-example.xes",
    "roadtraffic50": "tests/input_data/roadtraffic50traces.xes",
    "reviewing": "tests/input_data/reviewing.xes",
    "receipt": "tests/input_data/receipt.xes",
    "hospital_discharge": "tests/input_data/inh_res_nets/logs/hospital_discharge.xes",
    "insurance_claim": "tests/input_data/inh_res_nets/logs/insurance_claim.xes",
    "order_fulfillment": "tests/input_data/inh_res_nets/logs/order_fulfillment.xes",
    "a12": "tests/input_data/synthetic_logs/a12/a12f0n00.xes.gz",
    "a22": "tests/input_data/synthetic_logs/a22/a22f0n00.xes.gz",
    "a32": "tests/input_data/synthetic_logs/a32/a32f0n00.xes.gz",
}


def _log_stats(log_rel: str) -> dict[str, Any]:
    traces = RT.traces(log_rel)
    events = sum(len(t) for t in traces)
    alphabet = sorted({a for t in traces for a in t})
    return {
        "log_rel": log_rel,
        "traces": len(traces),
        "events": events,
        "alphabet": len(alphabet),
        "avg_trace_len": round(events / max(len(traces), 1), 2),
        "sha256": engine.sha256_file(engine.PINNED_ROOT / log_rel),
        "complexity_full": engine.complexity_of(events),
    }


def _replay_summary(log_rel: str, net_rel: str) -> dict[str, Any]:
    traces = RT.traces(log_rel)
    verdicts = RT.replay(traces, net_rel)
    bad = [i for i, v in enumerate(verdicts) if not v["trace_is_fit"]]
    return {
        "traces": len(traces),
        "fit": len(traces) - len(bad),
        "unfit": len(bad),
        "all_fit": not bad,
        "first_unfit_indices": bad[:10],
    }


def _legacy_log_usage() -> dict[str, Any]:
    """Read-only scan of legacy staging: which log/net rels appear in tasks."""
    used_logs: set[str] = set()
    used_nets: set[str] = set()
    task_count = 0
    qdir = LEGACY_STAGING / "qualified"
    for path in sorted(qdir.glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_count += 1
                mat = rec.get("public", {}).get("task_material", {})
                if "log" in mat:
                    used_logs.add(mat["log"].get("path", "?").replace(
                        str(engine.PINNED_ROOT) + "/", ""))
                if "net" in mat:
                    used_nets.add(mat["net"].get("path", "?").replace(
                        str(engine.PINNED_ROOT) + "/", ""))
    return {
        "legacy_qualified_tasks": task_count,
        "used_log_rels": sorted(used_logs),
        "used_net_rels": sorted(used_nets),
    }


def _trace_overlap(a_rel: str, b_rel: str) -> dict[str, Any]:
    a = [tuple(t) for t in RT.traces(a_rel)]
    b = [tuple(t) for t in RT.traces(b_rel)]
    set_a, set_b = set(a), set(b)
    return {
        "a": a_rel,
        "b": b_rel,
        "a_traces": len(a),
        "b_traces": len(b),
        "shared_trace_variants": len(set_a & set_b),
        "a_prefix_equals_b_prefix": a[: len(b)] == b if len(a) >= len(b) else b[: len(a)] == a,
        "identical_multisets": sorted(a) == sorted(b),
    }


def _footprints_probe(log_rel: str, net_rel: str) -> dict[str, Any]:
    """Probe the pm4py footprints discovery + conformance API surface."""
    try:
        fp_alg = RT._mods["fp_alg"]
        fp_conf = RT._mods["fp_conf"]
        net, im, fm = RT.net(net_rel)
        model_fp = fp_alg.apply(net, im, fm, variant=fp_alg.Variants.PETRI_NET)
        traces = RT.traces(log_rel)
        log_fp = fp_alg.apply(RT.event_log(traces), variant=fp_alg.Variants.ENTIRE_EVENT_LOG)
        conf = fp_conf.apply(log_fp, model_fp)
        # conf over trace-by-trace footprints for per-trace verdicts
        trace_fp = fp_alg.apply(RT.event_log(traces), variant=fp_alg.Variants.TRACE_BY_TRACE)
        per_trace = fp_conf.apply(trace_fp, model_fp)
        return {
            "ok": True,
            "model_footprints_keys": sorted(model_fp.keys()),
            "log_footprints_keys": sorted(log_fp.keys()),
            "entire_log_conf_keys": sorted(conf.keys()) if isinstance(conf, dict) else str(type(conf)),
            "per_trace_count": len(per_trace),
            "per_trace_sample": [
                {
                    "is_fit": bool(r.get("is_fit")),
                    "violations_keys": sorted(r.get("violations", {}).keys())
                    if isinstance(r.get("violations"), dict) else str(type(r.get("violations"))),
                }
                for r in per_trace[:3]
            ],
            "unfit_traces": sum(1 for r in per_trace if not r.get("is_fit")),
        }
    except Exception as exc:  # honest probe failure
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}


def run_scan(out_path: Path) -> dict[str, Any]:
    t0 = time.time()
    report: dict[str, Any] = {
        "scan": "l12-expand20k-source-scan/1",
        "pinned_root": str(engine.PINNED_ROOT),
        "pm4py_version": engine.PM4PY_VERSION,
        "pm4py_commit": engine.PM4PY_COMMIT,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logs: dict[str, Any] = {}
    for key, rel in NEW_LOG_CANDIDATES.items():
        try:
            logs[key] = _log_stats(rel)
        except Exception as exc:
            logs[key] = {"log_rel": rel, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    report["new_logs"] = logs

    gold: dict[str, Any] = {}
    for key, pair in GOLD_PAIR_CANDIDATES.items():
        try:
            gold[key] = {
                "log": pair["log"], "net": pair["net"],
                "net_sha256": engine.sha256_file(engine.PINNED_ROOT / pair["net"]),
                **_replay_summary(pair["log"], pair["net"]),
            }
        except Exception as exc:
            gold[key] = {"log": pair["log"], "net": pair["net"],
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    report["gold_pair_candidates"] = gold

    noisy: dict[str, Any] = {}
    for key, pair in NOISY_PAIR_CANDIDATES.items():
        try:
            noisy[key] = {
                "log": pair["log"], "net": pair["net"],
                **_replay_summary(pair["log"], pair["net"]),
            }
        except Exception as exc:
            noisy[key] = {"log": pair["log"], "net": pair["net"],
                          "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    report["noisy_pair_candidates"] = noisy

    report["trace_overlap"] = [
        _trace_overlap(LEGACY_LOGS["roadtraffic50"], "tests/input_data/roadtraffic100traces.xes"),
        _trace_overlap(LEGACY_LOGS["roadtraffic50"], "tests/input_data/roadtraffic_10000.xes.gz"),
        _trace_overlap("tests/input_data/fairness/hiring_log_high.xes.gz",
                       "tests/input_data/fairness/lending_log_high.xes.gz"),
        _trace_overlap("tests/input_data/fairness/hiring_log_high.xes.gz",
                       "tests/input_data/fairness/hospital_log_high.xes.gz"),
        _trace_overlap("tests/input_data/fairness/hiring_log_high.xes.gz",
                       "tests/input_data/fairness/renting_log_high.xes.gz"),
    ]

    report["legacy_usage"] = _legacy_log_usage()

    report["footprints_probe"] = _footprints_probe(
        "tests/input_data/synthetic_logs/a12/a12f0n05.xes.gz",
        "tests/input_data/synthetic_logs/a12/a12.pnml",
    )

    report["wall_seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="L12 20K expansion source scan")
    ap.add_argument("--out", default=str(EXPANSION_DIR / "scan_report.json"))
    args = ap.parse_args()
    report = run_scan(Path(args.out))
    summary = {
        "new_logs_ok": sum(1 for v in report["new_logs"].values() if "error" not in v),
        "gold_pairs_all_fit": sorted(
            k for k, v in report["gold_pair_candidates"].items() if v.get("all_fit")),
        "noisy_pairs_mixed": sorted(
            k for k, v in report["noisy_pair_candidates"].items()
            if v.get("unfit", 0) > 0 and v.get("fit", 0) > 0),
        "footprints_ok": report["footprints_probe"].get("ok"),
        "wall_seconds": report["wall_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
