"""Registries for the L12 20K expansion (facts verified by scan.py, 2026-10-31).

All units come from the pinned pm4py snapshot ``pm4py-process@tag-2.7.23.11``
(same frozen source as the legacy 8K lane; NEW units, never legacy units).
Scan evidence: ``data/erule/review5/build8000/expansion/lanes/L12/scan_report.json``.

Verified facts behind each registry:
- NEW_GOLD_PAIRS: strict token-based replay (TBR) fitness 1.0 on ALL traces
  (scan: roadtraffic100 100/100, roadtraffic10000 10002/10002).
  ``stochastic_running_example`` was excluded: its visible transition labels
  are identical to legacy ``running-example.pnml`` (behaviorally the same net).
  The seven inh_res log+net pairs are NOT gold (2-10 of 30 traces fit) — they
  are mixed noisy pairs and are registered as such.
- NEW_LOGS: 18 event logs that parse with concept:name activities and were
  never used by any legacy L12 task (legacy used 13 log files; the three
  inh_res logs registered-but-unused by legacy families are included — scan
  ``legacy_usage.used_log_rels`` proves no legacy task references them).
  ``xes_20.xes`` is excluded: its events carry no ``concept:name``.
- NEW_NOISY_PAIRS: 18 log+net pairs with mixed strict-TBR fitness
  (some fit, some unfit) — 11 new synthetic noise levels (a12/a22/a32 f0n20,
  f0n50; the whole a42 family) + the 7 inh_res pairs.
- FP_CONF_PAIRS: noisy pairs where pm4py footprints model discovery
  (reachability graph) terminates and per-trace footprints conformance is
  mixed. a42 nets time out (>120 s reachability build); inh_res nets raise
  on model discovery (not sound WF-nets) — both excluded, honestly.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
STAGING_20K = REPO_ROOT / "bank" / "staging_20k" / "l12"
EXPANSION_DIR = (
    REPO_ROOT / "data" / "erule" / "review5" / "build8000" / "expansion" / "lanes" / "L12"
)

DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
SEED_BASE = 20261101
SOURCE_SNAPSHOT_ID = "pm4py-process@tag-2.7.23.11"
SEMANTIC_PROFILE = "pm4py-conformance-offline/1"
GENERATOR_VERSION = "l12-expand20k/1.0.0"
PROJECTION_RULE = "first_n_traces_document_order"

NEW_GOLD_PAIRS: dict[str, dict[str, str]] = {
    "roadtraffic100": {
        "log": "tests/input_data/roadtraffic100traces.xes",
        "net": "tests/input_data/roadtraffic.pnml",
    },
    "roadtraffic10000": {
        "log": "tests/input_data/roadtraffic_10000.xes.gz",
        "net": "tests/input_data/roadtraffic.pnml",
    },
}

NEW_LOGS: dict[str, str] = {
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
    "hospital_discharge": "tests/input_data/inh_res_nets/logs/hospital_discharge.xes",
    "insurance_claim": "tests/input_data/inh_res_nets/logs/insurance_claim.xes",
    "order_fulfillment": "tests/input_data/inh_res_nets/logs/order_fulfillment.xes",
}

NEW_NOISY_PAIRS: dict[str, dict[str, str]] = {
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
}

#: noisy pairs usable for footprints-conformance isolation (see module doc).
FP_CONF_PAIRS: dict[str, dict[str, str]] = {
    k: NEW_NOISY_PAIRS[k]
    for k in ("a12_n20", "a12_n50", "a22_n20", "a22_n50", "a32_n20", "a32_n50")
}


def projection_ladder(total: int) -> list[int]:
    """Deterministic first-N cutoffs, complexity-aware, answer-scale bounded.

    - small logs (<= 100 traces): five ~evenly spaced cutoffs from 2..total
      (legacy-style fractional ladder, floored at 2);
    - medium logs (<= 2000 traces): {2, 25%, 50%, 75%, 100%};
    - big logs (> 2000 traces): bounded ladder {2, 10, 50, 200, 1000} so the
      exact answer artifacts stay within the legacy lane's observed scale
      while still reaching medium/high complexity tiers.
    """
    if total <= 100:
        out: list[int] = []
        for k in range(1, 6):
            n = max(2, round(total * k / 5))
            if n <= total and n not in out:
                out.append(n)
        return out or [min(2, total)]
    if total <= 2000:
        cand = [2, max(3, total // 4), max(4, total // 2),
                max(5, 3 * total // 4), total]
    else:
        cand = [2, 10, 50, 200, 1000]
    out = []
    for n in cand:
        n = max(2, min(n, total))
        if n not in out:
            out.append(n)
    return out
