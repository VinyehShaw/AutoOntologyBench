"""L12 engine: pm4py-native process-mining operations over the pinned snapshot.

Single-runtime design: pm4py import costs ~30-80s on NAS IO, so the runtime is a
lazy module-level singleton. Every native verdict in lane L12 funnels through
the helpers here; all are deterministic (fixed pm4py 2.7.23.11, fixed snapshot,
no wall-clock or randomness inside the native calls).

Artifact registries are scoped to ``tests/input_data/`` of the pinned pm4py
tree: GOLD_PAIRS replay at fitness 1.0 under strict token-based replay (proven
at generation time, re-proven by the verifier); NOISY_PAIRS mix fitting and
non-fitting traces under the same model; LOGS covers log-only audit families.
a42 is deliberately excluded from GOLD_PAIRS: its own noise-free log does not
fully replay on its shipped model at this pm4py version (honest exclusion).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from source_registry.fetchers import l12_pm4py as fetcher

ENGINE = "pm4py"
PM4PY_VERSION = "2.7.23.11"
PM4PY_COMMIT = fetcher.PINNED_COMMIT
PINNED_ROOT = fetcher.PINNED_ROOT

#: Strict token-replay parameters: unknown activities and remaining tokens both
#: break fitness (pm4py defaults do NOT penalize unknown activities).
STRICT_TBR_PARAMS = {
    "disable_variants": True,
    "consider_remaining_in_fitness": True,
    "consider_activities_not_in_model_in_fitness": True,
    "show_progress_bar": False,
}

_DECLARE_NEGATIVE_TEMPLATES = ("noncoexistence", "nonsuccession", "nonchainsuccession")

GOLD_PAIRS: dict[str, dict[str, str]] = {
    "running_example": {
        "log": "tests/input_data/running-example.xes",
        "net": "tests/input_data/running-example.pnml",
    },
    "roadtraffic50": {
        "log": "tests/input_data/roadtraffic50traces.xes",
        "net": "tests/input_data/roadtraffic.pnml",
    },
    "a12": {
        "log": "tests/input_data/synthetic_logs/a12/a12f0n00.xes.gz",
        "net": "tests/input_data/synthetic_logs/a12/a12.pnml",
    },
    "a22": {
        "log": "tests/input_data/synthetic_logs/a22/a22f0n00.xes.gz",
        "net": "tests/input_data/synthetic_logs/a22/a22.pnml",
    },
    "a32": {
        "log": "tests/input_data/synthetic_logs/a32/a32f0n00.xes.gz",
        "net": "tests/input_data/synthetic_logs/a32/a32.pnml",
    },
}

LOGS: dict[str, str] = {
    "running_example": GOLD_PAIRS["running_example"]["log"],
    "roadtraffic50": GOLD_PAIRS["roadtraffic50"]["log"],
    "reviewing": "tests/input_data/reviewing.xes",
    "receipt": "tests/input_data/receipt.xes",
    "hospital_discharge": "tests/input_data/inh_res_nets/logs/hospital_discharge.xes",
    "insurance_claim": "tests/input_data/inh_res_nets/logs/insurance_claim.xes",
    "order_fulfillment": "tests/input_data/inh_res_nets/logs/order_fulfillment.xes",
    "a12": GOLD_PAIRS["a12"]["log"],
    "a22": GOLD_PAIRS["a22"]["log"],
    "a32": GOLD_PAIRS["a32"]["log"],
}

NOISY_PAIRS: dict[str, dict[str, str]] = {
    "a12_n05": {
        "net": GOLD_PAIRS["a12"]["net"],
        "log": "tests/input_data/synthetic_logs/a12/a12f0n05.xes.gz",
    },
    "a12_n10": {
        "net": GOLD_PAIRS["a12"]["net"],
        "log": "tests/input_data/synthetic_logs/a12/a12f0n10.xes.gz",
    },
    "a22_n05": {
        "net": GOLD_PAIRS["a22"]["net"],
        "log": "tests/input_data/synthetic_logs/a22/a22f0n05.xes.gz",
    },
    "a22_n10": {
        "net": GOLD_PAIRS["a22"]["net"],
        "log": "tests/input_data/synthetic_logs/a22/a22f0n10.xes.gz",
    },
    "a32_n05": {
        "net": GOLD_PAIRS["a32"]["net"],
        "log": "tests/input_data/synthetic_logs/a32/a32f0n05.xes.gz",
    },
    "a32_n10": {
        "net": GOLD_PAIRS["a32"]["net"],
        "log": "tests/input_data/synthetic_logs/a32/a32f0n10.xes.gz",
    },
    "receipt_onevar": {
        "net": "tests/input_data/receipt_one_variant.pnml",
        "log": "tests/input_data/receipt.xes",
    },
}


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Pm4pyRuntime:
    """Lazy singleton holding the pm4py module + parsed-artifact caches."""

    def __init__(self) -> None:
        self._pm4py: Any = None
        self._mods: dict[str, Any] = {}
        self._traces: dict[str, list[list[str]]] = {}
        self._nets: dict[str, tuple[Any, Any, Any]] = {}
        self._net_labels: dict[str, dict[str, str]] = {}

    def _load(self) -> None:
        if self._pm4py is not None:
            return
        import pm4py  # noqa: PLC0415 - deliberate lazy import (costly on NAS)
        from pm4py.algo.conformance.declare import (  # noqa: PLC0415
            algorithm as decl_conf,
        )
        from pm4py.algo.conformance.footprints import (  # noqa: PLC0415
            algorithm as fp_conf,
        )
        from pm4py.algo.conformance.log_skeleton import (  # noqa: PLC0415
            algorithm as lsk_conf,
        )
        from pm4py.algo.conformance.tokenreplay import (  # noqa: PLC0415
            algorithm as token_replay,
        )
        from pm4py.algo.discovery.declare import algorithm as decl_disc  # noqa: PLC0415
        from pm4py.algo.discovery.dfg import algorithm as dfg_alg  # noqa: PLC0415
        from pm4py.algo.discovery.footprints import algorithm as fp_alg  # noqa: PLC0415
        from pm4py.algo.discovery.log_skeleton import (  # noqa: PLC0415
            algorithm as lsk_alg,
        )
        from pm4py.objects.log.importer.xes import (  # noqa: PLC0415
            importer as xes_importer,
        )
        from pm4py.objects.log.obj import EventLog, Event, Trace  # noqa: PLC0415
        from pm4py.objects.petri_net.importer import (  # noqa: PLC0415
            importer as pnml_importer,
        )

        self._pm4py = pm4py
        self._mods = {
            "decl_conf": decl_conf,
            "fp_conf": fp_conf,
            "lsk_conf": lsk_conf,
            "token_replay": token_replay,
            "decl_disc": decl_disc,
            "dfg_alg": dfg_alg,
            "fp_alg": fp_alg,
            "lsk_alg": lsk_alg,
            "xes_importer": xes_importer,
            "pnml_importer": pnml_importer,
            "EventLog": EventLog,
            "Trace": Trace,
            "Event": Event,
        }

    # -- artifact loading -------------------------------------------------------
    def traces(self, log_rel: str) -> list[list[str]]:
        """Pinned log as a list of activity-name traces (cached)."""
        if log_rel in self._traces:
            return self._traces[log_rel]
        self._load()
        path = PINNED_ROOT / log_rel
        if not path.is_file():
            raise FileNotFoundError("pinned log missing: %s" % path)
        log = self._mods["xes_importer"].apply(
            str(path), parameters={"show_progress_bar": False}
        )
        traces = [[e["concept:name"] for e in tr] for tr in log]
        self._traces[log_rel] = traces
        return traces

    def net(self, net_rel: str, relabel: dict[str, str] | None = None) -> tuple[Any, Any, Any]:
        """Pinned PNML net as (net, initial_marking, final_marking).

        ``relabel`` maps transition ids to replacement labels; relabeled nets are
        never cached (mutation isolation), base nets are cached read-only.
        """
        self._load()
        path = PINNED_ROOT / net_rel
        if not path.is_file():
            raise FileNotFoundError("pinned net missing: %s" % path)
        net, im, fm = self._mods["pnml_importer"].apply(str(path))
        if relabel:
            for t in net.transitions:
                if t.name in relabel:
                    t.label = relabel[t.name]
            return net, im, fm
        if net_rel not in self._nets:
            self._nets[net_rel] = (net, im, fm)
            self._net_labels[net_rel] = {
                t.name: t.label for t in net.transitions if t.label is not None
            }
        return self._nets[net_rel]

    def visible_transitions(self, net_rel: str) -> dict[str, str]:
        """transition id -> label for every visible (non-tau) transition."""
        self.net(net_rel)
        return dict(self._net_labels[net_rel])

    def event_log(self, traces: list[list[str]]) -> Any:
        self._load()
        EventLog, Trace, Event = (
            self._mods["EventLog"],
            self._mods["Trace"],
            self._mods["Event"],
        )
        out = EventLog()
        for names in traces:
            tr = Trace()
            for n in names:
                tr.append(Event({"concept:name": n}))
            out.append(tr)
        return out

    # -- native checks ----------------------------------------------------------
    def replay(self, traces: list[list[str]], net_rel: str) -> list[dict[str, Any]]:
        """Strict token-based replay; one verdict dict per trace."""
        self._load()
        net, im, fm = self.net(net_rel)
        log = self.event_log(traces)
        res = self._mods["token_replay"].apply(
            log, net, im, fm, parameters=dict(STRICT_TBR_PARAMS)
        )
        return [
            {
                "trace_is_fit": bool(r["trace_is_fit"]),
                "trace_fitness": round(float(r["trace_fitness"]), 6),
                "missing_tokens": int(r["missing_tokens"]),
                "remaining_tokens": int(r["remaining_tokens"]),
            }
            for r in res
        ]

    def replay_on_relabeled(
        self, traces: list[list[str]], net_rel: str, relabel: dict[str, str]
    ) -> list[dict[str, Any]]:
        self._load()
        net, im, fm = self.net(net_rel, relabel=relabel)
        log = self.event_log(traces)
        res = self._mods["token_replay"].apply(
            log, net, im, fm, parameters=dict(STRICT_TBR_PARAMS)
        )
        return [
            {
                "trace_is_fit": bool(r["trace_is_fit"]),
                "trace_fitness": round(float(r["trace_fitness"]), 6),
                "missing_tokens": int(r["missing_tokens"]),
                "remaining_tokens": int(r["remaining_tokens"]),
            }
            for r in res
        ]

    def dfg(self, traces: list[list[str]]) -> dict[str, Any]:
        self._load()
        dfg_alg = self._mods["dfg_alg"]
        log = self.event_log(traces)
        dfg = dfg_alg.apply(log, variant=dfg_alg.Variants.FREQUENCY)
        sa = self._pm4py.get_start_activities(log)
        ea = self._pm4py.get_end_activities(log)
        return {
            "edges": sorted([[a, b, c] for (a, b), c in dfg.items()]),
            "start_activities": dict(sorted(sa.items())),
            "end_activities": dict(sorted(ea.items())),
        }

    def variants(self, traces: list[list[str]]) -> list[list[Any]]:
        """Canonical variant partition: sorted [variant_string, trace_indices, count]."""
        self._load()
        log = self.event_log(traces)
        index_of = {id(tr): i for i, tr in enumerate(log)}
        raw = self._pm4py.get_variants(log)
        out: list[list[Any]] = []
        for key, val in raw.items():
            idx = sorted(index_of[id(t)] for t in val)
            out.append([",".join(key), idx, len(val)])
        return sorted(out, key=lambda r: (r[1][0], r[0]))

    def endpoints(self, traces: list[list[str]]) -> dict[str, Any]:
        self._load()
        log = self.event_log(traces)
        starts = self._pm4py.get_start_activities(log)
        ends = self._pm4py.get_end_activities(log)
        counts = self._pm4py.get_event_attribute_values(log, "concept:name")
        return {
            "start_activities": dict(sorted(starts.items())),
            "end_activities": dict(sorted(ends.items())),
            "activity_counts": dict(sorted(counts.items())),
        }

    def footprints_log(self, traces: list[list[str]]) -> dict[str, Any]:
        self._load()
        fp_alg = self._mods["fp_alg"]
        fp = fp_alg.apply(
            self.event_log(traces), variant=fp_alg.Variants.ENTIRE_EVENT_LOG
        )
        return {
            k: sorted([sorted(list(p)) for p in v]) if isinstance(v, set) else v
            for k, v in fp.items()
        }

    def skeleton(self, traces: list[list[str]]) -> dict[str, Any]:
        self._load()
        return dict(self._mods["lsk_alg"].apply(self.event_log(traces)))

    def skeleton_check(self, traces: list[list[str]], skel: dict[str, Any]) -> list[dict[str, Any]]:
        self._load()
        res = self._mods["lsk_conf"].apply(self.event_log(traces), skel)
        return [
            {
                "is_fit": bool(r["is_fit"]),
                "no_dev_total": int(r["no_dev_total"]),
                "deviations": [str(d) for d in r["deviations"]],
                "dev_fitness": round(float(r["dev_fitness"]), 6),
            }
            for r in res
        ]

    def declare_model(self, traces: list[list[str]]) -> dict[str, Any]:
        self._load()
        model = dict(self._mods["decl_disc"].apply(self.event_log(traces)))
        # Discovery and the conformance checker disagree on the negative
        # templates (a mined noncoexistence(a,b) fires on traces that contain
        # both a and b even though discovery derived it from the same log), so
        # the lane DECLARE spec keeps positive templates only — every retained
        # constraint is self-consistently checkable.
        return {
            k: v
            for k, v in model.items()
            if k not in _DECLARE_NEGATIVE_TEMPLATES and isinstance(v, dict) and v
        }

    def declare_check(self, traces: list[list[str]], model: dict[str, Any]) -> list[dict[str, Any]]:
        self._load()
        res = self._mods["decl_conf"].apply(self.event_log(traces), model)
        return [
            {
                "is_fit": bool(r["is_fit"]),
                "no_dev_total": int(r["no_dev_total"]),
                "deviations": [str(d) for d in r["deviations"]],
                "dev_fitness": round(float(r["dev_fitness"]), 6),
            }
            for r in res
        ]


RUNTIME = Pm4pyRuntime()


def unit_id_for(rel: str) -> str:
    return "%s/%s" % (fetcher.SOURCE_ID, rel)


def project_first_n(traces: list[list[str]], n: int) -> list[list[str]]:
    return traces[: max(2, min(n, len(traces)))]


def complexity_of(scope_events: int) -> str:
    if scope_events < 100:
        return "low"
    if scope_events < 1500:
        return "medium"
    return "high"
