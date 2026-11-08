"""L08 20K-expansion native verifier: cql-execution + independent recomputation.

Mirrors the legacy L08 verifier design for the ten NEW rule kinds:

- ``prove_exists``: executes the task's ELM over its synthetic FHIR panel with
  the vendored cql-execution engine (Node), recomputes the same rule with the
  engine-independent Python path (``recompute_20k``), and requires EXACT
  agreement (differential gate) plus a non-trivial answer set.
- ``certify_semantic_relevance``: counterfactual witness — kind-specific
  perturbations (threshold step, operator flip, lookback change, min_count
  shift, direction flip, band shrink/expand, age shift, exclusion toggle)
  must change the engine's answer on at least one patient.
- ``verify_terminal``: application tasks — the submitted ``answer_ids`` set
  must equal the engine-derived reference set exactly; repair tasks — the
  submitted rule parameters {op, threshold, lookback_days, min_count} are
  rebuilt into ELM, executed by the engine, and must reproduce the intended
  classification.

No LLM output is ever used as gold. All engine runs are deterministic (frozen
reference date REF_DATE_20K; no wall-clock ELM).

This module deliberately does NOT export ``PLUGIN``/``get_plugin()`` so the
8K plugin discovery (which resolves ``benchmark.plugins.verifier.l08`` to the
legacy ``verifier.PLUGIN``) is unaffected.
"""
from __future__ import annotations

import copy
from typing import Any

from benchmark.plugins.verifier.l08 import elm_20k
from benchmark.plugins.verifier.l08 import recompute_20k
from benchmark.plugins.verifier.l08.runner import (
    NodeEngineRunner, engine_versions, job_cache_key,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

PROFILE_20K = "cql-fhir-r4-explicit-paths/2"

#: other encounter classes for the enc_gap class-swap counterfactual
_ENC_CLASS_ALTERNATIVES = ("AMB", "EMER", "IMP", "HH")


def perturbations_20k(rule: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic counterfactual perturbations of a 20K rule (name, mutated)."""
    out: list[tuple[str, dict[str, Any]]] = []
    kind = rule["kind"]
    step = _threshold_step(rule)

    def add(name: str, mutated: dict[str, Any]) -> None:
        out.append((name, copy.deepcopy(mutated)))

    if rule.get("threshold") is not None:
        for delta, tag in ((step, "up"), (-step, "down")):
            mutated = dict(rule, threshold=_num(rule["threshold"]) + delta)
            add("threshold_%s(%+d)" % (tag, delta), mutated)
        flipped = dict(rule, op={"<=": "<", "<": "<=", ">=": ">", ">": ">="}[rule["op"]])
        add("op_flip(%s->%s)" % (rule["op"], flipped["op"]), flipped)
    if rule.get("lookback_days"):
        for factor, tag in ((2, "doubled"), (0, "zero"), (0.5, "halved")):
            mutated = dict(rule, lookback_days=int(rule["lookback_days"]) * factor)
            add("lookback_%s(%d)" % (tag, mutated["lookback_days"]), mutated)
    if kind in ("obs_count", "obs_count_repair"):
        for delta, tag in ((1, "up"), (-1, "down")):
            mutated = dict(rule, min_count=int(rule["min_count"]) + delta)
            add("min_count_%s(%+d)" % (tag, delta), mutated)
    if kind == "obs_trend":
        mutated = dict(rule, direction="worsen" if rule.get("direction", "improve") == "improve" else "improve")
        add("direction_flip(%s)" % mutated["direction"], mutated)
    if kind == "enc_gap":
        for alt in _ENC_CLASS_ALTERNATIVES:
            if alt != rule["enc_class"]:
                add("class_swap(%s)" % alt, dict(rule, enc_class=alt))
    if kind == "age_cond":
        for delta, tag in ((5, "up"), (-5, "down"), (10, "up10")):
            mutated = dict(rule, min_age_years=max(18, int(rule["min_age_years"]) + delta))
            add("age_%s(%+d y)" % (tag, delta), mutated)
    if kind == "obs_range":
        s = step
        add("band_shrink", dict(rule, lo=_num(rule["lo"]) + s, hi=_num(rule["hi"]) - s))
        add("band_expand", dict(rule, lo=_num(rule["lo"]) - s, hi=_num(rule["hi"]) + s))
    if rule.get("exclude") is not None:
        add("exclusion_toggled", dict(rule, exclude=not rule["exclude"]))
    return out


def _num(v: Any) -> float:
    return float(v)


def _threshold_step(rule: dict[str, Any]) -> float:
    """Counterfactual step: 15% of the |threshold| (or band midpoint for ranges)."""
    if rule.get("threshold") is not None:
        t = _num(rule["threshold"])
    else:
        t = (_num(rule.get("lo", 0)) + _num(rule.get("hi", 0))) / 2.0
    return float(max(1.0, round(abs(t) * 0.15)))


class L0820kNativeVerifier:
    """Native verifier for the L08 20K-expansion rule kinds (see docstring)."""

    def __init__(self, runner: NodeEngineRunner | None = None,
                 results_cache: dict[str, dict[str, bool]] | None = None) -> None:
        self._runner = runner
        self._cache: dict[str, dict[str, bool]] = results_cache if results_cache is not None else {}

    @property
    def runner(self) -> NodeEngineRunner:
        if self._runner is None:
            self._runner = NodeEngineRunner()
        return self._runner

    # ------------------------------------------------------------- core engine

    def engine_evaluate(self, rule: dict[str, Any],
                        patients: list[dict[str, Any]]) -> dict[str, bool]:
        """Execute the rule's ELM over the panel (cached by content hash)."""
        elm_lib = elm_20k.build_elm_20k(rule)
        key = job_cache_key(elm_lib, patients)
        if key not in self._cache:
            self._cache[key] = self.runner.evaluate(elm_lib, patients, job_id=key[:24])
        return self._cache[key]

    def python_evaluate(self, rule: dict[str, Any],
                        patients: list[dict[str, Any]]) -> dict[str, bool]:
        return recompute_20k.recompute_20k(rule, patients)

    def batch_evaluate(self, pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
                       *, chunk: int = 100) -> int:
        """Prime the cache with one node invocation per ``chunk`` jobs.

        Returns the number of engine jobs actually executed (uncached ones).
        """
        jobs: list[dict[str, Any]] = []
        keys: list[str] = []
        for rule, patients in pairs:
            elm_lib = elm_20k.build_elm_20k(rule)
            key = job_cache_key(elm_lib, patients)
            if key in self._cache:
                continue
            jobs.append({"job_id": key[:24], "library": elm_lib, "patients": patients})
            keys.append(key)
        executed = 0
        for i in range(0, len(jobs), chunk):
            batch = jobs[i:i + chunk]
            batch_keys = keys[i:i + chunk]
            out = self.runner.run_jobs(batch)
            key_by_job_id = {j["job_id"]: k for j, k in zip(batch, batch_keys)}
            for res in out.get("results", []):
                if not res.get("ok"):
                    continue  # leave uncached: the lazy path will surface the error
                result: dict[str, bool] = {}
                for pid, exprs in res.get("patient_results", {}).items():
                    val = exprs.get("Eligible", exprs.get("Flagged"))
                    result[pid] = bool(val) if val is not None else False
                self._cache[key_by_job_id[res["job_id"]]] = result
            executed += len(batch)
        return executed

    # ------------------------------------------------------- protocol surface

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        rule, patients = ast["rule"], ast["patients"]
        try:
            engine = self.engine_evaluate(rule, patients)
        except Exception as exc:  # engine failure = not provable
            return ProofOrCounterexample(exists=False, reason="ENGINE_ERROR:%s" % exc,
                                         evidence={"error": str(exc)})
        py = self.python_evaluate(rule, patients)
        agree = engine == py
        answer = sorted(pid for pid, v in engine.items() if v)
        n = len(engine)
        evidence = {
            "engine": engine_versions(),
            "engine_answer": answer,
            "python_answer": sorted(pid for pid, v in py.items() if v),
            "differential_agree": agree,
            "panel_size": n,
            "answer_size": len(answer),
        }
        if not agree:
            return ProofOrCounterexample(exists=False, reason="DIFFERENTIAL_MISMATCH",
                                         evidence=evidence)
        if len(answer) < 2 or len(answer) > n - 2:
            return ProofOrCounterexample(exists=False, reason="TRIVIAL:answer_boundary",
                                         evidence=evidence)
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def certify_semantic_relevance(self, grounded_task: GroundedTask) -> SemanticWitness:
        ast = grounded_task.ast.ast
        rule, patients = ast["rule"], ast["patients"]
        base = self.engine_evaluate(rule, patients)
        base_answer = frozenset(pid for pid, v in base.items() if v)
        tried: list[dict[str, Any]] = []
        for name, mutated in perturbations_20k(rule):
            try:
                mutated_result = self.engine_evaluate(mutated, patients)
            except Exception as exc:
                tried.append({"counterfactual": name, "error": str(exc)})
                continue
            mutated_answer = frozenset(pid for pid, v in mutated_result.items() if v)
            tried.append({
                "counterfactual": name,
                "answer_size": len(mutated_answer),
                "symmetric_difference": len(base_answer ^ mutated_answer),
            })
            if mutated_answer != base_answer:
                return SemanticWitness(ok=True, witness={
                    "base_answer_size": len(base_answer),
                    "sensitive_counterfactual": name,
                    "symmetric_difference": len(base_answer ^ mutated_answer),
                    "all_counterfactuals": tried,
                })
        return SemanticWitness(ok=False, witness={
            "base_answer_size": len(base_answer),
            "all_counterfactuals": tried,
            "reason": "no perturbation changed the engine answer",
        })

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        ast = task.ast.ast
        rule, patients = ast["rule"], ast["patients"]
        reference = self.engine_evaluate(rule, patients)
        ref_set = frozenset(pid for pid, v in reference.items() if v)
        submission = submitted_state_or_answer if isinstance(submitted_state_or_answer, dict) else {}
        regime = task.ast.regime
        if regime == "repair":
            return self._verify_repair(ast, submission, ref_set)
        answer_ids = submission.get("answer_ids")
        if not isinstance(answer_ids, list):
            return NativeVerdict(valid=False, detail={"error": "answer_ids must be a list"})
        got = frozenset(str(x) for x in answer_ids)
        valid = got == ref_set
        return NativeVerdict(valid=valid, detail={
            "expected_size": len(ref_set),
            "received_size": len(got),
            "missing": sorted(ref_set - got),
            "extra": sorted(got - ref_set),
        })

    def _verify_repair(self, ast: dict[str, Any], submission: dict[str, Any],
                       intended: frozenset[str]) -> NativeVerdict:
        base_rule = ast["rule"]
        params = ("op", "threshold", "lookback_days", "min_count")
        missing = [p for p in params if p not in submission]
        if missing:
            return NativeVerdict(valid=False, detail={"error": "missing params: %s" % missing})
        try:
            rebuilt = copy.deepcopy(base_rule)
            rebuilt["op"] = str(submission["op"])
            rebuilt["threshold"] = float(submission["threshold"])
            rebuilt["lookback_days"] = int(submission["lookback_days"])
            rebuilt["min_count"] = int(submission["min_count"])
            if rebuilt["op"] not in (">=", ">", "<=", "<"):
                raise ValueError("invalid op %r" % rebuilt["op"])
            if rebuilt["threshold"] < 0 or rebuilt["lookback_days"] < 0:
                raise ValueError("negative threshold/lookback")
            if rebuilt["min_count"] < 1:
                raise ValueError("min_count must be >= 1")
        except (TypeError, ValueError) as exc:
            return NativeVerdict(valid=False, detail={"error": "bad params: %s" % exc})
        result = self.engine_evaluate(rebuilt, ast["patients"])
        got = frozenset(pid for pid, v in result.items() if v)
        return NativeVerdict(valid=got == intended, detail={
            "expected_size": len(intended),
            "received_size": len(got),
            "missing": sorted(intended - got),
            "extra": sorted(got - intended),
        })
