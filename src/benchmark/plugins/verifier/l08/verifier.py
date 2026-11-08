"""L08 native verifier: cql-execution engine + independent Python recomputation.

Verification design (lane contract: "actual CQL execution + independent
recomputation path"):

- ``prove_exists``: executes the task's ELM over its synthetic FHIR panel with
  the vendored cql-execution engine (Node), recomputes the same rule with the
  engine-independent Python path (``recompute.py``), and requires EXACT
  agreement (differential gate) plus a non-trivial answer set (at least 2 and
  at most panel_size-2 patients eligible).
- ``certify_semantic_relevance``: counterfactual witness — perturbing the rule
  (threshold step, operator flip, lookback change, exclusion toggle) must
  change the engine's answer on at least one patient (rule is semantically
  load-bearing for this panel).
- ``verify_terminal``: application tasks — the submitted ``answer_ids`` set
  must equal the engine-derived reference set exactly; repair tasks — the
  submitted rule parameters are rebuilt into ELM, executed by the engine, and
  must reproduce the intended classification.

No LLM output is ever used as gold. All engine runs are deterministic (frozen
reference date; no wall-clock ELM).
"""
from __future__ import annotations

import copy
from typing import Any

from benchmark.plugins.verifier.l08 import elm as elm_mod
from benchmark.plugins.verifier.l08 import recompute
from benchmark.plugins.verifier.l08.runner import (
    NodeEngineRunner, engine_versions, job_cache_key,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

PROFILE = "cql-fhir-r4-explicit-paths/1"


def perturbations(rule: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic counterfactual perturbations of a rule (name, mutated rule)."""
    out: list[tuple[str, dict[str, Any]]] = []
    step = _threshold_step(rule)
    if rule.get("threshold") is not None:
        for delta, tag in ((step, "up"), (-step, "down")):
            mutated = copy.deepcopy(rule)
            mutated["threshold"] = _num(rule["threshold"]) + delta
            out.append(("threshold_%s(%+d)" % (tag, delta), mutated))
        flipped = copy.deepcopy(rule)
        flipped["op"] = {"<=": "<", "<": "<=", ">=": ">", ">": ">="}[rule["op"]]
        out.append(("op_flip(%s->%s)" % (rule["op"], flipped["op"]), flipped))
    if rule.get("lookback_days"):
        for factor, tag in ((2, "doubled"), (0, "zero"), (0.5, "halved")):
            mutated = copy.deepcopy(rule)
            mutated["lookback_days"] = int(rule["lookback_days"]) * factor
            out.append(("lookback_%s(%d)" % (tag, mutated["lookback_days"]), mutated))
    elif rule["kind"] == "cond_active":
        # no-lookback condition rules: adding an onset window is the
        # clinically meaningful counterfactual (older onsets drop out)
        for lb in (90, 365):
            mutated = copy.deepcopy(rule)
            mutated["lookback_days"] = lb
            out.append(("onset_window_added(%d)" % lb, mutated))
    if rule.get("exclude") is not None:
        mutated = copy.deepcopy(rule)
        mutated["exclude"] = not rule["exclude"]
        out.append(("exclusion_toggled", mutated))
    return out


def _num(v: Any) -> float:
    return float(v)


def _threshold_step(rule: dict[str, Any]) -> float:
    """Counterfactual step: 15% of |threshold| (crosses the 6% value grid)."""
    t = _num(rule.get("threshold", 10))
    step = max(1.0, round(abs(t) * 0.15))
    return float(step)


class L08NativeVerifier:
    """NativeVerifierPlugin for L08 (see module docstring for the gate design)."""

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

    def engine_evaluate(self, rule: dict[str, Any], patients: list[dict[str, Any]]) -> dict[str, bool]:
        """Execute the rule's ELM over the panel (cached by content hash)."""
        elm_lib = elm_mod.build_elm(rule)
        key = job_cache_key(elm_lib, patients)
        if key not in self._cache:
            self._cache[key] = self.runner.evaluate(elm_lib, patients, job_id=key[:24])
        return self._cache[key]

    def python_evaluate(self, rule: dict[str, Any], patients: list[dict[str, Any]]) -> dict[str, bool]:
        return recompute.recompute(rule, patients)

    def batch_evaluate(self, pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
                       *, chunk: int = 100) -> int:
        """Prime the cache with one node invocation per ``chunk`` jobs.

        Returns the number of engine jobs actually executed (uncached ones).
        """
        from benchmark.plugins.verifier.l08.elm import build_elm

        jobs: list[dict[str, Any]] = []
        keys: list[str] = []
        for rule, patients in pairs:
            elm_lib = build_elm(rule)
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
                    # leave uncached: the lazy path will surface the error
                    continue
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
        for name, mutated in perturbations(rule):
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
        params = ("op", "threshold", "lookback_days", "exclude")
        missing = [p for p in params if p not in submission]
        if missing:
            return NativeVerdict(valid=False, detail={"error": "missing params: %s" % missing})
        try:
            rebuilt = copy.deepcopy(base_rule)
            rebuilt["op"] = str(submission["op"])
            rebuilt["threshold"] = float(submission["threshold"])
            rebuilt["lookback_days"] = int(submission["lookback_days"])
            rebuilt["exclude"] = bool(submission["exclude"])
            if rebuilt["op"] not in (">=", ">", "<=", "<"):
                raise ValueError("invalid op %r" % rebuilt["op"])
            if rebuilt["threshold"] < 0 or rebuilt["lookback_days"] < 0:
                raise ValueError("negative threshold/lookback")
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


PLUGIN = L08NativeVerifier()
