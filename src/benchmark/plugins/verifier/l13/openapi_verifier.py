"""L13 native verifier: deterministic OpenAPI contract truth.

The verifier is the private oracle. Every verdict is *recomputed* from the
pinned snapshot bytes (unit digests are re-checked against the index): the
contract in each task is re-derived from the spec, jsonschema 4.26.0 performs
instance validation, and exact-set recomputation covers selection/enumeration
families. No stored evidence is trusted; no LLM output is gold.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import jsonschema

from benchmark.plugins.source.l13 import contracts, oas
from benchmark.plugins.source.l13.apis_guru import ApisGuruSourcePlugin
from benchmark.plugins.source.l13.oas import OasError
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)
from source_registry.fetchers.l13_apis_guru import PINNED_COMMIT

ENGINE = "openapi-validator(jsonschema-%s)" % jsonschema.__version__

_SUBMIT_KEY = {
    "param_contract": "parameters",
    "body_contract": "body",
    "response_construction": "instance",
    "version_migration": None,  # per mode
    "schema_mapping": "instance",
    "operation_selection": "operation_keys",
    "security_requirement": "alternatives",
}


class OpenAPIContractVerifier:
    """NativeVerifierPlugin for lane L13."""

    def __init__(self, oracle_dir: str | Path) -> None:
        self.oracle_dir = Path(oracle_dir)
        self.oracle_dir.mkdir(parents=True, exist_ok=True)
        self.oracle_dir.chmod(0o700)
        self.source = ApisGuruSourcePlugin()

    # -- contract re-derivation -------------------------------------------------
    def _load(self, ast: dict[str, Any], which: str = "") -> dict[str, Any]:
        unit_id = ast["unit_id" + which]
        digest = ast["unit_digest" + which]
        payload_ref = unit_id.split("/", 1)[1]
        path = self.source.spec_path(payload_ref)
        actual = oas.sha256_file(path)
        if actual != digest:
            raise OasError("unit digest mismatch: %s" % unit_id)
        return oas.load_spec(path)

    def _site(self, spec: dict[str, Any], coordinate: dict[str, Any]):
        for site in oas.iter_operations(spec):
            if site.path == coordinate["path"] and site.method == coordinate["method"]:
                return site
        raise OasError("operation not found: %s %s" % (coordinate.get("method"), coordinate.get("path")))

    def rederive(self, ast: dict[str, Any]) -> dict[str, Any]:
        """Recompute contract(s) and answer material from the pinned specs."""
        kind = ast["goal_kind"]
        coord = ast["coordinate"]
        spec = self._load(ast)
        out: dict[str, Any] = {}
        if kind == "param_contract":
            out["contract"] = contracts.param_contract(contracts.task_params(self._site(spec, coord), spec))
        elif kind == "body_contract":
            site = self._site(spec, coord)
            if site.request_body_schema is None:
                raise OasError("operation has no request body")
            out["contract"] = {"kind": "body_contract",
                               "body_schema": oas.normalize_schema(site.request_body_schema, spec)}
        elif kind == "response_construction":
            site = self._site(spec, coord)
            if coord["code"] not in site.response_schemas:
                raise OasError("response code %s absent" % coord["code"])
            out["contract"] = {"kind": "response_construction", "code": coord["code"],
                               "response_schema": oas.normalize_schema(
                                   site.response_schemas[coord["code"]], spec, direction="response")}
        elif kind == "operation_selection":
            feats = [contracts.op_features(s) for s in oas.iter_operations(spec)]
            out["contract"] = {"kind": "operation_selection",
                               "operations": [{k: f[k] for k in ("operation_key", "path", "method", "parameters")}
                                              for f in feats]}
            out["answer"] = contracts.apply_predicate(feats, ast["constraints"]["predicate"])
            out["relaxed_answer"] = contracts.apply_predicate(
                feats, ast["constraints"]["predicate"], relaxed=True)
        elif kind == "security_requirement":
            site = self._site(spec, coord)
            out["contract"] = {"kind": "security_requirement",
                               "security_schemes": site.security_schemes,
                               "operation_security": site.operation.get("security"),
                               "global_security": spec.get("security")}
            out["answer"] = contracts.normalized_security(site.security)
        elif kind == "version_migration":
            spec_a = self._load(ast, "_b")
            site_b = self._site(spec, coord)
            site_a = self._site(spec_a, coord)
            if coord["mode"] == "params":
                out["contract"] = contracts.param_contract(contracts.task_params(site_b, spec))
                out["contract_b"] = contracts.param_contract(contracts.task_params(site_a, spec_a))
            else:
                out["contract"] = {"kind": "body_contract",
                                   "body_schema": oas.normalize_schema(site_b.request_body_schema, spec)}
                out["contract_b"] = {"kind": "body_contract",
                                     "body_schema": oas.normalize_schema(site_a.request_body_schema, spec_a)}
        elif kind == "schema_mapping":
            spec_s = self._load(ast, "_b")
            out["contract"] = {"kind": "schema_mapping",
                               "target_schema": self._component(spec, coord["schema_b"])}
            out["contract_b"] = {"kind": "schema_mapping",
                                 "source_schema": self._component(spec_s, coord["schema_a"])}
        else:
            raise OasError("unknown goal_kind %r" % kind)
        return out

    @staticmethod
    def _component(spec: dict[str, Any], name: str) -> dict[str, Any]:
        schemas = (spec.get("components") or {}).get("schemas") or {}
        if name not in schemas:
            raise OasError("component schema %r absent" % name)
        return oas.normalize_schema(schemas[name], spec)

    # -- NativeVerifierPlugin -----------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        ev: dict[str, Any] = {"engine": ENGINE, "source_commit": PINNED_COMMIT,
                              "goal_kind": ast["goal_kind"]}
        try:
            re = self.rederive(ast)
        except OasError as exc:
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:%s" % str(exc)[:200])
        if oas.sha256_json(re["contract"]) != oas.sha256_json(ast["contract"]):
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:contract_drift")
        if "contract_b" in re and (
                ast.get("contract_b") is None
                or oas.sha256_json(re["contract_b"]) != oas.sha256_json(ast["contract_b"])):
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:contract_b_drift")
        try:
            gate = _GATES[ast["goal_kind"]](self, ast, re, ev)
        except OasError as exc:
            return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:%s" % str(exc)[:200])
        if gate is not None:
            return ProofOrCounterexample(False, ev, gate)
        return ProofOrCounterexample(True, ev)

    def write_oracle(self, grounded_task: GroundedTask, evidence: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "lane_id": "L13",
            "family_id": grounded_task.ast.family_id,
            "goal_kind": ast["goal_kind"],
            "engine": ENGINE,
            "source_commit": PINNED_COMMIT,
            "unit_id": ast["unit_id"],
            "unit_digest": ast["unit_digest"],
            "unit_id_b": ast.get("unit_id_b"),
            "unit_digest_b": ast.get("unit_digest_b"),
            "coordinate": ast["coordinate"],
            "contract": ast["contract"],
            "contract_b": ast.get("contract_b"),
            "initial": ast.get("initial"),
            "reference": ast["reference"],
            "guards": ast["guards"],
            "constraints": ast["constraints"],
            "witness": ast["witness"],
        }
        path = self.oracle_dir / ("%s.json" % grounded_task.oracle_ref)
        payload["oracle_sha256"] = oas.sha256_json(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path

    def load_oracle(self, oracle_ref: str) -> dict[str, Any]:
        return json.loads((self.oracle_dir / ("%s.json" % oracle_ref)).read_text(encoding="utf-8"))

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        oracle = self.load_oracle(task.oracle_ref)
        checks: list[dict[str, Any]] = []
        kind = oracle["goal_kind"]
        submitted = submitted_state_or_answer or {}
        key = _SUBMIT_KEY[kind] or ("parameters" if oracle["coordinate"].get("mode") == "params" else "body")
        answer = submitted.get(key)
        if answer is None:
            return NativeVerdict(False, {"error": "missing submission key %s" % key})
        _TERMINAL[kind](self, oracle, answer, checks)
        valid = all(c["pass"] for c in checks)
        return NativeVerdict(valid, {"checks": checks, "engine": ENGINE})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        oracle = self.load_oracle(task.oracle_ref)
        kind = oracle["goal_kind"]
        try:
            payload = _WITNESS[kind](self, oracle)
        except OasError as exc:
            return SemanticWitness(False, {"reason": str(exc)[:200]})
        return SemanticWitness(bool(payload.pop("ok")), payload)


# -- prove gates ---------------------------------------------------------------
def _gate_param(verifier, ast, re, ev):
    c = ast["contract"]
    errs_initial = contracts.check_param_assignment(c, ast["initial"])
    errs_ref = contracts.check_param_assignment(c, ast["reference"])
    ev["initial_errors"] = errs_initial[:8]
    ev["reference_errors"] = errs_ref[:8]
    if not errs_initial:
        return "TRIVIAL:initial_already_valid"
    if errs_ref:
        return "UNSOLVABLE:reference_fails"
    if ast["initial"] == ast["reference"]:
        return "TRIVIAL:no_change"
    if not _guards_hold(ast["guards"], ast["initial"], ast["reference"]):
        return "UNSOLVABLE:guard_inconsistent"
    return None


def _gate_body(verifier, ast, re, ev):
    schema = ast["contract"]["body_schema"]
    errs_initial = oas.validate_instance(schema, ast["initial"])
    errs_ref = oas.validate_instance(schema, ast["reference"])
    ev["initial_errors"] = errs_initial[:8]
    ev["reference_errors"] = errs_ref[:8]
    if not errs_initial:
        return "TRIVIAL:initial_already_valid"
    if errs_ref:
        return "UNSOLVABLE:reference_fails"
    if ast["initial"] == ast["reference"]:
        return "TRIVIAL:no_change"
    if not _guards_hold(ast["guards"], ast["initial"], ast["reference"]):
        return "UNSOLVABLE:guard_inconsistent"
    return None


def _gate_response(verifier, ast, re, ev):
    schema = ast["contract"]["response_schema"]
    errs_ref = oas.validate_instance(schema, ast["reference"])
    ev["reference_errors"] = errs_ref[:8]
    if errs_ref:
        return "UNSOLVABLE:reference_fails"
    if oas.validate_instance(schema, {}) == []:
        return "TRIVIAL:contract_without_teeth"
    if not _extras_hold(ast["constraints"].get("extra") or [], ast["reference"]):
        return "UNSOLVABLE:reference_violates_extra_constraints"
    return None


def _gate_selection(verifier, ast, re, ev):
    answer, relaxed = re["answer"], re["relaxed_answer"]
    ev["answer_size"] = len(answer)
    ev["relaxed_size"] = len(relaxed)
    ev["total_operations"] = len(re["contract"]["operations"])
    if sorted(answer) != sorted(ast["reference"]):
        return "UNSOLVABLE:reference_mismatch"
    if not answer:
        return "UNSOLVABLE:empty_answer"
    if len(answer) >= ev["total_operations"]:
        return "TRIVIAL:answer_is_universe"
    if set(answer) >= set(relaxed):
        return "NON_SEMANTIC:predicate_relaxation_invariant"
    return None


def _gate_security(verifier, ast, re, ev):
    if re["answer"] != ast["reference"]:
        return "UNSOLVABLE:reference_mismatch"
    ev["alternatives"] = len(re["answer"])
    if not re["answer"]:
        return "TRIVIAL:unsecured_operation"
    return None


def _gate_migration(verifier, ast, re, ev):
    mode = ast["coordinate"]["mode"]
    ca, cb = ast["contract_b"], ast["contract"]
    check_a, check_b = (_param_check(ca), _param_check(cb)) if mode == "params" \
        else (_body_check(ca), _body_check(cb))
    if check_a(ast["initial"]):
        return "UNSOLVABLE:initial_invalid_under_source"
    if not check_b(ast["initial"]):
        return "TRIVIAL:initial_already_valid_under_target"
    if check_b(ast["reference"]):
        return "UNSOLVABLE:reference_fails_target"
    if not _guards_hold(ast["guards"], ast["initial"], ast["reference"]):
        return "UNSOLVABLE:guard_inconsistent"
    return None


def _gate_mapping(verifier, ast, re, ev):
    target = ast["contract"]["target_schema"]
    mapping = ast["constraints"]["mapping"]
    if oas.validate_instance(target, ast["reference"]):
        return "UNSOLVABLE:reference_fails_target"
    if not oas.validate_instance(target, ast["initial"]):
        return "TRIVIAL:unmapped_instance_already_valid"
    if set(ast["reference"]) != set(mapping):
        return "UNSOLVABLE:key_set_mismatch"
    for t, s in mapping.items():
        if ast["reference"].get(t) != ast["initial"].get(s):
            return "UNSOLVABLE:mapping_value_mismatch"
    return None


def _param_check(contract):
    return lambda value: contracts.check_param_assignment(contract, value)


def _body_check(contract):
    return lambda value: oas.validate_instance(contract["body_schema"], value)


def _guards_hold(guards, initial, final) -> bool:
    for g in guards:
        name = g["name"]
        if not isinstance(initial, dict) or not isinstance(final, dict):
            return False
        if initial.get(name) != final.get(name):
            return False
    return True


def _extras_hold(extras, instance) -> bool:
    if not isinstance(instance, dict):
        return not extras
    for ex in extras:
        if ex["kind"] == "include_property" and ex["name"] not in instance:
            return False
        if ex["kind"] == "array_length" and len(instance.get(ex["name"]) or []) != ex["length"]:
            return False
    return True


_GATES = {
    "param_contract": _gate_param,
    "body_contract": _gate_body,
    "response_construction": _gate_response,
    "operation_selection": _gate_selection,
    "security_requirement": _gate_security,
    "version_migration": _gate_migration,
    "schema_mapping": _gate_mapping,
}

# -- terminal checks -------------------------------------------------------------
def _term_param(verifier, oracle, answer, checks):
    errs = contracts.check_param_assignment(oracle["contract"], answer)
    checks.append({"check": "param_contract", "pass": not errs, "errors": errs[:8]})
    _term_guards(oracle, answer, checks)


def _term_body(verifier, oracle, answer, checks):
    errs = oas.validate_instance(oracle["contract"]["body_schema"], answer)
    checks.append({"check": "body_schema", "pass": not errs, "errors": errs[:8]})
    _term_guards(oracle, answer, checks)


def _term_response(verifier, oracle, answer, checks):
    errs = oas.validate_instance(oracle["contract"]["response_schema"], answer)
    checks.append({"check": "response_schema", "pass": not errs, "errors": errs[:8]})
    extras = oracle["constraints"].get("extra") or []
    checks.append({"check": "extra_constraints", "pass": _extras_hold(extras, answer)})


def _term_selection(verifier, oracle, answer, checks):
    ok = isinstance(answer, list) and sorted(map(str, answer)) == sorted(oracle["reference"])
    checks.append({"check": "exact_operation_set", "pass": ok})


def _term_security(verifier, oracle, answer, checks):
    norm = contracts.normalized_security(answer)
    checks.append({"check": "exact_security_alternatives", "pass": norm == oracle["reference"]})


def _term_migration(verifier, oracle, answer, checks):
    mode = oracle["coordinate"]["mode"]
    if mode == "params":
        _term_param(verifier, oracle, answer, checks)
    else:
        _term_body(verifier, oracle, answer, checks)


def _term_mapping(verifier, oracle, answer, checks):
    target = oracle["contract"]["target_schema"]
    mapping = oracle["constraints"]["mapping"]
    errs = oas.validate_instance(target, answer)
    checks.append({"check": "target_schema", "pass": not errs, "errors": errs[:8]})
    exact = isinstance(answer, dict) and set(answer) == set(mapping)
    checks.append({"check": "exact_keys", "pass": exact})
    ok_values = isinstance(answer, dict) and all(
        answer.get(t) == (oracle["initial"] or {}).get(s) for t, s in mapping.items())
    checks.append({"check": "mapped_values", "pass": ok_values})


def _term_guards(oracle, answer, checks):
    for g in oracle["guards"]:
        name = g["name"]
        ok = isinstance(answer, dict) and answer.get(name) == (oracle["initial"] or {}).get(name)
        checks.append({"check": "guard", "name": name, "pass": ok})


_TERMINAL = {
    "param_contract": _term_param,
    "body_contract": _term_body,
    "response_construction": _term_response,
    "operation_selection": _term_selection,
    "security_requirement": _term_security,
    "version_migration": _term_migration,
    "schema_mapping": _term_mapping,
}

# -- semantic witnesses ----------------------------------------------------------
def _relaxed_contract(contract: dict[str, Any], relax: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(contract)
    if "remove_required" in relax:
        for p in out.get("parameters", []):
            if p["name"] in relax["remove_required"]:
                p["required"] = False
    if "replace_schema" in relax:
        for p in out.get("parameters", []):
            if p["name"] in relax["replace_schema"]:
                p["schema"] = {}
    if "body_remove_required" in relax:
        schema = out.get("body_schema", {})
        schema["required"] = [r for r in schema.get("required", [])
                              if r not in relax["body_remove_required"]]
    if "body_replace_schema" in relax:
        props = out.get("body_schema", {}).get("properties", {})
        for name in relax["body_replace_schema"]:
            if name in props:
                props[name] = {}
    if "replace_whole_schema" in relax:
        for key in ("body_schema", "response_schema", "target_schema"):
            if key in out:
                out[key] = {}
    return out


def _witness_repair(verifier, oracle) -> dict[str, Any]:
    relax = dict(oracle["witness"].get("relax") or {})
    if "replace_schema" in relax and oracle["goal_kind"] == "response_construction":
        relax = {"replace_whole_schema": relax}
    relaxed = _relaxed_contract(oracle["contract"], relax)
    if oracle["goal_kind"] == "security_requirement":
        real = contracts.normalized_security(oracle["contract"].get("operation_security")
                                             or oracle["contract"].get("global_security"))
        return {"ok": bool(real), "mode": "strip_security",
                "real_alternatives": len(real), "relaxed_alternatives": 0}
    initial = oracle["initial"]
    if oracle["contract"]["kind"] == "param_contract":
        real_errs = contracts.check_param_assignment(oracle["contract"], initial)
        relaxed_errs = contracts.check_param_assignment(relaxed, initial)
    else:
        schema_key = "body_schema" if "body_schema" in oracle["contract"] else "response_schema"
        real_errs = oas.validate_instance(oracle["contract"][schema_key], initial)
        relaxed_errs = oas.validate_instance(relaxed[schema_key], initial)
    return {"ok": bool(real_errs) and not relaxed_errs, "mode": "relaxed_contract",
            "real_error_count": len(real_errs), "relaxed_error_count": len(relaxed_errs)}


def _witness_selection(verifier, oracle) -> dict[str, Any]:
    ast_like = {"goal_kind": "operation_selection", "unit_id": oracle["unit_id"],
                "unit_digest": oracle["unit_digest"], "coordinate": oracle["coordinate"],
                "constraints": oracle["constraints"]}
    feats = [contracts.op_features(s)
             for s in oas.iter_operations(verifier._load(ast_like))]
    answer = contracts.apply_predicate(feats, oracle["constraints"]["predicate"])
    relaxed = contracts.apply_predicate(feats, oracle["constraints"]["predicate"], relaxed=True)
    return {"ok": set(answer) < set(relaxed), "mode": "relaxed_predicate",
            "answer_size": len(answer), "relaxed_size": len(relaxed)}


def _witness_migration(verifier, oracle) -> dict[str, Any]:
    mode = oracle["coordinate"]["mode"]
    check_a = _param_check(oracle["contract_b"]) if mode == "params" else _body_check(oracle["contract_b"])
    check_b = _param_check(oracle["contract"]) if mode == "params" else _body_check(oracle["contract"])
    old_ok = not check_a(oracle["initial"])
    new_ok = not check_b(oracle["initial"])
    return {"ok": old_ok and not new_ok, "mode": "version_contrast",
            "valid_under_old": old_ok, "valid_under_new": new_ok}


def _witness_mapping(verifier, oracle) -> dict[str, Any]:
    target = oracle["contract"]["target_schema"]
    unmapped_errs = oas.validate_instance(target, oracle["initial"])
    mapped_errs = oas.validate_instance(target, oracle["reference"])
    return {"ok": bool(unmapped_errs) and not mapped_errs, "mode": "unmapped_fails",
            "unmapped_error_count": len(unmapped_errs)}


_WITNESS = {
    "param_contract": _witness_repair,
    "body_contract": _witness_repair,
    "response_construction": _witness_repair,
    "security_requirement": _witness_repair,
    "operation_selection": _witness_selection,
    "version_migration": _witness_migration,
    "schema_mapping": _witness_mapping,
}
