"""Verifier extensions for the L13 20K expansion.

Subclasses the legacy L13 verifier and adds the new contract kinds introduced
by the expansion families:

  status_selection         exact set of status codes accepting an instance
  media_type_selection     exact set of request media types accepting an instance
  header_construction      response-header value vs. header schema
  version_migration/mode=response  response payload across pinned versions

It also adds two body-contract witness relaxations used by x02/x03
(`body_drop_additional_properties`, `body_nested_remove_required`).

All answers are recomputed from pinned spec bytes; the stored oracle is never
trusted for a verdict.
"""
from __future__ import annotations

import copy
from typing import Any

from benchmark.plugins.source.l13 import oas
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13_exp import extract
from benchmark.plugins.verifier.l13.openapi_verifier import (
    ENGINE, OpenAPIContractVerifier,
)
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

NEW_KINDS = ("status_selection", "media_type_selection", "header_construction")
SUBMIT_KEY = {
    "status_selection": "status_codes",
    "media_type_selection": "media_types",
    "header_construction": "header_value",
}
NEW_BODY_RELAX = ("body_drop_additional_properties", "body_nested_remove_required")


def _guards_hold(guards, initial, final) -> bool:
    for g in guards:
        name = g["name"]
        if not isinstance(initial, dict) or not isinstance(final, dict):
            return False
        if initial.get(name) != final.get(name):
            return False
    return True


def _is_new(ast: dict[str, Any]) -> bool:
    kind = ast["goal_kind"]
    return kind in NEW_KINDS or (
        kind == "version_migration" and ast["coordinate"].get("mode") == "response")


class ExpansionVerifier(OpenAPIContractVerifier):
    """NativeVerifierPlugin for the additive L13 expansion families."""

    # -- contract re-derivation -------------------------------------------------
    def rederive(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["goal_kind"]
        coord = ast["coordinate"]
        if kind in NEW_KINDS:
            return self._rederive_new(ast)
        if kind == "version_migration" and coord.get("mode") == "response":
            return self._rederive_response_migration(ast)
        return super().rederive(ast)

    def _rederive_new(self, ast: dict[str, Any]) -> dict[str, Any]:
        kind = ast["goal_kind"]
        coord = ast["coordinate"]
        spec = self._load(ast)
        site = self._site(spec, coord)
        if kind == "status_selection":
            schemas = extract.response_schema_map(site, spec)
            inst = ast.get("initial")
            answer = sorted(k for k, s in schemas.items()
                            if oas.validate_instance(s, inst) == [])
            return {"contract": {"kind": "status_selection", "response_schemas": schemas},
                    "answer": answer}
        if kind == "media_type_selection":
            schemas = extract.request_media_schemas(site, spec)
            inst = ast.get("initial")
            answer = sorted(k for k, s in schemas.items()
                            if oas.validate_instance(s, inst) == [])
            return {"contract": {"kind": "media_type_selection", "media_schemas": schemas},
                    "answer": answer}
        if kind == "header_construction":
            code = coord["code"]
            name = coord["header"]
            headers = extract.response_headers(site, spec, code)
            if name not in headers:
                raise OasError("header %r absent for %s" % (name, code))
            return {"contract": {"kind": "header_construction", "header_name": name,
                                 "header_schema": headers[name]}}
        raise OasError("unknown expansion kind %r" % kind)

    def _rederive_response_migration(self, ast: dict[str, Any]) -> dict[str, Any]:
        coord = ast["coordinate"]
        code = coord["code"]
        spec = self._load(ast)
        spec_a = self._load(ast, "_b")
        site = self._site(spec, coord)
        site_a = self._site(spec_a, coord)
        if code not in site.response_schemas or code not in site_a.response_schemas:
            raise OasError("response code %r absent in one version" % code)
        sch = oas.normalize_schema(site.response_schemas[code], spec, direction="response")
        sch_a = oas.normalize_schema(site_a.response_schemas[code], spec_a,
                                     direction="response")
        return {
            "contract": {"kind": "response_construction", "code": code,
                         "response_schema": sch},
            "contract_b": {"kind": "response_construction", "code": code,
                           "response_schema": sch_a},
        }

    # -- prove_exists -----------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        if not _is_new(ast):
            return super().prove_exists(grounded_task)
        ev: dict[str, Any] = {"engine": ENGINE, "goal_kind": ast["goal_kind"]}
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
        kind = ast["goal_kind"]
        if kind == "status_selection":
            gate = self._gate_selection_set(ast, re, ev, "response_schemas")
        elif kind == "media_type_selection":
            gate = self._gate_selection_set(ast, re, ev, "media_schemas")
        elif kind == "header_construction":
            gate = self._gate_header(ast)
        else:
            gate = self._gate_response_migration(ast)
        if gate is not None:
            return ProofOrCounterexample(False, ev, gate)
        return ProofOrCounterexample(True, ev)

    @staticmethod
    def _gate_selection_set(ast, re, ev, key) -> str | None:
        answer = re["answer"]
        ev["answer_size"] = len(answer)
        ev["universe_size"] = len(re["contract"][key])
        if sorted(map(str, answer)) != sorted(map(str, ast["reference"])):
            return "UNSOLVABLE:reference_mismatch"
        if not answer:
            return "UNSOLVABLE:empty_answer"
        if len(answer) >= ev["universe_size"]:
            return "TRIVIAL:answer_is_universe"
        return None

    @staticmethod
    def _gate_header(ast) -> str | None:
        schema = ast["contract"]["header_schema"]
        if oas.validate_instance(schema, ast["reference"]):
            return "UNSOLVABLE:reference_fails"
        bad = (ast.get("witness") or {}).get("bad_value")
        if oas.validate_instance(schema, bad) == []:
            return "TRIVIAL:contract_without_teeth"
        return None

    @staticmethod
    def _gate_response_migration(ast) -> str | None:
        old = ast["contract_b"]["response_schema"]
        new = ast["contract"]["response_schema"]
        if oas.validate_instance(old, ast["initial"]):
            return "UNSOLVABLE:initial_invalid_under_source"
        if oas.validate_instance(new, ast["initial"]) == []:
            return "TRIVIAL:initial_already_valid_under_target"
        if oas.validate_instance(new, ast["reference"]):
            return "UNSOLVABLE:reference_fails_target"
        if not _guards_hold(ast["guards"], ast["initial"], ast["reference"]):
            return "UNSOLVABLE:guard_inconsistent"
        return None

    # -- terminal replay --------------------------------------------------------
    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        oracle = self.load_oracle(task.oracle_ref)
        kind = oracle["goal_kind"]
        mode = oracle["coordinate"].get("mode")
        if kind not in NEW_KINDS and not (kind == "version_migration" and mode == "response"):
            return super().verify_terminal(task, submitted_state_or_answer)
        submitted = submitted_state_or_answer or {}
        key = SUBMIT_KEY.get(kind, "instance")
        answer = submitted.get(key)
        checks: list[dict[str, Any]] = []
        if answer is None:
            return NativeVerdict(False, {"error": "missing submission key %s" % key})
        if kind in ("status_selection", "media_type_selection"):
            ok = isinstance(answer, list) and sorted(map(str, answer)) == sorted(
                map(str, oracle["reference"]))
            checks.append({"check": kind, "pass": ok})
        elif kind == "header_construction":
            errs = oas.validate_instance(oracle["contract"]["header_schema"], answer)
            checks.append({"check": "header_schema", "pass": not errs, "errors": errs[:8]})
        else:
            errs = oas.validate_instance(
                oracle["contract"]["response_schema"], answer)
            checks.append({"check": "response_schema", "pass": not errs, "errors": errs[:8]})
            for g in oracle["guards"]:
                name = g["name"]
                ok = isinstance(answer, dict) and answer.get(name) == (
                    oracle.get("initial") or {}).get(name)
                checks.append({"check": "guard", "name": name, "pass": ok})
        return NativeVerdict(all(c["pass"] for c in checks),
                             {"checks": checks, "engine": ENGINE})

    # -- semantic witnesses -----------------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        oracle = self.load_oracle(task.oracle_ref)
        kind = oracle["goal_kind"]
        if kind in NEW_KINDS:
            return self._witness_new(oracle)
        if kind == "version_migration" and oracle["coordinate"].get("mode") == "response":
            return self._witness_response_migration(oracle)
        relax = (oracle.get("witness") or {}).get("relax") or {}
        if kind == "body_contract" and any(k in relax for k in NEW_BODY_RELAX):
            return self._witness_body_relax(oracle)
        return super().certify_semantic_relevance(task)

    @staticmethod
    def _witness_new(oracle) -> SemanticWitness:
        kind = oracle["goal_kind"]
        if kind == "header_construction":
            schema = oracle["contract"]["header_schema"]
            bad = (oracle.get("witness") or {}).get("bad_value")
            real = oas.validate_instance(schema, bad)
            relaxed = oas.validate_instance({}, bad)
            ok = bool(real) and not relaxed
            return SemanticWitness(ok, {"ok": ok, "mode": "header_relax",
                                        "real_error_count": len(real),
                                        "relaxed_error_count": len(relaxed)})
        key = "response_schemas" if kind == "status_selection" else "media_schemas"
        schemas = oracle["contract"][key]
        inst = oracle.get("initial")
        matched = sorted(k for k, s in schemas.items()
                         if oas.validate_instance(s, inst) == [])
        target = (oracle.get("witness") or {}).get(
            "code" if kind == "status_selection" else "media")
        if target not in schemas:
            return SemanticWitness(False, {"ok": False, "mode": "missing_target"})
        relaxed = copy.deepcopy(schemas[target])
        relaxed["required"] = []
        real = oas.validate_instance(schemas[target], inst)
        relaxed_errs = oas.validate_instance(relaxed, inst)
        ok = target not in matched and bool(real) and not relaxed_errs
        return SemanticWitness(ok, {"ok": ok, "mode": "relax_other_schema",
                                    "target": target, "matched": matched,
                                    "real_error_count": len(real)})

    @staticmethod
    def _witness_response_migration(oracle) -> SemanticWitness:
        old = oracle["contract_b"]["response_schema"]
        new = oracle["contract"]["response_schema"]
        old_ok = oas.validate_instance(old, oracle["initial"]) == []
        new_ok = oas.validate_instance(new, oracle["initial"]) == []
        ok = old_ok and not new_ok
        return SemanticWitness(ok, {"ok": ok, "mode": "version_contrast",
                                    "valid_under_old": old_ok,
                                    "valid_under_new": new_ok})

    @staticmethod
    def _witness_body_relax(oracle) -> SemanticWitness:
        schema = oracle["contract"]["body_schema"]
        relax = oracle["witness"]["relax"]
        relaxed = copy.deepcopy(schema)
        if "body_drop_additional_properties" in relax:
            relaxed["additionalProperties"] = True
        if "body_nested_remove_required" in relax:
            path = relax["body_nested_remove_required"]["path"]
            node = relaxed
            for step in path[:-1]:
                node = (node.get("properties") or {}).get(step) or {}
            node["required"] = [r for r in node.get("required") or [] if r != path[-1]]
        real = oas.validate_instance(schema, oracle["initial"])
        relax_errs = oas.validate_instance(relaxed, oracle["initial"])
        ok = bool(real) and not relax_errs
        return SemanticWitness(ok, {"ok": ok, "mode": "relaxed_contract",
                                    "real_error_count": len(real),
                                    "relaxed_error_count": len(relax_errs)})


__all__ = ["ExpansionVerifier", "NEW_KINDS", "SUBMIT_KEY"]