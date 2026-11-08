"""Answer-shape normalisation for the L13 (OpenAPI contract) adapter.

The L13 verifiers expect a *submission wrapper* whose single key depends on the
task's goal kind, e.g. ``{"body": ...}`` for a body contract, ``{"status_codes":
[...]}`` for a status-selection expansion, ``{"header_value": "..."}`` for a
header-construction expansion.  Some models instead return the raw payload
that belongs under that key (a raw body object, a bare status-code list, a bare
header value).  The native verifier then calls ``.get`` on that raw value and
raises ``AttributeError``/``TypeError`` -- a harness shape mismatch, not a lane
verdict.

This module normalises *only* unambiguous raw payloads:

* an already-wrapped dict (exactly one key, the expected one) is left
  untouched; a raw dict that merely *contains* that key is still wrapped;
* a list is wrapped under the expected key (a list can never be a wrapper);
* a raw parameters/body/instance object is wrapped under the expected key;
* a string is wrapped under ``header_value`` (that key holds one scalar);
* for ``body`` / ``instance`` keys a scalar is wrapped only when the oracle's
  own contract schema accepts that JSON primitive type.

Anything else is left verbatim; the adapter catches the verifier's shape error
and records a scored failure instead of letting it escape as ``ERROR``.

No answer content is invented: every wrapped value is exactly the model's
payload, and the lane verifier still computes the verdict from its pinned
oracle.
"""
from __future__ import annotations

from typing import Any

#: wrapper key per new/legacy L13 goal kind (mirrors both native verifiers).
_WRAPPER_KEY: dict[str, str] = {
    "param_contract": "parameters",
    "body_contract": "body",
    "response_construction": "instance",
    "schema_mapping": "instance",
    "operation_selection": "operation_keys",
    "security_requirement": "alternatives",
    "status_selection": "status_codes",
    "media_type_selection": "media_types",
    "header_construction": "header_value",
}

#: keys whose value is a JSON object/list instance (as opposed to a scalar or a
#: list of identifiers); used for the scalar-schema test.
_INSTANCE_KEYS = frozenset({"body", "instance"})

#: keys that accept a raw JSON object payload (the wrapper is a one-key dict).
_OBJECT_KEYS = frozenset({"parameters", "body", "instance"})

_JSON_TYPE = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}


def expected_wrapper_key(oracle_payload: dict[str, Any]) -> str | None:
    """Return the submission-wrapper key the verifier will read."""
    kind = oracle_payload.get("goal_kind")
    if kind == "version_migration":
        mode = (oracle_payload.get("coordinate") or {}).get("mode")
        # ExpansionVerifier handles response-mode migration itself and reads
        # ``instance``; the legacy path reads ``parameters`` for params mode
        # and ``body`` for every other mode.
        if mode == "response":
            return "instance"
        if mode == "params":
            return "parameters"
        return "body"
    if isinstance(kind, str):
        return _WRAPPER_KEY.get(kind)
    return None


def _schema_types(schema: dict[str, Any]) -> set[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return {t for t in declared if isinstance(t, str)}
    # No explicit type: JSON Schema treats it as unconstrained.
    return {"__any__"}


def _instance_schema(oracle_payload: dict[str, Any]) -> dict[str, Any] | None:
    contract = oracle_payload.get("contract")
    if not isinstance(contract, dict):
        return None
    for key in ("body_schema", "response_schema", "target_schema"):
        schema = contract.get(key)
        if isinstance(schema, dict):
            return schema
    return None


def _scalar_matches_schema(value: Any, schema: dict[str, Any] | None) -> bool:
    if schema is None:
        return False
    json_type = _JSON_TYPE.get(type(value))
    if json_type is None:
        return False
    types = _schema_types(schema)
    if "__any__" in types:
        return True
    if json_type in types:
        return True
    # JSON Schema "number" also accepts integers.
    return json_type == "integer" and "number" in types


def normalize_answer(answer: Any, oracle_payload: dict[str, Any]
                     ) -> tuple[Any, dict[str, Any] | None]:
    """Return ``(normalized_answer, note)``.

    ``note`` is ``None`` when the answer already has the verifier shape or when
    the shape is not safely normalisable.  It carries non-answer provenance
    only (original type / wrapper key) so the row can record what happened.
    """
    key = expected_wrapper_key(oracle_payload)
    if key is None:
        return answer, None

    # Already a submission wrapper: a dict whose *only* key is the expected
    # wrapper.  Extra keys mean this is the raw payload (which may itself
    # contain a property named like the wrapper), not a wrapper.
    if (isinstance(answer, dict) and len(answer) == 1 and key in answer):
        return answer, None

    note = {
        "wrapper_key": key,
        "original_type": type(answer).__name__,
    }

    # A list can never be a wrapper value; it is the model's raw payload.
    if isinstance(answer, list):
        return {key: answer}, note

    if isinstance(answer, dict) and key in _OBJECT_KEYS:
        return {key: answer}, note

    if isinstance(answer, str):
        # header_value is a single scalar by definition; body/instance strings
        # are accepted only when the oracle's own schema allows a string.
        if key == "header_value":
            return {key: answer}, note
        if key in _INSTANCE_KEYS and _scalar_matches_schema(
                answer, _instance_schema(oracle_payload)):
            return {key: answer}, note
        return answer, None

    if isinstance(answer, (int, float, bool)) and key in _INSTANCE_KEYS:
        if _scalar_matches_schema(answer, _instance_schema(oracle_payload)):
            return {key: answer}, note
        return answer, None

    return answer, None


__all__ = ["expected_wrapper_key", "normalize_answer"]