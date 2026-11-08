"""L13 answer-wrapper normalisation: both normalisers must agree with the
native verifier's ``submitted.get(<key>)`` read.

Covers the ``version_migration`` coordinate modes (params/response/body), raw
payloads that happen to contain the wrapper key as a legitimate property, and
already-wrapped submissions.
"""
from __future__ import annotations

import pytest

from benchmark.eval_20k import answer_contracts
from benchmark.eval_20k.adapters import l13_answer
from benchmark.eval_20k.tasks import TaskRecord


def _task(family: str, *, mode: str | None = None) -> TaskRecord:
    public = {"coordinate": {"mode": mode}} if mode is not None else {}
    return TaskRecord(
        task_id="L13-%s" % family,
        instance_id="i-%s" % family,
        lane_id="L13",
        family_id=family,
        base_goal_id="b-%s" % family,
        task_regime="application",
        complexity="low",
        semantic_profile="openapi/1",
        oracle_handle="or-%s" % family,
        public=public,
        raw={},
    )


def _oracle(kind: str, *, mode: str | None = None) -> dict:
    payload: dict = {"goal_kind": kind, "contract": {}}
    if mode is not None:
        payload["coordinate"] = {"mode": mode}
    return payload


@pytest.mark.parametrize(
    "mode,key",
    [("params", "parameters"), ("response", "instance"), ("body", "body")],
)
def test_version_migration_wrapper_key_matches_native(mode: str, key: str) -> None:
    family = "l13-f05-api-version-migration"
    assert answer_contracts._l13_wrapper_key(
        family, {"coordinate": {"mode": mode}}) == key
    assert l13_answer.expected_wrapper_key(
        _oracle("version_migration", mode=mode)) == key


@pytest.mark.parametrize(
    "family,kind,mode,answer,expected",
    [
        (
            "l13-f05-api-version-migration", "version_migration", "body",
            {"name": "v2"}, {"body": {"name": "v2"}},
        ),
        (
            "l13-f05-api-version-migration", "version_migration", "body",
            [1, 2], {"body": [1, 2]},
        ),
        (
            "l13-f05-api-version-migration", "version_migration", "params",
            {"limit": 5}, {"parameters": {"limit": 5}},
        ),
        (
            "l13-x08-response-version-migration", "version_migration",
            "response", [{"id": 1}], {"instance": [{"id": 1}]},
        ),
        (
            "l13-x02-additional-property-body-repair", "body_contract", None,
            {"body": {"x": 1}, "id": 2},
            {"body": {"body": {"x": 1}, "id": 2}},
        ),
        (
            "l13-f01-required-parameter-repair", "param_contract", None,
            {"parameters": "not-the-wrapper", "limit": 5},
            {"parameters": {"parameters": "not-the-wrapper", "limit": 5}},
        ),
        (
            "l13-x04-response-status-selection", "status_selection", None,
            ["200", "404"], {"status_codes": ["200", "404"]},
        ),
        (
            "l13-x02-additional-property-body-repair", "body_contract", None,
            {"body": {"x": 1}}, {"body": {"x": 1}},
        ),
    ],
)
def test_l13_raw_payload_wrapped_consistently(
        family: str, kind: str, mode: str | None, answer: object,
        expected: object) -> None:
    task = _task(family, mode=mode)
    normalized_contracts, _ = answer_contracts.normalize_answer(task, answer)
    normalized_adapter, _ = l13_answer.normalize_answer(
        answer, _oracle(kind, mode=mode))
    assert normalized_contracts == expected
    assert normalized_adapter == expected


def test_existing_single_key_wrapper_is_not_rewrapped() -> None:
    answer = {"body": {"x": 1}}
    task = _task("l13-x02-additional-property-body-repair")
    normalized_contracts, note = answer_contracts.normalize_answer(task, answer)
    assert normalized_contracts is answer
    assert note is None
    normalized_adapter, note = l13_answer.normalize_answer(
        answer, _oracle("body_contract"))
    assert normalized_adapter is answer
    assert note is None