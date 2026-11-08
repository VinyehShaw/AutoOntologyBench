"""Fixtures for the eval_20k harness tests.

``fixtures/tasks_smoke.jsonl`` holds three REAL L05 task rows from the
aob-20000-candidate-dev1 build (public projection + split="test"), and
``fixtures/oracle/l05/`` holds their REAL private oracle payloads copied from
the bank staging tree. Reference/negative answers are derived at test time
from those oracle payloads via the lane adapter — no answer is invented in
the test code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TASKS_SMOKE = FIXTURES / "tasks_smoke.jsonl"
ORACLE_ROOT = FIXTURES / "oracle"

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = REPO_ROOT / "bank" / "builds" / "aob-20000-candidate-dev1"


@pytest.fixture(scope="session")
def smoke_task_rows() -> list[dict]:
    return [json.loads(line) for line in
            TASKS_SMOKE.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="session")
def oracle_payloads() -> dict[str, dict]:
    payloads = {}
    for path in sorted(ORACLE_ROOT.glob("l05/*.json")):
        payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return payloads
