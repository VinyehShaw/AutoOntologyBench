"""L08 evidence re-check CLI: re-run native verification for a banked instance.

Usage (from repo root, the exact command stored in each qualified record's
``native_evidence.commands``):

    .venv/bin/python -m benchmark.plugins.verifier.l08.recheck \
        --bank bank/staging/l08 --instance <instance_id>

Reads the instance's evidence job file (ELM + synthetic FHIR panel), executes
it with the vendored cql-execution engine, recomputes it with the independent
Python path, and compares both against the private oracle's reference answer.
Prints a JSON verdict; exit code 0 iff every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.plugins.verifier.l08 import recompute
from benchmark.plugins.verifier.l08.elm import build_elm
from benchmark.plugins.verifier.l08.runner import NodeEngineRunner, engine_versions


def find_evidence(bank: Path, instance_id: str) -> Path:
    for cand in sorted(bank.glob("evidence/*/%s.json" % instance_id)):
        return cand
    raise FileNotFoundError("no evidence job for instance %s under %s" % (instance_id, bank))


def find_oracle(bank: Path, instance_id: str) -> Path:
    for cand in sorted(bank.glob("oracle/*/%s.json" % instance_id)):
        return cand
    raise FileNotFoundError("no oracle record for instance %s under %s" % (instance_id, bank))


def recheck(bank: Path, instance_id: str) -> dict[str, Any]:
    evidence_path = find_evidence(bank, instance_id)
    oracle_path = find_oracle(bank, instance_id)
    job = json.loads(evidence_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    rule = job["rule"]
    elm = job["library"]
    patients = job["patients"]

    runner = NodeEngineRunner()
    engine = runner.evaluate(elm, patients, job_id="recheck-%s" % instance_id)
    py = recompute.recompute(rule, patients)
    engine_answer = sorted(pid for pid, v in engine.items() if v)
    py_answer = sorted(pid for pid, v in py.items() if v)
    reference = sorted(oracle["reference_answer"])

    # ELM in the evidence file must be exactly what build_elm emits for the rule
    elm_rebuild_match = elm == build_elm(rule)

    checks = {
        "engine_vs_python": engine_answer == py_answer,
        "engine_vs_oracle": engine_answer == reference,
        "python_vs_oracle": py_answer == reference,
        "elm_rebuild_match": elm_rebuild_match,
        "panel_size": len(patients),
        "answer_size": len(reference),
    }
    return {
        "instance_id": instance_id,
        "evidence_path": str(evidence_path),
        "oracle_path": str(oracle_path),
        "engine": engine_versions(),
        "engine_answer": engine_answer,
        "python_answer": py_answer,
        "oracle_reference_answer": reference,
        "checks": checks,
        "verdict": "pass" if all(v for k, v in checks.items() if isinstance(v, bool)) else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L08 native evidence re-check")
    ap.add_argument("--bank", default="bank/staging/l08")
    ap.add_argument("--instance", required=True)
    args = ap.parse_args(argv)
    try:
        result = recheck(Path(args.bank), args.instance)
    except (FileNotFoundError, KeyError) as exc:
        print(json.dumps({"verdict": "error", "error": str(exc)}))
        return 2
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
