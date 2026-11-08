"""L14 native-evidence reverification CLI.

Recomputes the clean-parser reference answer for one shipped L14 task from
the pinned ChEMBL 37 SQLite bytes and compares it against the recorded oracle
answer. This is the exact command embedded in every QUALIFIED record's
``native_evidence.commands``; it is independently re-runnable from the repo
root with only ``.venv/bin/python``:

  .venv/bin/python -m benchmark.plugins.verifier.l14.reverify \
      --pin-root external_knowledge/pinned/chembl/chembl-37 \
      --oracle bank/staging_20k/l14/oracle/<oracle-file>.json

Exit code 0 iff the recomputed verdict is PASS and matches the recorded
reference verdict and answer hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L14 native evidence reverify")
    ap.add_argument("--pin-root", required=True)
    ap.add_argument("--oracle", required=True,
                    help="path to the task's oracle JSON (evaluator-private)")
    args = ap.parse_args(argv)

    from benchmark.plugins.source.l14.kb import load_kb
    from benchmark.plugins.verifier.l14 import plugin as verifier

    oracle_path = Path(args.oracle)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    family = oracle["family"]
    params = oracle["params"]
    expected = tuple(sorted(set(oracle["expected_answer"])))

    kb = load_kb(args.pin_root)
    ev = verifier.verify_reference(Path(args.pin_root), family, params,
                                   expected, kb)

    answer_sha = _sha(sorted(set(expected)))
    ok = (ev.get("verdict") == "PASS"
          and oracle.get("reference", {}).get("verdict") == "PASS"
          and ev.get("answer_sha256", answer_sha) == answer_sha
          and oracle.get("expected_answer_sha256") == answer_sha)
    out = {
        "task_id": oracle.get("task_id"),
        "family": family,
        "recomputed_verdict": ev.get("verdict"),
        "recomputed_answer_sha256": ev.get("answer_sha256", answer_sha),
        "recorded_answer_sha256": oracle.get("expected_answer_sha256"),
        "recorded_verdict": oracle.get("reference", {}).get("verdict"),
        "engine": ev.get("engine"),
        "db_manifest_sha": ev.get("db_manifest_sha"),
        "match": ok,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
