"""L16 per-task reverify CLI: recompute the oracle answer from the XML store.

Recomputes the reference answer for the task recorded in the given oracle JSON
from the pinned XML-parser store and requires exact sha256 agreement with the
recorded expected answer. This is the per-task repro command cited in each
task's ``native_evidence.commands``.

CLI:
  .venv/bin/python -m benchmark.plugins.verifier.l16.reverify \
      --pin-root external_knowledge/pinned/uniprot/uniprot-2026_03-sprot \
      --oracle bank/staging_20k/l16/oracle/<oracle-handle>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="L16 reverify one oracle record")
    ap.add_argument("--pin-root", required=True)
    ap.add_argument("--oracle", required=True)
    args = ap.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)

    from benchmark.plugins.verifier.l16 import reference as ref

    t0 = time.time()
    oracle_path = Path(args.oracle)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    family = oracle["family"]
    params = oracle["params"]
    expected = tuple(sorted(set(oracle["expected_answer"])))
    root = Path(args.pin_root)
    meta = dict(ref._db(root).execute(
        "SELECT key, value FROM meta").fetchall())
    got = ref.reference_answer(root, family, params)
    got_sha = _sha(sorted(set(got)))
    want_sha = oracle.get("expected_answer_sha256") or _sha(sorted(expected))
    ok = got_sha == want_sha and set(got) == set(expected)
    out = {
        "task_id": oracle.get("task_id"),
        "family": family,
        "verdict": "PASS" if ok else "FAIL",
        "engine": "sqlite3 over l16_uniprot_xml.sqlite "
                  f"({meta.get('parser', '?')}, release "
                  f"{meta.get('release_id', '?')})",
        "store_source_sha256": meta.get("source_sha256", ""),
        "expected_sha256": want_sha,
        "recomputed_sha256": got_sha,
        "answer_count": len(got),
        "witness_ok": bool((oracle.get("witness") or {}).get("ok")),
        "wall_seconds": round(time.time() - t0, 3),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
