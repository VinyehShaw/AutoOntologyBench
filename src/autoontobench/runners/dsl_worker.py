"""Sandboxed ``gk-dsl/1`` worker (owner: W13). Runs ONLY inside the child
process spawned by ``autoontobench.runners.dsl_consumer`` (or its documented
in-process fallback). Stdlib + ``dsl_program`` only — no rdflib/pySHACL — so
the child is fast and the interpreter under test is exactly this fixed code.

Modes:
- ``load``  : parse the ``code-dsl/1`` bundle envelope, fail-closed file-set
              gate (``program.json`` only), schema-validate the program.
- ``check`` : additionally evaluate ``(state, action)`` — the SAME serialized
              facts the shacl-rule/1 path consumes — via ``evaluate_program``.

Never touches native truth; prints exactly one ``AOB_RESULT_JSON:`` line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from autoontobench.runners.dsl_program import evaluate_program
from autoontobench.runners.dsl_schema import ProgramRejected, validate_program
from autoontobench.runners.ontology_bundle import BUNDLE_SCHEMA, MAX_BUNDLE_BYTES

RESULT_MARKER = "AOB_RESULT_JSON:"
DSL_PROFILE = "code-dsl/1"
DSL_FILES = ("program.json",)


class DslBundleRejected(ValueError):
    """The code-dsl/1 bundle envelope failed structural validation (fail-closed)."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("dsl bundle rejected: " + "; ".join(self.reasons))


def parse_dsl_bundle(bundle_bytes: bytes) -> dict[str, str]:
    """The code-dsl/1 file-set gate (mirrors ontology_bundle.parse_bundle)."""
    if len(bundle_bytes) > 4 * 1024 * 1024:
        raise DslBundleRejected(["bundle: %d bytes exceed cap" % len(bundle_bytes)])
    try:
        doc: Any = json.loads(bundle_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DslBundleRejected(["bundle: not a JSON document: %s" % exc]) from exc
    if not isinstance(doc, dict) or doc.get("bundle") != BUNDLE_SCHEMA:
        raise DslBundleRejected(["bundle: schema must be %r" % BUNDLE_SCHEMA])
    if doc.get("profile") != DSL_PROFILE:
        raise DslBundleRejected(["bundle: profile must be %r" % DSL_PROFILE])
    files = doc.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise DslBundleRejected(["bundle: files must be an object of name -> utf8 content"])
    unknown = sorted(set(files) - set(DSL_FILES))
    if unknown:
        raise DslBundleRejected(["bundle: unexpected files %s (fail-closed)" % unknown])
    missing = [name for name in DSL_FILES if name not in files]
    if missing:
        raise DslBundleRejected(["bundle: missing required files %s" % missing])
    return dict(files)


def run_job(mode: str, job: Mapping[str, Any]) -> dict[str, Any]:
    import base64

    files = parse_dsl_bundle(base64.b64decode(job["bundle_b64"]))
    try:
        program = validate_program(json.loads(files["program.json"]))
    except json.JSONDecodeError as exc:
        return {"ok": False, "reasons": ["program.json: not valid JSON: %s" % exc]}
    except ProgramRejected as exc:
        return {"ok": False, "reasons": exc.reasons}
    if mode == "load":
        return {
            "ok": True,
            "payload": {
                "program": program,
                "program_sha": "sha256:"
                + hashlib.sha256(files["program.json"].encode("utf-8")).hexdigest(),
            },
        }
    result = evaluate_program(program, job.get("state") or {}, job.get("action") or {})
    return {
        "ok": True,
        "decision": result["decision"],
        "status": result["status"],
        "violations": result["violations"],
        "unknowns": result["unknowns"],
        "report_hash": result["report_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsl_worker")
    parser.add_argument("--mode", choices=("load", "check"), required=True)
    parser.add_argument("job", type=str)
    args = parser.parse_args(argv)
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        out = run_job(args.mode, job)
    except DslBundleRejected as exc:
        out = {"ok": False, "reasons": exc.reasons}
    except Exception as exc:  # worker boundary: report, never crash mute
        out = {
            "ok": False,
            "error": "INFRA_ERROR",
            "reasons": ["%s: %s" % (type(exc).__name__, str(exc)[:300])],
        }
    print(RESULT_MARKER + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via sandbox tests
    raise SystemExit(main())
