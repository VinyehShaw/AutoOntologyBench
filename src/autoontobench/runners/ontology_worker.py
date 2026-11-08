"""Sandboxed ontology worker (owner: W6). Runs ONLY inside the child process
spawned by ``autoontobench.core.ontology_consumer`` (or in its documented
in-process fallback). This is the only W6 module that imports the heavy
erule/rdflib/pySHACL stack; the parent process stays free of it.

Modes:
- ``load``  : parse bundle -> erule contract validation -> canonical N-Triples
              payload (graphs are rebuilt parent-side only for the fallback).
- ``check`` : additionally evaluate ``(state, action)`` through
              ``erule.runtime.rule_interpreter.evaluate`` over PUBLIC facts.

Never touches native truth; prints exactly one ``AOB_RESULT_JSON:`` line.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from autoontobench.runners.ontology_bundle import (
    OPTIONAL_FILES,
    REQUIRED_FILES,
    BundleRejected,
    parse_bundle,
)

RESULT_MARKER = "AOB_RESULT_JSON:"


_PATH_RE = re.compile(r"(?<![\w.-])(?:/[\w.@+-]+)+/?")


def sanitize_reasons(reasons: list[str], *, cap: int = 20, width: int = 300) -> list[str]:
    return [_PATH_RE.sub("<path>", str(r))[:width] for r in reasons[:cap]]


def load_shacl_artifact(files: Mapping[str, str]) -> dict[str, Any]:
    from erule.contract.validator import ArtifactRejected, load_artifact
    from erule.runtime.raw_graph import canonical_ntriples

    phi: dict[str, Any] | None = None
    if "phi.json" in files:
        try:
            phi_doc = json.loads(files["phi.json"])
        except json.JSONDecodeError as exc:
            return {"ok": False, "reasons": ["phi.json: not valid JSON: %s" % exc]}
        if not isinstance(phi_doc, dict):
            return {"ok": False, "reasons": ["phi.json: must be a JSON object"]}
        phi = phi_doc
    with tempfile.TemporaryDirectory(prefix="aob-artifact-") as root:
        root_path = Path(root)
        for name in REQUIRED_FILES + ("mappings.json",):
            if name in files:
                (root_path / name).write_text(files[name], encoding="utf-8")
        try:
            artifact = load_artifact(root_path)  # default profile erule-core/1.0
        except ArtifactRejected as exc:
            return {"ok": False, "reasons": list(exc.reasons)}
    return {
        "ok": True,
        "ontology_nt": canonical_ntriples(artifact.ontology_graph),
        "shapes_nt": canonical_ntriples(artifact.shapes_graph),
        "mappings": artifact.mappings,
        "manifest": artifact.manifest,
        "contract_hash": artifact.contract_hash,
        "phi": phi,
    }


def evaluate_shacl(payload: Mapping[str, Any], state: Any, action: Any) -> dict[str, Any]:
    from types import SimpleNamespace

    from rdflib import Graph

    from erule.contract.validator import load_profile
    from erule.runtime.rule_interpreter import evaluate

    artifact = SimpleNamespace(
        ontology_graph=Graph().parse(data=payload["ontology_nt"], format="nt"),
        shapes_graph=Graph().parse(data=payload["shapes_nt"], format="nt"),
        mappings=payload.get("mappings"),
        manifest=payload["manifest"],
        contract_hash=payload["contract_hash"],
    )
    result = evaluate(
        state,
        action,
        artifact,
        profile=load_profile("erule-core/1.0"),
        phi=payload.get("phi"),
    )
    return {
        "ok": True,
        "decision": result.decision.value if result.decision is not None else None,
        "status": result.status.value,
        "violations": result.violations,
        "unknowns": result.unknowns,
        "report_hash": result.report_hash,
        "contract_hash": payload["contract_hash"],
    }


def run_job(mode: str, job: Mapping[str, Any]) -> dict[str, Any]:
    files = parse_bundle(base64.b64decode(job["bundle_b64"]))
    loaded = load_shacl_artifact(files)
    if not loaded.get("ok"):
        loaded["reasons"] = sanitize_reasons(loaded.get("reasons", []))
        return loaded
    if mode == "load":
        return {"ok": True, "payload": loaded}
    return evaluate_shacl(loaded, job.get("state"), job.get("action"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ontology_worker")
    parser.add_argument("--mode", choices=("load", "check"), required=True)
    parser.add_argument("job", type=str)
    args = parser.parse_args(argv)
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        out = run_job(args.mode, job)
    except BundleRejected as exc:
        out = {"ok": False, "reasons": sanitize_reasons(exc.reasons)}
    except Exception as exc:  # worker boundary: report, never crash mute
        out = {
            "ok": False,
            "error": "INFRA_ERROR",
            "reasons": sanitize_reasons(["%s: %s" % (type(exc).__name__, exc)]),
        }
    print(RESULT_MARKER + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via sandbox tests
    raise SystemExit(main())
