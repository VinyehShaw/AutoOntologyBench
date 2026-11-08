"""L08 Node engine runner: Python subprocess bridge to vendored cql-execution.

Locates the pinned Node binary under ``tools/node/`` (self-contained; no
system node, no Java), sets ``NODE_PATH`` to the vendored npm engine closure,
and feeds ELM libraries + synthetic FHIR R4 patient bundles to
``benchmark/plugins/verifier/l08/node/run_elm.cjs`` via a JSON job file.

Deterministic: lane ELM never uses Today()/Now(); the reference date is a
frozen literal (``elm.REF_DATE``) and lookback cutoffs are computed in Python.

Self-test: ``python -m benchmark.plugins.verifier.l08.runner --self-test``
executes a built-in ELM library over a built-in synthetic bundle and checks
the expected per-patient booleans (engine wiring proof).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
NODE_BIN = REPO_ROOT / "tools" / "node" / "node-v20.18.0-linux-x64" / "bin" / "node"
NODE_MODULES = (REPO_ROOT / "external_knowledge" / "pinned" / "cql-clinical" / "npm"
                / "node_modules")
RUN_SCRIPT = Path(__file__).resolve().parent / "node" / "run_elm.cjs"
NODE_RECEIPT = REPO_ROOT / "tools" / "node" / "NODE_RECEIPT.json"

JOB_TIMEOUT_S = 300


class EngineUnavailable(RuntimeError):
    """Raised when the pinned Node runtime or engine closure is missing."""


def engine_versions() -> dict[str, str]:
    if not NODE_BIN.is_file():
        raise EngineUnavailable("node binary missing: %s (see tools/node/NODE_RECEIPT.json)" % NODE_BIN)
    if not NODE_MODULES.is_dir():
        raise EngineUnavailable("engine closure missing: %s" % NODE_MODULES)
    node_ver = subprocess.run([str(NODE_BIN), "--version"], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    cql_ver = json.loads((NODE_MODULES / "cql-execution" / "package.json").read_text())["version"]
    fhir_ver = json.loads((NODE_MODULES / "cql-exec-fhir" / "package.json").read_text())["version"]
    return {"node": node_ver, "cql-execution": cql_ver, "cql-exec-fhir": fhir_ver}


def job_cache_key(elm: dict[str, Any], patients: list[dict[str, Any]]) -> str:
    blob = json.dumps({"elm": elm, "patients": patients}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class NodeEngineRunner:
    """Batch executor: ELM libraries over FHIR bundles via cql-execution."""

    def __init__(self, *, node_bin: Path | None = None, node_modules: Path | None = None,
                 script: Path | None = None) -> None:
        self.node_bin = Path(node_bin) if node_bin else NODE_BIN
        self.node_modules = Path(node_modules) if node_modules else NODE_MODULES
        self.script = Path(script) if script else RUN_SCRIPT
        if not self.node_bin.is_file():
            raise EngineUnavailable("node binary missing: %s" % self.node_bin)
        if not self.node_modules.is_dir():
            raise EngineUnavailable("node_modules missing: %s" % self.node_modules)
        if not self.script.is_file():
            raise EngineUnavailable("runner script missing: %s" % self.script)

    def run_jobs(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Run ``[{job_id, library, patients}]``; returns the parsed runner output."""
        if not jobs:
            return {"engine": engine_versions(), "results": []}
        payload = {"jobs": jobs}
        fd, tmp = tempfile.mkstemp(prefix="l08_job_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            env = dict(os.environ)
            env["NODE_PATH"] = str(self.node_modules)
            proc = subprocess.run(
                [str(self.node_bin), str(self.script), tmp],
                capture_output=True, text=True, timeout=JOB_TIMEOUT_S, env=env, cwd=str(REPO_ROOT),
            )
            if proc.returncode != 0:
                raise RuntimeError("node runner failed (rc=%d): %s" % (
                    proc.returncode, proc.stderr[-2000:]))
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("node runner stdout not JSON: %s; stdout=%r" % (
                    exc, proc.stdout[:500])) from exc
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def evaluate(self, elm: dict[str, Any], patients: list[dict[str, Any]],
                 *, job_id: str = "job") -> dict[str, bool]:
        """Execute one library over patient bundles -> {patient_id: Eligible bool}."""
        out = self.run_jobs([{"job_id": job_id, "library": elm, "patients": patients}])
        return _extract(out, job_id)


def _extract(out: dict[str, Any], job_id: str) -> dict[str, bool]:
    for res in out.get("results", []):
        if res.get("job_id") != job_id:
            continue
        if not res.get("ok"):
            raise RuntimeError("engine job %s failed: %s" % (job_id, res.get("error")))
        result: dict[str, bool] = {}
        for pid, exprs in res.get("patient_results", {}).items():
            val = exprs.get("Eligible", exprs.get("Flagged"))
            if val is None:
                result[pid] = False
            elif isinstance(val, bool):
                result[pid] = val
            else:
                raise RuntimeError("engine returned non-boolean for %s: %r" % (pid, val))
        return result
    raise RuntimeError("job %s missing from runner output" % job_id)


# ------------------------------------------------------------------- self-test

_SELF_TEST_ELM = {
    "library": {
        "identifier": {"id": "L08SelfTest", "version": "1.0.0"},
        "schemaIdentifier": {"id": "urn:hl7-org:elm", "version": "r1"},
        "usings": {"def": [
            {"localIdentifier": "System", "uri": "urn:hl7-org:elm-types:r1"},
            {"localIdentifier": "FHIR", "uri": "http://hl7.org/fhir", "version": "4.0.1"},
        ]},
        "codeSystems": {"def": [{"name": "LOINC", "id": "http://loinc.org", "accessLevel": "Public"}]},
        "codes": {"def": [{"name": "BP_DIA", "id": "8462-4", "display": "Diastolic blood pressure",
                           "codeSystem": {"name": "LOINC"}, "accessLevel": "Public"}]},
        "statements": {"def": [
            {"name": "Patient", "context": "Patient", "expression": {
                "type": "SingletonFrom", "operand": {"type": "Retrieve",
                                                     "dataType": "{http://hl7.org/fhir}Patient"}}},
            {"name": "Eligible", "context": "Patient", "accessLevel": "Public", "expression": {
                "type": "Exists", "operand": {"type": "Query", "source": [
                    {"alias": "O", "expression": {
                        "type": "Retrieve", "dataType": "{http://hl7.org/fhir}Observation",
                        "codeProperty": "code", "codeComparator": "in",
                        "codes": {"type": "ToList", "operand": {"type": "CodeRef", "name": "BP_DIA"}}}}],
                    "where": {"type": "And", "operand": [
                        {"type": "GreaterOrEqual", "operand": [
                            {"path": "valueQuantity.value.value", "type": "Property",
                             "source": {"name": "O", "type": "AliasRef"}},
                            {"valueType": "{urn:hl7-org:elm-types:r1}Decimal", "value": "90",
                             "type": "Literal"}]},
                        {"type": "Equal", "operand": [
                            {"path": "valueQuantity.unit.value", "type": "Property",
                             "source": {"name": "O", "type": "AliasRef"}},
                            {"valueType": "{urn:hl7-org:elm-types:r1}String", "value": "mm[Hg]",
                             "type": "Literal"}]}]}}}},
        ]},
    }
}


def _self_test_patients() -> list[dict[str, Any]]:
    """Five synthetic bundles with known expected Eligible values (see self_test)."""
    specs = [
        ("selftest-a", [("o1", "8462-4", 95, "2026-09-01")]),          # True (95 >= 90)
        ("selftest-b", [("o2", "8462-4", 85, "2026-09-02")]),          # False (85 < 90)
        ("selftest-c", [("o3", "8867-4", 120, "2026-09-03")]),         # False (wrong code)
        ("selftest-d", [("o4", "8462-4", 90, "2026-09-04")]),          # True (90 >= 90, boundary)
        ("selftest-e", []),                                            # False (no obs)
    ]
    bundles: list[dict[str, Any]] = []
    for pid, obs_specs in specs:
        resources = [{"resourceType": "Patient", "id": pid, "gender": "female",
                      "birthDate": "1975-06-01"}]
        for oid, code, value, day in obs_specs:
            resources.append({"resourceType": "Observation", "id": oid, "status": "final",
                              "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
                              "valueQuantity": {"value": value, "unit": "mm[Hg]"},
                              "effectiveDateTime": day,
                              "subject": {"reference": "Patient/%s" % pid}})
        bundles.append({"resourceType": "Bundle", "type": "collection", "id": "b-%s" % pid,
                    "entry": [{"resource": r} for r in resources]})
    return bundles


def self_test() -> dict[str, Any]:
    """Engine wiring self-test; returns {ok, engine, results, expected, mismatches}."""
    runner = NodeEngineRunner()
    expected = {"selftest-a": True, "selftest-b": False, "selftest-c": False,
                "selftest-d": True, "selftest-e": False}
    out = runner.run_jobs([{"job_id": "selftest", "library": _SELF_TEST_ELM,
                            "patients": _self_test_patients()}])
    got = _extract(out, "selftest")
    mismatches = {pid: {"expected": expected[pid], "got": got.get(pid)}
                  for pid in expected if got.get(pid) != expected[pid]}
    return {"ok": not mismatches, "engine": out.get("engine", engine_versions()),
            "results": got, "expected": expected, "mismatches": mismatches}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="L08 Node engine runner")
    ap.add_argument("--self-test", action="store_true", help="run engine wiring self-test")
    args = ap.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result["ok"] else 1
    print(json.dumps(engine_versions(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
