"""L01 task engine: mechanism mutators + pinned-OPA native evaluation helpers.

Every task instance is solver-grounded by *reverse generation*: start from an
upstream-authored compliant fixture (official ``example_allowed*.yaml`` case
object), apply a mechanism mutator to induce a real violation (natively proven
DENY), and pair it with a reference repair (natively proven ALLOW). My mutators
only *propose* edits; the pinned ``opa eval`` pipeline decides acceptance, so no
Rego semantics are re-implemented here.

Native pipeline (pinned):
  tools/bin/opa 0.70.0 ``eval --format json --input <doc> --data <unit rego files>
  data.<package>.violation`` — DENY iff violation set non-empty
  (``rule_ingestion.gatekeeper_adapter.NATIVE_DECISION_CONTRACT``).

YAML parsing of case objects uses PyYAML (available in .venv; the
``rule_ingestion.yaml_subset`` parser intentionally excludes flow mappings and
block scalars that upstream fixtures use).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from rule_ingestion.gatekeeper_adapter import GatekeeperRuleUnit, admission_input, parse_unit
from rule_ingestion.yaml_subset import parse_yaml_subset

REPO_ROOT = Path(__file__).resolve().parents[4]
GK_ROOT = (
    REPO_ROOT / "external_knowledge" / "pinned" / "gatekeeper-library" / "commit-a8d4799d8daf"
)
OPA_BIN = str(REPO_ROOT / "tools" / "bin" / "opa")
OPA_VERSION = "0.70.0"
GK_COMMIT = "a8d4799d8dafd47ec5fafe945ef277d329e2cceb"
LOG_ROOT = Path("/tmp/opencode/aob8000/l01_opa_logs")

ENGINE = "opa"


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadedUnit:
    """One parsed policy unit pinned to a sample constraint."""

    raw: GatekeeperRuleUnit
    unit: str
    sample: str

    @property
    def rule_unit_id(self) -> str:
        return self.raw.rule_unit_id

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.raw.parameters)

    @property
    def match_kinds(self) -> tuple[str, ...]:
        return self.raw.match_kinds


_UNIT_CACHE: dict[tuple[str, str], LoadedUnit] = {}


def load_unit(unit: str, sample: str) -> LoadedUnit:
    key = (unit, sample)
    if key not in _UNIT_CACHE:
        raw = parse_unit(GK_ROOT, unit, sample)
        if raw.source_commit != GK_COMMIT:
            raise ValueError("pin drift for %s" % unit)
        _UNIT_CACHE[key] = LoadedUnit(raw=raw, unit=unit, sample=sample)
    return _UNIT_CACHE[key]


def load_unit_with_params(loaded: LoadedUnit, parameters: dict[str, Any]) -> LoadedUnit:
    """Counterfactual unit view: same Rego, different constraint parameters."""
    import dataclasses

    raw = dataclasses.replace(loaded.raw, parameters=dict(parameters))
    return LoadedUnit(raw=raw, unit=loaded.unit, sample=loaded.sample)


class OpaRunError(RuntimeError):
    """The pinned opa eval pipeline failed (never silently mapped)."""


def _run_opa_eval(
    loaded: LoadedUnit,
    input_doc: dict[str, Any],
    *,
    input_id: str,
    extra_data: list[Path] | None = None,
) -> dict[str, Any]:
    """Pinned pipeline (mirrors native_runners/opa_runner.py) + optional extra data.

    Lane-local copy because ``native_runners.opa_runner.run_native_input`` has no
    extra-data seam (data/erule/.../lanes/L01/REQUESTS.md). Same command shape,
    same decision contract, same log discipline:

        opa eval --format json --input <doc> --data <unit rego files>
                 [--data <inventory.json>] data.<package>.violation

    DENY iff the violation set is non-empty; non-zero exit / unparseable output
    raises (fail-closed), never a fake DENY.
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in input_id)[:120]
    input_path = LOG_ROOT / ("%s.input.json" % safe_id)
    log_path = LOG_ROOT / ("%s.log" % safe_id)
    input_bytes = (json.dumps(input_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    input_path.write_bytes(input_bytes)
    input_sha = hashlib.sha256(input_bytes).hexdigest()

    command = [OPA_BIN, "eval", "--format", "json", "--input", str(input_path)]
    for rego_file in loaded.raw.rego_files:
        command += ["--data", str(rego_file)]
    for extra in extra_data or []:
        command += ["--data", str(extra)]
    command.append("data.%s.violation" % loaded.raw.package)

    header = [
        "command: %s" % " ".join(command),
        "cwd: %s" % GK_ROOT,
        "opa_version: %s" % OPA_VERSION,
        "input_sha256: %s" % input_sha,
    ]
    try:
        proc = subprocess.run(
            command, cwd=str(GK_ROOT), capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        log_path.write_text("\n".join(header) + "\nerror: timeout after 60s\n")
        raise OpaRunError("opa eval timeout (log: %s)" % log_path)
    stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode

    violations: list[dict[str, Any]] = []
    decision: str | None = None
    parse_note = ""
    if exit_code == 0:
        try:
            payload = json.loads(stdout)
            value = payload["result"][0]["expressions"][0]["value"]
            if not isinstance(value, list):
                raise ValueError("violation value is not a list: %r" % type(value))
            violations = value
            decision = "DENY" if violations else "ALLOW"
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            parse_note = "parse failure: %s: %s" % (type(exc).__name__, exc)

    lines = list(header)
    lines.append("exit_code: %d" % exit_code)
    if decision is not None:
        lines.append("violation_count: %d" % len(violations))
        lines.append("decision: %s" % decision)
    if parse_note:
        lines.append(parse_note)
    lines += ["--- stdout ---", stdout, "--- stderr ---", stderr]
    log_path.write_text("\n".join(lines) + "\n")

    if exit_code != 0:
        raise OpaRunError("opa eval exit %d (log: %s)" % (exit_code, log_path))
    if decision is None:
        raise OpaRunError("could not parse opa eval payload (log: %s)" % log_path)
    return {
        "decision": decision,
        "violation_count": len(violations),
        "violations": violations,
        "input_sha256": input_sha,
        "command": command,
        "exit_code": exit_code,
        "log_path": str(log_path),
    }


def opa_eval_object(
    loaded: LoadedUnit,
    obj: dict[str, Any],
    *,
    context: str,
    inventory: list[dict[str, Any]] | None = None,
    cluster_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One native call with recomputable evidence (command, sha, verdict).

    ``inventory`` carries the other live namespaced objects; it is synthesized
    into a Gatekeeper-sync-shaped ``data.inventory`` document
    (namespace/apiVersion/kind/name). ``cluster_objects`` populates
    ``data.inventory.cluster[apiVersion][kind][name]`` for cluster-scoped state
    (e.g. StorageClass), matching the sync layout the units read.
    """
    extra_data: list[Path] = []
    inventory_sha = None
    if inventory is not None or cluster_objects is not None:
        inv: dict[str, Any] = {"inventory": {}}
        if inventory is not None:
            inv["inventory"]["namespace"] = {}
            for other in inventory:
                meta = other.get("metadata") or {}
                ns = meta.get("namespace", "default")
                api = other.get("apiVersion", "v1")
                kind = other.get("kind", "")
                name = meta.get("name", "")
                if not kind or not name:
                    continue
                clean = {k: v for k, v in other.items() if not k.startswith("__")}
                bucket = (
                    inv["inventory"]["namespace"]
                    .setdefault(ns, {})
                    .setdefault(api, {})
                    .setdefault(kind, {})
                )
                bucket[name] = clean
        if cluster_objects:
            inv["inventory"]["cluster"] = {}
            for other in cluster_objects:
                meta = other.get("metadata") or {}
                api = other.get("apiVersion", "v1")
                kind = other.get("kind", "")
                name = meta.get("name", "")
                if not kind or not name:
                    continue
                clean = {k: v for k, v in other.items() if not k.startswith("__")}
                inv["inventory"]["cluster"].setdefault(api, {}).setdefault(kind, {})[name] = clean
        inventory_sha = sha256_json(inv)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        fd, inv_path_str = tempfile.mkstemp(prefix="l01_inv_", suffix=".json", dir=str(LOG_ROOT))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(inv, sort_keys=True))
        extra_data.append(Path(inv_path_str))

    input_doc = admission_input(loaded.raw, obj)
    api_version = str(obj.get("apiVersion", "v1"))
    group, _, version = api_version.partition("/")
    if not version:
        group, version = "", api_version
    input_doc["review"]["kind"] = {
        "kind": obj.get("kind", ""),
        "apiVersion": api_version,
        "group": group,
        "version": version,
    }
    outcome = _run_opa_eval(
        loaded,
        input_doc,
        input_id=context,
        extra_data=extra_data or None,
    )
    evidence = {
        "engine": ENGINE,
        "engine_version": OPA_VERSION,
        "rule_unit_id": loaded.rule_unit_id,
        "decision": outcome["decision"],
        "violation_count": outcome["violation_count"],
        "violations": [str(v.get("msg", ""))[:200] for v in outcome["violations"]],
        "input_sha256": outcome["input_sha256"],
        "command": outcome["command"],
        "exit_code": outcome["exit_code"],
        "log_path": outcome["log_path"],
    }
    if inventory_sha:
        evidence["inventory_sha256"] = inventory_sha
    return evidence


def eval_object_set(
    units: Iterable[LoadedUnit],
    obj: dict[str, Any],
    *,
    context: str,
    inventory: list[dict[str, Any]] | None = None,
    cluster_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Native verdicts of every governing unit on one object."""
    verdicts = []
    for loaded in units:
        if obj.get("kind") in loaded.match_kinds:
            verdicts.append(opa_eval_object(loaded, obj, context=context, inventory=inventory,
                                            cluster_objects=cluster_objects))
    compliant = all(v["decision"] == "ALLOW" for v in verdicts)
    return {"compliant": compliant, "verdicts": verdicts}


def load_case_objects(unit: str, sample: str) -> list[dict[str, Any]]:
    """Official suite case objects of one sample (upstream-authored fixtures).

    Inventory fixture files (``example_inventory_*.yaml``) are not case objects
    and are excluded; annotations are kept (they are governed fields, e.g. the
    httpsonly ``kubernetes.io/ingress.allow-http`` annotation).
    """
    sample_dir = GK_ROOT / "library" / unit / sample
    out: list[dict[str, Any]] = []
    for path in sorted(sample_dir.glob("*.yaml")):
        if path.name == "constraint.yaml" or "inventory" in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and doc.get("kind"):
            doc["__fixture__"] = path.name
            out.append(doc)
    return out


def load_inventory_fixtures(unit: str, sample: str) -> list[dict[str, Any]]:
    """Upstream ``example_inventory_*.yaml`` objects (Gatekeeper-sync fixtures)."""
    sample_dir = GK_ROOT / "library" / unit / sample
    out: list[dict[str, Any]] = []
    for path in sorted(sample_dir.glob("example_inventory*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and doc.get("kind"):
            out.append(doc)
    return out


def sample_dirs(unit: str) -> list[str]:
    base = GK_ROOT / "library" / unit / "samples"
    if not base.is_dir():
        return []
    return sorted(
        "samples/%s" % p.name for p in base.iterdir() if (p / "constraint.yaml").is_file()
    )


def constraint_doc(unit: str, sample: str) -> dict[str, Any]:
    path = GK_ROOT / "library" / unit / sample / "constraint.yaml"
    return parse_yaml_subset(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mechanism mutators: (violate, repair) proposals over governed fields.
# Each takes (obj, params, rng) and returns a NEW object or None if the
# proposal does not apply. Native evaluation decides validity downstream.
# ---------------------------------------------------------------------------

def _containers(obj: dict[str, Any]) -> list[dict[str, Any]]:
    spec = obj.get("spec") or {}
    out: list[dict[str, Any]] = []
    for field in ("containers", "initContainers", "ephemeralContainers"):
        out.extend(c for c in spec.get(field) or [] if isinstance(c, dict))
    template = (spec.get("template") or {}).get("spec") or {}
    for field in ("containers", "initContainers"):
        out.extend(c for c in template.get(field) or [] if isinstance(c, dict))
    return out


def _image_basename(image: str) -> str:
    name = image.rsplit("/", 1)[-1]
    return name.split("@", 1)[0].split(":", 1)[0] or "app"


def _set_image(container: dict[str, Any], image: str) -> None:
    container["image"] = image


def violate_registry(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    bad = ["quay.io/unapproved", "docker.io/library", "registry.evil.example"]
    c = rng.choice(cs)
    _set_image(c, "%s/%s:latest" % (rng.choice(bad), _image_basename(c.get("image", "app"))))
    return new


def repair_registry(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    repos = params.get("repos") or params.get("allowedImages") or []
    if not repos:
        return None
    new = copy.deepcopy(obj)
    for c in _containers(new):
        prefix = repos[rng.randrange(len(repos))].rstrip("/")
        image = str(c.get("image", ""))
        if not any(str(image).startswith(r.rstrip("/").rstrip("*")) for r in repos):
            _set_image(c, "%s/%s:v1" % (prefix, _image_basename(image)))
    return new


def violate_disallowed_repo(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    repos = params.get("repos") or []
    if not repos:
        return None
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    c = rng.choice(cs)
    _set_image(c, "%s/%s:latest" % (repos[0].rstrip("/"), _image_basename(c.get("image", "app"))))
    return new


def repair_disallowed_repo(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    repos = [r.rstrip("/") for r in (params.get("repos") or [])]
    new = copy.deepcopy(obj)
    for c in _containers(new):
        image = str(c.get("image", ""))
        if any(image.startswith(r) for r in repos):
            _set_image(c, "registry.corp.example/approved/%s:v1" % _image_basename(image))
    return new


def violate_tag(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    tags = params.get("tags") or ["latest"]
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    c = rng.choice(cs)
    base = str(c.get("image", "app")).split(":")[0].split("@")[0]
    _set_image(c, "%s:%s" % (base, rng.choice(tags)))
    return new


def repair_tag(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    tags = set(params.get("tags") or ["latest"])
    new = copy.deepcopy(obj)
    for c in _containers(new):
        image = str(c.get("image", ""))
        if ":" in image and image.rsplit(":", 1)[-1] in tags:
            _set_image(c, "%s:v%s" % (image.split(":")[0], rng.choice(["1.4.0", "2.0.1", "0.9.3"])))
        elif ":" not in image and "@" not in image:
            _set_image(c, "%s:v%s" % (image, rng.choice(["1.4.0", "2.0.1", "0.9.3"])))
    return new


def _digest(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(64))


def violate_digest(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    c = rng.choice(cs)
    image = str(c.get("image", "registry.example/app"))
    if "@" in image:
        image = image.split("@")[0]
    _set_image(c, image if ":" in image else image + ":v1")
    return new


def repair_digest(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    for c in _containers(new):
        image = str(c.get("image", "registry.example/app"))
        if "@" not in image:
            base = image.split(":")[0]
            _set_image(c, "%s@sha256:%s" % (base, _digest(rng)))
    return new


def _label_specs(params: dict[str, Any], key: str) -> list[dict[str, Any]]:
    out = params.get(key) or []
    return [s for s in out if isinstance(s, dict)]


def _regex_witness(pattern: str | None, rng: random.Random) -> str:
    """Witness string for the simple allowedRegex shapes used by the library."""
    if not pattern:
        return "aob-value"
    if pattern.startswith("^"):
        body = pattern.strip("^$")
        if body.endswith(".*"):
            return body[:-2] + "-x"
        return body.replace("\\", "") or "aob-value"
    return pattern


#: Ordered witness candidates per known governed key. Every repair proposal is
#: natively re-checked, so a wrong witness costs one dropped candidate, never a
#: false compliant label.
_LABEL_WITNESSES: dict[str, list[str]] = {
    "owner": ["alice.agilebank.demo", "bob.agilebank.demo", "carol.agilebank.demo"],
    "a8r.io/owner": ["alice@example.com", "platform-team", "bob@corp.example"],
    "a8r.io/runbook": ["https://runbook.example.com/db", "http://runbooks.corp.example/app"],
}


def _witness_candidates(spec: dict[str, Any], rng: random.Random) -> list[str]:
    key = spec.get("key", "")
    cands = list(_LABEL_WITNESSES.get(key, []))
    cands.append(_regex_witness(spec.get("allowedRegex"), rng))
    return cands


def violate_labels(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    specs = _label_specs(params, "labels")
    if not specs:
        return None
    new = copy.deepcopy(obj)
    meta = new.setdefault("metadata", {})
    labels = meta.setdefault("labels", {})
    spec = rng.choice(specs)
    key = spec.get("key", "owner")
    if rng.random() < 0.5 or not spec.get("allowedRegex"):
        labels.pop(key, None)
    else:
        labels[key] = "INVALID VALUE!!"
    return new


def repair_labels(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    specs = _label_specs(params, "labels")
    if not specs:
        return None
    new = copy.deepcopy(obj)
    labels = new.setdefault("metadata", {}).setdefault("labels", {})
    for spec in specs:
        key = spec.get("key")
        if not key:
            continue
        cur = labels.get(key)
        if cur is None or (spec.get("allowedRegex") and not _regex_fullmatch(spec["allowedRegex"], str(cur))):
            labels[key] = _witness_candidates(spec, rng)[0]
    return new


def _regex_fullmatch(pattern: str, value: str) -> bool:
    import re

    try:
        return re.fullmatch(pattern.strip("^$"), value) is not None or re.fullmatch(pattern, value) is not None
    except re.error:
        return False


def violate_annotations(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    specs = _label_specs(params, "annotations")
    if not specs:
        return None
    new = copy.deepcopy(obj)
    anns = new.setdefault("metadata", {}).setdefault("annotations", {})
    spec = rng.choice(specs)
    anns.pop(spec.get("key", "aob.io/owner"), None)
    return new


def repair_annotations(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    specs = _label_specs(params, "annotations")
    if not specs:
        return None
    new = copy.deepcopy(obj)
    anns = new.setdefault("metadata", {}).setdefault("annotations", {})
    for spec in specs:
        key = spec.get("key")
        if key and (key not in anns or (spec.get("allowedRegex") and not _regex_fullmatch(spec["allowedRegex"], str(anns.get(key))))):
            anns[key] = _witness_candidates(spec, rng)[0]
    return new


def _replica_bounds(params: dict[str, Any]) -> tuple[int, int] | None:
    ranges = params.get("ranges")
    if ranges and isinstance(ranges, list):
        r0 = ranges[0]
        return int(r0.get("min_replicas", 1)), int(r0.get("max_replicas", 5))
    if "minReplicas" in params or "maxReplicas" in params:
        return int(params.get("minReplicas", 1)), int(params.get("maxReplicas", 5))
    return None


def _replica_spec(obj: dict[str, Any]) -> dict[str, Any] | None:
    return obj.get("spec") if isinstance(obj.get("spec"), dict) else None


def violate_replicas(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    bounds = _replica_bounds(params)
    spec = _replica_spec(obj)
    if bounds is None or spec is None:
        return None
    lo, hi = bounds
    new = copy.deepcopy(obj)
    spec = new["spec"]
    spec["replicas"] = hi + rng.randint(1, 4) if rng.random() < 0.6 else max(lo - rng.randint(1, 2), 0)
    return new


def repair_replicas(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    bounds = _replica_bounds(params)
    spec = _replica_spec(obj)
    if bounds is None or spec is None:
        return None
    lo, hi = bounds
    new = copy.deepcopy(obj)
    cur = new["spec"].get("replicas")
    if not isinstance(cur, int) or cur < lo or cur > hi:
        new["spec"]["replicas"] = min(max(lo, 1), hi)
    return new


_PROBE_KINDS = {"readinessProbe": {"httpGet": {"path": "/ready", "port": 8080}},
                "livenessProbe": {"httpGet": {"path": "/healthz", "port": 8080}}}


def violate_probes(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    probes = params.get("probes") or ["readinessProbe"]
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    for c in cs:
        c.pop(rng.choice(probes), None)
    return new


def repair_probes(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    probes = params.get("probes") or ["readinessProbe"]
    new = copy.deepcopy(obj)
    for c in _containers(new):
        for p in probes:
            if p in _PROBE_KINDS and (p not in c or not c[p]):
                c[p] = copy.deepcopy(_PROBE_KINDS[p])
    return new


def violate_sa_token(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    spec["automountServiceAccountToken"] = True
    return new


def repair_sa_token(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    if spec.get("automountServiceAccountToken") is not False:
        spec["automountServiceAccountToken"] = False
    return new


def _make_violate_service_type(banned: str):
    def fn(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        new = copy.deepcopy(obj)
        new.setdefault("spec", {})["type"] = banned
        return new

    return fn


def _make_repair_service_type():
    def fn(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        new = copy.deepcopy(obj)
        spec = new.setdefault("spec", {})
        if spec.get("type") in ("NodePort", "LoadBalancer"):
            spec["type"] = "ClusterIP"
            spec.pop("externalTrafficPolicy", None)
        return new

    return fn


def violate_external_ip(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    allowed = set(params.get("allowedIPs") or [])
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    candidate = "192.0.2.%d" % rng.randint(1, 250)
    if candidate in allowed:
        candidate = "198.51.100.%d" % rng.randint(1, 250)
    if candidate in allowed:
        return None
    spec["externalIPs"] = [candidate]
    return new


def repair_external_ip(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    allowed = params.get("allowedIPs") or []
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    ips = [ip for ip in spec.get("externalIPs") or [] if ip in allowed]
    if ips != spec.get("externalIPs"):
        spec["externalIPs"] = ips
        if not ips:
            spec.pop("externalIPs")
    return new


def _ingress_hosts(obj: dict[str, Any]) -> list[str]:
    rules = (obj.get("spec") or {}).get("rules") or []
    return [r.get("host", "") for r in rules if isinstance(r, dict)]


def violate_wildcard_ingress(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    rules = (new.setdefault("spec", {})).setdefault("rules", [])
    if not rules:
        rules.append({"host": "", "http": {"paths": []}})
    rules[0]["host"] = "*.wild.example.com" if rng.random() < 0.7 else ""
    return new


def repair_wildcard_ingress(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    rules = (new.get("spec") or {}).get("rules") or []
    for rule in rules:
        host = rule.get("host", "")
        if host.startswith("*") or host == "":
            rule["host"] = "app-%d.example.com" % rng.randint(1, 999)
    return new


def violate_https(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    spec.pop("tls", None)
    return new


def repair_https(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """k8shttpsonly: needs spec.tls non-empty AND the allow-http=false annotation."""
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    anns = new.setdefault("metadata", {}).setdefault("annotations", {})
    if not spec.get("tls"):
        hosts = _ingress_hosts(new)
        spec["tls"] = [{"hosts": [h for h in hosts if h], "secretName": "tls-secret"}]
    anns["kubernetes.io/ingress.allow-http"] = "false"
    return new


def violate_ephemeral(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    for c in cs:
        c.setdefault("resources", {}).setdefault("limits", {})["ephemeral-storage"] = "10Gi"
    return new


def repair_ephemeral(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    bound = str(params.get("ephemeral-storage", "1Gi"))
    new = copy.deepcopy(obj)
    for c in _containers(new):
        limits = c.setdefault("resources", {}).setdefault("limits", {})
        limits["ephemeral-storage"] = bound
    return new


def _ratio_params(params: dict[str, Any]) -> tuple[float, float]:
    ratio = float(params.get("ratio", 1))
    cpu_ratio = float(params.get("cpuRatio", params.get("ratio", 1)))
    return ratio, cpu_ratio


def violate_ratio(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """k8scontainerratios denies when limit > ratio*request (cpuRatio for cpu)."""
    _mem_ratio, cpu_ratio = _ratio_params(params)
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    for c in cs:
        res = c.setdefault("resources", {})
        res["requests"] = {"cpu": "100m", "memory": "128Mi"}
        res["limits"] = {"cpu": "%dm" % int(100 * (cpu_ratio + rng.randint(2, 5))),
                         "memory": "128Mi"}
    return new


def repair_ratio(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    mem_ratio, cpu_ratio = _ratio_params(params)
    new = copy.deepcopy(obj)
    for c in _containers(new):
        res = c.setdefault("resources", {})
        res["requests"] = {"cpu": "100m", "memory": "128Mi"}
        res["limits"] = {"cpu": "%dm" % int(100 * cpu_ratio),
                         "memory": "%dMi" % int(128 * mem_ratio)}
    return new


def violate_limits(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    for c in cs:
        c.setdefault("resources", {}).setdefault("limits", {})["cpu"] = "8"
        c["resources"]["limits"]["memory"] = "64Gi"
    return new


def repair_limits(bound_key: str) -> Callable[[dict[str, Any], dict[str, Any], random.Random], dict[str, Any] | None]:
    def fn(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        new = copy.deepcopy(obj)
        cpu = str(params.get("cpu", "500m"))
        mem = str(params.get("memory", "512Mi"))
        for c in _containers(new):
            res = c.setdefault("resources", {})
            res.setdefault(bound_key, {})["cpu"] = cpu
            res[bound_key]["memory"] = mem
        return new

    return fn


def violate_storageclass(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    allowed = params.get("allowedStorageClasses")
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    if allowed:
        spec["storageClassName"] = "fast-unapproved"
    else:
        spec.pop("storageClassName", None)  # presence-required sample
    return new


def repair_storageclass(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    allowed = params.get("allowedStorageClasses") or []
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    cur = spec.get("storageClassName")
    if allowed and cur not in allowed:
        spec["storageClassName"] = allowed[0]
    elif not allowed and cur is None:
        spec["storageClassName"] = "somestorageclass"
    return new


def unique_host_repair(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    rules = (new.get("spec") or {}).get("rules") or []
    name = (new.get("metadata") or {}).get("name", "ing")
    for rule in rules:
        rule["host"] = "%s-%d.example.com" % (name, rng.randint(100, 999))
    return new


#: Mechanism table: mechanism key -> (unit, violate, repair, cross_object).
MechanismSpec = tuple[str, Callable[..., Any], Callable[..., Any], bool]

MECHANISMS: dict[str, MechanismSpec] = {
    "allowedrepos": ("general/allowedrepos", violate_registry, repair_registry, False),
    "disallowedrepos": ("general/disallowedrepos", violate_disallowed_repo, repair_disallowed_repo, False),
    "disallowedtags": ("general/disallowedtags", violate_tag, repair_tag, False),
    "imagedigests": ("general/imagedigests", violate_digest, repair_digest, False),
    "requiredlabels": ("general/requiredlabels", violate_labels, repair_labels, False),
    "requiredannotations": ("general/requiredannotations", violate_annotations, repair_annotations, False),
    "replicalimits": ("general/replicalimits", violate_replicas, repair_replicas, False),
    "requiredprobes": ("general/requiredprobes", violate_probes, repair_probes, False),
    "automount-sa-token": (
        "general/automount-serviceaccount-token",
        violate_sa_token,
        repair_sa_token,
        False,
    ),
    "block-nodeport": (
        "general/block-nodeport-services",
        _make_violate_service_type("NodePort"),
        _make_repair_service_type(),
        False,
    ),
    "block-loadbalancer": (
        "general/block-loadbalancer-services",
        _make_violate_service_type("LoadBalancer"),
        _make_repair_service_type(),
        False,
    ),
    "externalip": ("general/externalip", violate_external_ip, repair_external_ip, False),
    "block-wildcard-ingress": (
        "general/block-wildcard-ingress",
        violate_wildcard_ingress,
        repair_wildcard_ingress,
        False,
    ),
    "httpsonly": ("general/httpsonly", violate_https, repair_https, False),
    "ephemeralstoragelimit": (
        "general/ephemeralstoragelimit",
        violate_ephemeral,
        repair_ephemeral,
        False,
    ),
    "containerresourceratios": (
        "general/containerresourceratios",
        violate_ratio,
        repair_ratio,
        False,
    ),
    "containerlimits": (
        "general/containerlimits",
        violate_limits,
        repair_limits("limits"),
        False,
    ),
    "containerrequests": (
        "general/containerrequests",
        violate_limits,
        repair_limits("requests"),
        False,
    ),
    "storageclass": ("general/storageclass", violate_storageclass, repair_storageclass, False),
    "uniqueingresshost": (
        "general/uniqueingresshost",
        violate_wildcard_ingress,
        unique_host_repair,
        True,
    ),
}
