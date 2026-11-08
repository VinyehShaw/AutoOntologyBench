"""L02 task engine: pinned cel-go native evaluation + VAP policy loading + mutators.

Native engine: ``cel_helper`` (cel-go v0.26.1, pinned binary at
``data/erule/review5/software/cel/helper/cel_helper``) wrapped by
``autoontobench.core.domains.cel``. Scope is the honest
``cel-expression-subset/1`` profile: dyn-typed ``object`` / ``params`` /
``oldObject`` / ``request`` only — no ``namespaceObject``, no ``variables``
declarations, no authorizer, no CRD schema type-checking. Tasks in this lane
never claim full Kubernetes admission semantics.

Policy material is loaded from the pinned kubernetes/website shard
(``external_knowledge/pinned/kubernetes-vap-cel/commit-ed70e311737a``):
validation/matchCondition/messageExpression strings are verbatim upstream;
task objects are synthetic K8s-shaped manifests (initial states are synthetic
by design; the *rules* are the external knowledge).

Every native call is evidence-logged: the expression file and input JSON land
under ``/tmp/opencode/aob8000/l02_cel_logs`` keyed by sha256 (written by
``autoontobench.core.domains.cel``), and each returned record carries the
re-runnable helper command plus input/expression digests.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from autoontobench.core.domains import cel

REPO_ROOT = Path(__file__).resolve().parents[4]
PINNED_ROOT = (
    REPO_ROOT
    / "external_knowledge"
    / "pinned"
    / "kubernetes-vap-cel"
    / "commit-ed70e311737a"
)
UPSTREAM_COMMIT = "ed70e311737a341574ed5758b7af24e3bba232cb"
LOG_ROOT = Path("/tmp/opencode/aob8000/l02_cel_logs")

ENGINE = "cel-go"
ENGINE_VERSION = cel.CEL_GO_VERSION  # v0.26.1
PROFILE = cel.PROFILE  # cel-expression-subset/1

SOURCE_ID = "kubernetes-vap-cel"
RELEASE_ID = "commit-ed70e311737a"
SOURCE_SNAPSHOT_ID = "%s@%s" % (SOURCE_ID, RELEASE_ID)


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CelRunError(RuntimeError):
    """The pinned cel-go pipeline failed (never silently mapped)."""


# ---------------------------------------------------------------------------
# Native evaluation with evidence records
# ---------------------------------------------------------------------------

def cel_check(expression: str) -> dict[str, Any]:
    """Compile-only screen. Raises CelRunError when the helper itself fails."""
    try:
        outcome = cel.check(expression, LOG_ROOT)
    except cel.CelRunError as exc:
        raise CelRunError(str(exc)) from exc
    return {
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "mode": "check",
        "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "output_type": outcome.output_type,
        "ok": outcome.ok,
        "exit_code": outcome.exit_code,
        "error": outcome.error,
        "command": [str(cel.HELPER), "check", "<expr.cel>"],
    }


def cel_eval(expression: str, inputs: dict[str, Any], *, input_id: str) -> dict[str, Any]:
    """One native eval with recomputable evidence (command, sha, value).

    ``inputs`` carries the dyn activation subset: any of ``object``, ``params``,
    ``oldObject``, ``request``. Evaluation errors (exit 2, ``ok=false``) are
    fail-closed: they raise CelRunError, never a fake verdict.
    """
    try:
        outcome = cel.evaluate(expression, inputs, LOG_ROOT)
    except cel.CelRunError as exc:
        raise CelRunError("expr=%r id=%s: %s" % (expression[:80], input_id, exc)) from exc
    if not outcome.ok:
        raise CelRunError(
            "cel eval rejected (expr=%r id=%s): %s" % (expression[:80], input_id, outcome.error)
        )
    blob = json.dumps({"expression": expression, "inputs": inputs}, sort_keys=True)
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "mode": "eval",
        "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "input_sha256": sha,
        "output_type": outcome.output_type,
        "value": outcome.value,
        "exit_code": outcome.exit_code,
        "expr_path": str(LOG_ROOT / (sha[:16] + ".cel")),
        "input_path": str(LOG_ROOT / (sha[:16] + ".input.json")),
        "command": [str(cel.HELPER), "eval", "<expr.cel>", "<input.json>"],
    }


def eval_bool(expression: str, inputs: dict[str, Any], *, input_id: str) -> dict[str, Any]:
    """Bool-typed eval; non-bool output is a contract violation (raises)."""
    rec = cel_eval(expression, inputs, input_id=input_id)
    if rec["output_type"] != "bool" or not isinstance(rec["value"], bool):
        raise CelRunError(
            "expected bool output, got %r (%r) for %r"
            % (rec["output_type"], rec["value"], expression[:80])
        )
    return rec


def policy_decision(
    unit: "PolicyUnit",
    inputs: dict[str, Any],
    *,
    input_id: str,
) -> dict[str, Any]:
    """ALLOW/DENY verdict of one policy unit on one input set.

    DENY iff any validation expression evaluates to false. Evaluation inputs
    are ``{object, params, request}`` (dyn subset).
    """
    merged = dict(inputs)
    merged.setdefault("params", {})
    verdicts = []
    for expr in unit.validations:
        rec = eval_bool(expr, merged, input_id=input_id)
        verdicts.append({
            "expression_sha256": rec["expression_sha256"],
            "input_sha256": rec["input_sha256"],
            "value": rec["value"],
            "exit_code": rec["exit_code"],
            "expr_path": rec["expr_path"],
            "input_path": rec["input_path"],
        })
    decision = "ALLOW" if all(v["value"] for v in verdicts) else "DENY"
    return {
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "policy_id": unit.policy_id,
        "decision": decision,
        "verdicts": verdicts,
        "command": [str(cel.HELPER), "eval", "<expr.cel>", "<input.json>"],
    }


# ---------------------------------------------------------------------------
# Policy loading from the pinned shard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyUnit:
    """One validation-bearing unit with verbatim upstream expressions."""

    policy_id: str
    source_path: str  # repo-relative path inside the pinned shard
    kind: str  # vap_policy | doc_expression_set
    validations: tuple[str, ...]
    message_expressions: tuple[str, ...] = ()
    match_conditions: tuple[tuple[str, str], ...] = ()  # (name, expression)
    param_kind: str | None = None
    message_kind: str = "violation"  # violation (on DENY) | audit (on evaluate)
    notes: str = ""

    @property
    def source_unit_ref(self) -> str:
        return "%s/%s" % (SOURCE_ID, self.source_path)


_POLICY_CACHE: dict[str, PolicyUnit] | None = None


def _vap_validations(doc: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for v in (doc.get("spec") or {}).get("validations") or []:
        expr = v.get("expression")
        if isinstance(expr, str) and expr.strip():
            out.append(expr.strip())
    return tuple(out)


def _vap_message_expressions(doc: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    spec = doc.get("spec") or {}
    for v in spec.get("validations") or []:
        expr = v.get("messageExpression")
        if isinstance(expr, str) and expr.strip() and expr.strip() not in out:
            out.append(expr.strip())
    for a in spec.get("auditAnnotations") or []:
        expr = a.get("valueExpression")
        if isinstance(expr, str) and expr.strip() and expr.strip() not in out:
            out.append(expr.strip())
    return tuple(out)


def _vap_match_conditions(doc: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for mc in (doc.get("spec") or {}).get("matchConditions") or []:
        name, expr = mc.get("name"), mc.get("expression")
        if isinstance(name, str) and isinstance(expr, str) and expr.strip():
            out.append((name.strip(), expr.strip()))
    return tuple(out)


def _load_yaml(rel: str) -> dict[str, Any]:
    text = (PINNED_ROOT / rel).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise CelRunError("pinned yaml did not parse to a mapping: %s" % rel)
    return doc


def load_policies() -> dict[str, PolicyUnit]:
    """Parse the pinned shard into policy units (verbatim expressions).

    Every admitted expression is proven compilable under the pinned dyn
    profile by a real ``cel_check`` call at load time; failure raises (the
    pinned shard drifted) rather than silently dropping a rule.
    """
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    policies: dict[str, PolicyUnit] = {}

    def _admit(unit: PolicyUnit) -> None:
        for expr in unit.validations + unit.message_expressions:
            rec = cel_check(expr)
            if not rec["ok"]:
                raise CelRunError(
                    "upstream expression no longer compiles under pinned profile: %r (%s): %s"
                    % (expr[:80], unit.source_path, rec["error"])
                )
        for _name, expr in unit.match_conditions:
            rec = cel_check(expr)
            if not rec["ok"]:
                raise CelRunError(
                    "upstream matchCondition no longer compiles: %r (%s): %s"
                    % (expr[:80], unit.source_path, rec["error"])
                )
        policies[unit.policy_id] = unit

    rel = "content/en/examples/validatingadmissionpolicy/basic-example-policy.yaml"
    _admit(PolicyUnit(
        policy_id="vap-basic-replica-limit",
        source_path=rel,
        kind="vap_policy",
        validations=_vap_validations(_load_yaml(rel)),
        notes="official basic example: static replica ceiling (<=5) on deployments",
    ))

    rel = "content/en/examples/access/deployment-replicas-policy.yaml"
    _admit(PolicyUnit(
        policy_id="vap-param-replica-limit",
        source_path=rel,
        kind="vap_policy",
        validations=_vap_validations(_load_yaml(rel)),
        message_expressions=_vap_message_expressions(_load_yaml(rel)),
        param_kind="ReplicaLimit",
        notes="paramKind ReplicaLimit; messageExpression over params.maxReplicas",
    ))

    rel = "content/en/examples/access/validating-admission-policy-audit-annotation.yaml"
    _admit(PolicyUnit(
        policy_id="vap-audit-annotation",
        source_path=rel,
        kind="vap_policy",
        validations=_vap_validations(_load_yaml(rel)),
        message_expressions=_vap_message_expressions(_load_yaml(rel)),
        message_kind="audit",
        notes="audit annotation valueExpression over object.spec.replicas",
    ))

    rel = "content/en/examples/access/validating-admission-policy-match-conditions.yaml"
    _admit(PolicyUnit(
        policy_id="vap-match-conditions-demo",
        source_path=rel,
        kind="vap_policy",
        validations=_vap_validations(_load_yaml(rel)),
        match_conditions=_vap_match_conditions(_load_yaml(rel)),
        notes="3 upstream matchConditions + demo name/namespace validation",
    ))

    rel = "content/en/examples/access/image-matches-namespace-environment.policy.yaml"
    # Only the dyn-profile-compiling fragment is admitted; the full policy uses
    # namespaceObject/variables (out of profile, recorded as non-compiling in
    # the fetch index).
    _admit(PolicyUnit(
        policy_id="vap-image-env-exempt-fragment",
        source_path=rel,
        kind="vap_policy",
        validations=(
            "'exempt' in object.metadata.labels && object.metadata.labels['exempt'] == 'true'",
        ),
        notes="exempt-label fragment of the image-environment policy (dyn subset)",
    ))

    # CRD-style validation expressions, verbatim from the pinned CEL reference
    # doc (content/en/docs/reference/using-api/cel.md).
    rel = "content/en/docs/reference/using-api/cel.md"
    for pid, expr, note in (
        ("doc-replica-envelope",
         "object.minReplicas <= object.replicas && object.replicas <= object.maxReplicas",
         "replicas inside [minReplicas, maxReplicas]"),
        ("doc-ttl-expiry",
         "has(object.expired) && object.created + object.ttl < object.expired",
         "expiry beyond created+ttl"),
        ("doc-primary-cluster",
         "size(object.clusters.filter(c, c.name == object.primary)) == 1",
         "exactly one cluster named by object.primary"),
        ("doc-state-counts",
         "'Available' in object.stateCounts",
         "stateCounts must contain Available"),
        ("doc-xor-lists",
         "(size(object.list1) == 0) != (size(object.list2) == 0)",
         "exactly one of list1/list2 is empty"),
        ("doc-health-prefix",
         "object.health.startsWith('ok')",
         "health string starts with ok"),
        ("doc-widgets-exists",
         "object.widgets.exists(w, w.key == 'x' && w.foo < 10)",
         "some widget key=x with foo<10"),
        ("doc-name-prefix",
         "object.metadata.name.startsWith(object.prefix)",
         "metadata.name starts with object.prefix"),
        ("doc-set-disjoint",
         "object.set1.all(e, !(e in object.set2))",
         "set1 disjoint from set2"),
    ):
        _admit(PolicyUnit(
            policy_id=pid, source_path=rel, kind="doc_expression_set",
            validations=(expr,), notes="CRD validation doc expression: " + note,
        ))

    rel = "content/en/docs/reference/access-authn-authz/validating-admission-policy.md"
    _admit(PolicyUnit(
        policy_id="doc-env-label-gate",
        source_path=rel,
        kind="doc_expression_set",
        validations=("has(object.metadata.labels) && 'example.com/environment' in object.metadata.labels",),
        notes="VAP doc: environment label must be present",
    ))
    _POLICY_CACHE = policies
    return policies


# ---------------------------------------------------------------------------
# Synthetic object bases (task instances are synthetic; rules are upstream)
# ---------------------------------------------------------------------------

NAME_POOL = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def deployment(name: str, namespace: str, replicas: int,
               labels: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(labels or {})},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {"containers": [{"name": "app", "image": "registry.example/%s:v1" % name}]},
            },
        },
    }


def replica_schedule(name: str, min_replicas: int, replicas: int, max_replicas: int) -> dict[str, Any]:
    """CRD-shaped object matching the doc replica-envelope expression."""
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "ReplicaSchedule",
        "metadata": {"name": name, "namespace": "default"},
        "minReplicas": min_replicas,
        "replicas": replicas,
        "maxReplicas": max_replicas,
    }


def ttl_object(name: str, created: int, ttl: int, expired: int | None) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "apiVersion": "stable.example.com/v1",
        "kind": "LeaseRecord",
        "metadata": {"name": name, "namespace": "default"},
        "created": created,
        "ttl": ttl,
    }
    if expired is not None:
        obj["expired"] = expired
    return obj


# ---------------------------------------------------------------------------
# Mechanism mutators: (violate, repair) proposals over governed fields.
# Each takes (obj, params, rng) and returns a NEW object or None when the
# proposal does not apply. Native evaluation decides acceptance downstream.
# ---------------------------------------------------------------------------

def violate_replica_static(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new["spec"]["replicas"] = 5 + rng.randint(1, 20)
    return new


def repair_replica_static(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new["spec"]["replicas"] = rng.randint(1, 5)
    return new


def violate_replica_param(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    ceiling = int(params.get("maxReplicas", 3))
    new = copy.deepcopy(obj)
    new["spec"]["replicas"] = ceiling + rng.randint(1, 15)
    return new


def repair_replica_param(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    ceiling = int(params.get("maxReplicas", 3))
    new = copy.deepcopy(obj)
    new["spec"]["replicas"] = rng.randint(1, max(ceiling, 1))
    return new


def violate_env_label(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    labels = new["metadata"].setdefault("labels", {})
    labels.pop("example.com/environment", None)
    if rng.random() < 0.4:
        new["metadata"].pop("labels", None)
    return new


def repair_env_label(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    labels = new["metadata"].setdefault("labels", {})
    labels["example.com/environment"] = rng.choice(["test", "staging", "prod"])
    return new


def violate_demo_name(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new["metadata"]["name"] = "demo-%s" % new["metadata"]["name"]
    new["metadata"]["namespace"] = rng.choice(["staging", "prod", "default"])
    return new


def repair_demo_name(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new["metadata"]["name"] = new["metadata"]["name"].replace("demo-", "").replace("demo", "svc")
    return new


def violate_envelope(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    hi = int(new["maxReplicas"])
    lo = int(new["minReplicas"])
    new["replicas"] = hi + rng.randint(1, 5) if rng.random() < 0.6 else max(lo - rng.randint(1, 3), 0)
    return new


def repair_envelope(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    lo, hi = int(new["minReplicas"]), int(new["maxReplicas"])
    new["replicas"] = rng.randint(lo, max(lo, hi))
    return new


def violate_ttl(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    if rng.random() < 0.5:
        new.pop("expired", None)
    else:
        new["expired"] = int(new["created"]) + int(new["ttl"]) - rng.randint(0, 30)
    return new


def repair_ttl(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new["expired"] = int(new["created"]) + int(new["ttl"]) + rng.randint(1, 600)
    return new


MechanismSpec = tuple[str, Callable[..., Any], Callable[..., Any]]

# mechanism key -> (policy_id, violate, repair)
MECHANISMS: dict[str, MechanismSpec] = {
    "static-replica-limit": ("vap-basic-replica-limit", violate_replica_static, repair_replica_static),
    "param-replica-limit": ("vap-param-replica-limit", violate_replica_param, repair_replica_param),
    "env-label-gate": ("doc-env-label-gate", violate_env_label, repair_env_label),
    "demo-name-namespace": ("vap-match-conditions-demo", violate_demo_name, repair_demo_name),
    "replica-envelope": ("doc-replica-envelope", violate_envelope, repair_envelope),
    "ttl-expiry": ("doc-ttl-expiry", violate_ttl, repair_ttl),
}
