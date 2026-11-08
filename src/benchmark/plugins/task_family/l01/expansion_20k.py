"""L01 20K expansion: eight new Gatekeeper policy families over the pinned library.

Adds genuinely new constraint mechanisms that the legacy 8K L01 build never used
(Pod Security Standards + RBAC + the v2 image allowlist). Nothing here renames or
re-parameterises a legacy family: every mechanism maps to a distinct upstream
``src/<unit>/src.rego`` that is absent from the legacy ``MECHANISMS`` table.

The module registers itself into the *existing* lane registries
(``engine.MECHANISMS`` / ``families.FAMILY_BY_ID``) at import time, so the legacy
``GatekeeperTaskFamilies.propose`` / ``ground`` and the pinned OPA verifier are
reused unchanged. Registration is in-memory only; no legacy file or artifact is
modified.

Expansion quota: 89 new instances (7 families x 11 + 1 family x 12).
"""
from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from benchmark.plugins.task_family.l01 import engine
from benchmark.plugins.task_family.l01 import families as fam
from benchmark.plugins.task_family.l01.families import FamilySpec

LANE_ID = "L01"
SEMANTIC_PROFILE = fam.SEMANTIC_PROFILE
GENERATOR_VERSION = "l01-families-20k/1.0.0"


def _containers(obj: dict[str, Any]) -> list[dict[str, Any]]:
    return engine._containers(obj)


def _unexempt(container: dict[str, Any]) -> None:
    """Force a container image that no ``exemptImages`` entry can match.

    Several PSS samples ship an exempt fixture; the task generator also draws
    those as base objects, so the mutators normalise the image first. The native
    OPA check still decides whether the violation is real.
    """
    container["image"] = "nginx:1.25"


# ---------------------------------------------------------------------------
# Mutators: (violate, repair) proposals over governed fields.
# ---------------------------------------------------------------------------

def violate_privileged(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    container = rng.choice(cs)
    _unexempt(container)
    container.setdefault("securityContext", {})["privileged"] = True
    return new


def repair_privileged(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    for container in _containers(new):
        _unexempt(container)
        container.setdefault("securityContext", {})["privileged"] = False
    return new


def violate_capabilities(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    container = rng.choice(cs)
    caps = container.setdefault("securityContext", {}).setdefault("capabilities", {})
    caps["add"] = ["SYS_ADMIN"]
    caps["drop"] = []
    return new


def repair_capabilities(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    allowed = list(params.get("allowedCapabilities") or [])
    drops = list(params.get("requiredDropCapabilities") or ["ALL"])
    for container in _containers(new):
        caps = container.setdefault("securityContext", {}).setdefault("capabilities", {})
        caps["add"] = allowed[:1]
        caps["drop"] = drops
    return new


def violate_seccomp(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    container = rng.choice(cs)
    _unexempt(container)
    container.setdefault("securityContext", {})["seccompProfile"] = {"type": "Unconfined"}
    return new


def repair_seccomp(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    profiles = [p for p in (params.get("allowedProfiles") or []) if p != "Localhost"]
    profile = profiles[0] if profiles else "RuntimeDefault"
    for container in _containers(new):
        _unexempt(container)
        container.setdefault("securityContext", {})["seccompProfile"] = {"type": profile}
    return new


def violate_host_namespaces(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    spec["hostPID"] = True
    spec["hostIPC"] = True
    return new


def repair_host_namespaces(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    spec = new.setdefault("spec", {})
    spec["hostPID"] = False
    spec["hostIPC"] = False
    return new


def violate_readonly_rootfs(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    for container in cs:
        _unexempt(container)
        container.setdefault("securityContext", {})["readOnlyRootFilesystem"] = False
    return new


def repair_readonly_rootfs(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    for container in _containers(new):
        _unexempt(container)
        container.setdefault("securityContext", {})["readOnlyRootFilesystem"] = True
    return new


def violate_forbidden_sysctls(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new.setdefault("spec", {}).setdefault("securityContext", {})["sysctls"] = [
        {"name": "kernel.msgmax", "value": "65536"},
    ]
    return new


def repair_forbidden_sysctls(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    allowed = list(params.get("allowedSysctls") or ["net.ipv4.tcp_syncookies"])
    new.setdefault("spec", {}).setdefault("securityContext", {})["sysctls"] = [
        {"name": allowed[0], "value": "1"},
    ]
    return new


def violate_anonymous_binding(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    new.setdefault("roleRef", {})["name"] = "cluster-role-disallowed"
    return new


def repair_anonymous_binding(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    allowed = list(params.get("allowedRoles") or ["cluster-role-1"])
    new.setdefault("roleRef", {})["name"] = allowed[0]
    return new


def violate_allowedimages_v2(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    cs = _containers(new)
    if not cs:
        return None
    rng.choice(cs)["image"] = "nginx:latest"
    return new


def repair_allowedimages_v2(obj: dict[str, Any], params: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    new = copy.deepcopy(obj)
    images = list(params.get("allowedImages") or [])
    prefix = next((i[:-1] for i in images if i.endswith("*")), images[0] if images else "openpolicyagent/")
    for container in _containers(new):
        container["image"] = "%sapp:v1" % prefix
    return new


#: mechanism key -> (unit, violate, repair, cross_object)
NEW_MECHANISMS: dict[str, tuple[str, Callable[..., Any], Callable[..., Any], bool]] = {
    "privileged-containers": (
        "pod-security-policy/privileged-containers",
        violate_privileged, repair_privileged, False),
    "capabilities": (
        "pod-security-policy/capabilities",
        violate_capabilities, repair_capabilities, False),
    "seccompv2": (
        "pod-security-policy/seccompv2",
        violate_seccomp, repair_seccomp, False),
    "host-namespaces": (
        "pod-security-policy/host-namespaces",
        violate_host_namespaces, repair_host_namespaces, False),
    "read-only-root-filesystem": (
        "pod-security-policy/read-only-root-filesystem",
        violate_readonly_rootfs, repair_readonly_rootfs, False),
    "forbidden-sysctls": (
        "pod-security-policy/forbidden-sysctls",
        violate_forbidden_sysctls, repair_forbidden_sysctls, False),
    "disallowanonymous": (
        "general/disallowanonymous",
        violate_anonymous_binding, repair_anonymous_binding, False),
    "allowedreposv2": (
        "general/allowedreposv2",
        violate_allowedimages_v2, repair_allowedimages_v2, False),
}

NEW_PREFERRED_SAMPLES: dict[str, list[str]] = {
    "privileged-containers": ["samples/psp-privileged-container"],
    "capabilities": ["samples/psp-capabilities-restricted"],
    "seccompv2": ["samples/psp-seccomp-restricted"],
    "host-namespaces": ["samples/psp-host-namespace"],
    "read-only-root-filesystem": ["samples/psp-readonlyrootfilesystem"],
    "forbidden-sysctls": ["samples/psp-forbidden-sysctls"],
    "disallowanonymous": ["samples/no-anonymous-bindings"],
    "allowedreposv2": ["samples/repo-must-be-openpolicyagent"],
}

#: Eight new families; low/medium/high instance counts sum to 89.
NEW_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l01-f21-privileged-container-removal", fam.REGIME_REPAIR, "repair_compliance",
        ("privileged-containers",),
        "Remove privileged execution from every container so the privileged-container "
        "admission policy passes.",
        2, 5, 4),
    FamilySpec(
        "l01-f22-capability-restriction", fam.REGIME_REPAIR, "repair_compliance",
        ("capabilities",),
        "Restrict container Linux capabilities to the allowed set and drop every required "
        "capability.",
        2, 5, 4),
    FamilySpec(
        "l01-f23-seccomp-profile-enforcement", fam.REGIME_REPAIR, "repair_compliance",
        ("seccompv2",),
        "Set an approved seccomp profile on every container so the seccomp policy passes.",
        2, 5, 4),
    FamilySpec(
        "l01-f24-host-namespace-isolation", fam.REGIME_REPAIR, "repair_compliance",
        ("host-namespaces",),
        "Isolate workloads from the host PID and IPC namespaces.",
        2, 5, 4),
    FamilySpec(
        "l01-f25-readonly-rootfs-enforcement", fam.REGIME_REPAIR, "repair_compliance",
        ("read-only-root-filesystem",),
        "Enforce a read-only root filesystem for every container.",
        2, 5, 4),
    FamilySpec(
        "l01-f26-forbidden-sysctl-blocking", fam.REGIME_REPAIR, "repair_compliance",
        ("forbidden-sysctls",),
        "Remove forbidden sysctls and keep only the explicitly allowed sysctl set.",
        2, 5, 4),
    FamilySpec(
        "l01-f27-anonymous-binding-repair", fam.REGIME_REPAIR, "repair_compliance",
        ("disallowanonymous",),
        "Qualify RBAC bindings so anonymous or unauthenticated subjects are never bound "
        "to an unsanctioned role.",
        2, 5, 4),
    FamilySpec(
        "l01-f28-image-allowlist-v2", fam.REGIME_REPAIR, "repair_compliance",
        ("allowedreposv2",),
        "Move every workload image onto the allowedImages prefixes enforced by the v2 "
        "image allowlist.",
        3, 5, 4),
)

NEW_FAMILY_IDS: tuple[str, ...] = tuple(f.family_id for f in NEW_FAMILIES)

_ORIGINAL_PICK_SAMPLE = fam._pick_sample


def _revalidate_chain(target: Path) -> None:
    """List every ancestor from GK_ROOT down to ``target`` to clear stale
    NFSv3 negative dentries before the actual file lookup."""
    try:
        rel = target.relative_to(engine.GK_ROOT)
    except ValueError:
        return
    cur = engine.GK_ROOT
    for part in rel.parts:
        cur = cur / part
        try:
            os.listdir(cur)
        except OSError:
            pass


def _load_unit_resilient(unit: str, sample: str, attempts: int = 5) -> Any:
    """Load a pinned unit, tolerating stale NFSv3 negative dentries."""
    path = engine.GK_ROOT / "library" / unit / sample / "constraint.yaml"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            _revalidate_chain(path.parent)
            return engine.load_unit(unit, sample)
        except Exception as exc:  # noqa: BLE001 - transient mount lookups
            last = exc
            time.sleep(0.05 * (attempt + 1))
    raise last  # type: ignore[misc]


def _pick_sample_20k(mechanism: str, rng: random.Random) -> str | None:
    """Lock new mechanisms to their one real sample (never fall back to a
    parameter-only sample directory that carries no case fixtures)."""
    if mechanism not in NEW_PREFERRED_SAMPLES:
        return _ORIGINAL_PICK_SAMPLE(mechanism, rng)
    unit = engine.MECHANISMS[mechanism][0]
    errors: list[str] = []
    for sample in NEW_PREFERRED_SAMPLES[mechanism]:
        try:
            _load_unit_resilient(unit, sample)
            return sample
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append("%s: %s" % (sample, exc))
    raise RuntimeError(
        "cannot load pinned sample for %s (%s): %s"
        % (mechanism, unit, "; ".join(errors)))


def register() -> None:
    """Idempotently install the new mechanisms/families into the lane registries."""
    for key, spec in NEW_MECHANISMS.items():
        engine.MECHANISMS.setdefault(key, spec)
    for key, samples in NEW_PREFERRED_SAMPLES.items():
        fam.PREFERRED_SAMPLES.setdefault(key, samples)
    for spec in NEW_FAMILIES:
        if spec.family_id not in fam.FAMILY_BY_ID:
            fam.FAMILIES = fam.FAMILIES + (spec,)
            fam.FAMILY_BY_ID[spec.family_id] = spec
    fam._pick_sample = _pick_sample_20k
    for key in NEW_PREFERRED_SAMPLES:
        unit = engine.MECHANISMS[key][0]
        for sample in NEW_PREFERRED_SAMPLES[key]:
            _revalidate_chain(engine.GK_ROOT / "library" / unit / sample)


def expected_total() -> int:
    return sum(f.total for f in NEW_FAMILIES)


def list_families() -> list[str]:
    return list(NEW_FAMILY_IDS)