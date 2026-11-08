"""Public-facts serialization bridge (owner: W6; Review-5 M3).

Closes the M2 live-seam gap (``data/erule/review5/evaluation/REPORT.md` §5):
``check_action_with_ontology`` hands the ontology consumer PUBLIC facts, but
the Graw serializer (``erule.runtime.raw_graph``) admits only
scalar/list/``$ref`` values, so a nested native observation (e.g. Gatekeeper
``{"objects": ...}`` with ``labels``/``containers``) or a nested action
(``set`` map) honestly PARSE_ERRORs. This module is the additive seam: a
per-domain **PublicFactsAdapter** projects the raw public observation and the
proposed action into the domain's documented serialized vocabulary BEFORE the
consumer sees them.

One-directional rule (hard): the bridge feeds ONLY the ontology-opinion path
(``check_action_with_ontology``). Native truth (``perform_action`` /
``probe_native``) always receives the raw action; ``scoring.interventions``
receives explicitly serialized replay steps. Adapted facts never reach the
native domain.

Adapter convention (for W2/W3/W4 domain modules):

    def to_public_facts(
        state_payload: dict, action: dict
    ) -> tuple[dict, dict]:  # (facts_state, facts_action)
        ...

A domain module that exposes ``to_public_facts`` is picked up by the
``module`` convention below; any composition root may alternatively call
``register_adapter(key, fn)`` with ``key`` = the domain's module path or its
``DOMAIN_ID`` attribute. An adapter must raise :class:`AdapterDeclined` for
any state/action shape outside its documented vocabulary — the session then
falls back to the RAW passthrough (marker ``serialization: "raw"``), which
may honestly PARSE_ERROR downstream. An adapter must never guess or drop
meaning silently; under-approximation belongs in the documented vocabulary,
not in silent key-dropping.

Resolution order (first hit wins):

1. ``registered`` — explicit ``register_adapter`` (key = module path or
   ``DOMAIN_ID``);
2. ``module`` — the domain's own module exposes ``to_public_facts``;
3. ``default`` — a W6-shipped default adapter (currently: Gatekeeper v1,
   because the serialized gk vocabulary is W6-defined (M2) and W2's domain
   module is frozen to W6);
4. ``raw`` — passthrough fallback, marker ``serialization: "raw"``.

The marker is reported verbatim in the ``check_action_with_ontology``
payload (``serialization`` field) so run evidence records which path
produced the facts. See
``data/erule/review5/evaluation/serialization_bridge.md``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: Adapter signature: (state_payload, action) -> (facts_state, facts_action).
PublicFactsAdapter = Callable[
    [dict[str, Any], dict[str, Any]], "tuple[dict[str, Any], dict[str, Any]]"
]


class AdapterDeclined(ValueError):
    """The adapter cannot serialize this state/action shape within its
    documented vocabulary. The caller falls back to the raw passthrough
    (which may honestly PARSE_ERROR downstream — never a guessed verdict)."""


@dataclass(frozen=True)
class ResolvedAdapter:
    adapter: PublicFactsAdapter | None
    marker: str  # "registered:<key>" | "module:<path>" | "default:<key>" | "raw"
    origin: str  # registered | module | default | raw
    key: str | None


_REGISTRY: dict[str, PublicFactsAdapter] = {}
_DEFAULT_ADAPTERS: dict[str, PublicFactsAdapter] = {}


def register_adapter(key: str, adapter: PublicFactsAdapter) -> None:
    """Explicitly register ``adapter`` for a domain module path or DOMAIN_ID.

    Overwriting an existing key is allowed (a domain team taking ownership of
    its vocabulary re-registers from its own composition root).
    """
    if not key:
        raise ValueError("adapter key must be a non-empty string")
    _REGISTRY[key] = adapter


def _domain_keys(domain: object) -> list[str]:
    keys = [type(domain).__module__]
    domain_id = getattr(domain, "DOMAIN_ID", None)
    if isinstance(domain_id, str) and domain_id:
        keys.append(domain_id)
    return keys


def resolve_adapter(domain: object) -> ResolvedAdapter:
    """Resolve the adapter for a live domain instance (see module docstring)."""
    keys = _domain_keys(domain)
    for key in keys:
        if key in _REGISTRY:
            return ResolvedAdapter(_REGISTRY[key], "registered:%s" % key, "registered", key)
    module = sys.modules.get(type(domain).__module__)
    hook = getattr(module, "to_public_facts", None) if module is not None else None
    if callable(hook):
        module_path = type(domain).__module__
        return ResolvedAdapter(hook, "module:%s" % module_path, "module", module_path)
    for key in keys:
        if key in _DEFAULT_ADAPTERS:
            return ResolvedAdapter(_DEFAULT_ADAPTERS[key], "default:%s" % key, "default", key)
    return ResolvedAdapter(None, "raw", "raw", None)


def project_for_ontology(
    domain: object, state_payload: Mapping[str, Any], action: Mapping[str, Any]
) -> "tuple[dict[str, Any], dict[str, Any], str]":
    """Project raw public facts through the resolved adapter.

    Returns ``(facts_state, facts_action, marker)``. On no adapter or
    :class:`AdapterDeclined`, returns the inputs unchanged with marker
    ``"raw"``.
    """
    resolved = resolve_adapter(domain)
    if resolved.adapter is None:
        return dict(state_payload), dict(action), "raw"
    try:
        facts_state, facts_action = resolved.adapter(dict(state_payload), dict(action))
    except AdapterDeclined:
        return dict(state_payload), dict(action), "raw"
    return facts_state, facts_action, resolved.marker


# ---- Gatekeeper serialized vocabulary v1 (single definition; W6-shipped default) ----
#
# Identical to the W6 M2 O-attempt vocabulary (previously defined in
# ``tests/agent/fixtures_gk_real.py``, which now re-exports these functions).
# The Graw serializer admits scalar/list/``$ref`` values only, so the nested
# Gatekeeper public observation is flattened deterministically:
#   per object ``type``/``name``/``namespace``/``label_<k>``/``image_<i>``/
#   ``replicas``/``service_type``; per action ``kind``(=op) + ``object``
#   (``$ref``) + ``idempotence_key`` + ``set_<dotted path normalized>``.
# ``op``↔``kind`` bridge: the W6 action envelope keys on ``kind``, the
# Gatekeeper domain on ``op``; bridged actions carry BOTH with identical
# values (a mismatch is declined, never guessed).
#
# Documented under-approximations (v1): object-level ``apiVersion``,
# ``initContainers``/``ephemeralContainers``, and container ``name``s are not
# projected (the O encodes clauses only over the projected fields — the same
# facts the M2 replay consumed). STATE projection is allowed to be a
# documented subset; the ACTION envelope is NOT: any action key outside
# {op, kind, object, idempotence_key, set} (e.g. ``value`` of create_object,
# ``remove`` of patch_object) changes what native truth would apply while the
# O would see nothing of it — the adapter declines rather than check a
# different action than the one performed.

GATEKEEPER_DOMAIN_MODULE = "autoontobench.core.domains.gatekeeper"
_GK_ACTION_KEYS = frozenset({"op", "kind", "object", "idempotence_key", "set"})
_GK_STATE_KEYS = frozenset({"objects"})


def _norm_path(path: str) -> str:
    return path.replace("[", "_").replace("]", "").replace(".", "_")


def gatekeeper_facts_state(state_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Gatekeeper public observation payload -> Graw-serializable flat records."""
    unknown = set(state_payload) - _GK_STATE_KEYS
    if unknown:
        raise AdapterDeclined("gatekeeper state keys outside v1 vocabulary: %s" % sorted(unknown))
    objects = state_payload.get("objects") or {}
    if not isinstance(objects, Mapping):
        raise AdapterDeclined("gatekeeper state 'objects' must be a mapping")
    facts: dict[str, Any] = {}
    for oid, entry in sorted(objects.items()):
        if not isinstance(oid, str) or not isinstance(entry, Mapping):
            raise AdapterDeclined("gatekeeper object records must be string-keyed mappings")
        record: dict[str, Any] = {
            "type": entry.get("kind"),
            "name": entry.get("name"),
            "namespace": entry.get("namespace", "default"),
        }
        labels = entry.get("labels") or {}
        if not isinstance(labels, Mapping):
            raise AdapterDeclined("gatekeeper labels must be a mapping")
        for key, value in sorted(labels.items()):
            if not isinstance(key, str):
                raise AdapterDeclined("gatekeeper label keys must be strings")
            record["label_%s" % key] = value
        containers = entry.get("containers") or []
        if not isinstance(containers, list):
            raise AdapterDeclined("gatekeeper containers must be a list")
        for index, container in enumerate(containers):
            if not isinstance(container, Mapping):
                raise AdapterDeclined("gatekeeper container entries must be mappings")
            record["image_%d" % index] = container.get("image")
        for field in ("replicas", "service_type"):
            if field in entry:
                record[field] = entry[field]
        facts[oid] = record
    return facts


def gatekeeper_facts_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Native Gatekeeper action -> serialized vocabulary (v1: patch/delete envelope + set)."""
    unknown = set(action) - _GK_ACTION_KEYS
    if unknown:
        raise AdapterDeclined(
            "gatekeeper action keys outside v1 vocabulary: %s (e.g. create_object 'value' "
            "or patch 'remove' are not serializable in v1)" % sorted(unknown)
        )
    op, kind = action.get("op"), action.get("kind")
    if op is None and kind is None:
        raise AdapterDeclined("gatekeeper action carries neither 'op' nor 'kind'")
    if op is not None and kind is not None and op != kind:
        raise AdapterDeclined("op/kind bridge mismatch: %r != %r" % (op, kind))
    obj = action.get("object")
    if not isinstance(obj, str) or not obj:
        raise AdapterDeclined("gatekeeper action.object must be a non-empty object id string")
    out: dict[str, Any] = {
        "kind": op if op is not None else kind,
        "object": {"$ref": obj},
        "idempotence_key": action.get("idempotence_key", ""),
    }
    set_map = action.get("set") or {}
    if not isinstance(set_map, Mapping):
        raise AdapterDeclined("gatekeeper action.set must be a mapping of dotted path -> value")
    for path, value in sorted(set_map.items()):
        if not isinstance(path, str):
            raise AdapterDeclined("gatekeeper action.set keys must be dotted-path strings")
        out["set_%s" % _norm_path(path)] = value
    return out


def gatekeeper_to_public_facts(
    state_payload: dict[str, Any], action: dict[str, Any]
) -> "tuple[dict[str, Any], dict[str, Any]]":
    """Gatekeeper PublicFactsAdapter entry point (registered default)."""
    return gatekeeper_facts_state(state_payload), gatekeeper_facts_action(action)


# W6-shipped default (see module docstring, resolution rule 3). W2's domain
# module stays untouched; if it later grows its own ``to_public_facts``, the
# ``module`` convention takes precedence over this default automatically.
_DEFAULT_ADAPTERS[GATEKEEPER_DOMAIN_MODULE] = gatekeeper_to_public_facts
