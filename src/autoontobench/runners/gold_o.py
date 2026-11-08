"""Gold-O bundles for the four pinned Gatekeeper rule units (owner: W13).

Single source of truth for the W13 gold-O arm set (G2_FULL_MATRIX_PROPOSAL P1
arm GOLD + the CORRUPT/REPAIRED mechanistic pair). Encodes the pinned sample
constraints of ``references/sources/gatekeeper-library`` @ a8d4799d8daf (see
``data/erule/review5/software/SIMULATION_ASSUMPTIONS.md`` §3) as one
``shacl-rule/1`` artifact per unit over the W6 serialized Gatekeeper
vocabulary v1 (``autoontobench.runners.serialization`` — the SAME adapter the
in-loop ``check_action_with_ontology`` seam consumes).

Semantics encoded (post-patch admission, mirroring the native contract
"the post-patch object must be natively compliant"): each rule joins the
proposed ``patch_object`` action to its target object and evaluates the
EFFECTIVE post-patch field (``COALESCE(action override, current state)``):

- ``allowedrepos``        (k8sallowedrepos, Pod, ns default, repos
  ["openpolicyagent/"]): every (single-container, v1 vocabulary) Pod image
  must keep prefix "openpolicyagent/"; absent image is a violation.
- ``requiredlabels``      (k8srequiredlabels, Pod, key-only sample): the Pod
  must carry the label key ``pizza`` post-patch (any value).
- ``replicalimits``       (k8sreplicalimits, Deployment, ranges [{3,50}]):
  effective replicas in [3, 50]; absent replicas is a violation.
- ``block-nodeport-services`` (k8sblocknodeport, Service): effective
  ``spec.type`` must not be "NodePort" (absent type passes, matching the
  native OPA rule, which only fires on type == NodePort).

pySHACL 0.40.1 quirk (pinned; recorded for reproducibility): its SPARQL
pre-flight rejects any query whose raw text matches a case-insensitive
``SERVICE[\\s<]`` regex, so ``gk:Service`` would be misread as a federated
query. The nodeport rule therefore writes the class as the inline IRI
``<urn:aob:gk#Service>`` (identical semantics; "Service>" defeats the regex).
The class keeps its faithful name ``gk:Service`` in ontology.ttl.

Documented under-approximations (inherited from vocabulary v1, NOT silent):
single-container projection (``image_0`` only — every pilot Pod has exactly
one container, verified against all 12 blueprint initial states);
``create_object`` values and patch ``remove`` are not serializable (the
adapter declines them honestly); other action kinds (``delete_object``,
``create_object``) are uncovered by the shapes and yield UNKNOWN.

Variants per unit: ``gold`` (faithful clause), ``corrupt`` (ONE durable
governed semantic mutation, see ``MUTATIONS``), ``repaired`` (gold semantics
restored under a renamed shape — content-distinct, decision-identical, the
W11 fixtures_gk_real precedent). Composition for a task = the union of the
shapes/mappings of the units governing that task's blueprint
(``unit_clause_counts``); ontology.ttl/phi.json are shared verbatim.
"""
# allow: SIZE_OK — pure data tables (the four gold-O SPARQL/TTL rule texts,
# the mutation/witness/mapping tables); executable logic is ~60 LOC
# (corrupt_select / bundle_files / manifest / governed_*).
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

GK_BASE = "urn:aob:gk#"
PROFILE = "shacl-rule/1"

#: The four pinned units (short id -> full source unit id).
UNIT_IDS: dict[str, str] = {
    "allowedrepos": "gatekeeper/general/allowedrepos",
    "requiredlabels": "gatekeeper/general/requiredlabels",
    "replicalimits": "gatekeeper/general/replicalimits",
    "block-nodeport-services": "gatekeeper/general/block-nodeport-services",
}
UNITS: tuple[str, ...] = tuple(UNIT_IDS)

VARIANTS: tuple[str, ...] = ("gold", "corrupt", "repaired")

# ---- shared vocabulary (identical in every per-unit and composed bundle) ----

ONTOLOGY_TTL = """\
@prefix gk: <urn:aob:gk#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

gk:K8sObject a owl:Class .
gk:Pod a owl:Class ; rdfs:subClassOf gk:K8sObject .
gk:Deployment a owl:Class ; rdfs:subClassOf gk:K8sObject .
gk:Service a owl:Class ; rdfs:subClassOf gk:K8sObject .
gk:Action a owl:Class .
gk:PatchObjectAction a owl:Class ; rdfs:subClassOf gk:Action .
gk:object a owl:ObjectProperty .
gk:idempotence_key a owl:DatatypeProperty .
gk:name a owl:DatatypeProperty .
gk:namespace a owl:DatatypeProperty .
gk:label_pizza a owl:DatatypeProperty .
gk:image_0 a owl:DatatypeProperty .
gk:replicas a owl:DatatypeProperty .
gk:service_type a owl:DatatypeProperty .
gk:set_metadata_labels_pizza a owl:DatatypeProperty .
gk:set_spec_containers_0_image a owl:DatatypeProperty .
gk:set_spec_replicas a owl:DatatypeProperty .
gk:set_spec_type a owl:DatatypeProperty .
"""

PHI = {
    "schema": "erule-phi/0.1",
    "base": GK_BASE,
    "types": {
        "patch_object": "PatchObjectAction",
        "Pod": "Pod",
        "Deployment": "Deployment",
        "Service": "Service",
    },
}
PHI_JSON = json.dumps(PHI, indent=2, sort_keys=True)

# ---- per-unit gold rules (SPARQL SELECT bodies over the serialized facts) ----
#
# Placeholders in each select: none — the strings below are the exact rule
# text; corrupt variants are produced by ONE exact replacement (MUTATIONS).

_SHAPE_HEADER = """\
@prefix gk: <urn:aob:gk#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

"""

_SHAPE_TMPL = """\
gk:{shape} a sh:NodeShape ;
  sh:targetClass gk:PatchObjectAction ;
  sh:sparql [
    a sh:SPARQLConstraint ;
    sh:message "{message}" ;
    sh:select \"\"\"{select}\"\"\" ;
  ] .
"""

_RULES: dict[str, dict[str, str]] = {
    "allowedrepos": {
        "shape": "AllowedReposPostState",
        "message": "post-patch Pod container image outside the allowed repos (openpolicyagent/)",
        "select": """PREFIX gk: <urn:aob:gk#>
SELECT $this WHERE {
  $this gk:object ?obj .
  ?obj a gk:Pod ; gk:namespace "default" .
  OPTIONAL { $this gk:set_spec_containers_0_image ?newimg . }
  OPTIONAL { ?obj gk:image_0 ?curimg . }
  BIND(COALESCE(?newimg, ?curimg) AS ?effimg)
  FILTER(!BOUND(?effimg) || !STRSTARTS(STR(?effimg), "openpolicyagent/"))
}""",
    },
    "requiredlabels": {
        "shape": "RequiredLabelsPostState",
        "message": "post-patch Pod still lacks the required label key (pizza)",
        "select": """PREFIX gk: <urn:aob:gk#>
SELECT $this WHERE {
  $this gk:object ?obj .
  ?obj a gk:Pod ; gk:namespace "default" .
  OPTIONAL { $this gk:set_metadata_labels_pizza ?newlab . }
  OPTIONAL { ?obj gk:label_pizza ?curlab . }
  FILTER(!BOUND(?newlab) && !BOUND(?curlab))
}""",
    },
    "replicalimits": {
        "shape": "ReplicaLimitsPostState",
        "message": "post-patch Deployment replicas outside the allowed range [3, 50]",
        "select": """PREFIX gk: <urn:aob:gk#>
SELECT $this WHERE {
  $this gk:object ?obj .
  ?obj a gk:Deployment .
  OPTIONAL { $this gk:set_spec_replicas ?newrep . }
  OPTIONAL { ?obj gk:replicas ?currep . }
  BIND(COALESCE(?newrep, ?currep) AS ?effrep)
  FILTER(!BOUND(?effrep) || ?effrep < 3 || ?effrep > 50)
}""",
    },
    "block-nodeport-services": {
        "shape": "BlockNodePortPostState",
        "message": "post-patch Service type is NodePort (blocked)",
        "select": """PREFIX gk: <urn:aob:gk#>
SELECT $this WHERE {
  $this gk:object ?obj .
  ?obj a <urn:aob:gk#Service> .
  OPTIONAL { $this gk:set_spec_type ?newtype . }
  OPTIONAL { ?obj gk:service_type ?curtype . }
  BIND(COALESCE(?newtype, ?curtype) AS ?efftype)
  FILTER(?efftype = "NodePort")
}""",
    },
}

#: The CORRUPT mutations: ONE durable governed semantic edit per unit, applied
#: as exact string replacements to the gold rule text (every needle must occur
#: exactly once — asserted at build time). Each is chosen so the governed
#: plan actions of the pilot blueprints flip ALLOW(gold) -> DENY(corrupt)
#: and/or the unit's negative witness flips DENY(gold) -> ALLOW(corrupt).
MUTATIONS: dict[str, dict[str, Any]] = {
    "allowedrepos": {
        "edits": [('"openpolicyagent/"', '"evilcorp/"')],
        "description": "allowed repo prefix changed: openpolicyagent/ -> evilcorp/",
    },
    "requiredlabels": {
        "edits": [
            ("set_metadata_labels_pizza", "set_metadata_labels_pasta"),
            ("gk:label_pizza", "gk:label_pasta"),
        ],
        "description": "required label key swapped: pizza -> pasta (action field + state field)",
    },
    "replicalimits": {
        "edits": [
            (
                "FILTER(!BOUND(?effrep) || ?effrep < 3 || ?effrep > 50)",
                "FILTER(BOUND(?effrep) && ?effrep >= 3 && ?effrep <= 50)",
            )
        ],
        "description": "range check inverted: the in-range window [3,50] becomes the DENY condition",
    },
    "block-nodeport-services": {
        "edits": [
            ('FILTER(?efftype = "NodePort")', 'FILTER(?efftype = "ClusterIP")')
        ],
        "description": "blocked service type swapped: NodePort -> ClusterIP",
    },
}

#: Per-unit governed NEGATIVE witnesses: (blueprint, patch action) that the
#: gold O must DENY (and native truth denies too — recorded side by side).
NEGATIVE_WITNESSES: dict[str, dict[str, Any]] = {
    "allowedrepos": {
        "blueprint": "gk_blueprint_07",
        "action": {
            "op": "patch_object",
            "object": "pod/web-1",
            "idempotence_key": "w13-neg-allowedrepos",
            "set": {"spec.containers[0].image": "evilcorp/server:v9"},
        },
        "why": "bad repo image: post-patch Pod image keeps the evilcorp/ prefix",
    },
    "requiredlabels": {
        "blueprint": "gk_blueprint_03",
        "action": {
            "op": "patch_object",
            "object": "pod/web-1",
            "idempotence_key": "w13-neg-requiredlabels",
            "set": {"metadata.labels.app": "web"},
        },
        "why": "missing required label: the patch leaves the Pod without the pizza key",
    },
    "replicalimits": {
        "blueprint": "gk_blueprint_04",
        "action": {
            "op": "patch_object",
            "object": "deployment/app",
            "idempotence_key": "w13-neg-replicalimits",
            "set": {"spec.replicas": 100},
        },
        "why": "out-of-range replicas: 100 exceeds the allowed maximum 50",
    },
    "block-nodeport-services": {
        "blueprint": "gk_blueprint_06",
        "action": {
            "op": "patch_object",
            "object": "service/web-svc",
            "idempotence_key": "w13-neg-nodeport",
            "set": {"spec.type": "NodePort"},
        },
        "why": "NodePort service: the effective post-patch type is the blocked one",
    },
}

#: Per-unit mappings entries (mappings/1.0; unioned on composition).
_MAPPINGS_BY_UNIT: dict[str, list[dict[str, str]]] = {
    "allowedrepos": [
        {"source": "action", "field": "set_spec_containers_0_image",
         "term": "set_spec_containers_0_image", "concept": "PatchObjectAction"},
        {"source": "state", "field": "image_0", "term": "image_0", "concept": "Pod"},
    ],
    "requiredlabels": [
        {"source": "action", "field": "set_metadata_labels_pizza",
         "term": "set_metadata_labels_pizza", "concept": "PatchObjectAction"},
        {"source": "state", "field": "label_pizza", "term": "label_pizza", "concept": "Pod"},
    ],
    "replicalimits": [
        {"source": "action", "field": "set_spec_replicas",
         "term": "set_spec_replicas", "concept": "PatchObjectAction"},
        {"source": "state", "field": "replicas", "term": "replicas", "concept": "Deployment"},
    ],
    "block-nodeport-services": [
        {"source": "action", "field": "set_spec_type",
         "term": "set_spec_type", "concept": "PatchObjectAction"},
        {"source": "state", "field": "service_type", "term": "service_type", "concept": "Service"},
    ],
}

_SHARED_MAPPING_ENTRIES: list[dict[str, str]] = [
    {"source": "action", "field": "object", "term": "object", "concept": "PatchObjectAction"},
    {"source": "action", "field": "idempotence_key",
     "term": "idempotence_key", "concept": "PatchObjectAction"},
    {"source": "state", "field": "type", "term": "type", "concept": "K8sObject"},
    {"source": "state", "field": "name", "term": "name", "concept": "K8sObject"},
    {"source": "state", "field": "namespace", "term": "namespace", "concept": "K8sObject"},
]


def corrupt_select(unit: str) -> str:
    """Apply the unit's single durable mutation to the gold rule text."""
    select = _RULES[unit]["select"]
    for needle, replacement in MUTATIONS[unit]["edits"]:
        if select.count(needle) != 1:
            raise ValueError(
                "mutation needle %r for %r must occur exactly once in the gold rule"
                % (needle, unit)
            )
        select = select.replace(needle, replacement)
    return select


def _shape_block(unit: str, variant: str) -> str:
    rule = _RULES[unit]
    shape = rule["shape"] + ("Repaired" if variant == "repaired" else "")
    select = corrupt_select(unit) if variant == "corrupt" else rule["select"]
    return _SHAPE_TMPL.format(shape=shape, message=rule["message"], select=select)


def mappings_doc(units: tuple[str, ...] | list[str]) -> dict[str, Any]:
    entries = list(_SHARED_MAPPING_ENTRIES)
    for unit in units:
        entries += _MAPPINGS_BY_UNIT[unit]
    return {"schema": "mappings/1.0", "mappings": entries}


def shapes_ttl(units: tuple[str, ...] | list[str], variant: str = "gold") -> str:
    if variant not in VARIANTS:
        raise ValueError("unknown gold-O variant %r" % variant)
    return _SHAPE_HEADER + "\n".join(_shape_block(unit, variant) for unit in units)


def manifest(unit_ids: tuple[str, ...] | list[str], variant: str, *,
             ontology_bytes: bytes, shapes_bytes: bytes,
             mappings_bytes: bytes) -> dict[str, Any]:
    """erule-artifact-manifest/1.0 with the contract hash computed over the
    exact file bytes (the same recompute the harness performs on edit)."""
    from erule.contract.validator import compute_contract_hash

    doc: dict[str, Any] = {
        "schema": "erule-artifact-manifest/1.0",
        "rdf_syntax": "turtle",
        "owl_profile": "owl2rl",
        "shacl_profile": "erule-core/1.0",
        "mapping_schema_version": "mappings/1.0",
        "engine": "pyshacl",
        "world_spec_version": "gatekeeper-library@a8d4799d8daf",
        "split_version": "dev",
        "provenance": {
            "real_source": True,
            "variant": variant,
            "units": list(unit_ids),
            "encodes": "post-patch admission clauses of the pinned gatekeeper-library sample constraints",
            "vocabulary": "W6 serialized gk vocabulary v1 (autoontobench.runners.serialization)",
        },
    }
    doc["contract_hash"] = compute_contract_hash(
        ontology_bytes=ontology_bytes,
        shapes_bytes=shapes_bytes,
        mappings_bytes=mappings_bytes,
        manifest=doc,
    )
    return doc


def bundle_files(units: tuple[str, ...] | list[str], variant: str = "gold") -> dict[str, str]:
    """The artifact file set for the given units+variant (gold-O source of truth)."""
    if variant not in VARIANTS:
        raise ValueError("unknown gold-O variant %r" % variant)
    units = tuple(units)
    unknown = sorted(set(units) - set(UNITS))
    if unknown:
        raise ValueError("unknown units %s (declared: %s)" % (unknown, list(UNITS)))
    shapes = shapes_ttl(units, variant)
    mappings = json.dumps(mappings_doc(units), indent=2, sort_keys=True)
    doc = manifest(
        tuple(UNIT_IDS[u] for u in units), variant,
        ontology_bytes=ONTOLOGY_TTL.encode("utf-8"),
        shapes_bytes=shapes.encode("utf-8"),
        mappings_bytes=mappings.encode("utf-8"),
    )
    return {
        "ontology.ttl": ONTOLOGY_TTL,
        "shapes.ttl": shapes,
        "manifest.json": json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False),
        "mappings.json": mappings,
        "phi.json": PHI_JSON,
    }


def unit_files(unit: str, variant: str = "gold") -> dict[str, str]:
    return bundle_files((unit,), variant)


#: Mapping from the dev split's full unit ids back to the short ids here.
def unit_of_source_id(source_unit_id: str) -> str:
    for short, full in UNIT_IDS.items():
        if source_unit_id == full:
            return short
    raise ValueError("unknown source rule unit %r" % source_unit_id)


def governed_units(blueprint_doc: Mapping[str, Any]) -> tuple[str, ...]:
    """The units governing a blueprint, read off its ``unit_clause_counts``."""
    return tuple(
        sorted(unit_of_source_id(u) for u in blueprint_doc.get("unit_clause_counts", {}))
    )


def governed_blueprints(unit: str, bp_root: Path) -> list[str]:
    """All blueprint ids under ``bp_root`` governed by ``unit`` (sorted)."""
    out: list[str] = []
    for path in sorted(bp_root.glob("gk_blueprint_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if unit in governed_units(doc):
            out.append(doc["blueprint_id"])
    return out
