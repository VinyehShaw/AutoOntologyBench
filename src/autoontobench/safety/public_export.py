"""Allowlist-based public world builder (owner: W1, G0 gate).

Builds `worlds/public/<world_id>/` from an original contaminated world dir.
The public projection is constructed from an explicit ALLOWLIST (never by
copy-then-delete; Review-5B Implementation §5 + leak_fix_design.md §2).

Public keeps: schema/world_id/family/domain_candidate/version, action_space,
raw_initial_state, admissible_probe_api, public_evidence, hashes (commitments
only), split/split_version, namespace_alias_map, provenance_category, and
native_oracle_ref.{decision_contract, opa_version}. Everything else moves to
the private root via `private_store`.

`artifact/shapes.ttl` is rebuilt as a STUB: only `sh:NodeShape` shells with
`sh:targetClass`, no `sh:sparql`/`sh:select`/`sh:message` constraint bodies.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

WORLD_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        "schema",
        "world_id",
        "family",
        "domain_candidate",
        "version",
        "action_space",
        "raw_initial_state",
        "admissible_probe_api",
        "public_evidence",
        "expert_reference_w3c",
        "hashes",
        "split",
        "split_version",
        "namespace_alias_map",
        "provenance_category",
        "native_oracle_ref",
    }
)

NATIVE_ORACLE_PUBLIC_KEYS: frozenset[str] = frozenset({"decision_contract", "opa_version"})

MANIFEST_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        "schema",
        "contract_hash",
        "engine",
        "owl_profile",
        "shacl_profile",
        "rdf_syntax",
        "mapping_schema_version",
        "validator_versions",
        "world_spec_version",
        "split_version",
    }
)

_STUB_HEADER = (
    "# STUB (Review-5 G0 repack): constraint bodies removed.\n"
    "# Only sh:NodeShape shells + sh:targetClass are public; gold constraints\n"
    "# live in the private root (expert_reference_w3c/), never in the agent view.\n"
)

_ONTOLOGY_STUB_HEADER = (
    "# STUB (Review-5 G0 repack): source-provenance fingerprint triples removed\n"
    "# (predicates matching source_?sha / source_?commit, e.g. goslice:sourceSha256).\n"
    "# Values survive only in the private root (expert_reference_w3c/ +\n"
    "# artifact_provenance.json). All other triples are byte-equivalent to source.\n"
)

# Matches predicate local names carrying source fingerprints (REPACK_REPORT v2, D3).
SOURCE_PROVENANCE_PREDICATE = re.compile(r"source_?sha|source_?commit", re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_world_json(world: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a parsed world.json into (public_projection, private_remainder)."""
    public = {k: v for k, v in world.items() if k in WORLD_PUBLIC_KEYS}
    private = {k: v for k, v in world.items() if k not in WORLD_PUBLIC_KEYS}
    oracle = public.get("native_oracle_ref")
    if isinstance(oracle, dict):
        public["native_oracle_ref"] = {
            k: v for k, v in oracle.items() if k in NATIVE_ORACLE_PUBLIC_KEYS
        }
    ref = public.get("expert_reference_w3c")
    if isinstance(ref, dict):
        # pointer removed: the gold artifact directory lives under the private root
        public["expert_reference_w3c"] = {"contract_hash": ref.get("contract_hash")}
    return public, private


def project_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split artifact/manifest.json into (public, private provenance block)."""
    public = {k: v for k, v in manifest.items() if k in MANIFEST_PUBLIC_KEYS}
    private = {k: v for k, v in manifest.items() if k not in MANIFEST_PUBLIC_KEYS}
    return public, private


def stub_shapes_turtle(shapes_path: Path) -> str:
    """Return a stub Turtle document: NodeShape shells + targetClass only."""
    from rdflib import Graph
    from rdflib.namespace import RDF, SH

    graph = Graph()
    graph.parse(shapes_path.as_posix(), format="turtle")
    stub = Graph()
    for prefix, ns in graph.namespaces():
        stub.bind(prefix, ns)
    for shape in sorted(set(graph.subjects(RDF.type, SH.NodeShape))):
        stub.add((shape, RDF.type, SH.NodeShape))
        for cls in sorted(set(graph.objects(shape, SH.targetClass))):
            stub.add((shape, SH.targetClass, cls))
    return _STUB_HEADER + stub.serialize(format="turtle")


def stub_ontology_turtle(ontology_path: Path) -> str:
    """Return the ontology minus source-provenance fingerprint triples.

    Uniform conservative rule for every profile: any triple whose predicate
    matches SOURCE_PROVENANCE_PREDICATE is removed from the public artifact.
    When nothing matches, the original text is returned unchanged.
    """
    from rdflib import Graph

    text = ontology_path.read_text(encoding="utf-8")
    graph = Graph()
    graph.parse(data=text, format="turtle")
    hits = [
        triple
        for triple in graph
        if SOURCE_PROVENANCE_PREDICATE.search(str(triple[1]))
    ]
    if not hits:
        return text
    for triple in hits:
        graph.remove(triple)
    for prefix, ns in graph.namespaces():
        graph.bind(prefix, ns)
    return _ONTOLOGY_STUB_HEADER + graph.serialize(format="turtle")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
