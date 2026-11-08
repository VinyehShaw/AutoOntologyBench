"""L04 source plugin: streaming inventory + index of Rhea and ChEBI pinned units.

All parsing is streaming (ElementTree ``iterparse`` / line-by-line TSV): no
whole-file-into-memory parsing. The Rhea RDF (≈2.05M triples) is walked once;
per-reaction/per-compound records are emitted as JSONL.gz index shards under
``<snapshot>/indexes/``. A KB access layer (`load_kb`) lazy-loads the small
derived indexes used by the task-family plugin.

Unit kinds emitted:
  rhea_reaction            (master / directional / bidirectional; equation,
                            status, isTransport, isChemicallyBalanced, ec,
                            citations, sides, children, parents)
  rhea_participant         (rhea_id, side, participant, compound, chebi)
  rhea_compound            (compound iri, chebi, name, formula, charge)
  rhea_xref                (rhea2xrefs.tsv rows: rhea, direction, master, db, id)
  rhea_go_link             (rhea2go.tsv)
  rhea_direction_row       (rhea-directions.tsv)
  rhea_relationship        (rhea-relationships.tsv)
  rhea_chebi_smiles        (rhea-chebi-smiles.tsv)
  chebi_entity             (chebi_lite.owl: id, iri, label, parents, deprecated)
  chebi_name_row           (chebiId_name.tsv)
  chebi_ph73_mapping       (chebi_pH7_3_mapping.tsv)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import IndexManifest, SourceSnapshot, SourceUnit

RH = "http://rdf.rhea-db.org/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
CHEBI_IRI = "http://purl.obolibrary.org/obo/CHEBI_"

INDEX_SUBDIR = "indexes"

KINDS = (
    "rhea_reaction", "rhea_participant", "rhea_compound", "rhea_xref",
    "rhea_go_link", "rhea_direction_row", "rhea_relationship",
    "rhea_chebi_smiles", "chebi_entity", "chebi_name_row", "chebi_ph73_mapping",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


# ---------------------------------------------------------------------------
# Rhea RDF streaming parse
# ---------------------------------------------------------------------------

_MULTIVAL_KEYS = {
    "contains", "contains1", "contains2", "contains3", "contains4", "contains5",
    "contains6", "contains7", "contains8", "contains9", "containsN",
    "side", "substrates", "products", "substratesOrProducts",
    "directionalReaction", "bidirectionalReaction", "citation", "ec",
    "seeAlso", "subClassOf", "transformableTo", "reactivePart",
}
_RESOURCE_VALUED = {
    "contains", "contains1", "contains2", "contains3", "contains4", "contains5",
    "contains6", "contains7", "contains8", "contains9", "containsN",
    "side", "substrates", "products", "substratesOrProducts",
    "directionalReaction", "bidirectionalReaction", "citation", "ec",
    "seeAlso", "subClassOf", "transformableTo", "reactivePart",
    "compound", "status", "chebi", "underlyingChebi", "location",
}


def load_rdf_records(rdf_gz: Path) -> dict[str, dict[str, Any]]:
    """Stream the frozen Rhea RDF once, merging split ``rdf:Description`` blocks
    by ``rdf:about`` into whole-file merged records.

    The Rhea RDF/XML serializes one resource across several Description blocks
    (e.g. a reaction side's ``rh:contains`` fan-out); merging by ``about`` is
    therefore mandatory. ElementTree ``iterparse`` keeps *file* memory bounded
    (elements are cleared as parsed); the merged record map holds ~0.6M small
    dicts (reactions/sides/participants/compounds), which fits comfortably in
    the build host's RAM. The graph is never materialized as triples here.
    """
    rh, rdf, rdfs = "{%s}" % RH, "{%s}" % RDF_NS, "{%s}" % RDFS_NS
    records: dict[str, dict[str, Any]] = {}
    with _open(rdf_gz) as fh:
        for _ev, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != rdf + "Description":
                continue
            about = elem.get(rdf + "about") or ""
            current = records.get(about)
            if current is None:
                current = records[about] = {"about": about}
            for child in elem:
                tag = child.tag
                if not (tag.startswith(rh) or tag.startswith(rdfs)):
                    continue
                key = tag.split("}", 1)[1]
                if key in _RESOURCE_VALUED:
                    val: Any = child.get(rdf + "resource") or ""
                else:
                    val = child.text or ""
                if key in _MULTIVAL_KEYS:
                    bucket = current.setdefault(key, [])
                    if val not in bucket:
                        bucket.append(val)
                elif key not in current:
                    current[key] = val
                elif current[key] != val:
                    prev = current[key]
                    if isinstance(prev, list):
                        if val not in prev:
                            prev.append(val)
                    else:
                        current[key] = [prev, val]
            elem.clear()
    return records


def reaction_units(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Project a merged Description record into a reaction unit (or None)."""
    about = rec["about"]
    accession = rec.get("accession")
    if not (about.startswith(RH) and isinstance(accession, str)
            and accession.startswith("RHEA:")):
        return None
    subs = [s for s in rec.get("subClassOf", []) if isinstance(s, str)]
    rhea_classes = {s.rsplit("/", 1)[-1] for s in subs
                    if s.startswith(RH) and not s.rsplit("/", 1)[-1].isdigit()}
    parent_reactions = [s for s in subs
                        if s.startswith(RH) and s.rsplit("/", 1)[-1].isdigit()]
    children_dir = rec.get("directionalReaction", [])
    children_bi = rec.get("bidirectionalReaction", [])
    if isinstance(children_dir, str):
        children_dir = [children_dir]
    if isinstance(children_bi, str):
        children_bi = [children_bi]
    if "BidirectionalReaction" in rhea_class_set(rec):
        kind = "bidirectional"
    elif "DirectionalReaction" in rhea_class_set(rec):
        kind = "directional"
    else:
        kind = "master"
    return {
        "rhea_id": accession,
        "iri": about,
        "kind": kind,
        "equation": rec.get("equation", ""),
        "status": (rec.get("status") or "").rsplit("/", 1)[-1],
        "is_transport": rec.get("isTransport") == "true",
        "is_balanced": rec.get("isChemicallyBalanced") == "true",
        "ec": sorted(e.rsplit("/", 1)[-1] for e in rec.get("ec", [])),
        "citations": sorted(c.rsplit("/", 1)[-1] for c in rec.get("citation", [])),
        "see_also": sorted(rec.get("seeAlso", [])),
        "sides": sorted(rec.get("side", [])),
        "substrates": sorted(rec.get("substrates", [])),
        "products": sorted(rec.get("products", [])),
        "children_lr_rl": sorted(c.rsplit("/", 1)[-1] for c in children_dir),
        "children_bi": sorted(c.rsplit("/", 1)[-1] for c in children_bi),
        "parent_reactions": sorted(p.rsplit("/", 1)[-1] for p in parent_reactions),
        "reaction_classes": sorted(rhea_classes),
        "transformable_to": sorted(t.rsplit("/", 1)[-1]
                                   for t in rec.get("transformableTo", [])),
    }


def rhea_class_set(rec: dict[str, Any]) -> set[str]:
    return {s.rsplit("/", 1)[-1] for s in rec.get("subClassOf", [])
            if isinstance(s, str) and s.startswith(RH)}


def compound_unit(rec: dict[str, Any]) -> dict[str, Any] | None:
    about = rec["about"]
    accession = rec.get("accession")
    if not (about.startswith(RH + "Compound_") and isinstance(accession, str)
            and accession.startswith("CHEBI:")):
        return None
    return {
        "compound": about,
        "chebi": accession,
        "name": rec.get("name", ""),
        "formula": rec.get("formula", ""),
        "charge": rec.get("charge", ""),
    }


# ---------------------------------------------------------------------------
# TSV streaming
# ---------------------------------------------------------------------------

def iter_tsv(path: Path, header: list[str] | None = None) -> Iterator[dict[str, str]]:
    """Stream TSV rows as dicts. ``header=None`` reads the header from line 1;
    pass an explicit header for headerless upstream files."""
    with path.open(newline="", encoding="utf-8") as fh:
        if header is None:
            header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header):
                cells += [""] * (len(header) - len(cells))
            yield dict(zip(header, cells))


# ---------------------------------------------------------------------------
# ChEBI OWL (chebi_lite) streaming parse
# ---------------------------------------------------------------------------

def iter_chebi_entities(owl_gz: Path) -> Iterator[dict[str, Any]]:
    owl, rdf, rdfs = "{%s}" % OWL_NS, "{%s}" % RDF_NS, "{%s}" % RDFS_NS
    with _open(owl_gz) as fh:
        for _ev, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != owl + "Class":
                continue  # do not clear: subClassOf/label children must survive
                          # until their enclosing owl:Class end event is processed
            about = elem.get(rdf + "about") or ""
            if not about.startswith(CHEBI_IRI):
                elem.clear()
                continue
            chebi_id = "CHEBI:" + about.rsplit("_", 1)[-1]
            label = ""
            parents: list[str] = []
            deprecated = False
            for child in elem:
                if child.tag == rdfs + "label" and not label:
                    label = child.text or ""
                elif child.tag == rdfs + "subClassOf":
                    res = child.get(rdf + "resource") or ""
                    if res.startswith(CHEBI_IRI):
                        parents.append("CHEBI:" + res.rsplit("_", 1)[-1])
                elif child.tag == owl + "deprecated":
                    deprecated = (child.text or "").strip() == "true"
            elem.clear()
            yield {"chebi": chebi_id, "iri": about, "label": label,
                   "parents": sorted(parents), "deprecated": deprecated}


# ---------------------------------------------------------------------------
# Index build (one pass per source file)
# ---------------------------------------------------------------------------

def build_index(snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest:
    root = Path(snapshot.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    shard_shas: dict[str, str] = {}

    def write_shard(kind: str, rows: Iterator[dict[str, Any]]) -> None:
        path = out_dir / f"{kind}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            n = 0
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                n += 1
        counts[kind] = n
        shard_shas[kind] = _sha256_file(path)

    rdf_path = root / "rhea" / "rdf" / "rhea.rdf.gz"
    # single streaming pass over the RDF, fan-out to reaction/participant/compound
    records = load_rdf_records(rdf_path)
    reactions: list[dict[str, Any]] = []
    compounds: list[dict[str, Any]] = []
    sides: dict[str, list[str]] = {}
    part2comp: dict[str, str] = {}
    for about, rec in records.items():
        if not about.startswith(RH):
            continue
        base = about.rsplit("/", 1)[-1]
        if base.startswith("Participant_"):
            comp = rec.get("compound")
            if isinstance(comp, str):
                part2comp[about] = comp
            continue
        if "_" in base:
            _num, _sep, letter = base.rpartition("_")
            if letter in ("L", "R"):
                contains = sorted({p for key, vals in rec.items()
                                   if key.startswith("contains")
                                   for p in (vals if isinstance(vals, list) else [vals])})
                if contains:
                    sides[about] = contains
                continue
        ru = reaction_units(rec)
        if ru is not None:
            reactions.append(ru)
            continue
        cu = compound_unit(rec)
        if cu is not None:
            compounds.append(cu)

    def _rxn_rows() -> Iterator[dict[str, Any]]:
        for r in sorted(reactions, key=lambda r: int(r["rhea_id"].split(":")[1])):
            yield r

    def _cmp_rows() -> Iterator[dict[str, Any]]:
        for c in sorted(compounds, key=lambda c: c["compound"]):
            yield c

    participants: list[dict[str, Any]] = []
    comp2chebi = {c["compound"]: c["chebi"] for c in compounds}
    for side, parts in sides.items():
        base = side.rsplit("/", 1)[-1]
        num, _sep, letter = base.rpartition("_")
        if not (num.isdigit() and letter in ("L", "R")):
            continue
        for part in parts:
            comp = part2comp.get(part)
            chebi = comp2chebi.get(comp or "")
            if comp is None or chebi is None:
                continue  # polymers/macromolecules: no published ChEBI identity
            participants.append({
                "rhea_id": f"RHEA:{num}", "side": letter, "participant": part,
                "compound": comp, "chebi": chebi})
    participants.sort(key=lambda r: (int(r["rhea_id"].split(":")[1]), r["side"],
                                     r["participant"]))

    write_shard("rhea_reaction", _rxn_rows())
    write_shard("rhea_compound", _cmp_rows())
    write_shard("rhea_participant", iter(participants))

    tsv_specs = {
        "rhea_xref": ("rhea/tsv/rhea2xrefs.tsv", None),
        "rhea_go_link": ("rhea/tsv/rhea2go.tsv", None),
        "rhea_direction_row": ("rhea/tsv/rhea-directions.tsv", None),
        "rhea_relationship": ("rhea/tsv/rhea-relationships.tsv", None),
        # headerless upstream files (verified against Rhea release-142 download docs)
        "rhea_chebi_smiles": ("rhea/tsv/rhea-chebi-smiles.tsv", ["CHEBI", "SMILES"]),
        "chebi_name_row": ("rhea/tsv/chebiId_name.tsv", ["CHEBI_ID", "NAME"]),
        "chebi_ph73_mapping": ("rhea/tsv/chebi_pH7_3_mapping.tsv", None),
    }
    for kind, (rel, header) in tsv_specs.items():
        write_shard(kind, iter_tsv(root / rel, header))

    write_shard("chebi_entity", iter_chebi_entities(root / "chebi" / "chebi_lite.owl.gz"))

    manifest_payload = {
        "source_id": snapshot.source_id,
        "release_id": snapshot.release_id,
        "out_dir": str(out_dir),
        "kind_counts": counts,
        "shard_sha256": shard_shas,
        "rhea_rdf_sha256": _sha256_file(rdf_path),
        "chebi_owl_sha256": _sha256_file(root / "chebi" / "chebi_lite.owl.gz"),
    }
    manifest_payload["manifest_sha256"] = _sha256_bytes(
        json.dumps(manifest_payload, sort_keys=True).encode())
    (out_dir / "index_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IndexManifest(source_id=snapshot.source_id, release_id=snapshot.release_id,
                         out_dir=str(out_dir), unit_count=sum(counts.values()),
                         kind_counts=counts)


def iter_source_units(snapshot: SourceSnapshot) -> Iterator[SourceUnit]:
    """Replay the built index shards as SourceUnit stream (build if absent)."""
    out_dir = Path(snapshot.root) / INDEX_SUBDIR
    if not (out_dir / "index_manifest.json").is_file():
        build_index(snapshot, out_dir=out_dir)
    for kind in KINDS:
        shard = out_dir / f"{kind}.jsonl.gz"
        if not shard.is_file():
            continue
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                uid = row.get("rhea_id") or row.get("chebi") or row.get("compound") \
                    or row.get("RHEA_ID") or row.get("CHEBI") or row.get("FROM_REACTION_ID") \
                    or _sha256_bytes(line.encode())[:16]
                yield SourceUnit(
                    unit_id=f"{kind}/{uid}", kind=kind,
                    payload_ref=f"{shard}#{uid}",
                    digest=_sha256_bytes(line.encode()),
                    metadata={"release_id": snapshot.release_id})
