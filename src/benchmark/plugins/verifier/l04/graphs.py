"""L04 native evidence graphs: lazy rdflib loaders over the pinned sources.

Graph inventory:
  G_rhea    — the pinned ``rhea/rdf/rhea.rdf.gz`` parsed directly by rdflib
              (≈2.05M triples; the real upstream RDF graph, no transformation).
  G_chebi   — the pinned ``chebi/chebi_lite.owl.gz`` parsed directly by rdflib.
  G_derived — TSV-carried published mappings that are not in the RDF
              (chebiId_name, rhea-chebi-smiles, chebi_pH7_3_mapping) serialized
              under the clearly-marked derived namespace
              ``http://autoontobench/ns/l04/derived/`` with per-file sha256
              provenance recorded in the graph manifest.

Graphs are loaded lazily once per worker process (module-level cache) and
content-hashed; every native evidence row records graph id + sha256 + engine
version so each verdict is independently recomputable.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

RH = "http://rdf.rhea-db.org/"
CHEBI_IRI = "http://purl.obolibrary.org/obo/CHEBI_"
L04D = "http://autoontobench/ns/l04/derived/"

_GRAPH_CACHE: dict[str, Any] = {}
_GRAPH_META: dict[str, dict[str, Any]] = {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rdf_graph_sha(graph_id: str) -> str:
    """Content identity of a loaded graph: sha256 over (source file sha256,
    triple count, engine, derivation flag). Parsing rdflib deterministically
    from the same pinned bytes yields the same graph, so the source hash pins
    the graph content without an expensive canonical re-serialization."""
    meta = _GRAPH_META[graph_id]
    payload = json.dumps(meta, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_rhea_graph(pin_root: Path):
    return _load_rdf(pin_root, "rhea", Path("rhea/rdf/rhea.rdf.gz"))


def load_chebi_graph(pin_root: Path):
    return _load_rdf(pin_root, "chebi", Path("chebi/chebi_lite.owl.gz"))


def _load_rdf(pin_root: Path, graph_id: str, rel: Path):
    import rdflib
    if graph_id in _GRAPH_CACHE:
        return _GRAPH_CACHE[graph_id]
    path = Path(pin_root) / rel
    g = rdflib.Graph()
    with gzip.open(path, "rb") as fh:
        g.parse(data=fh.read(), format="xml")
    _GRAPH_CACHE[graph_id] = g
    _GRAPH_META[graph_id] = {
        "graph_id": graph_id,
        "source_file": str(rel),
        "source_sha256": _sha256_file(path),
        "triples": len(g),
        "engine": f"rdflib {rdflib.__version__}",
        "derived": False,
    }
    return g


def load_derived_graph(pin_root: Path):
    """Derived TSV triples (non-RDF published mappings), clearly namespaced."""
    import rdflib
    from rdflib import Literal, URIRef
    if "derived" in _GRAPH_CACHE:
        return _GRAPH_CACHE["derived"]
    root = Path(pin_root)
    g = rdflib.Graph()
    l04d = rdflib.Namespace(L04D)
    sources: dict[str, str] = {}

    def add_tsv(rel: str, header: list[str] | None, emit) -> None:
        path = root / rel
        sources[rel] = _sha256_file(path)
        with path.open(newline="", encoding="utf-8") as fh:
            hdr = header or fh.readline().rstrip("\n").split("\t")
            for line in fh:
                cells = line.rstrip("\n").split("\t")
                if len(cells) != len(hdr):
                    cells += [""] * (len(hdr) - len(cells))
                emit(dict(zip(hdr, cells)))

    add_tsv("rhea/tsv/chebiId_name.tsv", ["CHEBI_ID", "NAME"], lambda r: g.add((
            URIRef(CHEBI_IRI + r["CHEBI_ID"].split(":")[1]), l04d.rheaName,
            Literal(r["NAME"].strip()))))
    add_tsv("rhea/tsv/rhea-chebi-smiles.tsv", ["CHEBI", "SMILES"], lambda r: g.add((
            URIRef(CHEBI_IRI + r["CHEBI"].split(":")[1]), l04d.chebiSmiles,
            Literal(r["SMILES"]))))
    add_tsv("rhea/tsv/chebi_pH7_3_mapping.tsv", None, lambda r: g.add((
            URIRef(CHEBI_IRI + r["CHEBI"]), l04d.ph73Mapping,
            URIRef(CHEBI_IRI + r["CHEBI_PH7_3"]))))
    add_tsv("rhea/tsv/rhea2xrefs.tsv", None, lambda r: g.add((
            URIRef(RH + r["MASTER_ID"]), l04d.xref,
            Literal(f'{r["DB"]}:{r["ID"]}'))))
    _GRAPH_CACHE["derived"] = g
    _GRAPH_META["derived"] = {
        "graph_id": "derived",
        "source_files": sources,
        "triples": len(g),
        "engine": f"rdflib {rdflib.__version__}",
        "derived": True,
        "namespace": L04D,
    }
    return g


def graph_manifest() -> dict[str, Any]:
    return {gid: {**meta, "graph_sha256": rdf_graph_sha(gid)}
            for gid, meta in _GRAPH_META.items()}
