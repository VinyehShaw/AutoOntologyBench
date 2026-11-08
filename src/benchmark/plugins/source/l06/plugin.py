"""L06 source plugin: pin/index the GO snapshot and enumerate knowledge units.

``build_index`` is the heavy stage (run once per pinned release):
  1. OBO leg: stream-parse go-basic.obo -> terms + is_a/relation edges.
  2. OWL leg: rdflib-parse go-basic.owl -> is_a edges (direct subClassOf) and
     relation edges (owl:Restriction onProperty/someValuesFrom superclasses).
  3. Differential extraction check: the OBO edge set and the OWL edge set must
     be *exactly equal* per predicate; any mismatch fails the build honestly.
  4. Declared profile closures (go-rel-closure/1, see closure.py docstring).
  5. GAF parse: positive (NOT-qualified excluded) gene->term rows by aspect;
     cross-source checks vs the ontology (obsolete refs, alt_id refs).

Outputs under ``external_knowledge/indexes/gene-ontology/<release_id>/``:
  terms.jsonl, edges.jsonl.gz, differential.json, graph.json.gz (adjacency),
  closures.json.gz (isa_anc / partof_anc / reg_anc{,_pos,_neg} per term),
  depth.json, annotations.jsonl.gz, gene_terms.json.gz, manifest.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l06.closure import (
    REGULATES_KINDS,
    compute_depths,
    partof_ancestors,
    regulates_ancestors_fast,
    transitive_ancestors,
)
from benchmark.schema.plugin_protocols import (
    FetchReceipt,
    IndexManifest,
    SourceSnapshot,
    SourceUnit,
)
from source_registry.fetchers.l06_go import (
    INDEX_ROOT,
    RELEASE_ID,
    SOURCE_ID,
    GeneOntologySource,
    iter_gaf_rows,
    iter_obo_terms,
    load_snapshot,
)

GO_PREFIX = "http://purl.obolibrary.org/obo/GO_"
REL_IRI_TO_NAME = {
    "http://purl.obolibrary.org/obo/BFO_0000050": "part_of",
    "http://purl.obolibrary.org/obo/BFO_0000051": "has_part",
    "http://purl.obolibrary.org/obo/BFO_0000066": "occurs_in",
    "http://purl.obolibrary.org/obo/RO_0002211": "regulates",
    "http://purl.obolibrary.org/obo/RO_0002212": "negatively_regulates",
    "http://purl.obolibrary.org/obo/RO_0002213": "positively_regulates",
    "http://purl.obolibrary.org/obo/RO_0002215": "capable_of",
    "http://purl.obolibrary.org/obo/RO_0002216": "capable_of_part_of",
}
IAO_REPLACED_BY = "http://purl.obolibrary.org/obo/IAO_0100001"


def _go_id(uri: str) -> str | None:
    if uri.startswith(GO_PREFIX):
        return "GO:" + uri[len(GO_PREFIX):]
    return None


def extract_edges_owl(owl_path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    """rdflib leg: (is_a edges, relation edges) from go-basic.owl.

    is_a: (S rdfs:subClassOf O) with both GO classes.
    relation: (S rdfs:subClassOf _:r), (_:r owl:onProperty P),
              (_:r owl:someValuesFrom O) with P a known relation IRI and O a
              GO class -> (S, pred_name, O).
    """
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    g = rdflib.Graph()
    g.parse(str(owl_path), format="xml")
    isa: set[tuple[str, str]] = set()
    rel: set[tuple[str, str, str]] = set()
    for s, o in g.subject_objects(RDFS.subClassOf):
        if not isinstance(o, rdflib.term.Node):
            continue
        s_id = _go_id(str(s))
        if s_id is None:
            continue
        if (o, RDF.type, OWL.Restriction) in g:
            prop = g.value(o, OWL.onProperty)
            tgt = g.value(o, OWL.someValuesFrom)
            if prop is None or tgt is None:
                continue
            pred = REL_IRI_TO_NAME.get(str(prop))
            o_id = _go_id(str(tgt))
            if pred and o_id:
                rel.add((s_id, pred, o_id))
        else:
            o_id = _go_id(str(o))
            if o_id is not None:
                isa.add((s_id, o_id))
    return isa, rel


def extract_edges_obo(obo_path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]],
                                               dict[str, dict[str, Any]]]:
    """OBO leg: (is_a edges, relation edges, term metadata)."""
    isa: set[tuple[str, str]] = set()
    rel: set[tuple[str, str, str]] = set()
    terms: dict[str, dict[str, Any]] = {}
    for t in iter_obo_terms(obo_path):
        terms[t.go_id] = {
            "go_id": t.go_id, "name": t.name, "namespace": t.namespace,
            "obsolete": t.obsolete, "is_a": list(t.is_a),
            "replaced_by": list(t.replaced_by), "consider": list(t.consider),
            "alt_ids": list(t.alt_ids), "stanza_sha256": t.stanza_sha256,
        }
        for p in t.is_a:
            isa.add((t.go_id, p))
        for pred, o in t.relationships:
            rel.add((t.go_id, pred, o))
    return isa, rel, terms


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


class L06SourcePlugin:
    """SourcePlugin for the pinned GO + GOA snapshot."""

    source_id = SOURCE_ID

    def __init__(self, release_root: Path | None = None) -> None:
        src = GeneOntologySource()
        self.release_root = Path(release_root) if release_root else src.release_dir

    @property
    def obo_path(self) -> Path:
        return self.release_root / "ontology" / "go-basic.obo"

    @property
    def owl_path(self) -> Path:
        return self.release_root / "ontology" / "go-basic.owl"

    @property
    def gaf_path(self) -> Path:
        return self.release_root / "annotations" / "goa_human.gaf.gz"

    def enumerate_units(self, snapshot: SourceSnapshot | None = None) -> Iterator[SourceUnit]:
        for t in iter_obo_terms(self.obo_path):
            yield SourceUnit(
                unit_id=f"goterm:{t.go_id}",
                kind="go_term_obsolete" if t.obsolete else "go_term",
                payload_ref=f"ontology/go-basic.obo#{t.go_id}",
                digest=t.stanza_sha256,
                metadata={"name": t.name, "namespace": t.namespace},
            )

    # ------------------------------------------------------------------
    def pin_and_index(self, source_config: dict[str, Any] | None = None) -> SourceSnapshot:
        """Build-order §3.2 facade: verify the pinned snapshot (sha256 + sanity),
        confirm the reasoning index exists with a passing OBO/OWL differential,
        and return the typed SourceSnapshot. The heavy index build is a separate
        one-off step (``python -m benchmark.plugins.source.l06.plugin``); this
        method never re-downloads and never rebuilds a valid index."""
        cfg = source_config or {}
        src = GeneOntologySource()
        release = src.discover_releases()[0]
        if not src.release_dir.is_dir():
            raise FileNotFoundError(
                f"pinned snapshot missing: {src.release_dir} "
                "(run: .venv/bin/python -m source_registry.fetchers.l06_go fetch)")
        receipt = FetchReceipt(
            source_id=SOURCE_ID, release_id=release.release_id,
            status="FETCHED_COMPLETE", bytes_downloaded=0, file_count=0,
            cache_dir=str(src.release_dir), sha256_manifest="",
            fetched_at="")
        report = src.verify_snapshot(receipt)
        if not report.ok:
            raise RuntimeError(f"snapshot verification failed: {report.checks}")
        index_dir = Path(cfg.get("index_dir", INDEX_ROOT))
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.is_file():
            if cfg.get("build_index_if_missing", True):
                self.build_index(index_dir)
            else:
                raise FileNotFoundError(
                    f"reasoning index missing: {index_dir} "
                    "(run: .venv/bin/python -m benchmark.plugins.source.l06.plugin)")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            diff = manifest.get("differential", {})
            if not (diff.get("isa_equal") and diff.get("rel_equal")):
                raise RuntimeError(
                    "index differential check failed; rebuild the index")
        return load_snapshot()

    # ------------------------------------------------------------------
    def build_index(self, out_dir: Path, *, with_owl_leg: bool = True) -> IndexManifest:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        isa_obo, rel_obo, terms = extract_edges_obo(self.obo_path)
        t1 = time.time()

        live = {tid for tid, m in terms.items() if not m["obsolete"]}
        # Restrict task graph to non-obsolete terms (annotation targets are current).
        isa = {(s, o) for (s, o) in isa_obo if s in live and o in live}
        rel = {(s, p, o) for (s, p, o) in rel_obo if s in live and o in live}

        differential: dict[str, Any] = {"with_owl_leg": with_owl_leg}
        if with_owl_leg:
            isa_owl, rel_owl = extract_edges_owl(self.owl_path)
            t2 = time.time()
            isa_owl_live = {(s, o) for (s, o) in isa_owl if s in live and o in live}
            rel_owl_live = {(s, p, o) for (s, p, o) in rel_owl if s in live and o in live}
            differential.update({
                "isa_obo": len(isa), "isa_owl": len(isa_owl_live),
                "isa_equal": isa == isa_owl_live,
                "isa_only_obo": sorted(isa - isa_owl_live)[:20],
                "isa_only_owl": sorted(isa_owl_live - isa)[:20],
                "rel_obo": len(rel), "rel_owl": len(rel_owl_live),
                "rel_equal": rel == rel_owl_live,
                "rel_only_obo": sorted(rel - rel_owl_live)[:20],
                "rel_only_owl": sorted(rel_owl_live - rel)[:20],
                "owl_parse_seconds": round(t2 - t1, 1),
            })
            if not (differential["isa_equal"] and differential["rel_equal"]):
                _write_json(out_dir / "differential.json", differential)
                raise RuntimeError(
                    "OBO/OWL differential extraction mismatch; see differential.json")
        _write_json(out_dir / "differential.json", differential)

        isa_parents: dict[str, list[str]] = {}
        isa_children: dict[str, list[str]] = {}
        rel_out: dict[str, dict[str, list[str]]] = {}
        rel_in_reg: dict[str, list[str]] = {}
        for s, o in sorted(isa):
            isa_parents.setdefault(s, []).append(o)
            isa_children.setdefault(o, []).append(s)
        for s, p, o in sorted(rel):
            rel_out.setdefault(s, {}).setdefault(p, []).append(o)
            if p in REGULATES_KINDS:
                rel_in_reg.setdefault(o, []).append(s)

        t3 = time.time()
        isa_anc: dict[str, list[str]] = {}
        partof_anc: dict[str, list[str]] = {}
        reg_anc: dict[str, list[str]] = {}
        reg_pos_anc: dict[str, list[str]] = {}
        reg_neg_anc: dict[str, list[str]] = {}
        partof_targets = {s: by.get("part_of", []) for s, by in rel_out.items()}
        for tid in sorted(live):
            anc = transitive_ancestors(tid, isa_parents)
            isa_anc[tid] = sorted(anc)
            partof_anc[tid] = sorted(partof_ancestors(tid, isa_parents, partof_targets))
        isa_anc_frozen = {k: frozenset(v) for k, v in isa_anc.items()}
        isa_desc: dict[str, set[str]] = {}
        for t, anc in isa_anc.items():
            for a in anc:
                isa_desc.setdefault(a, set()).add(t)
        isa_desc_frozen = {k: frozenset(v) for k, v in isa_desc.items()}
        for tid in sorted(live):
            reg_anc[tid] = sorted(regulates_ancestors_fast(
                tid, isa_desc_frozen, rel_in_reg, isa_children))
            reg_pos_anc[tid] = sorted(regulates_ancestors_fast(
                tid, isa_desc_frozen,
                _reverse_rel(rel_out, "positively_regulates"), isa_children))
            reg_neg_anc[tid] = sorted(regulates_ancestors_fast(
                tid, isa_desc_frozen,
                _reverse_rel(rel_out, "negatively_regulates"), isa_children))
        t4 = time.time()
        depths = compute_depths(isa_parents)

        # GAF: positive annotation rows (NOT-qualifier excluded from positives),
        # plus cross-source diagnostics vs the ontology.
        n_rows = n_not = n_obs_ref = n_altid_ref = 0
        obs_ref_rows: list[dict[str, str]] = []
        gene_terms: dict[str, dict[str, set[str]]] = {}
        alt_of: dict[str, str] = {}
        for tid, m in terms.items():
            for alt in m["alt_ids"]:
                alt_of.setdefault(alt, tid)
        live_or_known = live | set(alt_of)
        ann_path = out_dir / "annotations.jsonl.gz"
        with gzip.open(ann_path, "wt", encoding="utf-8") as fh:
            for row in iter_gaf_rows(self.gaf_path):
                n_rows += 1
                go_id = row["go_id"]
                is_not = "NOT" in row["qualifier"].split("|")
                if is_not:
                    n_not += 1
                status = ("obsolete" if go_id in terms and terms[go_id]["obsolete"]
                          else "alt_id" if go_id in alt_of and go_id not in terms
                          else "live" if go_id in live else "unknown")
                if status == "obsolete":
                    n_obs_ref += 1
                    if len(obs_ref_rows) < 200000:
                        obs_ref_rows.append({**row, "replaced_by":
                                             terms[go_id]["replaced_by"],
                                             "consider": terms[go_id]["consider"]})
                elif status == "alt_id":
                    n_altid_ref += 1
                if not is_not and status in ("live", "alt_id") and go_id in live_or_known:
                    canon = go_id if go_id in live else alt_of[go_id]
                    rec = {"symbol": row["symbol"], "db_object_id": row["db_object_id"],
                           "go_id": canon, "aspect": row["aspect"],
                           "evidence": row["evidence"], "qualifier": row["qualifier"]}
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    gene_terms.setdefault(row["symbol"], {}).setdefault(
                        row["aspect"], set()).add(canon)
        t5 = time.time()

        # serialize
        with (out_dir / "terms.jsonl").open("w", encoding="utf-8") as fh:
            for tid in sorted(terms):
                fh.write(json.dumps(terms[tid], sort_keys=True) + "\n")
        with gzip.open(out_dir / "edges.jsonl.gz", "wt", encoding="utf-8") as fh:
            for s, o in sorted(isa):
                fh.write(json.dumps({"s": s, "p": "is_a", "o": o}) + "\n")
            for s, p, o in sorted(rel):
                fh.write(json.dumps({"s": s, "p": p, "o": o}) + "\n")
        _write_json_gz(out_dir / "graph.json.gz", {
            "isa_parents": isa_parents, "isa_children": isa_children,
            "rel_out": rel_out,
        })
        _write_json_gz(out_dir / "closures.json.gz", {
            "isa_anc": isa_anc, "partof_anc": partof_anc, "reg_anc": reg_anc,
            "reg_pos_anc": reg_pos_anc, "reg_neg_anc": reg_neg_anc,
        })
        _write_json(out_dir / "depth.json", depths)
        _write_json_gz(out_dir / "gene_terms.json.gz", {
            sym: {asp: sorted(ids) for asp, ids in sorted(by_asp.items())}
            for sym, by_asp in sorted(gene_terms.items())
        })
        with gzip.open(out_dir / "obsolete_refs.jsonl.gz", "wt", encoding="utf-8") as fh:
            for row in obs_ref_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        manifest = {
            "source_id": self.source_id, "release_id": RELEASE_ID,
            "profile": "go-rel-closure/1",
            "counts": {
                "terms_total": len(terms), "terms_live": len(live),
                "terms_obsolete": len(terms) - len(live),
                "isa_edges_live": len(isa), "rel_edges_live": len(rel),
                "gaf_rows": n_rows, "gaf_not_rows": n_not,
                "gaf_obsolete_refs": n_obs_ref, "gaf_altid_refs": n_altid_ref,
                "genes_annotated": len(gene_terms),
                "terms_with_partof_ancestors": sum(1 for v in partof_anc.values() if v),
                "terms_with_regulators": sum(1 for v in reg_anc.values() if v),
            },
            "differential": differential,
            "files": {p.name: _file_sha(p) for p in sorted(out_dir.iterdir()) if p.is_file()},
            "timings_s": {"obo_leg": round(t1 - t0, 1),
                          "closures": round(t4 - t3, 1),
                          "gaf": round(t5 - t4, 1)},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(out_dir / "manifest.json", manifest)
        return IndexManifest(
            source_id=self.source_id, release_id=RELEASE_ID, out_dir=str(out_dir),
            unit_count=len(terms) + 1,
            kind_counts={"go_term": len(live), "go_term_obsolete": len(terms) - len(live),
                         "gaf_annotations": 1})

    @staticmethod
    def load_index(out_dir: Path) -> dict[str, Any]:
        out_dir = Path(out_dir)
        with gzip.open(out_dir / "graph.json.gz", "rt", encoding="utf-8") as fh:
            graph = json.load(fh)
        with gzip.open(out_dir / "closures.json.gz", "rt", encoding="utf-8") as fh:
            closures = json.load(fh)
        with gzip.open(out_dir / "gene_terms.json.gz", "rt", encoding="utf-8") as fh:
            gene_terms = json.load(fh)
        terms: dict[str, dict[str, Any]] = {}
        with (out_dir / "terms.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                m = json.loads(line)
                terms[m["go_id"]] = m
        return {
            "manifest": json.loads((out_dir / "manifest.json").read_text()),
            "terms": terms,
            "depth": json.loads((out_dir / "depth.json").read_text()),
            "graph": graph,
            "closures": closures,
            "gene_terms": gene_terms,
        }


def _reverse_rel(rel_out: dict[str, dict[str, list[str]]], pred: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s, by in rel_out.items():
        for o in by.get(pred, ()):
            out.setdefault(o, []).append(s)
    return out


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build L06 GO index")
    ap.add_argument("--out", default=str(
        Path("external_knowledge/indexes/gene-ontology") / RELEASE_ID))
    ap.add_argument("--skip-owl-leg", action="store_true",
                    help="skip the rdflib differential leg (NOT for production)")
    args = ap.parse_args()
    plugin = L06SourcePlugin()
    manifest = plugin.build_index(Path(args.out), with_owl_leg=not args.skip_owl_leg)
    print(json.dumps({**manifest.__dict__}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
