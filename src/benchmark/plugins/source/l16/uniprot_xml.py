"""L16 XML parser: streaming UniProtKB XML -> canonical records (path B).

Independent second parse path over the pinned ``uniprot_sprot.xml.gz`` using
stdlib ``xml.etree.ElementTree.iterparse``. Emits the same canonical record
schema as ``uniprot_dat`` so the fixed-sample cross-check can require exact
agreement of the two independent parses.
"""
from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l16.records import (
    ECO_TO_GO_CODE, PE_TEXT_TO_LEVEL, XML_FEATURE_TO_DAT, canonical_record,
    norm_space, outermost_paren_groups,
)

NS = "{http://uniprot.org/uniprot}"


def _ns_of(el: ET.Element) -> str:
    """Namespace prefix of the entry (FTP release uses https://, REST http://)."""
    tag = el.tag
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


def _pos(el: ET.Element | None) -> str:
    if el is None:
        return "?"
    pos = el.get("position") or ""
    status = el.get("status") or "certain"
    if not pos:
        return "?"  # position-less endpoint is unknown regardless of status
    if status == "uncertain":
        return "?" + pos
    if status == "less than":
        return "<" + pos
    if status == "greater than":
        return ">" + pos
    return pos


def _feature_location(feat: ET.Element, ns: str) -> tuple[str, str]:
    loc = feat.find(f"{ns}location")
    if loc is None:
        return "?", "?"
    pos = loc.find(f"{ns}position")
    if pos is not None:
        p = _pos(pos)
        return p, p
    begin = _pos(loc.find(f"{ns}begin"))
    end = _pos(loc.find(f"{ns}end"))
    return begin, end


def parse_entry(el: ET.Element) -> dict[str, Any]:
    ns = _ns_of(el)
    dataset = el.get("dataset", "")
    entry_version = int(el.get("version", "0"))

    accessions = [a.text or "" for a in el.findall(f"{ns}accession")]
    entry_name_el = el.find(f"{ns}name")
    entry_name = entry_name_el.text or "" if entry_name_el is not None else ""

    protein_name = ""
    ec_numbers: list[str] = []
    protein_el = el.find(f"{ns}protein")
    if protein_el is not None:
        rec = protein_el.find(f"{ns}recommendedName")
        if rec is not None:
            fn = rec.find(f"{ns}fullName")
            if fn is not None and fn.text:
                protein_name = fn.text
        for ecn in protein_el.iter(f"{ns}ecNumber"):
            if ecn.text:
                ec_numbers.append(ecn.text.strip())

    genes_all: list[str] = []
    gene_synonyms: list[str] = []
    for gene in el.findall(f"{ns}gene"):
        for nm in gene.findall(f"{ns}name"):
            ntype = nm.get("type", "")
            if ntype == "primary":
                genes_all.append(nm.text or "")
            elif ntype == "synonym":
                gene_synonyms.append(nm.text or "")

    organism_el = el.find(f"{ns}organism")
    os_sci = os_common = ""
    os_synonyms: list[str] = []
    taxon_id = 0
    lineage: list[str] = []
    if organism_el is not None:
        for nm in organism_el.findall(f"{ns}name"):
            if nm.get("type") == "scientific":
                os_sci = nm.text or ""
            elif nm.get("type") == "common":
                os_common = nm.text or ""
            elif nm.get("type") == "synonym":
                os_synonyms.append(nm.text or "")
        for ref in organism_el.findall(f"{ns}dbReference"):
            if ref.get("type") == "NCBI Taxonomy":
                taxon_id = int(ref.get("id", "0"))
        lin = organism_el.find(f"{ns}lineage")
        if lin is not None:
            lineage = [t.text or "" for t in lin.findall(f"{ns}taxon")]
    sci_base, sci_groups = outermost_paren_groups(os_sci)
    os_names = sorted(
        set([sci_base] + sci_groups
            + ([os_common] if os_common else []) + os_synonyms) - {""})

    pe_el = el.find(f"{ns}proteinExistence")
    pe_level = PE_TEXT_TO_LEVEL.get(pe_el.get("type", "") if pe_el is not None
                                    else "", 0)

    keywords = [k.text or "" for k in el.findall(f"{ns}keyword")]

    # entry-level evidence map: key -> (eco, pubmed)
    evidence: dict[str, tuple[str, str]] = {}
    for ev in el.findall(f"{ns}evidence"):
        key = ev.get("key", "")
        eco = ev.get("type", "")
        pmid = ""
        src = ev.find(f"{ns}source")
        if src is not None:
            for ref in src.findall(f"{ns}dbReference"):
                if ref.get("type") == "PubMed":
                    pmid = ref.get("id", "")
        evidence[key] = (eco, pmid)

    def ev_sets(attr: str | None) -> tuple[list[str], list[str]]:
        ecos: list[str] = []
        pmids: list[str] = []
        for key in (attr or "").split():
            if key in evidence:
                e, p = evidence[key]
                if e:
                    ecos.append(e)
                if p:
                    pmids.append(p)
        return sorted(set(ecos)), sorted(set(pmids), key=int)

    comments_function: list[dict[str, Any]] = []
    catalytic_activities: list[dict[str, Any]] = []
    subcellular: list[dict[str, Any]] = []
    diseases: list[dict[str, Any]] = []
    isoforms: list[dict[str, Any]] = []
    for comment in el.findall(f"{ns}comment"):
        ctype = comment.get("type", "")
        if ctype == "function":
            for text_el in comment.findall(f"{ns}text"):
                ecos, pmids = ev_sets(text_el.get("evidence"))
                comments_function.append({
                    "text": norm_space(text_el.text or ""),
                    "ecos": ecos, "pubmeds": pmids,
                })
        elif ctype == "subcellular location":
            mol_el = comment.find(f"{ns}molecule")
            scope = mol_el.text or "" if mol_el is not None else ""
            for sub in comment.findall(f"{ns}subcellularLocation"):
                for loc in sub.findall(f"{ns}location"):
                    ecos, _pmids = ev_sets(loc.get("evidence"))
                    subcellular.append({
                        "scope": scope,
                        "term": norm_space(loc.text or ""),
                        "ecos": ecos,
                    })
        elif ctype == "disease":
            dis = comment.find(f"{ns}disease")
            if dis is None:
                continue
            name_el = dis.find(f"{ns}name")
            acr_el = dis.find(f"{ns}acronym")
            mim = ""
            for ref in dis.findall(f"{ns}dbReference"):
                if ref.get("type") == "MIM":
                    mim = ref.get("id", "")
            diseases.append({
                "name": norm_space(name_el.text or "" if name_el is not None else ""),
                "acronym": acr_el.text or "" if acr_el is not None else "",
                "mim": mim,
            })
        elif ctype == "alternative products":
            for iso in comment.findall(f"{ns}isoform"):
                id_el = iso.find(f"{ns}id")
                names = [n.text or "" for n in iso.findall(f"{ns}name")]
                isoforms.append({
                    "iso_id": id_el.text or "" if id_el is not None else "",
                    "name": names[0] if names else "",
                    "synonyms": names[1:],
                })
        elif ctype == "catalytic activity":
            for rxn in comment.findall(f"{ns}reaction"):
                text_el = rxn.find(f"{ns}text")
                rheas: list[str] = []
                ecs: list[str] = []
                for ref in rxn.findall(f"{ns}dbReference"):
                    rtype, rid = ref.get("type", ""), ref.get("id", "")
                    if rtype == "Rhea" and rid.startswith("RHEA:"):
                        rheas.append(rid)
                    elif rtype == "EC":
                        ecs.append(rid)
                ecos, pmids = ev_sets(rxn.get("evidence"))
                physio: list[dict[str, str]] = []
                for pr in comment.findall(f"{ns}physiologicalReaction"):
                    for ref in pr.findall(f"{ns}dbReference"):
                        rid = ref.get("id", "")
                        if ref.get("type") == "Rhea" and rid.startswith("RHEA:"):
                            physio.append({"direction": pr.get("direction", ""),
                                           "rhea_id": rid})
                catalytic_activities.append({
                    "text": norm_space(text_el.text or ""
                                       if text_el is not None else ""),
                    "rhea_ids": rheas, "ec": ecs,
                    "ecos": ecos, "pubmeds": pmids, "physio": physio,
                })

    features: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    for feat in el.findall(f"{ns}feature"):
        ftype_xml = feat.get("type", "")
        ftype = XML_FEATURE_TO_DAT.get(ftype_xml)
        if ftype is None:
            raise ValueError(
                f"unmapped XML feature type {ftype_xml!r} in {entry_name}")
        begin, end = _feature_location(feat, ns)
        features.append({"type": ftype, "begin": begin, "end": end})
        if ftype == "VARIANT":
            orig_el = feat.find(f"{ns}original")
            var_el = feat.find(f"{ns}variation")
            variants.append({
                "begin": begin, "end": end,
                "original": orig_el.text or "" if orig_el is not None else "",
                "variation": var_el.text or "" if var_el is not None else "",
                "var_id": feat.get("id", ""),
            })

    go_refs: list[dict[str, Any]] = []
    rhea_refs: list[str] = [r for c in catalytic_activities
                            for r in c["rhea_ids"]]
    xrefs: list[dict[str, str]] = []
    proteomes: list[str] = []
    for ref in el.findall(f"{ns}dbReference"):
        db, rid = ref.get("type", ""), ref.get("id", "")
        if db == "GO":
            term = ""
            eco = ""
            project = ""
            for prop in ref.findall(f"{ns}property"):
                if prop.get("type") == "term":
                    term = prop.get("value", "")
                elif prop.get("type") == "evidence":
                    eco = prop.get("value", "")
                elif prop.get("type") == "project":
                    project = prop.get("value", "")
            aspect = term[:1] if term[:2] in ("C:", "F:", "P:") else ""
            if eco:
                code = ECO_TO_GO_CODE.get(eco)
                if code is None:
                    raise ValueError(
                        f"unmapped GO evidence ECO {eco!r} in {entry_name}")
                ev = f"{code}:{project}" if project else code
            else:
                ev = ""
            go_refs.append({
                "id": rid, "aspect": aspect,
                "term": term[2:] if aspect else term,
                "evidence": ev,
            })
        elif db == "Rhea":
            rhea_refs.append(rid)
        elif db == "EC":
            continue  # EC dbReferences duplicate <ecNumber>; DAT keeps EC in DE
        elif db == "Proteomes":
            proteomes.append(rid)
            xrefs.append({"db": db, "id": rid})
        else:
            xrefs.append({"db": db, "id": rid})

    pubmed_ids: list[str] = []
    for ref_el in el.findall(f"{ns}reference"):
        cit = ref_el.find(f"{ns}citation")
        if cit is None:
            continue
        for ref in cit.findall(f"{ns}dbReference"):
            if ref.get("type") == "PubMed":
                pubmed_ids.append(ref.get("id", ""))

    seq_el = el.find(f"{ns}sequence")
    seq_len = int(seq_el.get("length", "0")) if seq_el is not None else 0
    seq_version = int(seq_el.get("version", "0")) if seq_el is not None else 0
    crc64 = seq_el.get("checksum", "") if seq_el is not None else ""

    return canonical_record(
        accession=accessions[0] if accessions else "",
        accessions=accessions,
        entry_name=entry_name,
        reviewed=(dataset == "Swiss-Prot"),
        entry_version=entry_version,
        sequence_version=seq_version,
        sequence_length=seq_len,
        sequence_crc64=crc64,
        protein_name=protein_name,
        gene_primary=genes_all[0] if genes_all else "",
        genes_all=genes_all,
        gene_synonyms=gene_synonyms,
        organism_scientific=os_sci,
        organism_common=os_common,
        organism_names=os_names,
        taxon_id=taxon_id,
        lineage=lineage,
        pe_level=pe_level,
        keywords=keywords,
        ec_numbers=ec_numbers,
        comments_function=comments_function,
        catalytic_activities=catalytic_activities,
        subcellular=subcellular,
        diseases=diseases,
        isoforms=isoforms,
        features=features,
        variants=variants,
        go_refs=go_refs,
        rhea_refs=rhea_refs,
        xrefs=xrefs,
        proteomes=proteomes,
        pubmeds=pubmed_ids,
    )


def _iter_entry_bytes(path: Path) -> Iterator[bytes]:
    """Yield raw <entry>...</entry> byte chunks from the (gzipped) XML.

    The UniProt XML is one entry per top-level element; entries start with
    ``<entry `` at column 0 and end with ``</entry>`` at column 0. Chunked byte
    scanning avoids holding the 10+ GiB decompressed stream in one document.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        stream = io.BufferedReader(fh, buffer_size=1 << 22)
        buf = bytearray()
        in_entry = False
        for line in stream:
            if not in_entry and line.startswith(b"<entry "):
                in_entry = True
                buf = bytearray(line)
            elif in_entry:
                buf.extend(line)
                if line.startswith(b"</entry>"):
                    yield bytes(buf)
                    in_entry = False
                    buf = bytearray()


def iter_xml(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    for chunk in _iter_entry_bytes(path):
        el = ET.fromstring(chunk)
        yield parse_entry(el)
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def main(argv: list[str]) -> int:
    import json
    import sys
    limit = int(argv[2]) if len(argv) > 2 else 3
    first = True
    count = 0
    for rec in iter_xml(Path(argv[1]), limit=limit):
        if first:
            print(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False))
            first = False
        count += 1
    print(f"parsed={count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
