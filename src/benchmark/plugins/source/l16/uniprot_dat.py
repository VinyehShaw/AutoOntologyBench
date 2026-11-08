"""L16 DAT parser: streaming UniProtKB flat-file -> canonical records (path A).

Primary parse path over the pinned ``uniprot_sprot.dat.gz``. Yields one
canonical record (``records.canonical_record``) per entry. Pure stdlib,
line-oriented state machine; fails loudly on structural surprises.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l16.records import (
    canonical_record, norm_space, outermost_paren_groups,
    strip_evidence_braces,
)

_ID_RE = re.compile(r"^ID   (\S+)\s+(Reviewed|Unreviewed);\s+(\d+) AA\.")
_DT_SEQ_RE = re.compile(r"sequence version (\d+)")
_DT_ENTRY_RE = re.compile(r"entry version (\d+)")
_PE_RE = re.compile(r"^PE   (\d):")
_OX_RE = re.compile(r"NCBI_TaxID=(\d+)")
_OS_RE = re.compile(r"^OS   (.*)$")
_DISEASE_RE = re.compile(
    r"^(?P<name>.*?)(?: \((?P<acr>[^()]*)\))?(?: \[MIM:(?P<mim>\d+)\])?:\s*(?P<rest>.*)$")
_FT_KEY_RE = re.compile(r"^FT   (\S+)\s+(\S.*)$")
_FT_LOC_CONT = re.compile(r"^FT                   (\S[^/]\S*)$")
_FT_QUAL = re.compile(r"^FT                   /(\w+)(?:=(.*))?$")
_VARIANT_AA_RE = re.compile(r"^([A-Za-z]+) -> ([A-Za-z]+)(?: \((.*)\))?\s*$")
_SQ_RE = re.compile(r"^SQ   SEQUENCE\s+(\d+) AA;\s+\d+ MW;\s+([0-9A-F]+) CRC64;")
_ISO_NAME_RE = re.compile(r"Name=([^;]+);")
_ISO_ID_RE = re.compile(r"IsoId=([^;]+);")
_ISO_SYN_RE = re.compile(r"Synonyms=([^;]+);")
_CAT_RHEA_RE = re.compile(r"RHEA:(\d+)")
_CAT_EC_RE = re.compile(r"EC=([0-9][0-9A-Za-z.\-]*)\s*;")
_CAT_TEXT_RE = re.compile(r"Reaction=(.*?);\s*(?:Xref=|EC=|Evidence=|$)")
_CAT_PHYSIO_RE = re.compile(
    r"PhysiologicalDirection=([^;]+?);\s*Xref=Rhea:(RHEA:\d+)")


def _parse_catalytic(text: str) -> dict[str, Any]:
    """One CATALYTIC ACTIVITY block -> reaction dict.

    Master Rhea ids come from the reaction's own ``Xref=`` segment only;
    ``PhysiologicalDirection`` segments carry directional child Rhea ids and
    are recorded under ``physio`` (mirroring XML ``physiologicalReaction``).
    """
    physio = [{"direction": m.group(1).strip(), "rhea_id": m.group(2)}
              for m in _CAT_PHYSIO_RE.finditer(text)]
    main = text.split("PhysiologicalDirection=")[0]
    tm = _CAT_TEXT_RE.search(main)
    rhea_ids = ["RHEA:" + m for m in _CAT_RHEA_RE.findall(main)]
    ecs = _CAT_EC_RE.findall(main)
    _clean, ecos, pmids = strip_evidence_braces(main)
    return {"text": norm_space(tm.group(1)) if tm else "",
            "rhea_ids": rhea_ids, "ec": ecs, "ecos": ecos,
            "pubmeds": pmids, "physio": physio}


def _split_list_field(blob: str, key: str) -> list[str]:
    m = re.search(re.escape(key) + r"=([^;]*);", blob)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def _parse_de(de_lines: list[str]) -> tuple[str, list[str]]:
    """Return (protein_name, ec_numbers) from the DE block."""
    blob = " ".join(line[5:].strip() for line in de_lines)
    blob = re.sub(r"\{[^{}]*\}", "", blob)  # evidence flags on names
    name = ""
    m = re.search(r"RecName: Full=([^;]+);", blob)
    if m:
        name = m.group(1).strip()
    ecs = re.findall(r"EC=([0-9][0-9A-Za-z.\-]*)\s*;", blob)
    return norm_space(name), sorted(set(ecs))


def _parse_gn(gn_lines: list[str]) -> tuple[str, list[str], list[str]]:
    blob = " ".join(line[5:].strip() for line in gn_lines)
    blob = re.sub(r"\{[^{}]*\}", "", blob)  # evidence flags on gene names
    names: list[str] = []
    for m in re.finditer(r"Name=([^;]*);", blob):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok:
                names.append(tok)
    syns: list[str] = []
    for m in re.finditer(r"Synonyms=([^;]*);", blob):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok:
                syns.append(tok)
    return (names[0], names, sorted(set(syns))) if names else ("", [], [])


_SCI_QUALIFIER = re.compile(
    r"^(strain|isolat|serovar|serotype|subspecies|subsp\.|variety|var\.|"
    r"cultivar|cv\.|forma|f\.|biovar|pathovar|pv\.|lineage|ecotype|clone|"
    r"haplotype|genotype|group|subgroup|subtype|isolate|sp\.|ssp\.)",
    re.IGNORECASE)


def _parse_os(os_lines: list[str]) -> tuple[str, str, list[str]]:
    """Split OS text into (scientific_display, common_display, name_set).

    The name set (base name + every outermost parenthesized group) is the
    cross-format identity: XML distributes the groups over
    scientific/common/synonym name elements while DAT packs them into one
    line, so group placement is display-only but the token set must agree
    exactly.
    """
    text = norm_space(" ".join(line[5:].strip() for line in os_lines)).rstrip(".")
    base, groups = outermost_paren_groups(text)
    name_set = sorted(set([base] + groups) - {""})
    if not groups:
        return base, "", name_set
    sci_groups = [g for g in groups if _SCI_QUALIFIER.match(g)]
    common_groups = [g for g in groups if not _SCI_QUALIFIER.match(g)]
    scientific = base
    if sci_groups:
        scientific = f"{base} " + " ".join(f"({g})" for g in sci_groups)
    common = common_groups[-1] if common_groups else ""
    return scientific.strip(), common, name_set


def _parse_cc_blocks(cc_lines: list[str]) -> list[tuple[str, str]]:
    """Split CC lines into (topic, full_text) blocks."""
    blocks: list[tuple[str, list[str]]] = []
    cur_topic = ""
    for line in cc_lines:
        body = line[5:].rstrip() if len(line) > 5 else ""
        m = re.match(r"^-!- ([A-Z/ ]+):\s*(.*)$", body)
        if m:
            cur_topic = m.group(1).strip()
            blocks.append((cur_topic, [m.group(2)]))
        elif blocks and cur_topic:
            blocks[-1][1].append(body.strip())
    return [(t, norm_space(" ".join(parts))) for t, parts in blocks]


def _split_subcell_locations(text: str) -> tuple[str, list[tuple[str, list[str]]], str]:
    """Parse one SUBCELLULAR LOCATION block -> (scope, [(term, ecos)], note).

    Layout: optional ``[Isoform N]: `` scope prefix; period-separated location
    sentences each with optional ``{ECO...}`` evidence; optional trailing
    ``Note=...`` free text (excluded from the cross-check).
    """
    scope = ""
    m = re.match(r"^\[([^\]]+)\]:\s*(.*)$", text)
    if m:
        scope, text = m.group(1).strip(), m.group(2)
    note = ""
    nm = re.search(r"\sNote=(.*)$", text)
    if nm:
        note = nm.group(1)
        text = text[: nm.start()].rstrip(". ")
    out: list[tuple[str, list[str]]] = []
    for seg in re.split(r"\.\s+", text):
        seg = seg.strip()
        if not seg:
            continue
        # A location sentence is "Term {eco}; Topology {eco}; Orientation {eco}":
        # only the first part (with its own braces) is the location term.
        term_part = seg.split(";")[0].strip()
        clean, ecos, _pmids = strip_evidence_braces(term_part)
        term = clean.rstrip(".").strip()
        # XML splits comma-joined components into separate <location> elements
        # and capitalizes the first character of each component.
        for comp in term.split(","):
            comp = comp.strip()
            if comp:
                out.append((comp[:1].upper() + comp[1:], ecos))
    return scope, out, note


def _parse_alt_products(text: str) -> list[dict[str, Any]]:
    """Parse an ALTERNATIVE PRODUCTS block into isoform dicts."""
    isoforms: list[dict[str, Any]] = []
    text = re.sub(r"\{[^{}]*\}", "", text)  # evidence flags on isoform names
    for chunk in re.split(r"(?<=;)\s+(?=Name=)", text):
        idm = _ISO_ID_RE.search(chunk)
        if not idm:
            continue
        nm = _ISO_NAME_RE.search(chunk)
        syns: list[str] = []
        sm = _ISO_SYN_RE.search(chunk)
        if sm:
            syns = [s.strip() for s in sm.group(1).split(",") if s.strip()]
        # multi-entry isoforms list several ids ("P0DPB4-1, Q3TI53-5"); XML
        # keeps the first id per isoform element.
        iso_id = idm.group(1).split(",")[0].strip()
        isoforms.append({
            "iso_id": iso_id,
            "name": nm.group(1).strip() if nm else "",
            "synonyms": syns,
        })
    return isoforms


def _finish_feature(feats: list[dict[str, Any]], variants: list[dict[str, Any]],
                    cur: dict[str, Any] | None) -> None:
    if not cur:
        return
    loc = cur.get("loc", "")
    loc = loc.replace(" ", "")
    # isoform-scoped position "P08195-1:264" -> canonical position "264"
    loc = re.sub(r"[A-Z][A-Z0-9]*-[0-9]+:", "", loc)
    if ".." in loc:
        begin, end = loc.split("..", 1)
    else:
        begin = end = loc
    feats.append({"type": cur["type"], "begin": begin, "end": end})
    if cur["type"] == "VARIANT":
        note = cur.get("note", "")
        original = variation = ""
        vm = _VARIANT_AA_RE.match(note)
        if vm:
            original, variation = vm.group(1), vm.group(2)
        variants.append({
            "begin": begin, "end": end,
            "original": original, "variation": variation,
            "var_id": cur.get("id", ""),
        })


def parse_entry(lines: list[str]) -> dict[str, Any]:
    """Parse one entry's DAT lines (terminator ``//`` excluded)."""
    id_m = _ID_RE.match(lines[0])
    if not id_m:
        raise ValueError(f"bad ID line: {lines[0][:80]!r}")
    entry_name, status, seq_len_id = id_m.group(1), id_m.group(2), int(id_m.group(3))

    accessions: list[str] = []
    seq_version = entry_version = 0
    de_lines: list[str] = []
    gn_lines: list[str] = []
    os_lines: list[str] = []
    lineage: list[str] = []
    taxon_id = 0
    pe_level = 0
    keywords: list[str] = []
    cc_lines: list[str] = []
    features: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    go_refs: list[dict[str, Any]] = []
    rhea_refs: list[str] = []
    xrefs: list[dict[str, str]] = []
    proteomes: list[str] = []
    pubmed_ids: list[str] = []
    crc64 = ""
    seq_len = seq_len_id

    cur_ft: dict[str, Any] | None = None
    i = 1
    n = len(lines)
    while i < n:
        line = lines[i]
        code = line[:2]
        if code == "FT":
            m = _FT_KEY_RE.match(line)
            if m:
                _finish_feature(features, variants, cur_ft)
                cur_ft = {"type": m.group(1), "loc": m.group(2).strip()}
            elif line.startswith("FT                   /"):
                qm = _FT_QUAL.match(line)
                if qm and cur_ft is not None:
                    qkey, qval = qm.group(1), (qm.group(2) or "").strip()
                    if qval.startswith('"') and not qval.endswith('"'):
                        j = i + 1
                        while j < n and not lines[j].rstrip().endswith('"'):
                            qval += " " + lines[j][21:].strip()
                            j += 1
                        if j < n:
                            qval += " " + lines[j][21:].strip()
                            i = j
                    cur_ft[qkey] = qval.strip('"')
            elif _FT_LOC_CONT.match(line) and cur_ft is not None:
                cur_ft["loc"] += line[21:].strip()
        else:
            if cur_ft is not None:
                _finish_feature(features, variants, cur_ft)
                cur_ft = None
            if code == "AC":
                accessions.extend(
                    t.strip() for t in line[5:].split(";") if t.strip())
            elif code == "DT":
                sm = _DT_SEQ_RE.search(line)
                if sm:
                    seq_version = int(sm.group(1))
                em = _DT_ENTRY_RE.search(line)
                if em:
                    entry_version = int(em.group(1))
            elif code == "DE":
                de_lines.append(line)
            elif code == "GN":
                gn_lines.append(line)
            elif code == "OS":
                os_lines.append(line)
            elif code == "OC":
                lineage.extend(
                    t.strip() for t in line[5:].split(";") if t.strip())
            elif code == "OX":
                xm = _OX_RE.search(line)
                if xm:
                    taxon_id = int(xm.group(1))
            elif code == "PE":
                pm = _PE_RE.match(line)
                if pm:
                    pe_level = int(pm.group(1))
            elif code == "KW":
                keywords.extend(
                    t.strip().rstrip(".") for t in line[5:].split(";")
                    if t.strip().rstrip("."))
            elif code == "CC":
                body = line[5:]
                if (body.startswith("---")
                        or "Copyrighted by the UniProt Consortium" in body
                        or "Distributed under the Creative Commons" in body):
                    i += 1
                    continue  # per-entry license trailer, not annotation content
                cc_lines.append(line)
            elif code == "DR":
                body = line[5:].rstrip().rstrip(".")
                parts = [p.strip() for p in body.split(";")]
                if len(parts) >= 2:
                    db, rid = parts[0], parts[1]
                    if db == "GO" and len(parts) >= 3:
                        term_field = parts[2]
                        aspect = term_field[:1] if term_field[:2] in (
                            "C:", "F:", "P:") else ""
                        term = term_field[2:] if aspect else term_field
                        go_refs.append({
                            "id": rid, "aspect": aspect, "term": term,
                            "evidence": parts[3] if len(parts) > 3 else "",
                        })
                    elif db == "Rhea":
                        rhea_refs.append(rid)
                    elif db == "Proteomes":
                        proteomes.append(rid)
                        xrefs.append({"db": db, "id": rid})
                    else:
                        xrefs.append({"db": db, "id": rid})
            elif code == "RX":
                for pm in re.findall(r"PubMed=(\d+)", line):
                    pubmed_ids.append(pm)
            elif code == "SQ":
                sm = _SQ_RE.match(line)
                if sm:
                    seq_len = int(sm.group(1))
                    crc64 = sm.group(2)
        i += 1
    if cur_ft is not None:
        _finish_feature(features, variants, cur_ft)

    protein_name, ec_numbers = _parse_de(de_lines)
    gene_primary, genes_all, gene_synonyms = _parse_gn(gn_lines)
    os_sci, os_common, os_names = _parse_os(os_lines)

    comments_function: list[dict[str, Any]] = []
    catalytic_activities: list[dict[str, Any]] = []
    subcellular: list[dict[str, Any]] = []
    diseases: list[dict[str, Any]] = []
    isoforms: list[dict[str, Any]] = []
    for topic, text in _parse_cc_blocks(cc_lines):
        if topic == "FUNCTION":
            clean, ecos, pmids = strip_evidence_braces(text)
            comments_function.append(
                {"text": clean, "ecos": ecos, "pubmeds": pmids})
        elif topic == "SUBCELLULAR LOCATION":
            scope, locs, _note = _split_subcell_locations(text)
            for term, ecos in locs:
                subcellular.append({"scope": scope, "term": term, "ecos": ecos})
        elif topic == "DISEASE":
            if text.startswith("Note="):
                continue
            dm = _DISEASE_RE.match(text)
            if not dm:
                continue
            name = norm_space(dm.group("name"))
            if not name:
                continue
            diseases.append({
                "name": name,
                "acronym": (dm.group("acr") or "").strip(),
                "mim": dm.group("mim") or "",
            })
        elif topic == "ALTERNATIVE PRODUCTS":
            isoforms.extend(_parse_alt_products(text))
        elif topic == "CATALYTIC ACTIVITY":
            catalytic_activities.append(_parse_catalytic(text))
        elif topic in ("SIMILARITY", "CAUTION", "SEQUENCE CAUTION"):
            pass  # evidence of other comment types is not normalized

    for cat in catalytic_activities:
        rhea_refs.extend(cat["rhea_ids"])

    if lineage:
        lineage[-1] = lineage[-1].rstrip(".")
    return canonical_record(
        accession=accessions[0] if accessions else "",
        accessions=accessions,
        entry_name=entry_name,
        reviewed=(status == "Reviewed"),
        entry_version=entry_version,
        sequence_version=seq_version,
        sequence_length=seq_len,
        sequence_crc64=crc64,
        protein_name=protein_name,
        gene_primary=gene_primary,
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


def iter_dat(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream canonical records from a (gzipped) UniProt DAT file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    buf: list[str] = []
    emitted = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("//"):
                if buf:
                    yield parse_entry(buf)
                    emitted += 1
                    buf = []
                    if limit is not None and emitted >= limit:
                        return
            elif line.startswith("ID   "):
                buf = [line.rstrip("\n")]
            elif buf and line.strip():
                buf.append(line.rstrip("\n"))
    if buf:
        yield parse_entry(buf)


def main(argv: list[str]) -> int:
    """Smoke: parse N entries, print canonical JSON of the first."""
    import json
    import sys
    limit = int(argv[2]) if len(argv) > 2 else 3
    path = Path(argv[1])
    first = True
    count = 0
    for rec in iter_dat(path, limit=limit):
        if first:
            print(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False))
            first = False
        count += 1
    print(f"parsed={count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
