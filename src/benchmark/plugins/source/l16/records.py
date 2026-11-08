"""L16 canonical normalized UniProt record schema + shared helpers.

Both independent parsers (``uniprot_dat`` over the flat file, ``uniprot_xml``
over the XML) must emit records through ``canonical_record`` so the fixed-sample
cross-check compares byte-identical canonical JSON projections.

Cross-check projection rule: free-text comment bodies (``function[].text``)
differ in representation between DAT (``{ECO:...|PubMed:...}`` inline braces)
and XML (``(PubMed:...)`` inline + evidence-key attributes) and are excluded
from equality; their structured evidence (ECO ids, PubMed ids) IS compared.
Everything else is compared exactly.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CANONICAL_SCHEMA_VERSION = "l16-record/1.0"

#: XML feature type -> canonical DAT feature key (UniProt controlled vocabulary).
XML_FEATURE_TO_DAT = {
    "initiator methionine": "INIT_MET",
    "signal peptide": "SIGNAL",
    "transit peptide": "TRANSIT",
    "propeptide": "PROPEP",
    "chain": "CHAIN",
    "peptide": "PEPTIDE",
    "topological domain": "TOPO_DOM",
    "transmembrane region": "TRANSMEM",
    "intramembrane region": "INTRAMEM",
    "domain": "DOMAIN",
    "repeat": "REPEAT",
    "region of interest": "REGION",
    "coiled-coil region": "COILED",
    "short sequence motif": "MOTIF",
    "compositionally biased region": "COMPBIAS",
    "DNA-binding region": "DNA_BIND",
    "zinc finger region": "ZN_FING",
    "calcium-binding region": "CA_BIND",
    "nucleotide phosphate-binding region": "NUC_BIND",
    "binding site": "BINDING",
    "active site": "ACT_SITE",
    "site": "SITE",
    "non-standard amino acid": "NON_STD",
    "modified residue": "MOD_RES",
    "lipid moiety-binding region": "LIPID",
    "glycosylation site": "CARBOHYD",
    "disulfide bond": "DISULFID",
    "cross-link": "CROSSLNK",
    "branch site": "BRANCH",
    "sequence variant": "VARIANT",
    "mutagenesis site": "MUTAGEN",
    "unsure residue": "UNSURE",
    "sequence conflict": "CONFLICT",
    "non-consecutive residues": "NON_CONS",
    "non-terminal residue": "NON_TER",
    "helix": "HELIX",
    "strand": "STRAND",
    "turn": "TURN",
    "metal ion-binding site": "METAL",
    "splice variant": "VAR_SEQ",
}

PE_TEXT_TO_LEVEL = {
    "evidence at protein level": 1,
    "evidence at transcript level": 2,
    "inferred from homology": 3,
    "predicted": 4,
    "uncertain": 5,
}

#: ECO id -> legacy GO evidence code (official GO/UniProt mapping; XML emits
#: ECO ids in GO-cross-reference evidence while DAT emits the legacy codes).
ECO_TO_GO_CODE = {
    "ECO:0000269": "EXP", "ECO:0000314": "IDA", "ECO:0000353": "IPI",
    "ECO:0000315": "IMP", "ECO:0000316": "IGI", "ECO:0000270": "IEP",
    "ECO:0006056": "HTP", "ECO:0007005": "HDA", "ECO:0007001": "HMP",
    "ECO:0007003": "HGI", "ECO:0007007": "HEP", "ECO:0000318": "IBA",
    "ECO:0000319": "IBD", "ECO:0000320": "IKR", "ECO:0000321": "IRD",
    "ECO:0000250": "ISS", "ECO:0000266": "ISO", "ECO:0000247": "ISA",
    "ECO:0000255": "ISM", "ECO:0000317": "IGC", "ECO:0000245": "RCA",
    "ECO:0000304": "TAS", "ECO:0000303": "NAS", "ECO:0000305": "IC",
    "ECO:0000307": "ND", "ECO:0000501": "IEA",
}

ECO_RE = re.compile(r"ECO:\d{7}")
PMID_RE = re.compile(r"PubMed:(\d+)")
WS_RE = re.compile(r"\s+")


def norm_space(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def outermost_paren_groups(text: str) -> tuple[str, list[str]]:
    """Balanced scan: (text minus outermost paren groups, group contents)."""
    groups: list[str] = []
    base_parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch == "(":
            if depth == 0:
                cur = []
            else:
                cur.append(ch)
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                groups.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        elif depth == 0:
            base_parts.append(ch)
        else:
            cur.append(ch)
    return "".join(base_parts).strip(), groups


def strip_evidence_braces(text: str) -> tuple[str, list[str], list[str]]:
    """Remove ``{ECO:...|PubMed:...}`` groups; return (clean_text, ecos, pmids)."""
    ecos: list[str] = []
    pmids: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        inner = match.group(1)
        ecos.extend(ECO_RE.findall(inner))
        pmids.extend(PMID_RE.findall(inner))
        return " "

    clean = re.sub(r"\{([^{}]*)\}", _sub, text)
    return (norm_space(clean),
            sorted(set(ecos)), sorted(set(pmids), key=int))


def canonical_record(**kw: Any) -> dict[str, Any]:
    """Build the canonical record; enforces schema keys and deterministic order.

    Subcellular locations are merged on (scope, term) with unioned ECO sets:
    DAT packs several evidence codes into one location sentence while XML emits
    one <location> element per evidence set.
    """
    subcell_merged: dict[tuple[str, str], set[str]] = {}
    for s in kw.get("subcellular", []):
        key = (s.get("scope") or "", norm_space(s.get("term", "")))
        subcell_merged.setdefault(key, set()).update(s.get("ecos", []))
    rec = {
        "schema": CANONICAL_SCHEMA_VERSION,
        "accession": kw["accession"],
        "accessions": sorted(kw.get("accessions", [])),
        "entry_name": kw["entry_name"],
        "reviewed": bool(kw.get("reviewed", True)),
        "entry_version": int(kw.get("entry_version", 0)),
        "sequence_version": int(kw.get("sequence_version", 0)),
        "sequence_length": int(kw.get("sequence_length", 0)),
        "sequence_crc64": kw.get("sequence_crc64", ""),
        "protein_name": norm_space(kw.get("protein_name", "")),
        "gene_primary": kw.get("gene_primary") or "",
        "genes_all": sorted(kw.get("genes_all", [])),
        "gene_synonyms": sorted(kw.get("gene_synonyms", [])),
        "organism_scientific": kw.get("organism_scientific", ""),
        "organism_common": kw.get("organism_common") or "",
        "organism_names": sorted(set(kw.get("organism_names", []))),
        "taxon_id": int(kw.get("taxon_id", 0)),
        "lineage": list(kw.get("lineage", [])),
        "pe_level": int(kw.get("pe_level", 0)),
        "keywords": sorted(kw.get("keywords", [])),
        "ec_numbers": sorted(kw.get("ec_numbers", [])),
        "comments_function": sorted(
            ({"text": norm_space(c.get("text", "")),
              "ecos": sorted(set(c.get("ecos", []))),
              "pubmeds": sorted(set(c.get("pubmeds", [])), key=int)}
             for c in kw.get("comments_function", [])),
            key=lambda c: c["text"]),
        "subcellular": sorted(
            ({"scope": scope, "term": term, "ecos": sorted(ecos)}
             for (scope, term), ecos in subcell_merged.items()),
            key=lambda s: (s["scope"], s["term"])),
        "diseases": sorted(
            ({"name": norm_space(d.get("name", "")),
              "acronym": d.get("acronym", ""),
              "mim": d.get("mim", "")}
             for d in kw.get("diseases", [])),
            key=lambda d: d["name"]),
        "isoforms": sorted(
            ({"iso_id": i["iso_id"], "name": i.get("name", ""),
              "synonyms": sorted(i.get("synonyms", []))}
             for i in kw.get("isoforms", [])),
            key=lambda i: i["iso_id"]),
        "catalytic_activities": sorted(
            ({"rhea_ids": sorted(set(c.get("rhea_ids", []))),
              "ec": sorted(set(c.get("ec", []))),
              "ecos": sorted(set(c.get("ecos", []))),
              "pubmeds": sorted(set(c.get("pubmeds", [])), key=int),
              "physio": sorted(
                  ({"direction": p.get("direction", ""),
                    "rhea_id": p.get("rhea_id", "")}
                   for p in c.get("physio", [])),
                  key=lambda p: (p["direction"], p["rhea_id"])),
              "text": norm_space(c.get("text", ""))}
             for c in kw.get("catalytic_activities", [])),
            key=lambda c: (tuple(c["rhea_ids"]), tuple(c["ec"]), c["text"])),
        "features": sorted(
            ({"type": f["type"], "begin": str(f["begin"]), "end": str(f["end"])}
             for f in kw.get("features", [])),
            key=lambda f: (f["type"], f["begin"], f["end"])),
        "variants": sorted(
            ({"begin": str(v["begin"]), "end": str(v["end"]),
              "original": v.get("original", ""), "variation": v.get("variation", ""),
              "var_id": v.get("var_id", "")}
             for v in kw.get("variants", [])),
            key=lambda v: (v["begin"], v["original"], v["variation"])),
        "go_refs": sorted(
            ({"id": g["id"], "aspect": g["aspect"], "term": g["term"],
              "evidence": g.get("evidence", "")}
             for g in kw.get("go_refs", [])),
            key=lambda g: (g["id"], g["term"], g["evidence"])),
        "rhea_refs": sorted(set(kw.get("rhea_refs", []))),
        "xrefs": sorted(
            ({"db": x["db"], "id": x["id"]} for x in kw.get("xrefs", [])),
            key=lambda x: (x["db"], x["id"])),
        "proteomes": sorted(set(kw.get("proteomes", []))),
        "pubmed_ids": sorted(set(str(p) for p in kw.get("pubmeds", [])),
                             key=int),
    }
    return rec


def projection_for_xcheck(rec: dict[str, Any]) -> dict[str, Any]:
    """Drop representation-sensitive free text before DAT/XML comparison.

    ``organism_scientific``/``organism_common`` are display fields (placement
    of parenthesized name groups differs between formats); identity content is
    compared exactly via the ``organism_names`` set instead.
    """
    proj = dict(rec)
    proj.pop("organism_scientific", None)
    proj.pop("organism_common", None)
    proj["comments_function"] = sorted(
        ({k: v for k, v in c.items() if k != "text"}
         for c in rec["comments_function"]),
        key=lambda c: json.dumps(c, sort_keys=True))
    proj["catalytic_activities"] = sorted(
        ({k: v for k, v in c.items() if k != "text"}
         for c in rec["catalytic_activities"]),
        key=lambda c: json.dumps(c, sort_keys=True))
    return proj


def canonical_json(rec: dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def record_sha256(rec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(rec).encode()).hexdigest()
