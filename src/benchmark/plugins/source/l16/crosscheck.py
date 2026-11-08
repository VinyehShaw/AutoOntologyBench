"""L16 DAT-vs-XML cross-check: independent parser agreement on a fixed sample.

Samples accessions deterministically (sha256(accession) hash sampling: uniform,
order-independent, reproducible) plus the first ``prefix`` entries of the DAT
file, parses both pinned sources with the two independent parsers, and requires
exact equality of ``projection_for_xcheck`` canonical JSON. Any mismatch is a
build blocker: the oracle store (DAT) and the reference store (XML) must agree
on every checked entry.

CLI:
  .venv/bin/python -m benchmark.plugins.source.l16.crosscheck \
      --pin-root external_knowledge/pinned/uniprot/uniprot-2026_03-sprot \
      --samples 300 --out <pin-root>/store/crosscheck_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l16.records import projection_for_xcheck
from benchmark.plugins.source.l16.uniprot_dat import iter_dat
from benchmark.plugins.source.l16.uniprot_xml import iter_xml


def _accept(accession: str, modulus: int) -> bool:
    return int(hashlib.sha256(accession.encode()).hexdigest()[:8], 16) % modulus == 0


def _sample_accessions(dat_path: Path, n_samples: int, modulus: int,
                       prefix: int) -> set[str]:
    picked: set[str] = set()
    count = 0
    for rec in iter_dat(dat_path):
        count += 1
        if count <= prefix or _accept(rec["accession"], modulus):
            picked.add(rec["accession"])
    return picked


def run(dat_path: Path, xml_path: Path, *, n_samples: int, prefix: int,
        out_path: Path | None) -> dict[str, Any]:
    t0 = time.time()
    modulus = max(2, 575_748 // max(n_samples, 1))
    sample = _sample_accessions(dat_path, n_samples, modulus, prefix)
    dat_records: dict[str, dict[str, Any]] = {}
    for rec in iter_dat(dat_path):
        if rec["accession"] in sample:
            dat_records[rec["accession"]] = rec
    xml_records: dict[str, dict[str, Any]] = {}
    for rec in iter_xml(xml_path):
        if rec["accession"] in sample:
            xml_records[rec["accession"]] = rec
    missing_xml = sorted(sample - set(xml_records))
    mismatches: list[dict[str, Any]] = []
    for acc in sorted(sample):
        if acc not in dat_records or acc not in xml_records:
            continue
        a = projection_for_xcheck(dat_records[acc])
        b = projection_for_xcheck(xml_records[acc])
        if a != b:
            fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            mismatches.append({"accession": acc, "fields": fields,
                               "dat": {f: a.get(f) for f in fields},
                               "xml": {f: b.get(f) for f in fields}})
    receipt = {
        "status": "PASS" if not mismatches and not missing_xml else "FAIL",
        "samples_target": n_samples,
        "sampled_accessions": len(sample),
        "dat_parsed": len(dat_records),
        "xml_parsed": len(xml_records),
        "missing_in_xml": missing_xml,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "modulus": modulus,
        "prefix": prefix,
        "wall_seconds": round(time.time() - t0, 1),
        "projection_note": "comments_function[].text excluded (representation "
                           "difference); all other canonical fields compared "
                           "exactly",
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="L16 DAT/XML cross-check")
    ap.add_argument("--pin-root", required=True)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--prefix", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)
    root = Path(args.pin_root)
    receipt = run(root / "uniprot_sprot.dat.gz", root / "uniprot_sprot.xml.gz",
                  n_samples=args.samples, prefix=args.prefix,
                  out_path=Path(args.out) if args.out else None)
    print(json.dumps({k: v for k, v in receipt.items() if k != "mismatches"},
                     indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
