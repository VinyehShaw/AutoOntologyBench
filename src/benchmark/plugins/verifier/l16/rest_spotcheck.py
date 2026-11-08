"""L16 REST spot-check: fixed 5-accession sample vs the pinned local store.

Per the lane contract, live UniProt REST/SPARQL is used ONLY for this fixed
cross-check sample (never as oracle truth). Compares identity fields
(accession, entry name, primary gene, taxon id, sequence length, reviewed
status) between rest.uniprot.org (current) and the pinned 2026_03 store.
Differences are expected if upstream has rotated past 2026_03; the receipt
records observed agreement verbatim. Best-effort: network failure is recorded,
not fatal.

CLI:
  .venv/bin/python -m benchmark.plugins.verifier.l16.rest_spotcheck \
      --pin-root external_knowledge/pinned/uniprot/uniprot-2026_03-sprot
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

FIXED_SAMPLE = ["P04637", "P0C9F0", "Q6GZX4", "P08195", "P12036"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="L16 REST fixed-sample cross-check")
    ap.add_argument("--pin-root", required=True)
    args = ap.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)
    root = Path(args.pin_root)
    db = sqlite3.connect(
        f"file:{root / 'store' / 'l16_uniprot_dat.sqlite'}?mode=ro", uri=True)
    rows = []
    for acc in FIXED_SAMPLE:
        local = db.execute(
            "SELECT entry_name, gene_primary, taxon_id, sequence_length, "
            "reviewed FROM entries WHERE accession=?", (acc,)).fetchone()
        local_rec = None
        if local:
            local_rec = {"entry_name": local[0], "gene_primary": local[1],
                         "taxon_id": local[2], "sequence_length": local[3],
                         "reviewed": bool(local[4])}
        rec: dict = {"accession": acc, "local": local_rec}
        try:
            proc = subprocess.run(
                ["curl", "-sSfL", "--connect-timeout", "20", "--max-time", "60",
                 "--retry", "2",
                 f"https://rest.uniprot.org/uniprotkb/{acc}.json"
                 "?fields=accession,id,gene_names,organism_id,length,reviewed"],
                capture_output=True, text=True, timeout=90)
            if proc.returncode == 0:
                j = json.loads(proc.stdout)
                remote = {
                    "entry_name": j.get("uniProtkbId"),
                    "gene_primary": (j.get("genes") or [{}])[0]
                                    .get("geneName", {}).get("value", ""),
                    "taxon_id": j.get("organism", {}).get("taxonId"),
                    "sequence_length": j.get("sequence", {}).get("length"),
                    "reviewed": "reviewed (Swiss-Prot)" in
                                (j.get("entryType") or ""),
                }
                rec["remote"] = remote
                rec["fields_agree"] = {
                    k: (local_rec or {}).get(k) == v for k, v in remote.items()}
                rec["all_agree"] = all(rec["fields_agree"].values())
            else:
                rec["remote_error"] = proc.stderr.strip()[:300]
        except Exception as exc:
            rec["remote_error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(rec)
    n_ok = sum(1 for r in rows if r.get("all_agree"))
    receipt = {
        "kind": "rest_spotcheck",
        "endpoint": "rest.uniprot.org/uniprotkb/{acc}.json",
        "sample": FIXED_SAMPLE,
        "note": "REST serves current data; pinned store is 2026_03. Agreement "
                "expected while 2026_03 is current; disagreements after an "
                "upstream rotation reflect release drift, not parser error.",
        "checked": len(rows),
        "all_agree": n_ok,
        "rows": rows,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = root / "store" / "rest_spotcheck_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"checked": len(rows), "all_agree": n_ok,
                      "errors": [r["accession"] for r in rows
                                 if "remote_error" in r]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
