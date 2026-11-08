"""figures: paper figure emission for the P8 exporter.

No plotting library is pinned in the autoonto runtime (matplotlib absent), so
figures are emitted as **valid minimal placeholder PDFs** produced by a
pure-Python writer. Each placeholder is explicitly labeled ``PENDING DATA`` and
carries the figure's contract name; no data curves are fabricated. When real
claimable runs exist, this module is the single seam where a tested plotting
backend (Shared Contract §G: "optionally redrawn by Codex's tested plotting
code from the same measured JSON") is wired in.

The PDF writer emits a single-page PDF 1.4 document (Helvetica text only) with
a correct xref table and trailer, so any conformant reader can open it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

# Canonical figure set (Shared Contract §G figure list, minus visual track).
FIGURE_SPECS: Sequence[tuple[str, str]] = (
    ("source_funnel", "Source qualification funnel"),
    ("main_tsr_delta", "Paired task-success delta by arm"),
    ("causal_interventions", "Causal corruption / repair interventions"),
    ("active_budget", "Active vs passive probe budget curve"),
    ("cost_reuse", "Ontology reuse amortization"),
    ("semantics_vs_utility", "Semantic fidelity vs task utility"),
)

_PENDING_LABEL = "PENDING DATA"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def render_placeholder_pdf(title: str, note: str) -> bytes:
    """Build a valid single-page PDF containing only 'PENDING DATA' text."""
    lines: List[str] = [_PENDING_LABEL, title, note]
    stream_parts = ["BT", "/F1 20 Tf", "72 740 Td"]
    for i, line in enumerate(lines):
        if i:
            stream_parts.append("0 -28 Td")
            stream_parts.append("/F1 12 Tf" if i else "/F1 20 Tf")
        stream_parts.append(f"({_pdf_escape(line)}) Tj")
    stream_parts.append("ET")
    stream = "\n".join(stream_parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def write_figures(output_dir: Path, note: str) -> List[str]:
    """Emit the canonical placeholder figure set; returns relative paths."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, title in FIGURE_SPECS:
        path = fig_dir / f"{name}.pdf"
        path.write_bytes(render_placeholder_pdf(title, note))
        written.append(f"figures/{name}.pdf")
    return written
