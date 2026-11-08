"""TASK_FACTORY_FUNNEL.csv updater (W8 M3).

Per-row stage counters for the task-factory funnel. M3 fills the previously
PENDING native/feasible columns from real evidence and the final qualified count
from the actual validator decisions; rejection/blocked reasons are exact.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

NATIVE_VALID_COL = 4
FEASIBLE_COL = 5
LEAK_FREE_COL = 8
FINAL_QUALIFIED_COL = 9
REASONS_COL = 10


@dataclass(frozen=True)
class FunnelUpdate:
    native_valid: str               # int as str, or "PENDING"
    feasible: str
    final_qualified: str
    rejection_reasons: str
    leak_free: str | None = None    # None keeps the existing cell


def update_funnel(path: Path, updates: Mapping[str, FunnelUpdate]) -> list[str]:
    """Patch rows whose `source` label starts with an update key. Returns keys applied."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(lines))
    if not rows:
        raise ValueError(f"empty funnel csv: {path}")
    applied: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        key = next((k for k in updates if row[0].startswith(k)), None)
        if key is None:
            continue
        update = updates[key]
        row[NATIVE_VALID_COL] = update.native_valid
        row[FEASIBLE_COL] = update.feasible
        if update.leak_free is not None:
            row[LEAK_FREE_COL] = update.leak_free
        row[FINAL_QUALIFIED_COL] = update.final_qualified
        row[REASONS_COL] = update.rejection_reasons
        applied.append(key)
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return applied


def append_funnel_rows(path: Path, new_rows: list[list[str]]) -> list[str]:
    """Append per-source rows (W14). Existing rows are never touched; a row whose
    label already exists is skipped so reruns stay idempotent. Returns labels added."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(lines))
    if not rows:
        raise ValueError(f"empty funnel csv: {path}")
    existing = {row[0] for row in rows[1:] if row}
    added: list[str] = []
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for row in new_rows:
            if not row or row[0] in existing:
                continue
            writer.writerow(row)
            existing.add(row[0])
            added.append(row[0])
    return added
