"""autoontobench.export — P8 paper-export package (W9).

Modules:
- aggregate: load + strict-validate append-only ``researcher_trials.jsonl`` rows;
- paper: emit the 13 fixed results interfaces (Shared Contract §F);
- figures: pure-Python PENDING-placeholder PDF writer.
"""
from autoontobench.export.aggregate import StrictExportError, load_runs
from autoontobench.export.paper import export_all, run_cli

__all__ = ["StrictExportError", "export_all", "load_runs", "run_cli"]
