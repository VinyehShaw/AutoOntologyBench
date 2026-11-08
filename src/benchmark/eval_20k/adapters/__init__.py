"""Per-lane adapter modules discovered dynamically by ``lanes.get_adapter``.

Contract for adapter authors: create ``<lane_lower>.py`` (e.g. ``l01.py``)
exporting ``class Adapter(benchmark.eval_20k.lanes.LaneAdapter)`` whose
constructor accepts a single ``lane_kw: dict[str, str] | None`` argument.
Absent modules/attributes fall back to the built-in L05/reflection registry;
a module that fails to import or construct becomes an explicit NOT_SUPPORTED
verdict with the precise reason — never a silent score, never a probe crash.
"""
from __future__ import annotations
