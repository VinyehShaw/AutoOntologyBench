"""Benchmark Factory (FINAL-BUILD-ORDER-v2 work package S10).

Shared machinery for the 8,000-task build: the Source Hub (registry loader +
source lock), the content-addressed external-knowledge cache, the persistent
SQLite-WAL build queue, the shard framework, the build/telemetry driver and
strict config loading. Lane-specific source/family/verifier plugins live under
``benchmark/plugins/`` (lane work packages) and ``benchmark/examples/byok/``.

Owner: S10. ``autoontobench/factory/{gates,release,reports}/`` belong to S11 and
are imported lazily, only when present.
"""
from __future__ import annotations

FACTORY_VERSION = "0.1.0"
GENERATOR_VERSION = f"aob-factory/{FACTORY_VERSION}"

__all__ = ["FACTORY_VERSION", "GENERATOR_VERSION"]
