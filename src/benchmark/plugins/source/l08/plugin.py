"""L08 source plugin: pinned clinical_quality_language + vendored JS engine.

Facade over ``source_registry.fetchers.l08_cql.CqlClinicalSource`` (the
locked ExternalKnowledgeSource implementation) so the factory's plugin
discovery finds the lane's source triple entry. The snapshot, npm engine
closure, per-file sha256 manifests and index already exist on disk
(FETCHED_COMPLETE); ``fetch`` is idempotent and re-verifies integrity.
"""
from __future__ import annotations

from source_registry.fetchers.l08_cql import CqlClinicalSource

PLUGIN = CqlClinicalSource()

__all__ = ["PLUGIN", "CqlClinicalSource"]
