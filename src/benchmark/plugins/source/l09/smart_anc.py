"""L09 source plugin: adapter over the pinned WHO SMART-ANC snapshot.

The fetch/verify/index lifecycle lives in ``source_registry/fetchers/l09_smart_anc.py``
(pinned commit 301a6b8b904361315ca38b28ee9fbd133f9d0e94, CC0-1.0). This module
exposes the snapshot to the task-family / verifier plugins: canonical URL index,
typed artifact loaders, and the knowledge-base views the deterministic native
checks recompute from (ELM statement sets, CQL declarations, ValueSet expansion,
PlanDefinition action graphs).
"""
from __future__ import annotations

from benchmark.schema.plugin_protocols import SourceSnapshot
from source_registry.fetchers import l09_smart_anc as fetcher

SOURCE_ID = fetcher.SOURCE_ID
LANE_ID = fetcher.LANE_ID
RELEASE_ID = fetcher.RELEASE_ID
SNAPSHOT_ID = "%s@%s" % (SOURCE_ID, RELEASE_ID)


class SmartAncSourcePlugin:
    """Source plugin handle; delegates the lifecycle to the lane fetcher.

    R2 facade: exposes the full locked ``ExternalKnowledgeSource`` protocol
    surface (describe/discover_releases/fetch/verify_snapshot/inventory) so
    factory discovery (§3.2 source surface, protocol shape) recognizes L09 as
    a conformant source lane. All lifecycle work stays in the fetcher; this
    class only forwards.
    """

    source_id = SOURCE_ID
    lane_id = LANE_ID

    def describe(self):
        return fetcher.SmartAncSource().describe()

    def discover_releases(self):
        return fetcher.SmartAncSource().discover_releases()

    def fetch(self, release, *, resume: bool, cache_dir):
        return fetcher.SmartAncSource().fetch(release, resume=resume, cache_dir=cache_dir)

    def verify_snapshot(self, receipt):
        return fetcher.SmartAncSource().verify_snapshot(receipt)

    def inventory(self, snapshot):
        return fetcher.SmartAncSource().inventory(snapshot)

    def snapshot(self) -> SourceSnapshot:
        return fetcher.load_snapshot()


PLUGIN = SmartAncSourcePlugin()

SNAPSHOT: SourceSnapshot = fetcher.load_snapshot()
