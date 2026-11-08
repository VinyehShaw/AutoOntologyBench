"""Paired statistics (owner: W6; Review-5B §11.5 + Shared Contract §F).

Statistical unit: repeated runs/paraphrases/seeds are aggregated within the
ORIGINATING SOURCE MECHANISM cluster first; cluster bootstrap is reported
only with an adequate number of independent clusters. Otherwise: raw paired
discordant counts and uncertainty explicitly NOT estimated — no marketing
CIs from 1–4 source units.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping

#: Minimum independent mechanism clusters before any bootstrap CI is emitted.
MIN_INDEPENDENT_CLUSTERS = 30


@dataclass(frozen=True)
class PairedBinarySummary:
    """Paired binary outcome comparison (arm A vs arm B over the same goals)."""

    n_pairs: int
    n_both_success: int
    n_both_failure: int
    n_a_only: int  # discordant: A success, B failure
    n_b_only: int  # discordant: B success, A failure
    delta: float  # raw difference in success rates over all pairs
    n_clusters: int
    ci95_delta: tuple[float, float] | None  # None => not estimable (too few clusters)
    note: str


def paired_binary_by_cluster(
    pairs: Iterable[tuple[str, bool, bool]],
    *,
    n_boot: int = 2000,
    seed: int = 137,
    min_clusters: int = MIN_INDEPENDENT_CLUSTERS,
) -> PairedBinarySummary:
    """Paired comparison of per-pair (cluster_id, a_success, b_success).

    Cluster bootstrap: resample CLUSTERS with replacement, recompute the
    pooled success-rate difference per resample, report the percentile
    interval. With fewer than ``min_clusters`` independent clusters the CI is
    withheld by design and only raw discordant counts are returned.
    """
    rows = [(str(cluster), bool(a), bool(b)) for cluster, a, b in pairs]
    n_pairs = len(rows)
    n_a_only = sum(1 for _, a, b in rows if a and not b)
    n_b_only = sum(1 for _, a, b in rows if b and not a)
    n_both_success = sum(1 for _, a, b in rows if a and b)
    n_both_failure = n_pairs - n_both_success - n_a_only - n_b_only
    delta = (n_a_only - n_b_only) / n_pairs if n_pairs else 0.0

    by_cluster: dict[str, list[tuple[bool, bool]]] = {}
    for cluster, a, b in rows:
        by_cluster.setdefault(cluster, []).append((a, b))
    clusters = sorted(by_cluster)

    if len(clusters) < min_clusters:
        return PairedBinarySummary(
            n_pairs=n_pairs,
            n_both_success=n_both_success,
            n_both_failure=n_both_failure,
            n_a_only=n_a_only,
            n_b_only=n_b_only,
            delta=delta,
            n_clusters=len(clusters),
            ci95_delta=None,
            note=(
                "no CI: %d independent clusters < %d required; raw discordant "
                "counts only (Shared Contract §F)" % (len(clusters), min_clusters)
            ),
        )

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = [by_cluster[rng.choice(clusters)] for _ in clusters]
        flat = [pair for group in sample for pair in group]
        a_rate = sum(1 for a, _ in flat if a) / len(flat)
        b_rate = sum(1 for _, b in flat if b) / len(flat)
        estimates.append(a_rate - b_rate)
    estimates.sort()
    lo = estimates[int(0.025 * n_boot)]
    hi = estimates[min(n_boot - 1, int(0.975 * n_boot))]
    return PairedBinarySummary(
        n_pairs=n_pairs,
        n_both_success=n_both_success,
        n_both_failure=n_both_failure,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        delta=delta,
        n_clusters=len(clusters),
        ci95_delta=(lo, hi),
        note="paired cluster bootstrap (n_boot=%d, seed=%d)" % (n_boot, seed),
    )


def aggregate_within_cluster(
    pairs: Iterable[tuple[str, bool, bool]],
) -> Mapping[str, tuple[float, float]]:
    """First-stage aggregation: per-cluster success rates for arms A and B."""
    by_cluster: dict[str, list[tuple[bool, bool]]] = {}
    for cluster, a, b in pairs:
        by_cluster.setdefault(str(cluster), []).append((bool(a), bool(b)))
    return {
        cluster: (
            sum(1 for a, _ in rows if a) / len(rows),
            sum(1 for _, b in rows if b) / len(rows),
        )
        for cluster, rows in sorted(by_cluster.items())
    }
