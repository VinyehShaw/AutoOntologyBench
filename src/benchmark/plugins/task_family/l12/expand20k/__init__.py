"""L12 20K expansion (lane target 1150 = legacy 350 + new 800).

New dataset_id ``autoontobench-bank-20000-candidate-dev1``, seed base 20261101.
Writes only to ``bank/staging_20k/l12/``; never touches ``bank/staging/l12``
(legacy 8K, frozen read-only) or any other lane.

New content axes beyond the legacy build (all from the same pinned pm4py
snapshot, tag 2.7.23.11 — new *units*, never bundled AGPL content):
- new event logs (bpic2012, helpdesk, fairness logs, inh_res logs, ...)
- new gold log+net pairs (inh_res nets, roadtraffic100/10000)
- new noisy pairs (a12/a22/a32 f0n20/f0n50, a42 family)
- new corruption modes (distant swap, segment reversal, move-to-end,
  duplicate purge, double deletion, dual relabel)
- new native operations (log footprints audit, footprints conformance)
"""
