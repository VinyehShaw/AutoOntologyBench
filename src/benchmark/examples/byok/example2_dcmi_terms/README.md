# BYOK Example 2 - Build a personal bank from a new, legally clean source

Spec section 6.4 example 2 (BYOK plugin): take a small, legal external
knowledge file with a public URL, write a new Source / TaskFamily / Verifier
plugin trio, generate real tasks for a new task family, verify them with a
native independent verifier, and export a personal benchmark package. The
source here is **DCMI Metadata Terms** (`dublin_core_terms.ttl`, CC-BY 4.0,
47,834 bytes), pinned by sha256
`13df401072dd7015bf9d75162f3e41c8138075304b7b9cc1aa1e9c16db976797`
(`dcterms:modified 2012-06-14`, retrieved 2026-10-27). The native engine is
rdflib; no external service is called at verification time.

This bank is a **personal BYOK artifact**: it never merges into the official
`aob-8000-rc1` bank and never joins any official leaderboard
(`docs/EXTENDING_AOB.md` section 7).

## What the pipeline does

1. **Fetch**: `dcmi_source.py` downloads the pinned Turtle document through
   the factory content-addressed cache (resume-safe, sha256-verified),
   materializes it at
   `external_knowledge/pinned/dcmi-terms/dcmi-2012061/dublin_core_terms.ttl`
   and writes a fetch receipt under `external_knowledge/fetch_receipts/`.
   With `--resume` and the snapshot already on disk, no network is used.
2. **Propose**: `dcmi_family.py` parses the ontology with rdflib and emits
   closure candidates. Two families are configured in `bank.yaml`
   (`dcmi-subclass-closure`, `dcmi-subproperty-closure`, seed 3, up to 6
   candidates each). Measured on the pinned snapshot: the subclass family
   proposes 2 candidates (DCMI declares few `rdfs:subClassOf` closures), the
   subproperty family proposes 6.
3. **Native verify**: `dcmi_verifier.py` recomputes the transitive closure
   from the pinned file with `rdflib.Graph.transitive_subjects` and requires
   the recomputed answer to equal the recorded one (`prove_exists`), and for
   the semantic witness removes one feeding edge and requires the closure to
   change (`certify_semantic_relevance`). The recorded answer is never trusted
   blindly.
4. **Package**: public tasks (goal + root IRI + relation + budget) in
   `out/public/tasks.jsonl`; expected closures only in
   `out/private_oracle/oracle.json` (directory mode 0700); per-task hashes and
   full source provenance (pin, sha256, license, owner) in
   `out/task_manifest.json`.

The checked-in `out/` directory holds **2 qualified tasks** built on
2026-10-27T22:10:47Z:

| task_id | root | expected transitive subclasses |
|---|---|---|
| `byok-example2-dcmi-closure-RightsStatement-3` | `dcterms:RightsStatement` | `dcterms:LicenseDocument` |
| `byok-example2-dcmi-closure-rdf-schema#Class-3` | `rdfs:Class` | `dcterms:AgentClass` |

## Run

```bash
cd /mnt/faster_nas/benyou.mys/git/oob
.venv/bin/python -m autoontobench source add --config \
  benchmark/examples/byok/example2_dcmi_terms/source.yaml
.venv/bin/python -m autoontobench bank generate --config \
  benchmark/examples/byok/example2_dcmi_terms/bank.yaml \
  --output benchmark/examples/byok/example2_dcmi_terms/out --resume
.venv/bin/python -m autoontobench bank validate \
  --input benchmark/examples/byok/example2_dcmi_terms/out --native-only
.venv/bin/python -m autoontobench bank package \
  --input benchmark/examples/byok/example2_dcmi_terms/out --release private-local
```

## Verified run (2026-10-29, real outputs)

All commands executed from the repo root with `.venv/bin/python` on
2026-10-29 ~23:50 BJT. Exit codes in parentheses.

```text
$ .venv/bin/python -m autoontobench source add --config benchmark/examples/byok/example2_dcmi_terms/source.yaml
[source add] registered dcmi-terms -> /mnt/faster_nas/benyou.mys/git/oob/bank/byok/sources.d/dcmi-terms.yaml
(0)

$ .venv/bin/python -m autoontobench bank generate --config benchmark/examples/byok/example2_dcmi_terms/bank.yaml --dry-run
DRY-RUN generate bank=byok-example2-dcmi source=dcmi-terms candidates=6 out=/mnt/faster_nas/benyou.mys/git/oob/benchmark/examples/byok/example2_dcmi_terms/out
(0)

$ .venv/bin/python -m autoontobench bank validate --input benchmark/examples/byok/example2_dcmi_terms/out --native-only
[bank validate] input=/mnt/faster_nas/benyou.mys/git/oob/benchmark/examples/byok/example2_dcmi_terms/out tasks=2 revalidated=2 problems=0 native_only=True
(0)

$ .venv/bin/python -m autoontobench bank package --input benchmark/examples/byok/example2_dcmi_terms/out --release private-local
[bank package] private-local <- benchmark/examples/byok/example2_dcmi_terms/out files=3 -> /mnt/faster_nas/benyou.mys/git/oob/bank/releases/private-local/byok-example2-dcmi
(0)
```

The fetch/verify stage of `bank generate` still works: with `--resume` the
pinned snapshot's sha256 was re-checked on disk (no network) and
`verify_snapshot` passed before generation proceeded.

### Known issue: `bank generate` currently exits 1 (plugin drift)

The real (non-dry-run) generate fails in the verifier:

```text
$ .venv/bin/python -m autoontobench bank generate --config benchmark/examples/byok/example2_dcmi_terms/bank.yaml --output benchmark/examples/byok/example2_dcmi_terms/out --resume
Traceback (most recent call last):
  ...
  File ".../example2_dcmi_terms/dcmi_verifier.py", line 37, in prove_exists
    actual = _closure(Path(ev["snapshot_root"]), ev["class_iri"])
KeyError: 'class_iri'
(1)
```

Cause: `dcmi_family.py` was revised on 2026-10-28 to emit `native_evidence`
keys `root_iri` / `expected_closure` / `relation` / `snapshot_root` and to add
the second family (`dcmi-subproperty-closure`), but `dcmi_verifier.py` still
reads the earlier keys `class_iri` / `expected_subclasses` and only recomputes
`rdfs:subClassOf` closures. The checked-in `out/` was produced on 2026-10-27 by
the earlier, consistent family revision, which is why `bank validate` (which
re-runs the verifier against the stored oracle records) passes 2/2. The failed
run leaves `out/` byte-identical (verified by diff against a pre-run backup);
it only enqueues one idempotent `BYOK:dcmi-terms` item in `bank/queue.db`.

The fix (align the verifier with the two-relation evidence schema) belongs to
the example/factory owner; per the build-plan lane rules this README does not
modify plugin code. Tracked in `docs/EXTENDING_AOB.md` section 5.3 and the DOC
lane STATUS.

## Outputs

```text
out/
├── public/tasks.jsonl          # 2 rows: task_id, family/goal/instance ids, regime,
│                               # semantic_profile, public{user_goal, ontology_ref,
│                               # class_iri, allowed_tools, budget}, ast_sha256, status
├── private_oracle/oracle.json  # mode 0700; per task: ast, oracle_ref, native_evidence
└── task_manifest.json          # bank_id byok-example2-dcmi, owner aob-factory-example,
                                # license CC-BY-4.0, source pin + sha256, plugin paths,
                                # qualified=2, per-task hashes, built_at
```

`bank package --release private-local` copies all three files plus a
`PACKAGE_MANIFEST.json` (sha256 per file) into
`bank/releases/private-local/byok-example2-dcmi/`. Note that `bank package`
does not re-seal directory permissions (the packaged `private_oracle/`
directory comes out 0755); see `docs/BENCHMARK_RELEASE.md` section 3.

## License

Upstream: DCMI Metadata Terms, **CC-BY 4.0**, attribution "Dublin Core
Metadata Initiative (DCMI), CC-BY 4.0" (carried in the source descriptor's
`attribution` / `usage_constraints`). This example redistributes no upstream
content in the git tree: only the fetch recipe (URL + pinned release +
sha256). The downloaded snapshot lives in the gitignored
`external_knowledge/` cache. The derived bank itself is declared CC-BY-4.0 in
`bank.yaml` (`license`) with `owner: aob-factory-example`; a personal bank
built from this example inherits whatever license its own `bank.yaml`
declares, and it stays out of the official bank and leaderboard.
