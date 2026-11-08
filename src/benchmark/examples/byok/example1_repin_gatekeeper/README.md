# BYOK Example 1 — Re-pin an official source at a different release

Spec section 6.4 example 1: take the official L01 source
(`open-policy-agent/gatekeeper-library`) but pin a **different** revision than
the Review-5 baseline pin (`a8d4799d...`, master @ 2026-10-21). This example
pins commit
[`1946bc28e1337b90b7e675e3f0fa94aa2e92fd6e`](https://github.com/open-policy-agent/gatekeeper-library/commit/1946bc28e1337b90b7e675e3f0fa94aa2e92fd6e)
(2026-09-28) and builds **10** real policy-repair candidates, natively verified
with the pinned OPA binary (`tools/bin/opa`, 0.70.0).

## What the pipeline does

1. **Fetch**: `gatekeeper_source.py` downloads the pinned tarball (sha256
   `5d809aad...a8435a7`) through the factory content-addressed cache with
   resume + sha256 verification, extracts to
   `external_knowledge/pinned/gatekeeper-library-repin-1946bc28/1946bc28e133/`,
   writes a fetch receipt.
2. **Propose**: `gk_policy_family.py` enumerates `(policy, sample)` pairs from
   `library/general/*/samples/*/`, parses the Rego embedded in each
   `template.yaml` (`spec.targets[].code[engine=Rego].source.rego`), and emits
   10 `CandidateTaskAST`s (seed 7, regime `repair`).
3. **Native verify**: `gk_opa_verifier.py` runs
   `opa eval data.<pkg>.violation` on the disallowed manifest (must be
   non-empty = non-trivial) and on the allowed manifest (must be empty =
   solvable). Both directions must hold.
4. **Package**: public tasks (goal + manifest + parameters + policy rego) in
   `out/public/tasks.jsonl`; expected violations + repaired manifest only in
   `out/private_oracle/oracle.json` (mode 0700); manifest with per-task hashes
   in `out/task_manifest.json`.

## Run

```bash
cd /mnt/faster_nas/benyou.mys/git/oob
.venv/bin/python -m autoontobench source add --config \
  benchmark/examples/byok/example1_repin_gatekeeper/source.yaml
.venv/bin/python -m autoontobench bank generate --config \
  benchmark/examples/byok/example1_repin_gatekeeper/bank.yaml \
  --output benchmark/examples/byok/example1_repin_gatekeeper/out --resume
.venv/bin/python -m autoontobench bank validate \
  --input benchmark/examples/byok/example1_repin_gatekeeper/out --native-only
.venv/bin/python -m autoontobench bank package \
  --input benchmark/examples/byok/example1_repin_gatekeeper/out --release private-local
```

## License

Upstream repository: Apache-2.0 (verify per-asset). This example redistributes
no upstream content in the git tree: only the fetch recipe (URL + pinned commit
+ sha256). The downloaded snapshot lives in the gitignored
`external_knowledge/` cache.
