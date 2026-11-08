# AutoOntologyBench

Evaluation code for **AutoOntologyBench**. All source is under `src/`.

## Quick start

### 1. Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull the dataset from HuggingFace

```bash
export HF_TOKEN="..."
pip install -U huggingface_hub
hf download Vinyeh/AutoOntologyBench \
  --repo-type dataset \
  --local-dir AutoOntologyBench
```

The task file is then:

```text
AutoOntologyBench/tasks/qualified_tasks_test.jsonl
```

This release contains **19,947 public tasks**.

### 3. Mount the private oracle

The public dataset contains no answers. Point the evaluator at the separate
private oracle bundle:

```bash
export EVAL_ORACLE_ROOTS="$PWD/../release_oracle/staging_20k:$PWD/../release_oracle/staging"
```

Never publish the oracle.

### 4. One-click evaluation

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

bash src/scripts/eval.sh \
  --provider openai \
  --model gpt-4.1 \
  --sample-per-lane 5
```

Or let the wrapper download the dataset automatically and then run:

```bash
bash src/scripts/eval_from_hf.sh \
  --provider openai \
  --model gpt-4.1 \
  --sample-per-lane 5
```

Full run:

```bash
bash src/scripts/eval.sh --sample-per-lane 0
```

Outputs go to `runs/eval_<UTC>/`.

## Metrics

Core `report.json`:

- `TSR`: task success rate
- `V-TSR`: verified task success rate, distinct from TSR
- `tasks`, `scored`, `success`, `verified_success`
- `missing`, `not_supported`, `oracle_missing`, `error`
- `unsafe_rows`, `unsafe_action_count`, `unsafe_rate`, `unsafe_unchecked`
- `costs`: tokens, API calls, wall seconds
- per-lane and per-family breakdowns

Offline `metrics/`:

- `cps.json` / `CPS.md`: tokens per verified success, API calls per success, wall seconds per success
- `batch_tsr.json`: TSR by batch/run
- `regime_breakdown.json`: TSR by application / repair / organization / construction
- `mapf1.json` / `MAPF1.md`: mapping micro-F1, macro-F1, coverage
- `mapf1_common7.json`: Common-7 F1 and missing-family coverage
- `frule.json` / `FRULE.md`: conditional rule-witness balanced F_rule and coverage

## Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
└── src/
    ├── autoontobench/     # model/API types used by the evaluator
    ├── benchmark/         # harness, adapters, verifiers, metrics
    ├── scripts/           # eval.sh / eval_from_hf.sh
    └── LICENSES/          # third-party source attribution
```

Research/experiment code, task data, run artifacts, internal docs, and oracle
payloads are not included.

The internal Python package path `benchmark.eval_20k` is a historical module
name and is unchanged for compatibility; the evaluator itself uses the
19,947-task public set.

## Tests

```bash
PYTHONPATH=src pytest src/benchmark/eval_20k/tests
```

Expected: `134 passed, 1 skipped`.

## GitHub

```bash
git init
git add .
git commit -m "AutoOntologyBench evaluation code"
git branch -M main
git remote add origin <GITHUB_REPOSITORY_URL>
git push -u origin main
```

## License

Code: Apache-2.0 (`LICENSE`). Dataset: CC BY 4.0. Third-party attribution:
`src/LICENSES/`.