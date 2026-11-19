# Core-track

Core-track is the **world-grounded W3C ontology** track of AutoOntologyBench.
It is separate from the Answer-20K suite: each world has its own ontology
artifact, SHACL shapes, and task set, so there is no single global ontology.

The public release contains the answer-level evaluator, the public task rows,
per-world public materials, and the construction/release tooling. It does not
contain labels, gold ontology bundles, the frozen executor, or the private
oracle.

## Scale

| item | value |
|---|---:|
| worlds / lanes | 16 (L01-L16) |
| task records | 19,950 |
| dev | 1,688 |
| hidden_iid | 10,774 |
| hidden_comp | 4,648 |
| hidden_transfer | 2,840 |
| templates | 9,217 (opaque public IDs) |
| template families | 163 |
| public TTL files | 32 (16 `ontology_init.ttl` + 16 `shapes_init.ttl`) |
| task context | 8,600 TTLs in the dataset `task_context.tar.gz` |

The scale target was 19,947. The assembled public release is 19,950 because
L06, L10 and L11 each contribute one extra record so paraphrase siblings share a
split. Do not claim an exact 19,947 Core count.

## Public vs private

Public (HuggingFace dataset):

- 19,950 label-stripped task rows in `tasks/tasks.jsonl`;
- per-world public materials under `worlds/*/public/`;
- per-world metadata, license, and attribution;
- task-context TTLs in `task_context.tar.gz`.

Private (mounted separately):

- `core_oracle.jsonl` with machine-replayed labels;
- gold/corrupt/repaired ontology bundles and mappings;
- pack-internal oracle material.

Never upload the private oracle to the public dataset repository.

## Data layout

The repository does not ship the public data. Download it from HuggingFace:

```bash
export HF_TOKEN="..."
pip install -U huggingface_hub
hf download Vinyeh/AutoOntologyBench \
  --repo-type dataset \
  --local-dir ../aob-data
```

If the Core release is stored in the standalone dataset repository instead:

```bash
hf download Vinyeh/AutoOntologyBench-Core \
  --repo-type dataset \
  --local-dir ../aob-data-core
```

`src/scripts/eval_core.sh` looks for the task file in:

1. `data/core/tasks/tasks.jsonl` (repository-local release or data copy);
2. `../aob-data/core_worlds/data/core/tasks/tasks.jsonl`;
3. `../aob-data/data/core/tasks/tasks.jsonl`;
4. `AutoOntologyBench-Core-data/data/core/tasks/tasks.jsonl`;
5. `release_core/tasks/tasks.jsonl`.

Override the selection with `--tasks` or `EVAL_TASKS`.

## Quick start

```bash
git clone <GITHUB_REPO_URL> AutoOntologyBench
cd AutoOntologyBench
pip install -r requirements.txt

hf download Vinyeh/AutoOntologyBench \
  --repo-type dataset \
  --local-dir ../aob-data

export EVAL_TASKS="../aob-data/core_worlds/data/core/tasks/tasks.jsonl"
export AOB_CORE_ORACLE="../aob-private/oracle/core_oracle.jsonl"

bash src/scripts/eval_core.sh \
  --provider openai \
  --model gpt-4.1 \
  --sample-per-world 10
```

The oracle path is an example. Keep the private oracle outside `../aob-data`;
the public dataset must not contain labels.

### OpenAI-compatible API

```bash
export OPENAI_API_KEY="..."
bash src/scripts/eval_core.sh \
  --provider openai \
  --model gpt-4.1 \
  --base-url https://api.openai.com/v1 \
  --sample-per-world 10
```

### Local vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model --port 8000

bash src/scripts/eval_core.sh \
  --provider vllm \
  --model /path/to/model \
  --base-url http://127.0.0.1:8000/v1 \
  --sample-per-world 10
```

vLLM uses `VLLM_API_KEY` if set, otherwise `OPENAI_API_KEY`, otherwise the
literal `EMPTY` key expected by many local servers. Add `--no-json-mode` if the
server rejects `response_format=json_object`.

### Anthropic Messages API

```bash
export ANTHROPIC_API_KEY="..."
bash src/scripts/eval_core.sh \
  --provider anthropic \
  --model claude-sonnet-4-5
```

### Full set, smoke, and filters

```bash
# 10 per world (160 tasks) is the default smoke
bash src/scripts/eval_core.sh --provider vllm --model local-model

# all 19,950 tasks
bash src/scripts/eval_core.sh --provider openai --model gpt-4.1 \
  --sample-per-world 0

# only some worlds or lanes
bash src/scripts/eval_core.sh --worlds world_L04,world_L15
bash src/scripts/eval_core.sh --lanes L04,L15
```

### Download-and-run wrapper

```bash
export AOB_CORE_ORACLE="../aob-private/oracle/core_oracle.jsonl"
bash src/scripts/eval_core_from_hf.sh \
  --provider vllm \
  --model local-model \
  --base-url http://127.0.0.1:8000/v1
```

The wrapper defaults to `Vinyeh/AutoOntologyBench` and `../aob-data`. Override
with `AOB_CORE_HF_REPO` and `AOB_CORE_DATA_DIR` (or `AOB_DATA_DIR`).

## Outputs

Artifacts go to `runs/eval_core_<UTC>/`:

```text
predictions.jsonl
tasks_sampled.jsonl
report.json
rows.jsonl
run.log
api_logs/            # if the selected client logs requests
```

## Metrics

`report.json` contains:

- `metrics.tasks`: number of selected tasks.
- `metrics.scored`: tasks with both a prediction and an oracle label.
- `metrics.missing`: tasks missing a prediction or an oracle row.
- `metrics.unknown_predictions`: predictions whose `task_id` is not in the
  selected task set.
- `metrics.success`: tasks whose every action is correct.
- `metrics.tsr` (TSR): `success / scored`. **The whole task counts only when
  every action is correct.**
- `metrics.step_accuracy`: `steps_correct / steps`. **The fraction of correct
  actions across all scored tasks.**
- `metrics.steps`, `metrics.steps_correct`: raw denominators.
- `per_world`, `per_lane`, `per_split`: each bucket reports `tasks`, `scored`,
  `success`, `tsr`, `steps`, `steps_correct`, `step_accuracy`.

Scoring is exact; no model judge is used.

## Custom harness

Any harness can produce a JSONL submission with one record per task:

```json
{"task_id": "world_L04:task_0001", "answer": ["CHEBI:15361"]}
```

Then run:

```bash
python -m benchmark.eval_core.run \
  --tasks ../aob-data/core_worlds/data/core/tasks/tasks.jsonl \
  --predictions runs/my_harness/predictions.jsonl \
  --oracle ../aob-private/oracle/core_oracle.jsonl \
  --out runs/my_harness/score
```

`task_id` is the public `release_task_id` (`<world_id>:<task_id>`).

Answer contract:

- one-step query: exact JSON payload (list/dict/string/number);
- one-step validation: `"ok"` or `"reject:<reason>"` (a boolean is also
  accepted);
- multi-step: one entry per action, or
  `{"outcomes": [{"ok": ..., "reason": ..., "payload": ...}, ...]}`.

This is the recommended way to connect a custom agent harness. It is not a
drop-in adapter for arbitrary harnesses: the harness must emit this schema.

## Rebuild the release

The public data is assembled from 17-file world packs:

```bash
python src/scripts/assemble_core_release.py \
  --pack <pack-dir> --pack <pack-dir> ... \
  --out data/core --force
python src/scripts/assemble_core_release.py --verify --out data/core
```

`src/scripts/build_core_oracle.py` rebuilds the private oracle from the packs.
The public archive does not contain the packs; the construction code and the
release contract are included for auditability.

## What this evaluator is not

The Core-track release is **not** a scored ontology-repair harness. The full
Core loop also has a frozen executor, gold/corrupt/repaired bundles, native
verifiers, and pack-level task context. Those private materials are not in the
public package. This evaluator scores **answers** against the private replay
oracle; it does not run the researcher/executor tool loop.

## License and academic use

- Code: Apache-2.0 (`LICENSE`).
- Project-authored Core data: CC BY 4.0 (`LICENSE-DATA.md` in the dataset).
- Upstream source terms: `LICENSES/` in the dataset and `src/LICENSES/` in this
  repository.

Academic research use is the intended use and is permitted without a separate
fee; see [`ACADEMIC_USE.md`](../ACADEMIC_USE.md). Upstream terms (including L12
pm4py AGPL, L13 per-spec licenses, and L06 GOA terms) remain in force.