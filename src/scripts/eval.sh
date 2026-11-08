#!/usr/bin/env bash
# eval.sh — one-click AutoOntologyBench model evaluation:
#   1. generate predictions via benchmark.eval_20k.model_runner against an
#      OpenAI-compatible endpoint or Anthropic Messages API
#      (keys are read from the provider-specific environment variables and
#      are never printed, logged or persisted by this script)
#   2. score them via benchmark.eval_20k.run against the mounted oracle roots
#
# Usage (from anywhere; resolves the repo root itself):
#   src/scripts/eval.sh                          # smoke: 50 tasks per lane
#   src/scripts/eval.sh --sample-per-lane 0      # full run (all tasks)
#   src/scripts/eval.sh --lanes L01,L05 --sample-per-lane 5
#   src/scripts/eval.sh --provider openai --model gpt-4.1 --base-url https://api.openai.com/v1
#   src/scripts/eval.sh --provider vllm --model local-model --base-url http://127.0.0.1:8000/v1
#   src/scripts/eval.sh --provider anthropic --model claude-sonnet-4-5
#
# Env overrides: EVAL_PROVIDER, EVAL_BASE_URL, EVAL_API_KEY_ENV, EVAL_TASKS,
# EVAL_ORACLE_ROOTS (required for scoring), EVAL_MODEL,
# EVAL_MAX_TOKENS (default 32768), PYTHON_BIN (default python3).
# Artifacts: runs/eval_<UTC timestamp>/{predictions.jsonl,
# tasks_sampled.jsonl, report.json, rows.jsonl, run.log, api_logs/}.
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
export PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p runs

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
TASKS="${EVAL_TASKS:-AutoOntologyBench/tasks/qualified_tasks_test.jsonl}"
ORACLE_ROOTS="${EVAL_ORACLE_ROOTS:-}"
PROVIDER="${EVAL_PROVIDER:-openai}"
BASE_URL="${EVAL_BASE_URL:-}"
API_KEY_ENV="${EVAL_API_KEY_ENV:-}"
MODEL="${EVAL_MODEL:-}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-32768}"
SAMPLE=50
LANE_SAMPLES="${EVAL_LANE_SAMPLES:-L11=10}"
LANES=""
WORKERS=8
RESUME=""
NO_JSON_MODE=""
THINKING=""
EXTRA_METRICS="${EVAL_EXTRA_METRICS:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --provider)        PROVIDER="$2"; shift 2 ;;
    --base-url)        BASE_URL="$2"; shift 2 ;;
    --api-key-env)     API_KEY_ENV="$2"; shift 2 ;;
    --sample-per-lane) SAMPLE="$2"; shift 2 ;;
    --lane-samples)    LANE_SAMPLES="$2"; shift 2 ;;
    --lanes)           LANES="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --workers)         WORKERS="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --no-json-mode)    NO_JSON_MODE="--no-json-mode"; shift ;;
    --thinking)        THINKING="--thinking"; shift ;;
    --no-extra-metrics) EXTRA_METRICS=0; shift ;;
    --resume)          RESUME="--resume"; shift ;;
    -h|--help)         sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "eval: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$PROVIDER" in
  openai|openai-compatible)
    [ -n "$MODEL" ] || MODEL="qwen3.8-max"
    if [ -z "$API_KEY_ENV" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "eval: OPENAI_API_KEY must be set (or pass --api-key-env)" >&2
      exit 3
    fi ;;
  vllm)
    [ -n "$MODEL" ] || { echo "eval: --model is required for vllm" >&2; exit 2; } ;;
  anthropic|claude)
    [ -n "$MODEL" ] || { echo "eval: --model is required for anthropic/claude" >&2; exit 2; }
    if [ -z "$API_KEY_ENV" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "eval: ANTHROPIC_API_KEY must be set (or pass --api-key-env)" >&2
      exit 3
    fi ;;
  *)
    echo "eval: unknown provider $PROVIDER (openai|vllm|anthropic)" >&2
    exit 2 ;;
esac

if [ ! -f "$TASKS" ]; then
  echo "eval: tasks file not found: $TASKS" >&2
  exit 2
fi

RUN_ID="${EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR=$(mktemp -d "runs/eval_${RUN_ID}_XXXXXX")
LOG="$RUN_DIR/run.log"

LANES_ARGS=()
if [ -n "$LANES" ]; then LANES_ARGS=(--lanes "$LANES"); fi
LANE_SAMPLE_ARGS=()
if [ -n "$LANE_SAMPLES" ] && [ "$SAMPLE" != "0" ]; then
  LANE_SAMPLE_ARGS=(--lane-sample "$LANE_SAMPLES")
fi
BASE_ARGS=()
if [ -n "$BASE_URL" ]; then BASE_ARGS=(--base-url "$BASE_URL"); fi
KEY_ARGS=()
if [ -n "$API_KEY_ENV" ]; then KEY_ARGS=(--api-key-env "$API_KEY_ENV"); fi
BACKEND_ARGS=(--provider "$PROVIDER" "${BASE_ARGS[@]}" "${KEY_ARGS[@]}" $NO_JSON_MODE $THINKING)

echo "== step 1/2: generate predictions (provider=$PROVIDER model=$MODEL max_tokens=$MAX_TOKENS sample_per_lane=$SAMPLE lane_samples=${LANE_SAMPLES:-none} lanes=${LANES:-all}) ==" | tee "$LOG"
"$PYTHON_BIN" -m benchmark.eval_20k.model_runner \
  --tasks "$TASKS" \
  --out "$RUN_DIR/predictions.jsonl" \
  --model "$MODEL" \
  --workers "$WORKERS" \
  --max-tokens "$MAX_TOKENS" \
  --sample-per-lane "$SAMPLE" \
  "${BACKEND_ARGS[@]}" \
  "${LANE_SAMPLE_ARGS[@]}" \
  "${LANES_ARGS[@]}" $RESUME 2>&1 | tee -a "$LOG"

ORACLE_ARGS=()
OLD_IFS="$IFS"; IFS=':'
for root in $ORACLE_ROOTS; do
  if [ -d "$root" ]; then ORACLE_ARGS+=(--oracle "$root"); fi
done
IFS="$OLD_IFS"
if [ "${#ORACLE_ARGS[@]}" -eq 0 ]; then
  echo "eval: EVAL_ORACLE_ROOTS must point to the private oracle bundle" >&2
  echo "eval: tried: $ORACLE_ROOTS" >&2
  exit 3
fi

echo "== step 2/2: score predictions (oracle roots: $ORACLE_ROOTS) ==" | tee -a "$LOG"
"$PYTHON_BIN" -m benchmark.eval_20k.run \
  --tasks "$RUN_DIR/tasks_sampled.jsonl" \
  --no-count-check \
  --predictions "$RUN_DIR/predictions.jsonl" \
  "${ORACLE_ARGS[@]}" \
  --no-repo-staging \
  --out "$RUN_DIR" 2>&1 | tee -a "$LOG"

"$PYTHON_BIN" - "$RUN_DIR/report.json" <<'PY' | tee -a "$LOG"
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    report = json.load(fh)
o = report["metrics"]["overall"]
fmt = lambda x: "n/a" if x is None else "%.4f" % x
print("== coverage ==")
print("tasks sampled : %d" % o["tasks"])
print("scored        : %d" % o["scored"])
print("missing       : %d" % o["missing"])
print("not_supported : %d" % o["not_supported"])
print("oracle_missing: %d" % o["oracle_missing"])
print("error         : %d" % o["error"])
print("TSR           : %s (n=%d)" % (fmt(o["tsr"]), o["tsr_denominator"]))
print("V-TSR         : %s (n=%d)" % (fmt(o["v_tsr"]), o["v_tsr_denominator"]))
PY

if [ "$EXTRA_METRICS" = "1" ]; then
  METRICS_DIR="$RUN_DIR/metrics"
  mkdir -p "$METRICS_DIR"
  FRULE_ORACLE_ARGS=()
  OLD_IFS="$IFS"; IFS=':'
  for root in $ORACLE_ROOTS; do
    if [ -d "$root" ]; then FRULE_ORACLE_ARGS+=(--oracle-root "$root"); fi
  done
  IFS="$OLD_IFS"

  echo "== step 3/3: extra metrics ==" | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_cost_cli \
    --root "$RUN_DIR" --out "$METRICS_DIR" 2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_batch \
    --run-dir "$RUN_DIR" --batch-label "$(basename "$RUN_DIR")" \
    --out "$METRICS_DIR/batch_tsr.json" 2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_regime \
    --runs-glob "$RUN_DIR/report.json" --tasks-dir "$RUN_DIR" \
    --out "$METRICS_DIR/regime_breakdown.json" 2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_mapf1 \
    --run-root "$RUN_DIR" \
    --out-json "$METRICS_DIR/mapf1.json" \
    --out-md "$METRICS_DIR/MAPF1.md" 2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_mapf1_common \
    --mapf1 "$METRICS_DIR/mapf1.json" \
    --allow-missing \
    --out "$METRICS_DIR/mapf1_common7.json" 2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_20k.metrics_frule \
    --run-root "$RUN_DIR" "${FRULE_ORACLE_ARGS[@]}" \
    --out-json "$METRICS_DIR/frule.json" \
    --out-md "$METRICS_DIR/FRULE.md" 2>&1 | tee -a "$LOG"
fi

echo "eval: artifacts in $RUN_DIR" | tee -a "$LOG"
