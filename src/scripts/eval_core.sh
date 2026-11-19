#!/usr/bin/env bash
# eval_core.sh — one-click Core-track answer evaluation.
#
#   1. optionally generate predictions from an OpenAI-compatible API,
#      a local vLLM OpenAI-compatible server, or Anthropic Messages API;
#   2. score them against a separately mounted private Core oracle.
#
# Usage (from anywhere; resolves the repo root itself):
#   src/scripts/eval_core.sh --oracle /path/to/core_oracle.jsonl
#   src/scripts/eval_core.sh --provider openai --model gpt-4.1 --base-url https://api.openai.com/v1
#   src/scripts/eval_core.sh --provider vllm --model local-model --base-url http://127.0.0.1:8000/v1
#   src/scripts/eval_core.sh --provider anthropic --model claude-sonnet-4-5
#   src/scripts/eval_core.sh --predictions runs/my_harness/predictions.jsonl --oracle ...
#
# The private oracle is required for scoring and is never downloaded or printed
# by this script. API keys are read from environment variables only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ "$(basename "$PARENT")" = "src" ] && [ -d "$PARENT/benchmark" ]; then
  ROOT="$(cd "$PARENT/.." && pwd)"
  SRC_ROOT="$PARENT"
else
  ROOT="$PARENT"
  SRC_ROOT="$ROOT"
fi
export PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p runs

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/.conda/envs/aob-core/bin/python" ]; then
    PYTHON_BIN="$ROOT/.conda/envs/aob-core/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

DEFAULT_TASKS=""
for candidate in \
  "$ROOT/data/core/tasks/tasks.jsonl" \
  "$ROOT/../aob-data/core_worlds/data/core/tasks/tasks.jsonl" \
  "$ROOT/../aob-data/data/core/tasks/tasks.jsonl" \
  "$ROOT/AutoOntologyBench-Core-data/data/core/tasks/tasks.jsonl" \
  "$ROOT/release_core/tasks/tasks.jsonl"; do
  if [ -f "$candidate" ]; then
    DEFAULT_TASKS="$candidate"
    break
  fi
done
if [ -z "$DEFAULT_TASKS" ]; then
  DEFAULT_TASKS="$ROOT/data/core/tasks/tasks.jsonl"
fi

TASKS="${EVAL_TASKS:-$DEFAULT_TASKS}"
ORACLE="${EVAL_ORACLE:-${AOB_CORE_ORACLE:-}}"
PROVIDER="${EVAL_PROVIDER:-openai}"
BASE_URL="${EVAL_BASE_URL:-}"
API_KEY_ENV="${EVAL_API_KEY_ENV:-}"
MODEL="${EVAL_MODEL:-qwen3.8-max}"
SAMPLE=10
WORLDS=""
LANES=""
WORKERS=4
MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}"
TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
PREDICTIONS=""
RESUME=""
NO_JSON_MODE=""
THINKING=""
OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --provider)        PROVIDER="$2"; shift 2 ;;
    --base-url)        BASE_URL="$2"; shift 2 ;;
    --api-key-env)     API_KEY_ENV="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --sample-per-world) SAMPLE="$2"; shift 2 ;;
    --worlds)          WORLDS="$2"; shift 2 ;;
    --lanes)           LANES="$2"; shift 2 ;;
    --workers)         WORKERS="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --oracle)          ORACLE="$2"; shift 2 ;;
    --predictions)     PREDICTIONS="$2"; shift 2 ;;
    --out)             OUT="$2"; shift 2 ;;
    --resume)          RESUME="--resume"; shift ;;
    --no-json-mode)    NO_JSON_MODE="--no-json-mode"; shift ;;
    --thinking)        THINKING="--thinking"; shift ;;
    -h|--help)         sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "eval_core: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$TASKS" ]; then
  echo "eval_core: tasks file not found: $TASKS" >&2
  exit 2
fi
if [ -z "$ORACLE" ]; then
  for candidate in \
    "$ROOT/../aob-private/oracle/core_oracle.jsonl" \
    "$ROOT/../aob-data/core_worlds/private/oracle/core_oracle.jsonl" \
    "$ROOT/../AutoOntologyBench-Core-private/oracle/core_oracle.jsonl" \
    "$ROOT/release_candidates/AutoOntologyBench-Core-private/oracle/core_oracle.jsonl"; do
    if [ -f "$candidate" ]; then
      ORACLE="$candidate"
      break
    fi
  done
fi
if [ -z "$ORACLE" ] || [ ! -e "$ORACLE" ]; then
  echo "eval_core: private oracle required; pass --oracle, EVAL_ORACLE, or AOB_CORE_ORACLE" >&2
  exit 3
fi

if [ -n "$PREDICTIONS" ] && [ ! -f "$PREDICTIONS" ]; then
  echo "eval_core: predictions file not found: $PREDICTIONS" >&2
  exit 3
fi

if [ -z "$OUT" ]; then
  RUN_ID="${EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  OUT=$(mktemp -d "runs/eval_core_${RUN_ID}_XXXXXX")
else
  mkdir -p "$OUT"
fi
LOG="$OUT/run.log"
mkdir -p "$OUT"

if [ -n "$PREDICTIONS" ]; then
  echo "== step 1/2: using supplied predictions ==" | tee "$LOG"
  SCORE_TASKS="$TASKS"
else
  case "$PROVIDER" in
    openai|openai-compatible)
      if [ -z "$API_KEY_ENV" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "eval_core: OPENAI_API_KEY must be set (or pass --api-key-env)" >&2
        exit 3
      fi ;;
    vllm)
      [ -n "$MODEL" ] || { echo "eval_core: --model is required for vllm" >&2; exit 2; } ;;
    anthropic|claude)
      [ -n "$MODEL" ] || { echo "eval_core: --model is required for anthropic" >&2; exit 2; }
      if [ -z "$API_KEY_ENV" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "eval_core: ANTHROPIC_API_KEY must be set (or pass --api-key-env)" >&2
        exit 3
      fi ;;
    *) echo "eval_core: unknown provider $PROVIDER" >&2; exit 2 ;;
  esac
  echo "== step 1/2: generate predictions (provider=$PROVIDER model=$MODEL sample_per_world=$SAMPLE) ==" | tee "$LOG"
  "$PYTHON_BIN" -m benchmark.eval_core.model_runner \
    --tasks "$TASKS" \
    --out "$OUT/predictions.jsonl" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --sample-per-world "$SAMPLE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    ${BASE_URL:+--base-url "$BASE_URL"} \
    ${API_KEY_ENV:+--api-key-env "$API_KEY_ENV"} \
    ${WORLDS:+--worlds "$WORLDS"} \
    ${LANES:+--lanes "$LANES"} \
    $RESUME $NO_JSON_MODE $THINKING 2>&1 | tee -a "$LOG"
  SCORE_TASKS="$OUT/tasks_sampled.jsonl"
  PREDICTIONS="$OUT/predictions.jsonl"
fi

echo "== step 2/2: score predictions (oracle mounted) ==" | tee -a "$LOG"
"$PYTHON_BIN" -m benchmark.eval_core.run \
  --tasks "$SCORE_TASKS" \
  --predictions "$PREDICTIONS" \
  --oracle "$ORACLE" \
  --no-count-check \
  --out "$OUT" 2>&1 | tee -a "$LOG"

echo "eval_core: artifacts in $OUT" | tee -a "$LOG"