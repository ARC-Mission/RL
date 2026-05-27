#!/bin/bash
# Submit generated-answer evals for one consolidated policy checkpoint.
#
# Usage:
#   scripts/berlin/run_sft_checkpoint_eval.sh /path/to/checkpoints/run/step_0100
#
# Requires checkpointing.save_consolidated=true so vLLM can load:
#   step_XXXX/policy/weights/model/consolidated

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/to/step_checkpoint [extra Hydra overrides...]"
  exit 2
fi

CKPT_PATH="$1"
shift

RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-hai008}"
RUN_NAME="${RUN_NAME:-eval-$(basename "$(dirname "$CKPT_PATH")")-$(basename "$CKPT_PATH")}"
LOG_ROOT="${LOG_ROOT:-/fast/project/HFMI_SynergyUnit/yll/logs}"

HF_MODEL="${HF_MODEL:-$CKPT_PATH/policy/weights/model/consolidated}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$CKPT_PATH/policy/tokenizer}"
EVAL_ROOT="${EVAL_ROOT:-$CKPT_PATH/evals}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-33280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TP_SIZE="${TP_SIZE:-1}"
EVAL_K_VALUE="${EVAL_K_VALUE:-1}"
EVAL_NUM_TESTS_PER_PROMPT="${EVAL_NUM_TESTS_PER_PROMPT:-4}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-${WANDB_PROJECT:-}}"
EVAL_WANDB_NAME="${EVAL_WANDB_NAME:-${WANDB_NAME:-}}"
EVAL_WANDB_ENTITY="${EVAL_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
EVAL_WANDB_RUN_ID="${EVAL_WANDB_RUN_ID:-${WANDB_RUN_ID:-}}"
EVAL_TRAIN_STEP="${EVAL_TRAIN_STEP:-}"

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: Ray launcher not found: $RAY_SUB"
  exit 1
fi
if [[ ! -d "$CKPT_PATH" ]]; then
  echo "ERROR: Checkpoint directory not found: $CKPT_PATH"
  exit 1
fi
if [[ ! -d "$HF_MODEL" && ( "$HF_MODEL" = /* || "$HF_MODEL" != */* ) ]]; then
  echo "ERROR: HF-consolidated model not found and value does not look like a HF repo id: $HF_MODEL"
  echo "       Set checkpointing.save_consolidated=true before training checkpoints are saved."
  exit 1
fi
if [[ ! -d "$TOKENIZER_PATH" && ( "$TOKENIZER_PATH" = /* || "$TOKENIZER_PATH" != */* ) ]]; then
  echo "ERROR: Tokenizer path not found and value does not look like a HF repo id: $TOKENIZER_PATH"
  exit 1
fi

mkdir -p "$EVAL_ROOT" "$LOG_ROOT/slurm" "$REPO_ROOT/scripts/logs"

cmd=(
  bash scripts/berlin/run_sft_checkpoint_eval_inner.sh
  "$CKPT_PATH"
  "$HF_MODEL"
  "$TOKENIZER_PATH"
  "$EVAL_ROOT"
  "$MAX_MODEL_LEN"
  "$MAX_NEW_TOKENS"
  "$TEMPERATURE"
  "$TP_SIZE"
  "$EVAL_K_VALUE"
  "$EVAL_NUM_TESTS_PER_PROMPT"
  "$@"
)

printf -v COMMAND '%q ' "${cmd[@]}"
export COMMAND
export GPUS_PER_NODE
export EVAL_WANDB_PROJECT
export EVAL_WANDB_NAME
export EVAL_WANDB_ENTITY
export EVAL_WANDB_RUN_ID
export EVAL_TRAIN_STEP

SBATCH_MEM="${SBATCH_MEM:-120G}"

sbatch_args=(
  --nodes=1
  --gres="gpu:$GPUS_PER_NODE"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$SBATCH_MEM"
  --time="$SBATCH_TIME"
  --job-name="$RUN_NAME"
)
if [[ -n "$SBATCH_EXCLUDE" ]]; then
  sbatch_args+=(--exclude="$SBATCH_EXCLUDE")
fi

sbatch \
  "${sbatch_args[@]}" \
  "$RAY_SUB"

echo "Submitted eval job $RUN_NAME for $CKPT_PATH"
