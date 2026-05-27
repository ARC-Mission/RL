#!/bin/bash
# Runs inside one Ray allocation and evaluates checkpoint specs serially.

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 CKPT_PATH|WANDB_PROJECT|WANDB_NAME|WANDB_RUN_ID [...]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-33280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TP_SIZE="${TP_SIZE:-1}"
EVAL_K_VALUE="${EVAL_K_VALUE:-1}"
EVAL_NUM_TESTS_PER_PROMPT="${EVAL_NUM_TESTS_PER_PROMPT:-4}"
CLEAN_EVAL_ROOT="${CLEAN_EVAL_ROOT:-0}"

for spec in "$@"; do
  IFS='|' read -r CKPT_PATH EVAL_WANDB_PROJECT EVAL_WANDB_NAME EVAL_WANDB_RUN_ID <<<"$spec"

  if [[ -z "${CKPT_PATH:-}" ]]; then
    echo "ERROR: empty checkpoint path in spec: $spec"
    exit 2
  fi

  HF_MODEL="$CKPT_PATH/policy/weights/model/consolidated"
  TOKENIZER_PATH="$CKPT_PATH/policy/tokenizer"
  EVAL_ROOT="$CKPT_PATH/evals"

  if [[ ! -d "$HF_MODEL" ]]; then
    echo "ERROR: model directory not found: $HF_MODEL"
    exit 1
  fi
  if [[ ! -d "$TOKENIZER_PATH" ]]; then
    echo "ERROR: tokenizer directory not found: $TOKENIZER_PATH"
    exit 1
  fi

  if [[ "$CLEAN_EVAL_ROOT" == "1" ]]; then
    rm -rf \
      "$EVAL_ROOT/math500" \
      "$EVAL_ROOT/aime2024" \
      "$EVAL_ROOT/aime2025" \
      "$EVAL_ROOT/aime2026" \
      "$EVAL_ROOT/.done"
  fi

  export EVAL_WANDB_PROJECT
  export EVAL_WANDB_NAME
  export EVAL_WANDB_RUN_ID

  echo "[INFO] Evaluating $CKPT_PATH"
  echo "[INFO] W&B project=$EVAL_WANDB_PROJECT name=$EVAL_WANDB_NAME run_id=$EVAL_WANDB_RUN_ID"

  "$SCRIPT_DIR/run_sft_checkpoint_eval_inner.sh" \
    "$CKPT_PATH" \
    "$HF_MODEL" \
    "$TOKENIZER_PATH" \
    "$EVAL_ROOT" \
    "$MAX_MODEL_LEN" \
    "$MAX_NEW_TOKENS" \
    "$TEMPERATURE" \
    "$TP_SIZE" \
    "$EVAL_K_VALUE" \
    "$EVAL_NUM_TESTS_PER_PROMPT"
done
