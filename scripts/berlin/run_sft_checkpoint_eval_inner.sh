#!/bin/bash
# Runs inside a Ray allocation. Evaluates one HF-consolidated policy checkpoint.

set -euo pipefail

CKPT_PATH="$1"
HF_MODEL="$2"
TOKENIZER_PATH="$3"
EVAL_ROOT="$4"
MAX_MODEL_LEN="$5"
MAX_NEW_TOKENS="$6"
TEMPERATURE="$7"
TP_SIZE="$8"
EVAL_K_VALUE="$9"
EVAL_NUM_TESTS_PER_PROMPT="${10}"
shift 10

ckpt_name="$(basename "$CKPT_PATH")"
if [[ "$ckpt_name" =~ ^step_0*([0-9]+)$ ]]; then
  TRAIN_STEP="${BASH_REMATCH[1]}"
else
  TRAIN_STEP="${EVAL_TRAIN_STEP:-0}"
fi

log_eval_to_wandb() {
  local dataset="$1"
  local score="$2"
  local num_tests="$3"
  local k_value="$4"

  if [[ -z "${EVAL_WANDB_RUN_ID:-}" || -z "${EVAL_WANDB_PROJECT:-}" ]]; then
    return 0
  fi

  uv run python - "$dataset" "$score" "$TRAIN_STEP" "$num_tests" "$k_value" <<'PY'
import os
import sys

import wandb

dataset, score, train_step, num_tests, k_value = sys.argv[1:]
project = os.environ["EVAL_WANDB_PROJECT"]
run_id = os.environ["EVAL_WANDB_RUN_ID"]
name = os.environ.get("EVAL_WANDB_NAME") or None
entity = os.environ.get("EVAL_WANDB_ENTITY") or None

run = wandb.init(project=project, entity=entity, id=run_id, name=name, resume="allow")
wandb.define_metric("train_step")
wandb.define_metric("eval/*", step_metric="train_step")
wandb.log(
    {
        "train_step": int(train_step),
        f"eval/{dataset}/pass_at_{k_value}": float(score),
        f"eval/{dataset}/num_tests_per_prompt": int(num_tests),
        f"eval/{dataset}/k_value": int(k_value),
    }
)
run.finish()
PY
}

datasets=(math500 aime2024 aime2025 aime2026)

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  num_tests="$EVAL_NUM_TESTS_PER_PROMPT"
  save_path="$EVAL_ROOT/$dataset"
  mkdir -p "$save_path"
  log_path="$save_path/run_eval.log"

  set +e
  uv run python examples/run_eval.py \
    --config examples/configs/evals/eval.yaml \
    generation.model_name="$HF_MODEL" \
    tokenizer.name="$TOKENIZER_PATH" \
    tokenizer.chat_template=default \
    data.prompt_file=examples/prompts/cot.txt \
    data.dataset_name="$dataset" \
    generation.vllm_cfg.max_model_len="$MAX_MODEL_LEN" \
    generation.max_new_tokens="$MAX_NEW_TOKENS" \
    generation.temperature="$TEMPERATURE" \
    generation.vllm_cfg.tensor_parallel_size="$TP_SIZE" \
    cluster.gpus_per_node="$TP_SIZE" \
    eval.metric=pass@k \
    eval.k_value="$EVAL_K_VALUE" \
    eval.num_tests_per_prompt="$num_tests" \
    eval.save_path="$save_path" \
    "$@" 2>&1 | tee "$log_path"
  eval_rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$eval_rc" -ne 0 ]]; then
    exit "$eval_rc"
  fi

  score="$(
    awk -F'[= (]' '/^score=/ {print $2}' "$log_path" | tail -n 1
  )"
  if [[ -n "$score" ]]; then
    log_eval_to_wandb "$dataset" "$score" "$num_tests" "$EVAL_K_VALUE"
  else
    echo "WARNING: Could not parse score from $log_path; skipping W&B eval metric."
  fi
done

touch "$EVAL_ROOT/.done"
