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

prepare_local_model_tokenizer() {
  if [[ ! -d "$HF_MODEL" || ! -d "$TOKENIZER_PATH" ]]; then
    return 0
  fi

  local tokenizer_files=(
    added_tokens.json
    chat_template.jinja
    merges.txt
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
  )

  local file
  for file in "${tokenizer_files[@]}"; do
    if [[ -f "$TOKENIZER_PATH/$file" && ! -e "$HF_MODEL/$file" ]]; then
      ln -s "$TOKENIZER_PATH/$file" "$HF_MODEL/$file"
    fi
  done
}

prepare_local_model_tokenizer

ckpt_name="$(basename "$CKPT_PATH")"
if [[ "$ckpt_name" == "step_0000_initial" ]]; then
  TRAIN_STEP="${EVAL_TRAIN_STEP:-0}"
  EVAL_WANDB_PREFIX="${EVAL_WANDB_PREFIX:-eval}"
elif [[ "$ckpt_name" =~ ^step_0*([0-9]+)$ ]]; then
  TRAIN_STEP="${BASH_REMATCH[1]}"
  EVAL_WANDB_PREFIX="${EVAL_WANDB_PREFIX:-eval}"
else
  TRAIN_STEP="${EVAL_TRAIN_STEP:-0}"
  EVAL_WANDB_PREFIX="${EVAL_WANDB_PREFIX:-eval}"
fi

log_eval_to_wandb() {
  local dataset="$1"
  local score="$2"
  local save_path="$3"
  local metric_name="$4"

  if [[ -z "${EVAL_WANDB_RUN_ID:-}" || -z "${EVAL_WANDB_PROJECT:-}" ]]; then
    return 0
  fi

  uv run python - "$dataset" "$score" "$TRAIN_STEP" "$EVAL_WANDB_PREFIX" "$save_path" "$TOKENIZER_PATH" "$metric_name" <<'PY'
import json
import os
import sys

import wandb
from transformers import AutoTokenizer

dataset, score, train_step, metric_prefix, save_path, tokenizer_path, metric_name = sys.argv[1:]
project = os.environ["EVAL_WANDB_PROJECT"]
run_id = os.environ["EVAL_WANDB_RUN_ID"]
name = os.environ.get("EVAL_WANDB_NAME") or None
entity = os.environ.get("EVAL_WANDB_ENTITY") or None

with open(os.path.join(save_path, "evaluation_data.json"), "r") as f:
    eval_data = json.load(f)["evaluation_data"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
generation_lengths = [
    len(tokenizer.encode(sample["response"], add_special_tokens=False))
    for sample in eval_data
]
mean_generation_length = sum(generation_lengths) / len(generation_lengths)

run = wandb.init(project=project, entity=entity, id=run_id, name=name, resume="allow")
wandb.define_metric("train_step")
wandb.define_metric(
    f"{metric_prefix}/{dataset}/{metric_name}",
    step_metric="train_step",
    step_sync=True,
)
wandb.define_metric(
    f"{metric_prefix}/{dataset}/mean_generation_length",
    step_metric="train_step",
    step_sync=True,
)
wandb.log(
    {
        "train_step": int(train_step),
        f"{metric_prefix}/{dataset}/{metric_name}": float(score),
        f"{metric_prefix}/{dataset}/mean_generation_length": mean_generation_length,
    }
)
run.finish()
PY
}

datasets=(math500 aime2024 aime2025 aime2026)

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  num_tests="$EVAL_NUM_TESTS_PER_PROMPT"
  k_value="$EVAL_K_VALUE"
  metric_name="pass_at_1_avg4"
  if [[ "$dataset" == "math500" ]]; then
    num_tests=1
    metric_name="pass_at_1_avg1"
  fi
  if [[ "$k_value" != "1" ]]; then
    echo "ERROR: Eval logging expects pass@1; got k_value=$k_value"
    exit 1
  fi
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
    eval.k_value="$k_value" \
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
    log_eval_to_wandb "$dataset" "$score" "$save_path" "$metric_name"
  else
    echo "WARNING: Could not parse score from $log_path; skipping W&B eval metric."
  fi
done

touch "$EVAL_ROOT/.done"
