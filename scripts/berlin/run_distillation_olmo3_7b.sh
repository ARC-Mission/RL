#!/bin/bash
# Submit OLMo3-7B v14-mix50-wu25 self-distillation on one Berlin H100 node.
#
# Training dynamics match Jupiter Qwen3-8B v14-mix50-wu25:
#   LR = 5e-6, warmup 25 steps, GBS = 64, reverse KL, topk = 512
#   teacher_student_prefix_fraction = 0.50
#   Traces from qwen3_8b_answer column
#
# Topology on 1 × 4 H100 80GB:
#   policy/teacher TP=2, CP=1 → DP=2
#   vLLM TP=2

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/berlin/distill-olmo3-7b-v14-mix50-wu25.yaml}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
RUN_NAME="${RUN_NAME:-d-olmo3-7b-v14-mix50-wu25-berlin}"
AUTO_EVAL_WATCHER="${AUTO_EVAL_WATCHER:-1}"
EVAL_AT_START="${EVAL_AT_START:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-distillation}"
WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_ID_PREFIX="$(printf '%s' "$RUN_NAME" | tr -cd '[:alnum:]_-' | cut -c1-40)"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID_PREFIX}-$(date +%Y%m%d%H%M%S)}"

CKPT_ROOT="${CKPT_ROOT:-/fast/project/HFMI_SynergyUnit/yll/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/fast/project/HFMI_SynergyUnit/yll/logs}"
CKPT_DIR="${CKPT_ROOT}/${RUN_NAME}"
INITIAL_EVAL_MODEL="${INITIAL_EVAL_MODEL:-allenai/Olmo-3-7B-Think}"
INITIAL_EVAL_TOKENIZER="${INITIAL_EVAL_TOKENIZER:-$INITIAL_EVAL_MODEL}"

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: Ray launcher not found: $RAY_SUB"
  exit 1
fi

for override in "$@"; do
  case "$override" in
    checkpointing.checkpoint_dir=*) CKPT_DIR="${override#checkpointing.checkpoint_dir=}" ;;
    logger.wandb.project=*) WANDB_PROJECT="${override#logger.wandb.project=}" ;;
    logger.wandb.name=*) WANDB_NAME="${override#logger.wandb.name=}" ;;
    logger.wandb.entity=*) WANDB_ENTITY="${override#logger.wandb.entity=}" ;;
  esac
done

mkdir -p "$CKPT_DIR" "$LOG_ROOT/slurm" "$LOG_ROOT/$RUN_NAME" "$REPO_ROOT/scripts/logs"

cmd=(
  env
  WANDB_RUN_ID="$WANDB_RUN_ID"
  WANDB_RESUME=allow
  uv run python examples/run_distillation.py
  --config "$CONFIG"
  cluster.num_nodes="$NODES"
  cluster.gpus_per_node="$GPUS_PER_NODE"
  checkpointing.checkpoint_dir="$CKPT_DIR"
  checkpointing.save_consolidated=true
  checkpointing.keep_top_k=null
  logger.log_dir="$LOG_ROOT/$RUN_NAME"
  logger.wandb.project="$WANDB_PROJECT"
  logger.wandb.name="$WANDB_NAME"
  "$@"
)

printf -v COMMAND '%q ' "${cmd[@]}"
export COMMAND
export GPUS_PER_NODE

submit_output="$(
  sbatch \
  --nodes="$NODES" \
  --gres="gpu:$GPUS_PER_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --time="$SBATCH_TIME" \
  --job-name="$RUN_NAME" \
  "$RAY_SUB"
)"
echo "$submit_output"

TRAINING_JOB_ID="$(awk '/Submitted batch job/ {print $4}' <<< "$submit_output" | tail -n 1)"

echo "Submitted $RUN_NAME with $NODES node(s), $GPUS_PER_NODE GPU(s)/node."
echo "W&B run id: $WANDB_RUN_ID"

if [[ "$AUTO_EVAL_WATCHER" == "1" ]]; then
  if [[ -z "$TRAINING_JOB_ID" ]]; then
    echo "WARNING: Could not parse training Slurm job id; not launching eval watcher."
    exit 0
  fi

  eval_watcher_args=()
  if [[ -n "${EVAL_WATCHER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    eval_watcher_args=($EVAL_WATCHER_ARGS)
  fi

  TRAINING_JOB_ID="$TRAINING_JOB_ID" \
  EVAL_WANDB_PROJECT="$WANDB_PROJECT" \
  EVAL_WANDB_NAME="$WANDB_NAME" \
  EVAL_WANDB_ENTITY="$WANDB_ENTITY" \
  EVAL_WANDB_RUN_ID="$WANDB_RUN_ID" \
    "$SCRIPT_DIR/submit_sft_checkpoint_eval_watcher.sh" "$CKPT_DIR" "${eval_watcher_args[@]}"

  if [[ "$EVAL_AT_START" == "1" ]]; then
    initial_eval_dir="$CKPT_DIR/step_0000_initial"
    mkdir -p "$initial_eval_dir"
    RUN_NAME="eval-start-$RUN_NAME" \
    HF_MODEL="$INITIAL_EVAL_MODEL" \
    TOKENIZER_PATH="$INITIAL_EVAL_TOKENIZER" \
    EVAL_ROOT="$initial_eval_dir/evals" \
    EVAL_TRAIN_STEP=0 \
    EVAL_WANDB_PROJECT="$WANDB_PROJECT" \
    EVAL_WANDB_NAME="$WANDB_NAME" \
    EVAL_WANDB_ENTITY="$WANDB_ENTITY" \
    EVAL_WANDB_RUN_ID="$WANDB_RUN_ID" \
      "$SCRIPT_DIR/run_sft_checkpoint_eval.sh" "$initial_eval_dir" "${eval_watcher_args[@]}"
  fi
fi
