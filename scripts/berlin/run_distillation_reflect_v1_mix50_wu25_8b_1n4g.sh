#!/bin/bash
# Submit Qwen3-8B reflection-guided self-distillation on one Berlin H100 node.
#
# Topology on 1 x 4 H100:
#   policy TP=2, CP=1 -> DP=2
#   teacher TP=2, CP=2 -> DP=1
#   vLLM TP=2
#
# Training dynamics match the v14-mix50-wu25 run except this wrapper
# limits training rollouts to 8k tokens at temperature 0.6.
# Only difference: teacher sees extracted <reflection> blocks instead
# of the full candidate trace, with a new prompt asking the teacher
# to calibrate reasoning effort based on the reflections.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/berlin/distill-qwen3-8b-reflect-v1-mix50-wu25-gen8k-fast-1n4g.yaml}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-8192}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.6}"
RUN_NAME="${RUN_NAME:-d-qwen3-8b-reflect-v1-mix50-wu25-gen8k-t06-fast-1n4g-berlin}"
QWEN3_TP_PLAN="${QWEN3_TP_PLAN:-examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable}"

CKPT_ROOT="${CKPT_ROOT:-/fast/project/HFMI_SynergyUnit/yll/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/fast/project/HFMI_SynergyUnit/yll/logs}"

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: Ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT_ROOT/$RUN_NAME" "$LOG_ROOT/slurm" "$LOG_ROOT/$RUN_NAME" "$REPO_ROOT/scripts/logs"

cmd=(
  uv run python examples/run_distillation.py
  --config "$CONFIG"
  cluster.num_nodes="$NODES"
  cluster.gpus_per_node="$GPUS_PER_NODE"
  policy.dtensor_cfg.custom_parallel_plan="$QWEN3_TP_PLAN"
  teacher.dtensor_cfg.custom_parallel_plan="$QWEN3_TP_PLAN"
  policy.generation.max_new_tokens="$TRAIN_MAX_NEW_TOKENS"
  policy.generation.temperature="$TRAIN_TEMPERATURE"
  checkpointing.checkpoint_dir="$CKPT_ROOT/$RUN_NAME"
  logger.log_dir="$LOG_ROOT/$RUN_NAME"
  logger.wandb.name="$RUN_NAME"
  "$@"
)

printf -v COMMAND '%q ' "${cmd[@]}"
export COMMAND
export GPUS_PER_NODE

sbatch \
  --nodes="$NODES" \
  --gres="gpu:$GPUS_PER_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --time="$SBATCH_TIME" \
  --job-name="$RUN_NAME" \
  "$RAY_SUB"

echo "Submitted $RUN_NAME with $NODES node(s), $GPUS_PER_NODE GPU(s)/node."
