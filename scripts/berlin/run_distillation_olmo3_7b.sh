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
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
RUN_NAME="${RUN_NAME:-d-olmo3-7b-v14-mix50-wu25-berlin}"

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
  --time="$SBATCH_TIME" \
  --job-name="$RUN_NAME" \
  "$RAY_SUB"

echo "Submitted $RUN_NAME with $NODES node(s), $GPUS_PER_NODE GPU(s)/node."
