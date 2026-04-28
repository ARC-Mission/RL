#!/bin/bash
# Submit Qwen3-8B chat-teacher self-distillation on one Berlin H100 node.
#
# Topology on 1 x 4 H100 80GB:
#   policy/teacher TP=2, CP=2 -> DP=1
#   vLLM TP=2
#
# The static teacher assistant context is read from qwen3_4b_answer.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/berlin/distill-8b-chat-teacher.yaml}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
RUN_NAME="${RUN_NAME:-distill-8b-chat-teacher-4btrace-berlin}"
ASSISTANT_CONTENT_MODE="${ASSISTANT_CONTENT_MODE:-full}"

CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: Ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT_ROOT/$RUN_NAME" "$LOG_ROOT/$RUN_NAME" "$REPO_ROOT/scripts/logs"

cmd=(
  uv run python examples/run_distillation.py
  --config "$CONFIG"
  cluster.num_nodes="$NODES"
  cluster.gpus_per_node="$GPUS_PER_NODE"
  data.default.teacher_refine_assistant_content_mode="$ASSISTANT_CONTENT_MODE"
  checkpointing.checkpoint_dir="$CKPT_ROOT/$RUN_NAME"
  logger.log_dir="$LOG_ROOT/$RUN_NAME"
  logger.wandb.name="$RUN_NAME"
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
