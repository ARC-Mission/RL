#!/bin/bash
# 8B multi-turn refine self-distillation on 4 Horeka nodes (16× A100).
# Run from srun --pty bash inside a 4-node salloc.
#
# Topology: TP=4, CP=2, vLLM TP=4 (yaml defaults) → DP=2 on 16 GPUs.
#
# Teacher message log (multi-turn refine mode):
#   [user(problem)]
#   [assistant(<think>{static dataset trace}</think>)]   <-- injected context
#   [user(refine_instruction from refine.txt)]
#   [assistant(live student rollout)]                    <-- presented to be scored

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

NAME="d-8b-refine-wu25-4n-horeka"
CKPT="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/checkpoints/${NAME}"
LOG="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/logs/${NAME}"
mkdir -p "$CKPT" "$LOG"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-refine-bs64.yaml"
TP_PLAN="examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable"

export COMMAND="uv run python examples/run_distillation.py \
  --config $CONFIG \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4 \
  policy.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  teacher.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  distillation.val_at_start=true \
  policy.scheduler.0.kwargs.total_iters=25 \
  'policy.scheduler.2.milestones=[25]' \
  checkpointing.checkpoint_dir=$CKPT \
  logger.log_dir=$LOG \
  logger.wandb.name=$NAME"

exec bash "$SCRIPT_DIR/ray.sub"
