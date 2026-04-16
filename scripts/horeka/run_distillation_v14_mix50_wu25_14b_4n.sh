#!/bin/bash
# 14B v14-mix50-wu25 self-distillation on 4 Horeka nodes (16× A100 80GB).
# Run from srun --pty bash inside a 4-node salloc.
#
# Topology: TP=4, CP=2, vLLM TP=4 → DP=2 on 16 GPUs.
# 14B is already memory-tight on GH200 96GB at TP=2 (Jupiter uses micro_batch=1
# and reduced dyn-batch tokens). On A100 80GB we add CP=2 to halve the
# per-GPU activation slice of the 33k-token teacher sequence.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

NAME="d-14b-v14-mix50-wu25-4n-horeka"
CKPT="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/checkpoints/${NAME}"
LOG="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/logs/${NAME}"
mkdir -p "$CKPT" "$LOG"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
TP_PLAN="examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable"

export COMMAND="uv run python examples/run_distillation.py \
  --config $CONFIG \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4 \
  policy.model_name=Qwen/Qwen3-14B \
  teacher.model_name=Qwen/Qwen3-14B \
  policy.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  teacher.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.dynamic_batching.train_mb_tokens=20480 \
  policy.dynamic_batching.logprob_mb_tokens=20480 \
  data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v14_independent_check_improved.txt \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_14b_answer=trace' \
  distillation.teacher_student_prefix_fraction=0.50 \
  distillation.val_at_start=true \
  policy.scheduler.0.kwargs.total_iters=25 \
  'policy.scheduler.2.milestones=[25]' \
  checkpointing.checkpoint_dir=$CKPT \
  logger.log_dir=$LOG \
  logger.wandb.name=$NAME"

exec bash "$SCRIPT_DIR/ray.sub"
