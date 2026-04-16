#!/bin/bash
# 32B v14-mix50-wu25 self-distillation on 8 Horeka nodes (32× A100 80GB).
# Run from srun --pty bash inside an 8-node salloc.
#
# Topology: TP=4, CP=2, vLLM TP=4 → DP=4 on 32 GPUs.
# A100 node has 4× 80GB GPUs so TP is capped at 4 by within-node NVLink;
# we use CP=2 (Jupiter GH200 could afford CP=1) to make the 33k teacher
# activations fit. micro_batch=1 + reduced dyn-batch tokens mirror Jupiter.
# Following Jupiter, use qwen3_14b_answer as the teacher trace for 32B.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

NAME="d-32b-v14-mix50-wu25-8n-horeka"
CKPT="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/checkpoints/${NAME}"
LOG="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/logs/${NAME}"
mkdir -p "$CKPT" "$LOG"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
TP_PLAN="examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable"

export COMMAND="uv run python examples/run_distillation.py \
  --config $CONFIG \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=4 \
  policy.model_name=Qwen/Qwen3-32B \
  teacher.model_name=Qwen/Qwen3-32B \
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
