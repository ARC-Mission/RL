#!/bin/bash
# 4B v14-mix50-wu25 self-distillation on 1 Horeka node (4× A100 80GB).
# Run from srun --pty bash inside a 1-node salloc.
#
# Topology: TP=1, CP=1, vLLM TP=1 → DP=4 on 4 GPUs.
# ~16 GB weights (policy + teacher) + activations for 33k teacher seq → fits at 80GB.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

NAME="d-4b-v14-mix50-wu25-1n-horeka"
CKPT="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/checkpoints/${NAME}"
LOG="/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/logs/${NAME}"
mkdir -p "$CKPT" "$LOG"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
TP_PLAN="examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable"

export COMMAND="uv run python examples/run_distillation.py \
  --config $CONFIG \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=4 \
  policy.model_name=Qwen/Qwen3-4B \
  teacher.model_name=Qwen/Qwen3-4B \
  policy.dtensor_cfg.tensor_parallel_size=1 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=2 \
  teacher.dtensor_cfg.context_parallel_size=2 \
  policy.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  teacher.dtensor_cfg.custom_parallel_plan=$TP_PLAN \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.55 \
  teacher.generation.vllm_cfg.gpu_memory_utilization=0.55 \
  policy.logprob_batch_size=1 \
  policy.train_micro_batch_size=1 \
  policy.logprob_chunk_size=512 \
  policy.dynamic_batching.train_mb_tokens=17408 \
  policy.dynamic_batching.logprob_mb_tokens=17408 \
  teacher.logprob_batch_size=1 \
  teacher.logprob_chunk_size=512 \
  teacher.dynamic_batching.logprob_mb_tokens=33280 \
  data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v14_independent_check_improved.txt \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_4b_answer=trace' \
  distillation.teacher_student_prefix_fraction=0.50 \
  distillation.val_at_start=false \
  policy.scheduler.0.kwargs.total_iters=25 \
  'policy.scheduler.2.milestones=[25]' \
  checkpointing.checkpoint_dir=$CKPT \
  logger.log_dir=$LOG \
  logger.wandb.name=$NAME"

exec bash "$SCRIPT_DIR/ray.sub"
