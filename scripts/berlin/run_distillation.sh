#!/bin/bash
#SBATCH --job-name=run_distillation 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/RL/scripts/logs/%j-%x.out
module purge
module load GCC/13.3.0
module load CUDA/12.6.0
module load git

cd /fast/project/HFMI_SynergyUnit/yll/RL

export HF_HOME=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface
export TORCH_CUDA_ARCH_LIST="9.0"
export NRL_FORCE_REBUILD_VENVS=true
uv run python examples/run_distillation.py \
    policy.model_name="Qwen/Qwen3-1.7B" \
    policy.max_total_sequence_length=8192 \
    policy.generation.max_new_tokens=8192 \
    policy.generation.vllm_cfg.max_model_len=32768 \
    policy.dynamic_batching.train_mb_tokens=8704 \
    policy.dynamic_batching.logprob_mb_tokens=8704 \
    teacher.dynamic_batching.logprob_mb_tokens=8704 \
    distillation.val_max_total_sequence_length=32768 \
    distillation.val_max_new_tokens=32768 \
    distillation.topk_logits_k=512 \
    policy.optimizer.kwargs.lr=2e-5 \
    loss_fn.kl_type=mixed
