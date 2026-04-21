#!/bin/bash
#SBATCH --job-name=egmd-8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=05:00:00
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/logs/slurm/%j-%x.out

set -euo pipefail

module purge
module load GCC/13.3.0
module load CUDA/12.6.0
module load git

# GCC module can set OMP_PROC_BIND=true which crashes Ray's RuntimeEnvAgent
unset OMP_PROC_BIND OMP_PLACES

REPO_ROOT="/fast/project/HFMI_SynergyUnit/yll/RL"
cd "$REPO_ROOT"

# ── Core env ────────────────────────────────────────────────────────────────
export HF_HOME="/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

# ── Ray settings ────────────────────────────────────────────────────────────
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_TMPDIR="/tmp/${USER}/ray-${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Tell Ray to only use the CPUs/GPUs SLURM allocated, not the full node.
# Without this, Ray auto-detects all 224 CPUs on the node and pre-starts
# 224 idle workers, causing massive contention under SLURM's cgroup limit.
export RAY_OVERRIDE_RESOURCES="{\"CPU\": ${SLURM_CPUS_PER_TASK}, \"GPU\": 8}"
export RAY_OBJECT_STORE_MEMORY=$((50 * 1024 * 1024 * 1024))

# ulimit per Ray best practices
# https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

# Clean up stale Ray state from previous jobs on this node
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_latest 2>/dev/null || true

# ── Cache directories (avoid contention on shared filesystem) ───────────────
JOB_CACHE_ROOT="/fast/project/HFMI_SynergyUnit/yll/.cache/jobs/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${JOB_CACHE_ROOT}/xdg-cache"
export XDG_CONFIG_HOME="${JOB_CACHE_ROOT}/xdg-config"
export NEMO_RL_VLLM_CACHE_BASE="${JOB_CACHE_ROOT}/vllm"
export NEMO_RL_VLLM_CONFIG_BASE="${JOB_CACHE_ROOT}/vllm-config"
export TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_ROOT}/torch_inductor"
export TRITON_CACHE_DIR="${JOB_CACHE_ROOT}/triton"
export FLASHINFER_CACHE_DIR="${JOB_CACHE_ROOT}/flashinfer"
export TORCH_EXTENSIONS_DIR="${JOB_CACHE_ROOT}/torch_extensions"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

mkdir -p "$JOB_CACHE_ROOT" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$NEMO_RL_VLLM_CACHE_BASE" "$NEMO_RL_VLLM_CONFIG_BASE" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$FLASHINFER_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# ── Output directories ──────────────────────────────────────────────────────
mkdir -p /fast/project/HFMI_SynergyUnit/yll/logs/slurm
mkdir -p /fast/project/HFMI_SynergyUnit/yll/logs/egmd-8b-rewrite
mkdir -p /fast/project/HFMI_SynergyUnit/yll/checkpoints/egmd-8b-rewrite

# ── Launch ──────────────────────────────────────────────────────────────────
CONFIG="examples/configs/opsd/berlin/egmd-8b-1n8g-rewrite.yaml"
QWEN3_TP_PLAN="examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable"

uv run python examples/run_distillation.py \
    --config "$CONFIG" \
    policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
    teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
    "$@"
