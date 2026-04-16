#!/bin/bash
#SBATCH --job-name=run_distillation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # More CPUs = faster parallel compilation of flash-attn/vLLM
#SBATCH --gres=gpu:4          # GPU required to compile flash-attn, deep_ep, deep_gemm
#SBATCH --time=01:00:00       # flash-attn ~1h, vLLM ~1-2h to compile from source
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-p0023960
#SBATCH --output=/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/scripts/logs/%j-%x.out

module purge
module load compiler/gnu/13
module load devel/cuda/12.9

# compiler/gnu/13 sets OMP_PROC_BIND=true which causes Ray's RuntimeEnvAgent
# to crash when runtime_env env_vars are used (the agent process inherits
# strict OpenMP thread affinity and hangs, closing the Raylet socket).
unset OMP_PROC_BIND
unset OMP_PLACES

# Make CUDA compiler visible to build tools (flash-attn, deep_ep, etc.)
# Use CUDA_DIR if set by the module, otherwise keep existing CUDA_HOME
export CUDA_HOME=${CUDA_DIR:-$CUDA_HOME}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /hkfs/work/workspace/scratch/tum_hki2875-myspace/RL

# Expose pip-installed NCCL headers/libs so Transformer Engine can compile.
# The nvidia-nccl-cu12 pip package provides nccl.h but the compiler doesn't
# know to look inside site-packages without an explicit include path.
NCCL_PKG_DIR=$(uv run python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null)
if [ -n "$NCCL_PKG_DIR" ]; then
  export CPATH=${NCCL_PKG_DIR}/include:${CPATH:-}
  export LD_LIBRARY_PATH=${NCCL_PKG_DIR}/lib:$LD_LIBRARY_PATH
fi

uv run ray stop --force 2>/dev/null || true

export HF_HOME=/hkfs/work/workspace/scratch/tum_hki2875-myspace/.cache/huggingface

# ── Force single-node SLURM context ──
# When running interactively inside a multi-node allocation (srun --pty bash),
# SLURM vars still advertise ALL allocated nodes. Override everything so Ray,
# vLLM, and PyTorch distributed only see this single node.
THIS_NODE=$(hostname -s)
export SLURM_JOB_NUM_NODES=1
export SLURM_NNODES=1
export SLURM_NTASKS=1
export SLURM_NPROCS=1
export SLURM_NODELIST=$THIS_NODE
export SLURM_JOB_NODELIST=$THIS_NODE
export SLURM_STEP_NODELIST=$THIS_NODE

# Prevent transformer-engine from failing its compiled-extension sanity check
# (the PyPI wheel may not match the node's exact CUDA toolkit).
export NVTE_SKIP_SANITY_CHECKS=1

# Diagnostics
echo "=== Node: $THIS_NODE ==="
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi not found"
echo "========================="

NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_distillation.py
