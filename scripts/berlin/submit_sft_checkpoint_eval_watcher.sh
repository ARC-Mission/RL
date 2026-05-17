#!/bin/bash
# Submit the checkpoint eval watcher as a lightweight Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/to/checkpoint_run_dir [extra eval overrides...]"
  exit 2
fi

CKPT_ROOT="$1"
shift

RUN_NAME="${RUN_NAME:-watch-eval-$(basename "$CKPT_ROOT")}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/scripts/logs}"
mkdir -p "$LOG_DIR"

cmd=(
  env
  TRAINING_JOB_ID="${TRAINING_JOB_ID:-}"
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-}"
  EVAL_WANDB_NAME="${EVAL_WANDB_NAME:-}"
  EVAL_WANDB_ENTITY="${EVAL_WANDB_ENTITY:-}"
  EVAL_WANDB_RUN_ID="${EVAL_WANDB_RUN_ID:-}"
  "$SCRIPT_DIR/watch_sft_checkpoints_eval.sh"
  "$CKPT_ROOT"
  "$@"
)
printf -v COMMAND '%q ' "${cmd[@]}"

sbatch \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --time="$SBATCH_TIME" \
  --partition=standard \
  --qos=low \
  --account=hfmi_synergyunit \
  --job-name="$RUN_NAME" \
  --output="$LOG_DIR/%j-%x.out" \
  --wrap="cd '$REPO_ROOT' && $COMMAND"

echo "Submitted watcher $RUN_NAME for $CKPT_ROOT"
