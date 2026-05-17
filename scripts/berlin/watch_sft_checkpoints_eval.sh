#!/bin/bash
# Poll a checkpoint directory and submit eval jobs for new step_* checkpoints.
#
# Usage:
#   scripts/berlin/watch_sft_checkpoints_eval.sh \
#     /fast/project/HFMI_SynergyUnit/yll/checkpoints/sft-qwen3-8b-openthoughts-rewrite-1n4g-berlin

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/to/checkpoint_run_dir [extra eval overrides...]"
  exit 2
fi

CKPT_ROOT="$1"
shift

POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_IDLE_POLLS="${MAX_IDLE_POLLS:-36}"
SUBMIT_FINAL="${SUBMIT_FINAL:-1}"
TRAINING_JOB_ID="${TRAINING_JOB_ID:-}"

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "ERROR: Checkpoint root not found: $CKPT_ROOT"
  exit 1
fi

training_job_is_active() {
  if [[ -z "$TRAINING_JOB_ID" ]]; then
    return 0
  fi
  [[ -n "$(squeue -h -j "$TRAINING_JOB_ID" 2>/dev/null)" ]]
}

if [[ -n "$TRAINING_JOB_ID" ]]; then
  echo "Watching checkpoints while Slurm job $TRAINING_JOB_ID is active."
fi

idle_polls=0
while true; do
  submitted_any=0

  shopt -s nullglob
  for ckpt in "$CKPT_ROOT"/step_*; do
    [[ -d "$ckpt" ]] || continue
    marker="$ckpt/evals/.submitted"
    done_marker="$ckpt/evals/.done"
    hf_model="$ckpt/policy/weights/model/consolidated"
    config_file="$ckpt/config.yaml"

    if [[ -f "$marker" || -f "$done_marker" ]]; then
      continue
    fi
    if [[ ! -f "$config_file" || ! -d "$hf_model" ]]; then
      continue
    fi

    mkdir -p "$ckpt/evals"
    "$SCRIPT_DIR/run_sft_checkpoint_eval.sh" "$ckpt" "$@"
    date -Is > "$marker"
    submitted_any=1
  done
  shopt -u nullglob

  if ! training_job_is_active; then
    echo "Training job $TRAINING_JOB_ID is no longer active; stopping checkpoint watcher."
    exit 0
  fi

  if [[ "$submitted_any" -eq 1 ]]; then
    idle_polls=0
  else
    idle_polls=$((idle_polls + 1))
  fi

  if [[ "$SUBMIT_FINAL" -eq 0 && "$idle_polls" -ge "$MAX_IDLE_POLLS" ]]; then
    echo "No new checkpoints for $((POLL_SECONDS * MAX_IDLE_POLLS)) seconds; exiting."
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
