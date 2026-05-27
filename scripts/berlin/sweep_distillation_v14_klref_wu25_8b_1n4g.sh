#!/bin/bash
# Sweep KL-ref penalty: 0.01, 0.02, 0.05, 0.10
# Each job: 1 node x 4 GPUs on Berlin, eval at start + steps 10, 25, then every 25.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_distillation_v14_klref005_wu25_8b_1n4g.sh"

declare -A KL_VALUES=(
  [001]=0.01
  [002]=0.02
  [005]=0.05
  [010]=0.10
)

for suffix in 001 002 005 010; do
  kl_val="${KL_VALUES[$suffix]}"
  run_name="d-qwen3-8b-v14-klref${suffix}-wu25-1n4g-berlin"

  echo "=== Launching $run_name (kl_ref=$kl_val) ==="

  RUN_NAME="$run_name" \
  WANDB_NAME="$run_name" \
    "$BASE_SCRIPT" \
    loss_fn.reference_policy_kl_penalty="$kl_val" \
    distillation.val_at_start=true

  echo ""
done

echo "All 4 sweep jobs submitted."
