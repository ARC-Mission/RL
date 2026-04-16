#!/bin/bash
# Allocate 2x4 accelerated-h100 nodes for 1 day inside tmux sessions.
# Each run uses a unique timestamp tag so it never collides with existing sessions/jobs.
# Usage: ./salloc_tmux.sh
# Jump out: Ctrl+b, then d
# Once attached, run:  srun --pty bash

NUM_SESSIONS=2
TAG=$(date +%m%d-%H%M%S)
CREATED=()

for i in $(seq 0 $((NUM_SESSIONS - 1))); do
    SESSION="gpu${i}-${TAG}"

    tmux new-session -d -s "$SESSION" \
        salloc --partition=accelerated \
               --account=hk-project-p0023960 \
               --nodes=4 \
               --gres=gpu:4 \
               --time=1-00:00:00 \
               --job-name="$SESSION"

    echo "Started tmux session: $SESSION  (4 nodes, 1 day)"
    CREATED+=("$SESSION")
done

echo ""
echo "  Sessions: ${CREATED[*]}"
echo "  Attach:   tmux attach -t ${CREATED[0]}"
echo "  Detach:   Ctrl+b, then d"
echo "  On node:  srun --pty bash"
echo "  Kill:     tmux kill-session -t ${CREATED[0]}"
echo "  Kill all: tmux kill-server"
