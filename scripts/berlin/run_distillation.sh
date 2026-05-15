#!/bin/bash
# Submit the current Berlin 8B chat-teacher distillation job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_distillation_chat_teacher_8b.sh" "$@"
