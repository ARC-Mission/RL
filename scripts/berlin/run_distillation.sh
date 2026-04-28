#!/bin/bash
# Backward-compatible entrypoint for Berlin distillation.
# Submits the current 8B chat-teacher job through the Ray launcher.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_distillation_chat_teacher_8b.sh"
