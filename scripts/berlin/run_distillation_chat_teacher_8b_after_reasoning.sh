#!/bin/bash
# Submit the Berlin 8B chat-teacher run where the static assistant context
# uses only the text after </think> when present.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASSISTANT_CONTENT_MODE="after_reasoning_if_present"
export RUN_NAME="${RUN_NAME:-distill-8b-chat-teacher-4banswer-berlin}"

exec "$SCRIPT_DIR/run_distillation_chat_teacher_8b.sh" "$@"
