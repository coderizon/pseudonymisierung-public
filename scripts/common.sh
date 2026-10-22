#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/engine}"
export PSEUDO_UI_STATE_DIR="${PSEUDO_UI_STATE_DIR:-$PROJECT_DIR/data/ui}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export PSEUDO_PYTHON="${PSEUDO_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [ ! -x "$PSEUDO_PYTHON" ]; then
    echo 'Python virtual environment missing. Read README.md first.' >&2
    exit 1
fi
