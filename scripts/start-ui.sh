#!/bin/sh
set -eu
. "$(dirname "$0")/common.sh"
exec "$PSEUDO_PYTHON" "$PROJECT_DIR/pseudo-ui/app.py" "$@"
