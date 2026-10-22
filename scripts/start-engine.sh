#!/bin/sh
set -eu
. "$(dirname "$0")/common.sh"
exec "$PSEUDO_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 "$@"
