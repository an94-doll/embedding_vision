#!/usr/bin/env bash
set -euo pipefail
editor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
editor_python="${CASSETTE_PYTHON:-python3}"
exec "$editor_python" "$editor_root/batch/review_server.py" "$@"
