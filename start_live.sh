#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${YOLO_PYTHON:-}" ]]; then
  yolo_python="$YOLO_PYTHON"
elif command -v python >/dev/null 2>&1; then
  yolo_python=python
else
  yolo_python=python3
fi
if ! command -v "$yolo_python" >/dev/null 2>&1; then
  echo "Python not found. Activate the installed environment or set YOLO_PYTHON." >&2
  exit 1
fi
exec "$yolo_python" -u "$package_root/scripts/view_d405_yolo.py" \
  --weights "$package_root/weights/cassette_lower_epoch173.pt" \
  --width 1280 --height 720 --fps 30 --display-fps 30 --record-mode annotated "$@"
