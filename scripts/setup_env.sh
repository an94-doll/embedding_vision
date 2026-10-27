#!/usr/bin/env bash
# Create a dedicated Linux x86_64 Conda environment; never install into system Python.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: bash scripts/setup_env.sh
       YOLO_ENV=yolo26_cassette bash scripts/setup_env.sh

Default environment name: yolo26. Creates Python 3.11 if missing; reuses an
existing Python 3.11 environment only when installed locked packages agree.
Conflicting environments are preserved: choose another YOLO_ENV to continue.
Uses official Conda, PyPI and PyTorch CUDA 12.8 sources. Does not install or
change the NVIDIA driver, system CUDA, system Python, or unrelated environments.
The exported lock matches the source machine; a fresh install was not tested.
HELP
  exit 0
fi
if (( $# )); then
  printf 'Unknown arguments. Run with --help.\n' >&2
  exit 2
fi

setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd -- "$setup_dir/.." && pwd)"
target_env="${YOLO_ENV:-yolo26}"
if [[ ! "$target_env" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || "$target_env" == base || "$target_env" == root ]]; then
  printf 'YOLO_ENV must be a dedicated Conda environment name (not base/root or a path).\n' >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'This exact CUDA lock targets Linux x86_64. Other platforms need a separate compatible lock.\n' >&2
  exit 2
fi

conda_bin=''
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_bin="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
else
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "$candidate" ]]; then
      conda_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "$conda_bin" ]]; then
  printf 'Conda was not found. Install Conda, initialize your shell, then rerun this script.\n' >&2
  exit 2
fi
conda_base="$("$conda_bin" info --base)"
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"
env_prefix="$(conda env list --json | "$conda_base/bin/python" -c '
import json, pathlib, sys
matches=[p for p in json.load(sys.stdin)["envs"] if pathlib.Path(p).name == sys.argv[1]]
if len(matches)>1:
    sys.exit("Ambiguous environment name; choose a unique YOLO_ENV.")
print(matches[0] if matches else "")
' "$target_env")"

new_environment=0
if [[ -z "$env_prefix" ]]; then
  conda create --name "$target_env" --override-channels \
    --channel https://repo.anaconda.com/pkgs/main python=3.11 pip --yes
  conda activate "$target_env"
  new_environment=1
else
  conda activate "$env_prefix"
fi
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Existing environment is not Python 3.11. It was preserved; select a new YOLO_ENV.')
print('Dedicated interpreter:', sys.executable)
PY

inspect_lock() {
  python - "$package_dir/requirements.lock.txt" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
missing, conflicts = [], []
for line in Path(sys.argv[1]).read_text().splitlines():
    line=line.strip()
    if not line or line.startswith(('#','-')):
        continue
    name, expected=line.split('==',1)
    try:
        actual=metadata.version(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
        continue
    if actual != expected:
        conflicts.append(f'{name}: installed {actual}, lock requires {expected}')
for name in ('opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless'):
    try:
        version=metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    conflicts.append(f'{name}=={version} shares cv2 with the required GUI opencv-python')
if conflicts:
    print('Package versions conflict with this lock:',file=sys.stderr)
    print('\n'.join(conflicts),file=sys.stderr)
    raise SystemExit(11)
if missing:
    print('Missing locked packages:', ', '.join(missing))
    raise SystemExit(10)
print('All locked package versions already match.')
PY
}

needs_install="$new_environment"
if (( ! new_environment )); then
  if inspect_lock; then
    needs_install=0
  else
    check_status=$?
    if (( check_status == 10 )); then
      needs_install=1
    else
      printf 'Preserved the existing environment. Rerun with a new YOLO_ENV name.\n' >&2
      exit "$check_status"
    fi
  fi
fi
if (( needs_install )); then
  python -m pip --isolated install \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128'
  python -m pip --isolated install --requirement "$package_dir/requirements.lock.txt"
fi
inspect_lock
python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import cv2
import pyrealsense2
import torch
import ultralytics
gui=next((line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith('GUI:')), 'GUI: unknown')
if gui.split(':',1)[1].strip().upper() in {'NONE','NO','UNKNOWN'}:
    raise SystemExit('OpenCV lacks a GUI backend; this viewer requires the GUI opencv-python wheel.')
print(gui)
for name in ('torch','torchvision','ultralytics','pyrealsense2','opencv-python'):
    print(f'{name}=={metadata.version(name)}')
print('PyTorch CUDA runtime:', torch.version.cuda)
print('Setup checks package imports only; live GPU, desktop display and D405 access require their own run.')
PY
printf '\nEnvironment ready. Activate it in your shell with:\n  conda activate %q\n' "$target_env"
