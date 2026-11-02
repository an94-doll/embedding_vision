#!/usr/bin/env python3
"""Verify this environment with real CUDA work; never fall back to CPU."""

import argparse
import ctypes
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def configure_local_cache():
    # Set before importing Ultralytics / matplotlib, without editing shared settings.
    for variable, subdir in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        path = ROOT / ".cache" / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["YOLO_OFFLINE"] = "true"


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def nvidia_info():
    info = {}
    commands = {
        "gpu_query": ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        "summary": ["nvidia-smi"],
    }
    for key, command in commands.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            info[key] = {"command": command, "returncode": result.returncode}
            if result.returncode:
                info[key]["error"] = (result.stderr or result.stdout).strip()
            elif key == "gpu_query":
                info[key]["columns"] = ["index", "name", "memory_total_mib", "memory_free_mib", "driver_version", "compute_capability"]
                info[key]["rows"] = [[part.strip() for part in line.split(",")] for line in result.stdout.strip().splitlines()]
            else:
                driver = re.search(r"Driver Version:\s*([\d.]+)", result.stdout)
                cuda = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
                info["driver_version"] = driver.group(1) if driver else None
                info["driver_supported_cuda_version"] = cuda.group(1) if cuda else None
        except (OSError, subprocess.TimeoutExpired) as exc:
            info[key] = {"command": command, "error": str(exc)}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/env_check.json", help="Absolute path or path relative to the project root")
    args = parser.parse_args()
    output = project_path(args.output)
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "failed",
        "project_dir": str(ROOT),
        "python": {"executable": sys.executable, "version": platform.python_version(), "prefix": sys.prefix},
        "system": {"platform": platform.platform(), "machine": platform.machine()},
        "device_requested": "cuda:0",
        "cpu_fallback_allowed": False,
    }
    stage = "imports"
    started = time.perf_counter()
    try:
        configure_local_cache()
        import torch
        import torchvision
        import ultralytics

        report["versions"] = {"torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__}
        report["torch_cuda_build_version"] = torch.version.cuda
        report["cuda_runtime_package_version"] = importlib.metadata.version("nvidia-cuda-runtime-cu12")
        report["nvidia"] = nvidia_info()
        report["cuda_available"] = torch.cuda.is_available()
        stage = "cuda_initialization"
        if not report["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; GPU acceptance cannot pass on CPU. Check device access and the NVIDIA driver.")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report["gpu"] = {
            "index": 0, "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "total_memory_mib": properties.total_memory / (1024 ** 2),
            "free_memory_bytes_before_test": free_bytes,
            "runtime_total_memory_bytes": total_bytes,
        }
        stage = "cuda_tensor_operation"
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        operation_started = time.perf_counter()
        a = torch.ones((128, 128), device="cuda:0", dtype=torch.float32)
        b = torch.full((128, 64), 0.5, device="cuda:0", dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize(0)
        operation_ms = (time.perf_counter() - operation_started) * 1000
        if any(t.device != torch.device("cuda:0") for t in (a, b, c)):
            raise RuntimeError("A CUDA input/output device check failed.")
        if not bool(torch.isfinite(c).all().item()) or not bool((c == 64.0).all().item()):
            raise RuntimeError("CUDA matrix multiplication produced an invalid result.")
        report["tensor_check"] = {
            "operation": "ones(128,128) @ full((128,64),0.5)",
            "dtype": str(c.dtype), "input_devices": [str(a.device), str(b.device)],
            "output_device": str(c.device), "output_shape": list(c.shape),
            "expected_value": 64.0, "observed_min": c.min().item(), "observed_max": c.max().item(),
            "all_finite": True, "numerically_correct": True, "synchronize_passed": True,
            "wall_time_ms_including_allocation_and_sync": operation_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "passed": True,
        }
        # Query the runtime actually loaded by this process, distinct from the
        # driver's supported CUDA version and torch's compile-time CUDA version.
        stage = "cuda_runtime_version"
        runtime_paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/libcudart.so" in line})
        if len(runtime_paths) != 1:
            raise RuntimeError(f"Expected one loaded CUDA runtime library, found: {runtime_paths}")
        runtime = ctypes.CDLL(runtime_paths[0])
        get_version = runtime.cudaRuntimeGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        version = ctypes.c_int()
        returncode = get_version(ctypes.byref(version))
        if returncode != 0:
            raise RuntimeError(f"cudaRuntimeGetVersion failed: {returncode}")
        report["loaded_cuda_runtime"] = {
            "library_path": runtime_paths[0], "version_integer": version.value,
            "version": f"{version.value // 1000}.{(version.value % 1000) // 10}",
            "api_returncode": returncode,
        }
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(output, report)
    print(json.dumps({"status": report["status"], "report": str(output), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
