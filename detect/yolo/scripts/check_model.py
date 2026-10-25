#!/usr/bin/env python3
"""Check original YOLO26s-pose weights using one synthetic image on cuda:0."""

import argparse
from datetime import datetime
import hashlib
import json
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from check_env import ROOT, configure_local_cache, project_path, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/yolo26s-pose.pt", help="Absolute path or path relative to the project root")
    parser.add_argument("--output", default="reports/model_check.json", help="Absolute path or path relative to the project root")
    args = parser.parse_args()
    weights, output = project_path(args.weights), project_path(args.output)
    report = {
        "schema_version": 1, "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "failed", "project_dir": str(ROOT), "python_executable": sys.executable,
        "weights": {"path": str(weights)}, "device_requested": "cuda:0",
        "cpu_fallback_allowed": False, "training_started": False,
    }
    stage = "weights_file"
    started = time.perf_counter()
    hook = None
    try:
        if not weights.is_file():
            raise FileNotFoundError(f"Local weights do not exist: {weights}; download the official checkpoint first.")
        report["weights"].update({"size_bytes": weights.stat().st_size, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
        stage = "imports"
        configure_local_cache()
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.engine.results import Results

        report["versions"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__}
        stage = "cuda_initialization"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU inference does not satisfy acceptance.")
        torch.cuda.set_device(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        stage = "model_loading"
        model = YOLO(str(weights))
        report["task"] = model.task
        head = model.model.model[-1]
        report["model_class"] = type(model.model).__name__
        report["head_class"] = type(head).__name__
        report["original_kpt_shape"] = list(head.kpt_shape)
        report["yaml_kpt_shape"] = list(model.model.yaml["kpt_shape"])
        if model.task != "pose" or type(head).__name__ != "Pose26":
            raise RuntimeError("The checkpoint is not a YOLO26 pose model.")
        if report["original_kpt_shape"] != [17, 3] or report["yaml_kpt_shape"] != [17, 3]:
            raise RuntimeError("Expected the original COCO keypoint configuration [17, 3].")
        model.to("cuda:0")
        model.model.float().eval()
        devices = sorted({str(p.device) for p in model.model.parameters()})
        dtypes = sorted({str(p.dtype) for p in model.model.parameters()})
        report["model_parameter_devices"] = devices
        report["model_parameter_dtypes"] = dtypes
        if devices != ["cuda:0"] or dtypes != ["torch.float32"]:
            raise RuntimeError("The model parameters are not all FP32 on cuda:0.")

        forward_inputs = []

        def record_forward(module, inputs):
            tensor = inputs[0]
            item = {"device": str(tensor.device), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            forward_inputs.append(item)
            if tensor.device != torch.device("cuda:0") or tensor.dtype != torch.float32:
                raise RuntimeError(f"Actual model forward input violates the GPU/FP32 requirement: {item}")
            if tuple(tensor.shape) != (1, 3, 640, 640):
                raise RuntimeError(f"Unexpected model input shape: {item['shape']}")

        hook = model.model.register_forward_pre_hook(record_forward)
        # A deterministic gray background and white rectangle, generated only in memory.
        image = np.full((640, 640, 3), 114, dtype=np.uint8)
        image[220:420, 120:520] = 255
        report["input"] = {"shape_hwc": list(image.shape), "dtype": str(image.dtype), "description": "gray 114 background; white rectangle y=220:420, x=120:520", "sha256": hashlib.sha256(image.tobytes()).hexdigest(), "saved": False}
        stage = "gpu_inference"
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            results = model.predict(source=image, imgsz=640, device=0, quantize=32, batch=1, stream=False, save=False, save_txt=False, save_crop=False, show=False, verbose=False)
        torch.cuda.synchronize(0)
        wall_ms = (time.perf_counter() - inference_started) * 1000
        report["forward_inputs_including_library_warmup"] = forward_inputs
        if not forward_inputs:
            raise RuntimeError("No actual model forward input was observed.")
        stage = "result_validation"
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Results):
            raise RuntimeError("Expected exactly one Ultralytics Results object.")
        result = results[0]
        if tuple(result.orig_shape) != (640, 640) or result.boxes is None or result.keypoints is None:
            raise RuntimeError("Missing pose output fields or invalid original image shape.")
        boxes, keypoints = result.boxes.data, result.keypoints.data
        count = len(result.boxes)
        if tuple(boxes.shape) != (count, 6) or tuple(keypoints.shape) != (count, 17, 3):
            raise RuntimeError(f"Invalid output shapes: boxes={tuple(boxes.shape)}, keypoints={tuple(keypoints.shape)}")
        if any(t.device != torch.device("cuda:0") for t in (boxes, keypoints)):
            raise RuntimeError("Result tensors are not on cuda:0.")
        if not all(bool(torch.isfinite(t).all().item()) for t in (boxes, keypoints)):
            raise RuntimeError("Inference returned non-finite output values.")
        report["prediction"] = {
            "result_type": type(result).__name__, "result_count": len(results), "detection_count": count,
            "boxes_shape": list(boxes.shape), "keypoints_shape": list(keypoints.shape),
            "boxes_device": str(boxes.device), "keypoints_device": str(keypoints.device),
            "all_finite": True, "synchronize_passed": True, "empty_detections_are_valid": True,
            "wall_time_ms_including_predictor_setup_and_warmup": wall_ms,
            "ultralytics_speed_ms": result.speed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        }
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        if hook is not None:
            hook.remove()
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(output, report)
    print(json.dumps({"status": report["status"], "report": str(output), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
