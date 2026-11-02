#!/usr/bin/env python3
"""Cache one fixed validation prediction pass for conservative deduplication checks."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from evaluate import IMAGE_SUFFIXES, read_truth

WEIGHTS = ROOT / "runs/pose/basket_white_box_2pt/weights/best.pt"
IMAGE_DIR = ROOT / "box_in_basket/dataset/images/val"
DESTINATION = Path(__file__).with_name("validation_predictions.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_path(image):
    return ROOT / "box_in_basket/dataset/labels/val" / image.with_suffix(".txt").name


def serialize(instance):
    return dict(box=instance.box.tolist(), points=instance.points.tolist(),
                confidence=float(instance.confidence), class_id=int(instance.class_id))


def main():
    images = sorted(p.resolve() for p in IMAGE_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) != 95:
        raise ValueError(f"Expected 95 validation images, found {len(images)}")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    device = 0 if torch.cuda.is_available() else "cpu"
    parameters = dict(imgsz=1280, conf=0.25, iou=0.7, max_det=300, quantize=32,
                      rect=True, augment=False, nms=True, half=False, device=device,
                      verbose=False, save=False)
    manifest = [dict(image=str(image.relative_to(ROOT)), image_sha256=sha256(image),
                     label_sha256=sha256(label_path(image))) for image in images]
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       torch=torch.__version__, ultralytics=ultralytics.__version__,
                       opencv=cv2.__version__, numpy=np.__version__,
                       cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
                       opencv_threads=cv2.getNumThreads(), cpu_count=os.cpu_count(),
                       device_name=torch.cuda.get_device_name(device) if device != "cpu" else platform.processor())
    key = dict(weights=str(WEIGHTS.relative_to(ROOT)), weights_sha256=sha256(WEIGHTS),
               parameters=parameters, manifest=manifest, environment=environment)
    if DESTINATION.is_file():
        existing = json.loads(DESTINATION.read_text())
        if existing.get("metadata", {}).get("cache_key") == key and len(existing.get("images", [])) == len(images):
            print(json.dumps(dict(status="reused", path=str(DESTINATION), images=len(images))), flush=True)
            return

    model = YOLO(str(WEIGHTS))
    if model.task != "pose" or list(model.model.model[-1].kpt_shape) != [2, 3]:
        raise ValueError("Expected two-endpoint pose checkpoint")
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if first is None:
        raise ValueError(f"Unreadable image: {images[0]}")
    # Predictor/model setup and explicit warm-up are excluded from recorded frame timings.
    model.predict(source=first, **parameters)
    if device != "cpu":
        torch.cuda.synchronize(device)
    print(json.dumps(dict(status="warmed", device=device, images=len(images))), flush=True)
    rows = []
    for index, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = frame.shape[:2]
        if device != "cpu":
            torch.cuda.synchronize(device)
        start = perf_counter()
        result = model.predict(source=frame, **parameters)[0]
        # CPU materialization also guarantees completion before the timer is stopped.
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        points = (result.keypoints.data.detach().cpu().numpy() if result.keypoints is not None
                  else np.empty((0, 2, 3), dtype=np.float32))
        if device != "cpu":
            torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - start) * 1000
        if points.shape != (len(boxes), 2, 3):
            raise ValueError(f"Unexpected endpoint shape {points.shape}: {image_path}")
        predictions = [dict(box=box.tolist(), points=point.tolist(), confidence=float(confidence),
                            class_id=int(class_id))
                       for box, point, confidence, class_id in zip(boxes, points, confidences, classes)]
        rows.append(dict(image=str(image_path.relative_to(ROOT)), width=width, height=height,
                         predictions=predictions, truth=[serialize(x) for x in read_truth(image_path, width, height)],
                         inference_ms=elapsed_ms))
        if index % 20 == 0 or index == len(images):
            print(json.dumps(dict(status="progress", completed=index, total=len(images))), flush=True)
    timings = np.array([row["inference_ms"] for row in rows])
    output = dict(metadata=dict(cache_key=key, names=model.names, warmup_calls=1,
                                timing_definition="perf_counter around predict and CPU result materialization; excludes image read, GT read, model initialization and one explicit warmup",
                                timing_summary_ms=dict(count=len(rows), mean=float(timings.mean()),
                                                       median=float(np.median(timings)), p95=float(np.percentile(timings, 95)),
                                                       minimum=float(timings.min()), maximum=float(timings.max()))), images=rows)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(DESTINATION)
    print(json.dumps(dict(status="written", path=str(DESTINATION), images=len(rows),
                         predictions=sum(len(row["predictions"]) for row in rows),
                         truth=sum(len(row["truth"]) for row in rows),
                         timing_summary_ms=output["metadata"]["timing_summary_ms"])), flush=True)


if __name__ == "__main__":
    main()
