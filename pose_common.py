"""Shared configuration, label parsing and result export for five-point Pose."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {0: "front", 1: "back"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
DEFAULT_WEIGHTS = ROOT / "runs/pose/front_back_pose/weights/best.pt"
PRETRAINED_WEIGHTS = ROOT.parent / "yolo_pose/weights/yolov8n-pose.pt"


def load_config(path):
    import yaml

    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("数据配置不存在：" + str(path))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("数据配置必须是 YAML 字典")
    names = config.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in names.items()}
    if names != NAMES or config.get("kpt_shape") != [5, 3]:
        raise ValueError("需要 0=front、1=back 和 kpt_shape: [5, 3]")
    if config.get("flip_idx") != [4, 3, 2, 1, 0]:
        raise ValueError("flip_idx 应为 [4, 3, 2, 1, 0]")
    root = Path(config.get("path", str(path.parent))).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    config["path"] = str(root.resolve())
    return config


def split_images(config, split):
    value = config.get(split)
    if not value:
        raise ValueError("数据配置未指定 " + split)
    values = value if isinstance(value, list) else [value]
    images = []
    for entry in values:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path(config["path"]) / path
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip()).expanduser()
                if not image.is_absolute():
                    image = path.parent / image
                images.append(image.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path.resolve())
        else:
            raise ValueError("图片目录或清单不存在：" + str(path))
    if not images:
        raise ValueError(split + " 集没有图片")
    if len(images) != len(set(images)):
        raise ValueError(split + " 集存在重复图片路径")
    return sorted(images)


def label_path(image):
    parts = list(Path(image).parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError("图片路径必须包含 images 目录：" + str(image))
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_label(path, width=1, height=1):
    path = Path(path)
    if not path.is_file():
        raise ValueError("缺少标签：" + str(path))
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("当前数据集不接受空标签：" + str(path))
    instances = []
    for number, line in enumerate(rows, 1):
        fields = list(map(float, line.split()))
        prefix = "{}:{}".format(path, number)
        if len(fields) != 20 or not all(math.isfinite(x) for x in fields):
            raise ValueError(prefix + " 应包含 20 个有限数值")
        cls, cx, cy, bw, bh = fields[:5]
        if cls not in NAMES or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(prefix + " 类别或检测框无效")
        xmin, ymin, xmax, ymax = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if min(xmin, ymin) < -1e-6 or max(xmax, ymax) > 1 + 1e-6:
            raise ValueError(prefix + " 检测框越界")
        points = []
        for i in range(5):
            x, y, visible = fields[5 + i * 3:8 + i * 3]
            if not (0 <= x <= 1 and 0 <= y <= 1 and visible in (0, 1, 2)):
                raise ValueError(prefix + " 关键点无效")
            if visible and not (xmin - 1e-6 <= x <= xmax + 1e-6 and ymin - 1e-6 <= y <= ymax + 1e-6):
                raise ValueError(prefix + " 关键点在框外")
            points.append([x * width, y * height, int(visible)])
        instances.append({"class_id": int(cls), "bbox_xyxy":
                          [xmin * width, ymin * height, xmax * width, ymax * height],
                          "keypoints": points})
    return instances


def preflight(config, splits):
    images_by_split, seen_paths, seen_hashes = {}, set(), {}
    for split in splits:
        images = split_images(config, split)
        for image in images:
            if not image.is_file():
                raise ValueError("图片不存在：" + str(image))
            if image in seen_paths:
                raise ValueError("不同集合包含相同图片路径：" + str(image))
            seen_paths.add(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError("图片内容重复：{} / {}".format(seen_hashes[digest], image))
            seen_hashes[digest] = image
            read_label(label_path(image))
        images_by_split[split] = images
    return images_by_split


def select_device(requested):
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if not requested.isdigit():
        raise ValueError("device 应为 auto、cpu 或单个 GPU 编号，例如 0")
    if not torch.cuda.is_available() or int(requested) >= torch.cuda.device_count():
        raise ValueError("请求的 CUDA 设备不可用：" + requested)
    return requested


def validate_model(model):
    names = {int(k): v for k, v in model.names.items()}
    if model.task != "pose" or list(model.model.model[-1].kpt_shape) != [5, 3] or names != NAMES:
        raise ValueError("需要训练后的两类别五点 Pose 权重（0 front，1 back）")


def detections_from_result(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.keypoints is None:
        raise ValueError("模型检测结果未包含关键点")
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    points = result.keypoints.data.cpu().tolist()
    if len(points) != len(boxes):
        raise ValueError("检测框与关键点数量不匹配")
    rows = []
    for box, cls, score, kpts in zip(boxes, classes, scores, points):
        if len(kpts) != 5 or any(len(p) != 3 for p in kpts):
            raise ValueError("关键点输出必须为 5×3")
        cls = int(cls)
        if cls not in NAMES:
            raise ValueError("未知类别编号")
        rows.append({"class_id": cls, "class_name": NAMES[cls],
                     "confidence": float(score), "bbox_xyxy": box,
                     "keypoints": [{"name": "P{}".format(i + 1), "x": p[0], "y": p[1],
                                    "confidence": p[2]} for i, p in enumerate(kpts)]})
    return rows


def draw_detections(image, detections, point_conf):
    import cv2

    canvas = image.copy()
    for detection in detections:
        color = (60, 220, 60) if detection["class_id"] == 0 else (220, 100, 240)
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        title = "{} {:.2f}".format(detection["class_name"], detection["confidence"])
        cv2.putText(canvas, title, (x1, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for point in detection["keypoints"]:
            if point["confidence"] < point_conf:
                continue
            x, y = int(round(point["x"])), int(round(point["y"]))
            c = (0, 220, 255) if point["name"] in ("P2", "P3", "P4") else (255, 200, 0)
            cv2.circle(canvas, (x, y), 4, c, -1)
            cv2.putText(canvas, point["name"], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return canvas


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def save_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_imgsz(size):
    if size <= 0 or size % 32:
        raise ValueError("imgsz 应为正的 32 倍数，例如 736；原图无需修改")
