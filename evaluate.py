#!/usr/bin/env python3
"""白盒两端点与铁篮四角点评估（兼容旧两点模型）。修改下方配置后直接运行 python evaluate.py。"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.check_env import configure_local_cache

configure_local_cache()

import cv2
import numpy as np
import yaml


# 配置集中在这里；像素阈值用于二维诊断，不代表机器人抓取精度已达标。
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/pose/white_box_basket_4pt/weights/best.pt"
DATA = ROOT / "box_in_basket/dataset/data.yaml"
SPLIT = "test"                         # 开发调参用 val，最终对比用 test。
IMGSZ = 1280
BATCH = 8
DEVICE = "auto"                        # 自动选择 GPU 0 或 CPU；也可直接填 0 / "cpu"。
WORKERS = 2
CONF = 0.25                            # 实际运行时的检测框阈值。
POINT_CONF = 0.50                      # 各有效端点/角点达到此阈值才视为可信。
MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DET = 300
PIXEL_THRESHOLDS = (2.0, 5.0, 10.0)
PAIR_ERROR_PX = 5.0                    # 两个端点各自的最大允许误差。
CENTER_ERROR_PX = 5.0
ANGLE_ERROR_DEG = 5.0                  # 无方向线段的角度误差，按180度周期计算。
RUN_STANDARD_MAP = True
AP_CONF = 0.001                        # mAP扫置信度曲线，不能用 CONF 截掉低分候选。
SAVE_STANDARD_PLOTS = True
SAVE_VISUALIZATIONS = True
PROJECT = ROOT / "runs/evaluate"
NAME = f"white_box_basket_4pt_{SPLIT}"  # 已有目录会自动递增，避免覆盖。

POINT_NAMES = ("left_endpoint", "right_endpoint")
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
CLASS_NAMES = {0: "white_box", 1: "basket"}
CORNER_ERROR_PX = 5.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INSTANCE_FIELDS = ("image gt_index pred_index matched_iou detection_confidence both_gt_known "
                   "geometry_evaluable both_pred_confident reversed_order pair_qualified localization_qualified "
                   "left_error_px right_error_px center_error_px angle_error_deg length_error_px "
                   "relative_length_error reason").split()
POINT_FIELDS = ("image gt_index keypoint gt_x gt_y pred_x pred_y point_confidence pixel_error available").split()
PREDICTION_FIELDS = ("image pred_index detection_confidence matched_gt_index best_gt_iou status "
                     "bbox_xyxy keypoints_xyc").split()


INSTANCE_FIELDS += ["class_id", "class_name", "all_corners_known", "all_corners_confident",
                    "corners_qualified", "max_corner_error_px"]
POINT_FIELDS += ["class_id", "class_name"]
PREDICTION_FIELDS += ["class_id", "class_name"]


def dataset_layout(config):
    names = config.get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else names
    shape = config.get("kpt_shape")
    if not ((shape == [2, 3] and names == {0: "white_box"})
            or (shape == [4, 3] and names in (CLASS_NAMES, {0: "basket"}))):
        raise ValueError("需要 white_box 单类两点、basket 单类四点，或 white_box/basket 双类四点配置")
    return names, shape


@dataclass
class Instance:
    box: np.ndarray                    # 原图 xyxy。
    points: np.ndarray                 # (K, 3)：GT第三列是可见性，预测第三列是置信度。
    confidence: float = 1.0
    class_id: int = 0


def semantic_instances(instances, names):
    """Map model-local IDs to evaluator IDs: white_box=0, basket=1."""
    mapping = {local: next(k for k, v in CLASS_NAMES.items() if v == name)
               for local, name in names.items()}
    return [Instance(x.box, x.points, x.confidence, mapping[x.class_id]) for x in instances]


def load_dataset():
    config = yaml.safe_load(DATA.read_text(encoding="utf-8-sig"))
    dataset_layout(config)
    root = Path(config.get("path", "."))
    root = root.resolve() if root.is_absolute() else (DATA.parent / root).resolve()
    sources = config[SPLIT]
    images = []
    for source in sources if isinstance(sources, list) else [sources]:
        source = root / source
        if source.is_dir():
            images.extend(p.resolve() for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif source.suffix.lower() == ".txt":
            images.extend((source.parent / line.strip()).resolve()
                          for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        else:
            images.append(source.resolve())
    if not images or len(images) != len(set(images)):
        raise ValueError("评估图片集合为空或包含重复路径")
    return config, sorted(images)


def read_truth(image, width, height, kpt_shape=(2, 3)):
    parts = list(image.parts)
    if "images" not in parts:
        raise ValueError("需要YOLO标准 images/... 与 labels/... 目录结构")
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    path = Path(*parts).with_suffix(".txt")
    lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:                       # 空标签是背景图，不能被排除。
        return []
    labels = np.asarray(lines, dtype=float)
    n = kpt_shape[0]
    if (labels.ndim != 2 or labels.shape[1] != 5 + n * 3 or not np.isfinite(labels).all()
            or not np.isin(labels[:, 0], [0] if n == 2 else [0, 1]).all()
            or np.any(labels[:, 3:5] <= 0)
            or not np.isin(labels[:, 7::3], [0, 1, 2]).all()):
        raise ValueError(f"标签必须符合 {n} 点Pose格式：" + str(path))
    truth = []
    for row in labels:
        cx, cy, w, h = row[1:5] * [width, height, width, height]
        points = row[5:].reshape(n, 3) * [width, height, 1]
        truth.append(Instance(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), points, class_id=int(row[0])))
    return truth


def box_ious(truth, predictions):
    if not truth or not predictions:
        return np.zeros((len(truth), len(predictions)))
    a, b = np.array([x.box for x in truth], dtype=float), np.array([x.box for x in predictions], dtype=float)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    union = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)[:, None] \
        + np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_instances(truth, predictions, threshold=MATCH_IOU):
    """先最大化合格匹配数量，再最大化总IoU；不使用端点误差挑选对应关系。"""
    from ultralytics.utils.ops import linear_sum_assignment

    ious = box_ious(truth, predictions)
    if not truth or not predictions:
        return {}, ious
    same_class = np.array([g.class_id for g in truth])[:, None] == [p.class_id for p in predictions]
    valid = (ious >= threshold) & same_class
    # 每个GT都有独立的未匹配列；bonus大于所有IoU收益之和，保证匹配数量优先。
    reward = np.where(valid, min(ious.shape) + 1 + ious, 0.0)
    cost = -np.concatenate([reward, np.zeros((len(truth), len(truth)))], axis=1)
    rows, cols = linear_sum_assignment(cost)
    return {int(g): int(p) for g, p in zip(rows, cols) if p < len(predictions) and valid[g, p]}, ious


def number(value):
    return float(value) if np.isfinite(value) else None


def ratio(a, b):
    return a / b if b else None


def distribution(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
            "rmse": float(np.sqrt(np.mean(values ** 2))) if len(values) else None}


def analyze_image(truth, predictions, width, height, image_name):
    matches, ious = match_instances(truth, predictions, MATCH_IOU)
    instance_rows, point_rows, prediction_rows = [], [], []
    for g, gt in enumerate(truth):
        p = matches.get(g)
        pred = predictions[p] if p is not None else None
        if gt.class_id == 1:
            row, rows = analyze_basket(gt, pred, g, p, ious[g, p] if p is not None else None,
                                       width, height, image_name)
            instance_rows.append(row)
            point_rows.extend(rows)
            continue
        # White-box placeholders never participate in endpoint or midpoint metrics.
        gt = Instance(gt.box, gt.points[:2], gt.confidence, gt.class_id)
        pred = Instance(pred.box, pred.points[:2], pred.confidence, pred.class_id) if pred is not None else None
        known = gt.points[:, 2] > 0
        errors = [None, None]
        available = [False, False]
        for k, name in enumerate(POINT_NAMES):
            point = pred.points[k] if pred is not None else None
            finite = point is not None and np.isfinite(point).all()
            if finite:
                available[k] = bool(point[2] >= POINT_CONF and 0 <= point[0] <= width and 0 <= point[1] <= height)
            if not known[k]:
                continue
            if finite:
                errors[k] = float(np.linalg.norm(point[:2] - gt.points[k, :2]))
            point_rows.append(dict(image=image_name, gt_index=g, keypoint=name, class_id=0, class_name="white_box",
                                   gt_x=float(gt.points[k, 0]), gt_y=float(gt.points[k, 1]),
                                   pred_x=number(point[0]) if point is not None else None,
                                   pred_y=number(point[1]) if point is not None else None,
                                   point_confidence=number(point[2]) if point is not None else None,
                                   pixel_error=errors[k], available=available[k]))
        both_known = bool(known.all())
        gt_vector = gt.points[1, :2] - gt.points[0, :2]
        gt_length = float(np.linalg.norm(gt_vector))
        geometry_evaluable = both_known and gt_length > 1e-6
        center_error = angle_error = length_error = relative_length_error = None
        reversed_order = bool(pred is not None and pred.points[0, 0] > pred.points[1, 0])
        if geometry_evaluable and all(e is not None for e in errors):
            vector = pred.points[1, :2] - pred.points[0, :2]
            length = float(np.linalg.norm(vector))
            center_error = float(np.linalg.norm(pred.points[:, :2].mean(0) - gt.points[:, :2].mean(0)))
            length_error = abs(length - gt_length)
            relative_length_error = length_error / gt_length
            if length > 1e-6:
                delta = math.degrees(math.atan2(vector[1], vector[0]) - math.atan2(gt_vector[1], gt_vector[0]))
                angle_error = abs((delta + 90) % 180 - 90)
        confident = all(available)
        pair_ok = bool(geometry_evaluable and angle_error is not None and confident and not reversed_order
                       and all(e is not None and e <= PAIR_ERROR_PX for e in errors))
        localization_ok = bool(pair_ok and center_error <= CENTER_ERROR_PX
                               and angle_error is not None and angle_error <= ANGLE_ERROR_DEG)
        reason = ("missed" if pred is None else "partial_gt" if not both_known else
                  "degenerate_gt_edge" if not geometry_evaluable else "unavailable_endpoint" if not confident else
                  "reversed_endpoint_order" if reversed_order else "degenerate_pred_edge" if angle_error is None else
                  "endpoint_error" if not pair_ok else
                  "geometry_error" if not localization_ok else "qualified")
        instance_rows.append(dict(image=image_name, gt_index=g, pred_index=p, class_id=0, class_name="white_box",
                                  matched_iou=float(ious[g, p]) if pred is not None else None,
                                  detection_confidence=pred.confidence if pred is not None else None,
                                  both_gt_known=both_known, geometry_evaluable=geometry_evaluable,
                                  both_pred_confident=confident, reversed_order=reversed_order,
                                  pair_qualified=pair_ok, localization_qualified=localization_ok,
                                  left_error_px=errors[0], right_error_px=errors[1], center_error_px=center_error,
                                  angle_error_deg=angle_error, length_error_px=length_error,
                                  relative_length_error=relative_length_error, reason=reason))
    reverse_matches = {p: g for g, p in matches.items()}
    for p, pred in enumerate(predictions):
        g = reverse_matches.get(p)
        same_class = [i for i, gt in enumerate(truth) if gt.class_id == pred.class_id]
        overlap = float(ious[same_class, p].max()) if same_class else 0.0
        status = "matched" if g is not None else "duplicate_candidate" if overlap >= MATCH_IOU else "false_positive"
        prediction_rows.append(dict(image=image_name, pred_index=p, class_id=pred.class_id,
                                    class_name=CLASS_NAMES[pred.class_id], detection_confidence=pred.confidence,
                                    matched_gt_index=g, best_gt_iou=overlap, status=status,
                                    bbox_xyxy=json.dumps([number(v) for v in pred.box]),
                                    keypoints_xyc=json.dumps([[number(v) for v in point] for point in pred.points])))
    return instance_rows, point_rows, prediction_rows


def summarize_white(instances, points, predictions):
    matched = sum(r["pred_index"] is not None for r in instances)
    pair_gt = sum(r["both_gt_known"] for r in instances)
    geometry_gt = sum(r["geometry_evaluable"] for r in instances)
    pair_ok = sum(r["pair_qualified"] for r in instances)
    localization_ok = sum(r["localization_qualified"] for r in instances)

    def point_metrics(rows):
        errors = [r["pixel_error"] for r in rows if r["pixel_error"] is not None]
        return {"evaluable_gt_points": len(rows), "matched_points": len(errors),
                "available_points": sum(r["available"] for r in rows),
                "pixel_error_raw_matched": distribution(errors),
                "pck_pixels": {str(t): {
                    "raw_matched": ratio(sum(e <= t for e in errors), len(errors)),
                    "raw_all_gt": ratio(sum(e <= t for e in errors), len(rows)),
                    "available_all_gt": ratio(sum(r["available"] and r["pixel_error"] is not None
                                                  and r["pixel_error"] <= t for r in rows), len(rows))}
                    for t in PIXEL_THRESHOLDS}}

    return {
        "detection": {"gt_instances": len(instances), "pred_instances": len(predictions),
                      "matched_instances": matched, "missed_instances": len(instances) - matched,
                      "false_positives_including_duplicates": len(predictions) - matched,
                      "duplicate_candidates": sum(r["status"] == "duplicate_candidate" for r in predictions),
                      "precision": ratio(matched, len(predictions)), "recall": ratio(matched, len(instances))},
        "endpoints": {"all": point_metrics(points), **{
            name: point_metrics([r for r in points if r["keypoint"] == name]) for name in POINT_NAMES}},
        "pairs": {"evaluable_gt_pairs": pair_gt, "partial_or_unknown_gt": len(instances) - pair_gt,
                  "degenerate_gt_edges": pair_gt - geometry_gt,
                  "both_confident_matched_pairs": sum(r["both_gt_known"] and r["both_pred_confident"] for r in instances),
                  "qualified_pairs": pair_ok, "qualified_pair_rate_all_evaluable_gt": ratio(pair_ok, pair_gt),
                  "reversed_order_matched": sum(r["reversed_order"] for r in instances)},
        "geometry": {"evaluable_gt_edges": geometry_gt,
                     "raw_matched": {key: distribution(r[key] for r in instances) for key in
                                     ("center_error_px", "angle_error_deg", "length_error_px", "relative_length_error")},
                     "qualified_localizations": localization_ok,
                     "qualified_localization_rate_all_evaluable_gt": ratio(localization_ok, geometry_gt)},
    }


def analyze_basket(gt, pred, g, p, iou, width, height, image_name):
    known = gt.points[:, 2] > 0
    errors, available, points = [], [], []
    for k, name in enumerate(CORNER_NAMES):
        point = pred.points[k] if pred is not None else None
        finite = point is not None and np.isfinite(point).all()
        error = float(np.linalg.norm(point[:2] - gt.points[k, :2])) if known[k] and finite else None
        good = bool(finite and point[2] >= POINT_CONF and 0 <= point[0] <= width and 0 <= point[1] <= height)
        errors.append(error)
        available.append(good)
        if known[k]:
            points.append(dict(image=image_name, gt_index=g, class_id=1, class_name="basket", keypoint=name,
                               gt_x=float(gt.points[k, 0]), gt_y=float(gt.points[k, 1]),
                               pred_x=number(point[0]) if point is not None else None,
                               pred_y=number(point[1]) if point is not None else None,
                               point_confidence=number(point[2]) if point is not None else None,
                               pixel_error=error, available=good))
    complete = bool(known.all())
    qualified = bool(complete and all(available) and all(e is not None and e <= CORNER_ERROR_PX for e in errors))
    row = dict(image=image_name, gt_index=g, pred_index=p, class_id=1, class_name="basket",
               matched_iou=float(iou) if iou is not None else None,
               detection_confidence=pred.confidence if pred is not None else None,
               all_corners_known=complete, all_corners_confident=all(available), corners_qualified=qualified,
               max_corner_error_px=max(errors) if all(e is not None for e in errors) else None,
               reason="missed" if pred is None else "partial_gt" if not complete else
                      "unavailable_corner" if not all(available) else "qualified" if qualified else "corner_error")
    return row, points


def detection_metrics(instances, predictions):
    matched = sum(r["pred_index"] is not None for r in instances)
    return dict(gt_instances=len(instances), pred_instances=len(predictions), matched_instances=matched,
                missed_instances=len(instances)-matched, false_positives_including_duplicates=len(predictions)-matched,
                duplicate_candidates=sum(r["status"] == "duplicate_candidate" for r in predictions),
                precision=ratio(matched, len(predictions)), recall=ratio(matched, len(instances)))


def summarize(instances, points, predictions):
    def select(rows, class_id):
        return [r for r in rows if r.get("class_id", 0) == class_id]
    white = summarize_white(select(instances, 0), select(points, 0), select(predictions, 0))
    baskets, corners = select(instances, 1), select(points, 1)
    def corner_metrics(rows):
        errors = [r["pixel_error"] for r in rows if r["pixel_error"] is not None]
        return dict(evaluable_gt_points=len(rows), matched_points=len(errors),
                    available_points=sum(r["available"] for r in rows), pixel_error_raw_matched=distribution(errors),
                    pck_pixels={str(t): dict(raw_matched=ratio(sum(e <= t for e in errors), len(errors)),
                        raw_all_gt=ratio(sum(e <= t for e in errors), len(rows)),
                        available_all_gt=ratio(sum(r["available"] and r["pixel_error"] is not None and
                                                  r["pixel_error"] <= t for r in rows), len(rows)))
                                for t in PIXEL_THRESHOLDS})
    complete = sum(r["all_corners_known"] for r in baskets)
    qualified = sum(r["corners_qualified"] for r in baskets)
    basket = dict(detection=detection_metrics(baskets, select(predictions, 1)),
                  corners={"all": corner_metrics(corners), **{name: corner_metrics(
                      [r for r in corners if r["keypoint"] == name]) for name in CORNER_NAMES}},
                  quadrilaterals=dict(evaluable_gt_baskets=complete, partial_or_unknown_gt=len(baskets)-complete,
                      all_confident_matched=sum(r["all_corners_known"] and r["all_corners_confident"] for r in baskets),
                      qualified_baskets=qualified, qualified_rate_all_evaluable_gt=ratio(qualified, complete)))
    return {**white, "detection": detection_metrics(instances, predictions),
            "by_class": {"white_box": white, "basket": basket}}


def draw_comparison(image, truth, predictions, instances, prediction_rows):
    """左图GT，右图预测；右图灰线是GT，蓝线连接GT中点与预测中点。"""
    left, right = image.copy(), image.copy()

    def draw_edge(canvas, points, visible, color):
        valid = [bool(v and np.isfinite(p[:2]).all()) for v, p in zip(visible, points)]
        edges = [(0, 1)] if len(points) == 2 else [(0, 1), (1, 2), (2, 3), (3, 0)]
        for a, b in edges:
            if valid[a] and valid[b]:
                cv2.line(canvas, tuple(np.rint(points[a, :2]).astype(int)),
                         tuple(np.rint(points[b, :2]).astype(int)), color, 1, cv2.LINE_AA)
        for k, (point, good) in enumerate(zip(points, valid)):
            if good:
                xy = tuple(np.rint(point[:2]).astype(int))
                cv2.circle(canvas, xy, 3 if k == 0 else 2, color, -1, cv2.LINE_AA)
                if len(points) == 4:
                    cv2.putText(canvas, str(k), (xy[0]+4, xy[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)

    for g, gt in enumerate(truth):
        gt = Instance(gt.box, gt.points[:2] if gt.class_id == 0 else gt.points, gt.confidence, gt.class_id)
        draw_edge(left, gt.points, gt.points[:, 2] > 0, (255, 255, 0))
        anchor = tuple(np.rint(gt.box[:2]).astype(int))
        cv2.putText(left, f"{CLASS_NAMES[gt.class_id]} G{g}", anchor, cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 0), 1)
        draw_edge(right, gt.points, gt.points[:, 2] > 0, (180, 180, 180))
        if instances[g]["pred_index"] is None:
            cv2.rectangle(right, anchor, tuple(np.rint(gt.box[2:]).astype(int)), (0, 0, 255), 1)
            cv2.putText(right, f"MISS G{g}", anchor, cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
    for p, (pred, record) in enumerate(zip(predictions, prediction_rows)):
        pred = Instance(pred.box, pred.points[:2] if pred.class_id == 0 else pred.points, pred.confidence, pred.class_id)
        g = record["matched_gt_index"]
        color = (0, 220, 0) if g is not None and instances[g].get("localization_qualified", instances[g].get("corners_qualified", False)) else (0, 200, 255)
        if g is None:
            color = (255, 0, 255)
        draw_edge(right, pred.points, [True] * len(pred.points), color)
        for point in pred.points:
            if np.isfinite(point).all() and point[2] < POINT_CONF:
                cv2.drawMarker(right, tuple(np.rint(point[:2]).astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
        if g is not None:
            gt = truth[g]
            if gt.class_id == 0 and instances[g]["geometry_evaluable"] and np.isfinite(pred.points[:, :2]).all():
                a = tuple(np.rint(gt.points[:2, :2].mean(0)).astype(int))
                b = tuple(np.rint(pred.points[:, :2].mean(0)).astype(int))
                cv2.line(right, a, b, (255, 100, 0), 1, cv2.LINE_AA)
        tag = f"P{p}->G{g}" if g is not None else f"P{p} {record['status']}"
        cv2.putText(right, CLASS_NAMES[pred.class_id] + " " + tag, tuple(np.rint(pred.box[:2]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
    canvas = np.concatenate([left, right], axis=1)
    canvas = cv2.copyMakeBorder(canvas, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(canvas, "LEFT: GT cyan | RIGHT: green=qualified, yellow=other, magenta=FP, red=miss/low confidence",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return canvas


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    if not WEIGHTS.is_file() or not DATA.is_file():
        raise FileNotFoundError("请在文件顶部配置有效的 WEIGHTS 和 DATA")
    if (SPLIT not in ("val", "test") or IMGSZ <= 0 or IMGSZ % 32 or BATCH < 1 or WORKERS < 0
            or not 0 <= AP_CONF <= CONF <= 1 or not 0 <= POINT_CONF <= 1
            or not 0 < MATCH_IOU <= 1 or not 0 < NMS_IOU <= 1 or MAX_DET < 1
            or min(*PIXEL_THRESHOLDS, PAIR_ERROR_PX, CENTER_ERROR_PX, ANGLE_ERROR_DEG, CORNER_ERROR_PX) <= 0):
        raise ValueError("顶部的评估参数范围无效")
    config, images = load_dataset()
    device = (0 if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    names, kpt_shape = dataset_layout(config)
    model = YOLO(str(WEIGHTS.resolve()))
    if model.task != "pose" or list(model.model.model[-1].kpt_shape) != kpt_shape or model.names != names:
        raise ValueError("权重类别名称和关键点布局必须与数据配置一致")
    destination = increment_path(PROJECT / NAME, mkdir=True)
    inference = dict(imgsz=IMGSZ, device=device, iou=NMS_IOU, max_det=MAX_DET,
                     quantize=32, rect=True, augment=False, nms=True)
    standard_metrics = None
    standard_by_class = None
    if RUN_STANDARD_MAP:
        metrics = model.val(data=str(DATA.resolve()), split=SPLIT, batch=BATCH, workers=WORKERS,
                            conf=AP_CONF, project=str(destination), name="standard", exist_ok=False,
                            plots=SAVE_STANDARD_PLOTS, **inference)
        standard_metrics = {key: number(float(value)) for key, value in metrics.results_dict.items()}
        standard_by_class = {names[int(class_id)]: dict(zip(metrics.keys,
                             [number(float(v)) for v in metrics.class_result(i)]))
                             for i, class_id in enumerate(metrics.ap_class_index)}
    all_instances, all_points, all_predictions, image_rows = [], [], [], []
    if SAVE_VISUALIZATIONS:
        (destination / "visualizations").mkdir()
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError("无法读取图片：" + str(path))
        height, width = image.shape[:2]
        truth = read_truth(path, width, height, kpt_shape)
        result = model.predict(source=image, conf=CONF, verbose=False, **inference)[0]
        predictions = []
        if len(result.boxes):
            if result.keypoints is None or tuple(result.keypoints.data.shape[1:]) != tuple(kpt_shape):
                raise ValueError("模型预测关键点布局与数据配置不一致")
            predictions = [Instance(box, points, float(conf), int(cls)) for box, points, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy(),
                result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        truth = semantic_instances(truth, names)
        predictions = semantic_instances(predictions, names)
        instances, points, prediction_rows = analyze_image(truth, predictions, width, height, str(path))
        all_instances.extend(instances)
        all_points.extend(points)
        all_predictions.extend(prediction_rows)
        per_image = summarize(instances, points, prediction_rows)
        image_rows.append(dict(image=str(path), width=width, height=height, **per_image["detection"],
                               qualified_pairs=per_image["pairs"]["qualified_pairs"],
                               qualified_localizations=per_image["geometry"]["qualified_localizations"],
                               qualified_baskets=per_image["by_class"]["basket"]["quadrilaterals"]["qualified_baskets"]))
        if SAVE_VISUALIZATIONS:
            comparison = draw_comparison(image, truth, predictions, instances, prediction_rows)
            if not cv2.imwrite(str(destination / "visualizations" / f"{index:04d}_{path.stem}.jpg"), comparison):
                raise OSError("匹配可视化保存失败")
        print(f"[{index}/{len(images)}] {path.name}: matched={per_image['detection']['matched_instances']}/{len(truth)}, "
              f"qualified_pairs={per_image['pairs']['qualified_pairs']}")
    report = dict(
        schema_version=3,
        report_class_names=CLASS_NAMES,
        weights=str(WEIGHTS.resolve()),
        data=str(DATA.resolve()), split=SPLIT, images=len(images), class_names=config["names"],
        ultralytics_version=ultralytics.__version__, coordinate_system="original image pixels",
        settings=dict(**inference, confidence=CONF, point_confidence=POINT_CONF, match_iou=MATCH_IOU,
                      ap_confidence=AP_CONF, pixel_thresholds=PIXEL_THRESHOLDS, pair_error_px=PAIR_ERROR_PX,
                      center_error_px=CENTER_ERROR_PX, angle_error_deg=ANGLE_ERROR_DEG, corner_error_px=CORNER_ERROR_PX),
        standard_yolo_metrics=standard_metrics, standard_yolo_metrics_by_class=standard_by_class, oks_sigmas=config.get("kpt_oks_sigmas") or [1 / kpt_shape[0]] * kpt_shape[0],
        policy={"class_ids": "CSV class IDs use white_box=0/basket=1; class_names records model-local IDs, report_class_names records CSV IDs.",
                "matching": "Class-aware global assignment: maximize match count, then total bbox IoU; endpoint errors are not used.",
                "raw_errors": "All matched, finite predictions, including low-confidence points. Endpoint IDs are never reordered.",
                "pck": "Pixel thresholds; all-GT denominators include misses. Available PCK also requires point confidence and in-image coordinates.",
                "pairs": "Both GT endpoints have visibility > 0; misses count as failures. Degenerate GT edges are reported and fail pair qualification.",
                "geometry": "Both GT endpoints known with nonzero length; raw errors include low-confidence matches. Angle is modulo 180 degrees.",
                "localization": "Both endpoints qualified AND center/angle tolerances met. Diagnostic 2D criterion, not robot grasp approval.",
                "basket": "Fixed corner IDs; only visibility > 0 points evaluated. Four-corner rate requires all four GT known, all predictions available and each error within corner_error_px; misses fail.",
                "legacy_sections": "Top-level endpoints/pairs/geometry describe white_box only; detection covers both classes. by_class separates all metrics.",
                "duplicates": "Unmatched predictions overlapping a same-class GT at MATCH_IOU are duplicate candidates, a subset of false positives.",
                "map": "Standard framework AP uses its own matching/OKS; its counts may differ from supplementary global matching. Not a hard pixel tolerance.",
                "split": "Random frame split; adjacent frames may share train/val/test. Use an independent capture for final acceptance."},
        **summarize(all_instances, all_points, all_predictions))
    (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_csv(destination / "per_image.csv", image_rows, list(image_rows[0]))
    save_csv(destination / "per_instance.csv", all_instances, INSTANCE_FIELDS)
    save_csv(destination / "per_keypoint.csv", all_points, POINT_FIELDS)
    save_csv(destination / "per_prediction.csv", all_predictions, PREDICTION_FIELDS)
    print("评估报告：", destination / "metrics.json")
    print("检测：", report["detection"])
    print("白盒两点：", report["pairs"])
    print("铁篮四角：", report["by_class"]["basket"]["quadrilaterals"])
    return destination


if __name__ == "__main__":
    main()
