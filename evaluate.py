#!/usr/bin/env python3
"""Evaluate detection, five-point localization and front/back recognition."""

import argparse
import math
import statistics
from pathlib import Path

from pose_common import (ROOT, DEFAULT_WEIGHTS, NAMES, check_imgsz, detections_from_result,
                         label_path, load_config, preflight, read_label, save_csv,
                         save_json, select_device, validate_model)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_instances(truth, predictions, threshold):
    """Class-independent, descending-IoU greedy one-to-one matching."""
    candidates = []
    for g, gt in enumerate(truth):
        for p, pred in enumerate(predictions):
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= threshold:
                candidates.append((-iou, g, p))
    used_gt, used_pred, matches = set(), set(), {}
    for negative_iou, g, p in sorted(candidates):
        if g not in used_gt and p not in used_pred:
            matches[g] = (p, -negative_iou)
            used_gt.add(g)
            used_pred.add(p)
    return matches, used_pred


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def point_summary(rows):
    matched = [r for r in rows if r["pixel_error"] is not None]
    values = [r["pixel_error"] for r in matched]
    correct = sum(r["normalized_error"] <= 0.05 for r in matched)
    return {
        "gt_points": len(rows), "matched_points": len(matched),
        "mean_pixel_error_matched": statistics.mean(values) if values else None,
        "median_pixel_error_matched": statistics.median(values) if values else None,
        "rmse_pixels_matched": math.sqrt(sum(v * v for v in values) / len(values)) if values else None,
        "pck_005_matched": ratio(correct, len(matched)),
        "pck_005_all_gt": ratio(correct, len(rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="补充识别/点位指标的目标置信度下限")
    parser.add_argument("--ap-conf", type=float, default=0.001, help="标准 mAP 评估的候选框阈值")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/evaluate")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    try:
        check_imgsz(args.imgsz)
        if not (0 <= args.ap_conf <= args.conf <= 1 and 0 < args.match_iou <= 1 and 0 < args.nms_iou <= 1):
            raise ValueError("置信度或 IoU 阈值无效")
        if args.batch < 1 or args.workers < 0:
            raise ValueError("batch 必须为正数，workers 不可为负")
        if not args.weights.is_file():
            raise ValueError("权重不存在：" + str(args.weights))
        import cv2
        from ultralytics import YOLO

        config = load_config(args.data)
        images = preflight(config, [args.split])[args.split]
        device = select_device(args.device)
        model = YOLO(str(args.weights.resolve()))
        validate_model(model)
        validation_run = {}

        def capture_validation_directory(validator):
            validation_run["save_dir"] = Path(validator.save_dir)

        model.add_callback("on_val_end", capture_validation_directory)
        print("评估集合={}，图片={}，设备={}".format(args.split, len(images), device))
        metrics = model.val(
            data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
            batch=args.batch, device=device, workers=args.workers, conf=args.ap_conf,
            iou=args.nms_iou, project=str(args.project.resolve()),
            name=args.name or (args.split + "_front_back"), exist_ok=False,
            plots=True, augment=False,
        )
        if "save_dir" not in validation_run:
            raise ValueError("评估完成后未获取到输出目录")
        destination = validation_run["save_dir"]
        confusion = {name: {"front": 0, "back": 0, "missed": 0} for name in NAMES.values()}
        unmatched_predictions = {name: 0 for name in NAMES.values()}
        image_rows, instance_rows, point_rows = [], [], []
        gt_total = pred_total = matched_total = face_correct = 0
        for index, image in enumerate(images, 1):
            canvas = cv2.imread(str(image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if canvas is None:
                raise ValueError("无法读取图片：" + str(image))
            height, width = canvas.shape[:2]
            truth = read_label(label_path(image), width, height)
            result = model.predict(source=canvas, imgsz=args.imgsz, device=device,
                                   conf=args.conf, iou=args.nms_iou, augment=False, verbose=False)[0]
            predictions = detections_from_result(result)
            matches, used_pred = match_instances(truth, predictions, args.match_iou)
            gt_total += len(truth)
            pred_total += len(predictions)
            matched_total += len(matches)
            image_correct = 0
            image_points = []
            for g, gt in enumerate(truth):
                matched = matches.get(g)
                pred = predictions[matched[0]] if matched else None
                true_name = NAMES[gt["class_id"]]
                predicted_name = pred["class_name"] if pred else "missed"
                confusion[true_name][predicted_name] += 1
                correct = pred is not None and pred["class_id"] == gt["class_id"]
                face_correct += int(correct)
                image_correct += int(correct)
                instance_rows.append({"image": str(image), "gt_index": g,
                                      "true_class": true_name, "predicted_class": predicted_name,
                                      "matched_iou": matched[1] if matched else None,
                                      "class_confidence": pred["confidence"] if pred else None,
                                      "face_correct": correct})
                box = gt["bbox_xyxy"]
                diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
                for k, (x, y, visibility) in enumerate(gt["keypoints"]):
                    if visibility == 0:
                        continue
                    point = pred["keypoints"][k] if pred else None
                    error = math.hypot(point["x"] - x, point["y"] - y) if point else None
                    row = {"image": str(image), "gt_index": g, "true_class": true_name,
                           "keypoint": "P{}".format(k + 1), "pixel_error": error,
                           "normalized_error": error / diagonal if error is not None else None,
                           "point_confidence": point["confidence"] if point else None,
                           "face_correct": correct}
                    point_rows.append(row)
                    image_points.append(row)
            for p, pred in enumerate(predictions):
                if p not in used_pred:
                    unmatched_predictions[pred["class_name"]] += 1
            image_rows.append({"image": str(image), "width": width, "height": height,
                               "gt_instances": len(truth), "pred_instances": len(predictions),
                               "matched_instances": len(matches), "face_correct": image_correct,
                               **point_summary(image_points)})
            print("[{}/{}] {}：匹配 {}/{}，正反正确 {}".format(
                index, len(images), image.name, len(matches), len(truth), image_correct))
        standard_metrics = {}
        for key, value in metrics.results_dict.items():
            value = float(value)
            standard_metrics[key] = value if math.isfinite(value) else None
        report = {
            "weights": str(args.weights.resolve()), "data": str(args.data.resolve()),
            "split": args.split, "images": len(images), "device": device,
            "standard_yolo_metrics": standard_metrics,
            "operating_point": {"confidence": args.conf, "nms_iou": args.nms_iou,
                                "match_iou": args.match_iou, "ap_confidence": args.ap_conf},
            "matching": "Class-independent greedy one-to-one, descending bbox IoU",
            "gt_instances": gt_total, "pred_instances": pred_total,
            "matched_instances": matched_total, "missed_instances": gt_total - matched_total,
            "unmatched_predictions": unmatched_predictions,
            "detection_precision_class_independent": ratio(matched_total, pred_total),
            "detection_recall_class_independent": ratio(matched_total, gt_total),
            "front_back_accuracy_matched": ratio(face_correct, matched_total),
            "correct_face_detection_rate_all_gt": ratio(face_correct, gt_total),
            "confusion_rows_truth_columns_prediction": confusion,
            "keypoints_all": point_summary(point_rows),
            "keypoints_per_point": {"P{}".format(k): point_summary(
                [r for r in point_rows if r["keypoint"] == "P{}".format(k)]) for k in range(1, 6)},
            "keypoints_per_class": {name: point_summary(
                [r for r in point_rows if r["true_class"] == name]) for name in NAMES.values()},
            "point_policy": "All visible GT points; no filtering by predicted keypoint confidence. Wrong-face matches remain included. Pixel means exclude misses; all-GT PCK counts misses as failures.",
            "pck_normalization": "GT bbox diagonal; threshold 0.05",
        }
        save_json(destination / "joint_metrics.json", report)
        save_csv(destination / "per_image.csv", list(image_rows[0]), image_rows)
        save_csv(destination / "per_instance.csv", ["image", "gt_index", "true_class", "predicted_class",
                                                     "matched_iou", "class_confidence", "face_correct"], instance_rows)
        save_csv(destination / "per_keypoint.csv", ["image", "gt_index", "true_class", "keypoint", "pixel_error",
                                                     "normalized_error", "point_confidence", "face_correct"], point_rows)
        print("评估报告：" + str(destination / "joint_metrics.json"))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
