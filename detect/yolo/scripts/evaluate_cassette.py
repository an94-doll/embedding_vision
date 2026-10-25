#!/usr/bin/env python3
"""Summarize a finished run and measure native-image pixel errors on its validation split."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--review-images", action="store_true", help="Create temporary overlays for manual inspection; remove them after review.")
    args = parser.parse_args()
    job = project_path(args.job)
    config = json.loads((job / "train_config.json").read_text())
    state = json.loads((job / "status.json").read_text())
    if state["status"] != "completed":
        raise RuntimeError("Training has not completed successfully.")
    run = Path(state["save_dir"])
    dataset = Path(config["snapshot_dataset"])
    configure_local_cache()
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    with (run / "results.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    best_row = max(rows, key=lambda r: r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(P)"])
    pose_row = max(rows, key=lambda r: r["metrics/mAP50-95(P)"])
    weights = run / "weights/best.pt"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    # Stripped checkpoints have epoch=-1. Check the selected epoch's stored metrics instead.
    for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)"):
        if abs(checkpoint["train_metrics"][key] - best_row[key]) > 2e-5:
            raise RuntimeError("Checkpoint metrics do not identify the expected best epoch.")
    del checkpoint
    samples = [s for s in json.loads((dataset / "samples.json").read_text()) if s["split"] == "val"]
    model = YOLO(str(weights)).to("cuda:0")
    objects, per_image, failures = [], [], []
    total_gt = total_predictions = 0
    geometry_counts = {"predictions": 0, "clockwise_convex": 0, "edge_12_is_long": 0, "both": 0}
    threshold_totals = {t: {"predictions": 0, "matched": 0, "all_four_within_5px": 0} for t in (0.25, 0.5, 0.75, 0.85, 0.9)}

    def ious(a, b):
        low = np.maximum(a[:, None, :2], b[None, :, :2])
        high = np.minimum(a[:, None, 2:], b[None, :, 2:])
        intersection = np.maximum(high - low, 0).prod(axis=-1)
        areas_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=-1)
        areas_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=-1)
        return intersection / np.maximum(areas_a[:, None] + areas_b[None, :] - intersection, 1e-12)

    def describe(values):
        values = np.asarray(values, dtype=float)
        return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)), "max": float(values.max())} if values.size else {"count": 0}

    def match_pairs(overlap):
        candidates = [(float(overlap[g, p]), int(g), int(p)) for g, p in np.argwhere(overlap >= 0.5)]
        used_gt, used_pred, matches = set(), set(), []
        for iou, g, p in sorted(candidates, reverse=True):
            if g not in used_gt and p not in used_pred:
                used_gt.add(g)
                used_pred.add(p)
                matches.append((g, p, iou))
        return used_gt, used_pred, matches

    for sample in samples:
        image_path = dataset / sample["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != sample["sha256"]:
            raise RuntimeError(f"Validation image changed: {image_path}")
        prediction = model.predict(str(image_path), imgsz=1280, device=0, quantize=32, conf=0.25, iou=0.7, rect=False, max_det=300, save=False, save_txt=False, save_crop=False, show=False, verbose=False)[0]
        torch.cuda.synchronize(0)
        boxes = prediction.boxes.xyxy.cpu().numpy()
        confidence = prediction.boxes.conf.cpu().numpy()
        points = prediction.keypoints.data[:, :, :2].cpu().numpy()
        gt = np.array([obj["points"] for obj in sample["objects"]], dtype=float)
        gt_boxes = np.concatenate((gt.min(axis=1), gt.max(axis=1)), axis=1)
        overlap = ious(gt_boxes, boxes)
        used_gt, used_pred, matches = match_pairs(overlap)
        for threshold, totals in threshold_totals.items():
            selected = np.flatnonzero(confidence >= threshold)
            _, _, selected_matches = match_pairs(overlap[:, selected])
            totals["predictions"] += len(selected)
            totals["matched"] += len(selected_matches)
            totals["all_four_within_5px"] += sum(bool((np.linalg.norm(points[selected[p]] - gt[g], axis=1) <= 5).all()) for g, p, _ in selected_matches)
        total_gt += len(gt)
        total_predictions += len(boxes)
        edges = np.roll(points, -1, axis=1) - points
        nxt = np.roll(edges, -1, axis=1)
        cross = edges[:, :, 0] * nxt[:, :, 1] - edges[:, :, 1] * nxt[:, :, 0]
        clockwise = (cross > 0).all(axis=1)
        lengths = np.linalg.norm(edges, axis=-1)
        long_edge = lengths[:, 0] > np.maximum(lengths[:, 1], lengths[:, 3])
        geometry_counts["predictions"] += len(points)
        geometry_counts["clockwise_convex"] += int(clockwise.sum())
        geometry_counts["edge_12_is_long"] += int(long_edge.sum())
        geometry_counts["both"] += int((clockwise & long_edge).sum())
        image_records = []
        for g, p, iou in matches:
            error = np.linalg.norm(points[p] - gt[g], axis=1)
            gt_edge, pred_edge = gt[g, 1] - gt[g, 0], points[p, 1] - points[p, 0]
            delta = np.degrees(np.arctan2(pred_edge[1], pred_edge[0]) - np.arctan2(gt_edge[1], gt_edge[0]))
            angle_error = abs((delta + 90) % 180 - 90)
            kp_box = np.concatenate((points[p].min(axis=0), points[p].max(axis=0)))[None, :]
            rec = {"sample_id": sample["sample_id"], "image": str(image_path), "kind": sample["kind"], "object_id": sample["objects"][g]["object_id"], "confidence": float(confidence[p]), "bbox_iou": iou, "keypoint_enclosure_iou": float(ious(gt_boxes[g:g+1], kp_box)[0, 0]), "corner_errors_px": error.tolist(), "mean_error_px": float(error.mean()), "max_error_px": float(error.max()), "angle_error_deg_mod_180": float(angle_error), "mean_error_relative_to_long_edge": float(error.mean() / np.linalg.norm(gt_edge)), "gt_points": gt[g].tolist(), "pred_points": points[p].tolist()}
            objects.append(rec)
            image_records.append(rec)
        missed = [{"sample_id": sample["sample_id"], "object_id": sample["objects"][g]["object_id"], "best_bbox_iou": float(overlap[g].max()) if len(boxes) else 0.0} for g in range(len(gt)) if g not in used_gt]
        false_positives = [{"sample_id": sample["sample_id"], "prediction_index": p, "confidence": float(confidence[p]), "best_bbox_iou": float(overlap[:, p].max()), "bbox_xyxy": boxes[p].tolist(), "keypoints_xy": points[p].tolist()} for p in range(len(boxes)) if p not in used_pred]
        failures.extend([{"type": "unmatched_ground_truth", **item} for item in missed])
        failures.extend([{"type": "unmatched_prediction", **item} for item in false_positives])
        errors = [e for obj in image_records for e in obj["corner_errors_px"]]
        per_image.append({"sample_id": sample["sample_id"], "kind": sample["kind"], "gt": len(gt), "predictions": len(boxes), "matched": len(matches), "false_positives": len(boxes) - len(matches), "missed_or_iou_below_threshold": len(gt) - len(matches), "corner_error_px": describe(errors)})
    if not objects:
        raise RuntimeError("No matched objects; cannot summarize conditional corner errors.")
    error_array = np.array([obj["corner_errors_px"] for obj in objects])
    maximum_errors = error_array.max(axis=1)
    matched = len(objects)
    thresholds = [1, 2, 3, 5, 10]
    for totals in threshold_totals.values():
        totals.update({"false_positives": totals["predictions"] - totals["matched"], "missed_or_iou_below_threshold": total_gt - totals["matched"], "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0, "recall": totals["matched"] / total_gt, "all_four_within_5px_fraction_of_gt": totals["all_four_within_5px"] / total_gt})
    report = {
        "evaluated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run_dir": str(run), "weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "training": {"epochs_completed": len(rows), "wall_seconds": state["elapsed_seconds"], "best_checkpoint_epoch": int(best_row["epoch"]), "best_checkpoint_selection": "box mAP50-95 + pose mAP50-95", "peak_pose_only_epoch": int(pose_row["epoch"]), "peak_pose_map50_95": pose_row["metrics/mAP50-95(P)"], "last_epoch_metrics": {k: v for k, v in rows[-1].items() if k.startswith("metrics/")}, "first_epoch_losses": {k: v for k, v in rows[0].items() if "loss" in k}, "last_epoch_losses": {k: v for k, v in rows[-1].items() if "loss" in k}},
        "official_best_checkpoint_validation": {"source": str(job / "status.json"), "images": len(samples), "objects": total_gt, "metrics": state["metrics"], "custom_oks_sigma": [0.25] * 4, "precision_recall_note": "Ultralytics F1-selected confidence operating point; not the fixed 0.25 threshold below."},
        "fixed_threshold_evaluation": {"conf": 0.25, "bbox_iou_match_threshold": 0.5, "matching": "descending IoU greedy one-to-one, same class", "imgsz": 1280, "rect": False, "quantize": 32, "device": "cuda:0", "images": len(samples), "ground_truth_objects": total_gt, "predictions": total_predictions, "matched": matched, "unmatched_ground_truth": total_gt - matched, "unmatched_predictions": total_predictions - matched, "precision": matched / total_predictions, "recall": matched / total_gt, "f1": 2 * matched / (total_predictions + total_gt), "pixel_error_definition": "Euclidean distance in native image pixels, strict corner indices 1..4, no permutation/alignment, all four points; conditional on matched boxes. Crop translation does not change pixel errors.", "corner_error_px": describe(error_array.ravel()), "corner_error_by_index_px": [describe(error_array[:, i]) for i in range(4)], "corner_fraction_within_px": {str(t): float((error_array <= t).mean()) for t in thresholds}, "objects_all_four_within_px": {str(t): {"count": int((maximum_errors <= t).sum()), "fraction_of_matched": float((maximum_errors <= t).mean()), "fraction_of_all_ground_truth": float((maximum_errors <= t).sum() / total_gt)} for t in thresholds}, "long_edge_angle_error_degrees_mod_180": describe([obj["angle_error_deg_mod_180"] for obj in objects]), "mean_corner_error_relative_to_long_edge": describe([obj["mean_error_relative_to_long_edge"] for obj in objects]), "bbox_iou": describe([obj["bbox_iou"] for obj in objects]), "keypoint_enclosure_iou": describe([obj["keypoint_enclosure_iou"] for obj in objects]), "geometry": geometry_counts, "by_image_kind": {kind: {"matched_objects": sum(obj["kind"] == kind for obj in objects), "corner_error_px": describe([e for obj in objects if obj["kind"] == kind for e in obj["corner_errors_px"]])} for kind in ("full_frame", "crop")}},
        "per_image": per_image, "unmatched_cases": failures,
        "confidence_threshold_comparison_same_predictions": threshold_totals,
        "worst_objects_by_mean_corner_error": sorted(objects, key=lambda obj: obj["mean_error_px"], reverse=True)[:8],
        "limitations": ["Only 16 validation images from one held-out video, 10 source frames of the same physical arrangement as training; no independent test set.", "The validation split was also used to select best.pt; these are validation results, not unbiased deployment estimates.", "Labels are approximate around blur/rounded corners; pixel error is measured against those annotations, not physical ground truth.", "Custom four-keypoint OKS sigma defaults to 0.25; high pose mAP does not guarantee all corners within a chosen pixel tolerance.", "Unmatched GTs are excluded from conditional pixel error; all-GT object success fractions include them as failures."],
    }
    write_report(job / "evaluation_summary.json", report)
    test_dir = job / "test"
    test_dir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    epochs = [r["epoch"] for r in rows]
    for key, label in [("metrics/mAP50-95(P)", "Pose mAP50-95"), ("metrics/mAP50-95(B)", "Box mAP50-95")]:
        axs[0].plot(epochs, [100 * r[key] for r in rows], label=label)
    axs[0].axvline(best_row["epoch"], color="gray", linestyle="--", alpha=0.7)
    axs[0].set(xlabel="Epoch", ylabel="Validation AP (%)", title=f"Best checkpoint: epoch {int(best_row['epoch'])}")
    axs[0].legend()
    axs[1].plot(epochs, [r["train/pose_loss"] for r in rows], label="Train pose loss")
    axs[1].plot(epochs, [r["val/pose_loss"] for r in rows], label="Validation pose loss")
    axs[1].set(xlabel="Epoch", ylabel="Pose loss", yscale="log", title="Pose loss convergence")
    axs[1].legend()
    flat = np.sort(error_array.ravel())
    axs[2].plot(flat, np.arange(1, len(flat) + 1) / len(flat) * 100)
    axs[2].axvline(3, color="gray", linestyle="--", alpha=0.7)
    axs[2].set(xlabel="Corner error (native pixels)", ylabel="Matched corners within error (%)", title=f"{matched}/{total_gt} objects matched at conf 0.25")
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(test_dir / "evaluation_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"report": str(job / "evaluation_summary.json"), "fixed_threshold_evaluation": report["fixed_threshold_evaluation"], "threshold_comparison": threshold_totals}, ensure_ascii=False, indent=2))
    if not args.review_images:
        return
    # Temporary evidence used for visual inspection; delete after inspection.
    worst = report["worst_objects_by_mean_corner_error"][:3]
    fig, axs = plt.subplots(1, len(worst), figsize=(13, 5), constrained_layout=True)
    for ax, obj in zip(axs, worst):
        ax.imshow(Image.open(obj["image"]))
        all_xy = np.array(obj["gt_points"] + obj["pred_points"])
        for key, color, label in [("gt_points", "lime", "Label"), ("pred_points", "red", "Prediction")]:
            p = np.array(obj[key])
            closed = np.concatenate((p, p[:1]))
            ax.plot(closed[:, 0], closed[:, 1], "o-", color=color, label=label, linewidth=1.3, markersize=4)
            for i, (x, y) in enumerate(p):
                ax.text(x, y - 2, str(i + 1), color=color, fontsize=9)
        low, high = all_xy.min(axis=0), all_xy.max(axis=0)
        ax.set_xlim(low[0] - 15, high[0] + 15)
        ax.set_ylim(high[1] + 15, low[1] - 15)
        ax.set_title(f"{obj['sample_id']} / {obj['object_id']}\nmean {obj['mean_error_px']:.2f}px; max {obj['max_error_px']:.2f}px", fontsize=9)
        ax.legend(fontsize=8)
    fig.savefig(test_dir / "temporary_worst_cases.png", dpi=160)
    plt.close(fig)
    full = next(s for s in samples if s["kind"] == "full_frame")
    fig, ax = plt.subplots(figsize=(6, 10), constrained_layout=True)
    ax.imshow(Image.open(dataset / full["image"]))
    for failure in failures:
        if failure["sample_id"] == full["sample_id"] and failure["type"] == "unmatched_prediction":
            xy = np.array(failure["keypoints_xy"])
            closed = np.concatenate((xy, xy[:1]))
            ax.plot(closed[:, 0], closed[:, 1], color="red", linewidth=1)
            ax.text(xy[0, 0], xy[0, 1], f"{failure['confidence']:.2f}", color="red", fontsize=8)
    ax.set_title("Full frame: unmatched predictions at conf 0.25")
    fig.savefig(test_dir / "temporary_full_frame_false_positives.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
