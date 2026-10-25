#!/usr/bin/env python3
"""Inspect a finished data_405 trial, export predictions and four-frame checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def match_reviewed_boxes(iou):
    """Exact one-to-one matching for the five reference boxes in each reviewed frame.

    Maximize IoU>=0.5 match count first, then summed IoU, without extra packages.
    """
    assert iou.shape[0] <= 12, "This exact small-set matcher is for the reviewed frames"
    states = {0: (0.0, [])}
    for p in range(iou.shape[1]):
        updated = dict(states)
        for mask, (score, pairs) in states.items():
            for g in range(iou.shape[0]):
                if mask & (1 << g) or iou[g, p] < 0.5:
                    continue
                new_mask = mask | (1 << g)
                new_score = score + float(iou[g, p])
                if new_mask not in updated or new_score > updated[new_mask][0]:
                    updated[new_mask] = (new_score, pairs + [(g, p)])
        states = updated
    return max(states.values(), key=lambda value: (len(value[1]), value[0]))[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    args = parser.parse_args()
    job = args.job.resolve()
    config = json.loads((job / "train_config.json").read_text())
    state = json.loads((job / "status.json").read_text())
    assert state["status"] == "completed", "Training has not completed"
    os.environ["YOLO_CONFIG_DIR"] = str(job / "cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(job / "cache/matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert torch.cuda.is_available()
    run = Path(state["save_dir"])
    dataset = Path(config["snapshot_dataset"])
    out = job / ("evaluation" if args.checkpoint == "best" else "evaluation_last")
    out.mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader((run / "results.csv").open())]
    assert len(history) == state["epochs_completed"]
    model = YOLO(str(run / "weights" / f"{args.checkpoint}.pt"))
    assert list(model.model.model[-1].kpt_shape) == [2, 3] and model.names == {0: "cassette_front"}
    best_row = history[state["best_epoch"] - 1]
    selected_epoch = state["best_epoch"] if args.checkpoint == "best" else state["epochs_completed"]
    selected_row = history[selected_epoch - 1]
    offset = int(config.get("completed_epochs_before_this_run", 0))
    for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)"):
        assert abs(float(model.ckpt["train_metrics"][key]) - selected_row[key]) < 1e-5, "Epoch/checkpoint disagreement"
    selected_metrics = state["metrics"]
    if args.checkpoint == "last":
        metrics = model.val(data=str(dataset / "data.yaml"), imgsz=1280, batch=8, device=0, workers=4,
                            rect=True, half=False, quantize=32, plots=False, save_json=False, verbose=False,
                            project=str(out), name="validation", exist_ok=True)
        selected_metrics = {str(k): float(v) for k, v in metrics.results_dict.items() if k != "fitness"}
    report = {"status": "completed", "run_dir": str(run), "job_dir": str(job),
              "epochs_completed": len(history), "best_epoch": state["best_epoch"],
              "completed_epochs_before_this_run": offset, "cumulative_epochs_completed": offset + len(history),
              "selected_checkpoint": args.checkpoint, "selected_epoch": selected_epoch,
              "selected_cumulative_epoch": offset + selected_epoch,
              "selected_checkpoint_validation_metrics": selected_metrics,
              "selected_epoch_csv_metrics": {k: v for k, v in selected_row.items() if k.startswith("metrics/")},
              "elapsed_seconds": state["elapsed_seconds"], "gpu": state["gpu"],
              "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
              "kpt_shape": [2, 3], "names": model.names, "default_oks_sigmas": [0.5, 0.5],
              "best_epoch_csv_metrics": {k: v for k, v in best_row.items() if k.startswith("metrics/")},
              "final_best_validation_metrics": state["metrics"], "dataset": json.loads((job / "dataset_audit.json").read_text()),
              "inference_settings": {"imgsz": 1280, "rect": True, "conf": 0.25, "endpoint_display_conf": 0.5, "quantize": 32},
              "limitations": ["Most labels are automatic candidates with known omissions and endpoint errors.",
                              "Validation is a later block of the same video, not an independent test set.",
                              "Pose mAP uses default OKS sigmas [0.5, 0.5], not a strict pixel-distance tolerance.",
                              "Four manually reviewed frames are consecutive validation images used in checkpoint selection, not an independent test."],
              "predictions": []}
    for tag in ("best", "last"):
        weights = run / "weights" / f"{tag}.pt"
        report[f"{tag}_weights"] = str(weights)
        report[f"{tag}_weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 23)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    manual_frames = {766, 767, 768, 769}
    distances, visible_confident_distances = [], []
    manual_totals = dict(images=0, ground_truth_objects=0, predicted_objects=0, matched_objects_iou50=0,
                         ground_truth_visible_endpoints=0, matched_visible_endpoints=0,
                         confident_endpoints_within_5px=0, confident_endpoints_within_10px=0)
    selection = [1, 200, 500, 700, 721, 745, 766, 767, 768, 769, 800, 850, 903]
    thumbs = []

    def draw_annotations(im, boxes, points, color, predicted=False, scores=None):
        d = ImageDraw.Draw(im)
        for n, (box, kp) in enumerate(zip(boxes, points), 1):
            visible = kp[:, 2] >= (0.5 if predicted else 2)
            d.rectangle(tuple(box), outline=color if visible.all() else "#ffca28", width=1)
            if visible.all():
                d.line([tuple(kp[0, :2]), tuple(kp[1, :2])], fill=color, width=2)
            for index, (x, y, _) in enumerate(kp):
                if visible[index]:
                    d.ellipse((x-3, y-3, x+3, y+3), fill=("#4bff87" if index == 0 else "#69b9ff"), outline="black")
            text = str(n) + (f" {scores[n-1]:.2f}" if predicted else "")
            d.text((max(0, box[0]), max(0, box[1]-13)), text, font=small, fill="white", stroke_width=1, stroke_fill="black")

    for frame in selection:
        split = "train" if frame <= 700 else "val"
        stem = f"color_{frame:06d}"
        src = dataset / "images" / split / f"{stem}.png"
        labels = np.loadtxt(dataset / "labels" / split / f"{stem}.txt", ndmin=2)
        gt_boxes = np.concatenate([labels[:, 1:3]-labels[:, 3:5]/2, labels[:, 1:3]+labels[:, 3:5]/2], 1) * [1280, 720, 1280, 720]
        gt_points = labels[:, 5:].reshape(-1, 2, 3) * [1280, 720, 1]
        result = model.predict(str(src), imgsz=1280, rect=True, device=0, quantize=32, half=False,
                               conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)[0]
        assert result.orig_shape == (720, 1280) and result.keypoints.data.shape[1:] == (2, 3)
        assert result.boxes.data.device.type == "cuda"
        boxes, points, scores = (x.cpu().numpy() for x in (result.boxes.xyxy, result.keypoints.data, result.boxes.conf))
        assert all(np.isfinite(x).all() for x in (boxes, points, scores))
        both_confident = (points[:, :, 2] >= 0.5).all(1)
        row = {"frame": frame, "split": split, "reviewed": frame in manual_frames, "reference_objects": len(labels),
               "predicted_left_right_order_violations": int((both_confident & (points[:, 0, 0] >= points[:, 1, 0])).sum()),
               "predicted_objects": len(boxes), "detections": [{"bbox_xyxy": b.tolist(), "keypoints_xy_confidence": p.tolist(), "confidence": float(s)} for b, p, s in zip(boxes, points, scores)]}
        if frame in manual_frames:
            pairs = []
            if len(boxes):
                low = np.maximum(gt_boxes[:, None, :2], boxes[None, :, :2])
                high = np.minimum(gt_boxes[:, None, 2:], boxes[None, :, 2:])
                intersection = np.maximum(high-low, 0).prod(2)
                union = (gt_boxes[:, 2:]-gt_boxes[:, :2]).prod(1)[:, None] + (boxes[:, 2:]-boxes[:, :2]).prod(1)[None, :] - intersection
                iou = intersection / np.maximum(union, 1e-8)
                pairs = match_reviewed_boxes(iou)
            manual_totals["images"] += 1
            manual_totals["ground_truth_objects"] += len(labels)
            manual_totals["predicted_objects"] += len(boxes)
            manual_totals["matched_objects_iou50"] += len(pairs)
            manual_totals["ground_truth_visible_endpoints"] += int((gt_points[:, :, 2] == 2).sum())
            row["matched_pairs"] = []
            for g, p in pairs:
                visible = gt_points[g, :, 2] == 2
                errors = np.linalg.norm(gt_points[g, visible, :2]-points[p, visible, :2], axis=1)
                confident = points[p, visible, 2] >= 0.5
                distances.extend(errors.tolist())
                visible_confident_distances.extend(errors[confident].tolist())
                manual_totals["matched_visible_endpoints"] += len(errors)
                for tolerance in (5, 10):
                    manual_totals[f"confident_endpoints_within_{tolerance}px"] += int(((errors <= tolerance) & confident).sum())
                row["matched_pairs"].append({"reference_index": g, "prediction_index": p, "iou": float(iou[g, p]), "visible_endpoint_errors_px": errors.tolist(), "predicted_endpoint_confident": confident.tolist()})
        original = Image.open(src).convert("RGB")
        left, right = original.copy(), original.copy()
        draw_annotations(left, gt_boxes, gt_points, "#38d1ed")
        draw_annotations(right, boxes, points, "#ff6b5f", predicted=True, scores=scores)
        canvas = Image.new("RGB", (2560, 780), "#12202e")
        canvas.paste(left, (0, 60)); canvas.paste(right, (1280, 60))
        draw = ImageDraw.Draw(canvas)
        label = "人工精修标注" if frame in manual_frames else "现有候选标注（未精修）"
        draw.text((16, 14), f"{stem} · {split} · {label}：{len(labels)} 个", font=font, fill="white")
        draw.text((1296, 14), f"{args.checkpoint} · 累计第 {offset + selected_epoch} 轮 · 预测 {len(boxes)} 个 · 阈值 0.25/0.5", font=font, fill="white")
        path = out / "predictions" / f"{stem}_comparison.jpg"
        canvas.save(path, quality=95)
        row["comparison"] = str(path)
        report["predictions"].append(row)
        canvas.thumbnail((1280, 390))
        thumbs.append(canvas)
    manual_totals["detection_recall_iou50_conf025"] = manual_totals["matched_objects_iou50"] / manual_totals["ground_truth_objects"]
    manual_totals["detection_precision_iou50_conf025"] = manual_totals["matched_objects_iou50"] / max(1, manual_totals["predicted_objects"])
    manual_totals["missed_objects_iou50"] = manual_totals["ground_truth_objects"] - manual_totals["matched_objects_iou50"]
    manual_totals["unmatched_predictions_iou50"] = manual_totals["predicted_objects"] - manual_totals["matched_objects_iou50"]
    for tolerance in (5, 10):
        manual_totals[f"all_visible_endpoints_within_{tolerance}px_fraction"] = manual_totals[f"confident_endpoints_within_{tolerance}px"] / manual_totals["ground_truth_visible_endpoints"]
    manual_totals["matched_visible_endpoint_error_mean_px"] = float(np.mean(distances)) if distances else None
    manual_totals["matched_visible_endpoint_error_median_px"] = float(np.median(distances)) if distances else None
    manual_totals["confident_visible_endpoint_error_median_px"] = float(np.median(visible_confident_distances)) if visible_confident_distances else None
    report["four_reviewed_validation_frames"] = manual_totals
    # A small montage remains readable; full-size comparisons are linked below.
    selected_thumbs = [thumbs[selection.index(i)] for i in (1, 700, 721, 766, 850, 903)]
    sheet = Image.new("RGB", (1280, 390*len(selected_thumbs)), "#12202e")
    for i, thumb in enumerate(selected_thumbs):
        sheet.paste(thumb, (0, 390*i))
    sheet.save(out / "comparison_overview.jpg", quality=94)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, names, title in zip(axes.flat, [("train/box_loss", "val/box_loss"), ("train/pose_loss", "val/pose_loss"), ("metrics/mAP50(B)", "metrics/mAP50-95(B)"), ("metrics/mAP50(P)", "metrics/mAP50-95(P)")], ["Box loss", "Pose loss", "Box mAP (candidate labels)", "Pose mAP (candidate labels; sigma=0.5)"]):
        for key in names:
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        ax.axvline(selected_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)
    source = Path(config["source_dataset"])
    guards = json.loads((job / "source_preservation_hashes.json").read_text())
    report["source_guard_files_unchanged"] = {rel: hashlib.sha256((source / rel).read_bytes()).hexdigest() == digest for rel, digest in guards.items()}
    summary_name = "training_summary.json" if args.checkpoint == "best" else "last_checkpoint_summary.json"
    (job / summary_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    gallery = "\n".join(f'<figure><a href="predictions/{Path(r["comparison"]).name}"><img loading="lazy" src="predictions/{Path(r["comparison"]).name}"></a><figcaption>第 {r["frame"]} 张 · {r["split"]}</figcaption></figure>' for r in report["predictions"])
    (out / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两端点 YOLO26 试训结果</title><style>body{margin:24px;background:#101923;color:#e9eef5;font:16px system-ui}img{width:100%;height:auto}figure{margin:24px 0}a{color:#6ed9fc}p{line-height:1.8}.note{color:#ffc56e}</style><h1>包埋盒两端点 · ' + args.checkpoint + f' · 累计第 {offset + selected_epoch} 轮</h1><p class="note">左侧为现有标注，右侧为模型预测。大部分标注尚未精修，自动评分不代表实际端点准确率。点击图片可放大。</p><p>绿点：左端；蓝点：右端；黄框：标注缺失端点，或预测端点置信度不足。此页仅查看预测，不会修改标注。</p><img src="training_curves.png" alt="本阶段训练曲线">' + gallery + '</html>')
    print(json.dumps({"epochs_completed": len(history), "cumulative_epochs_completed": offset + len(history), "selected_checkpoint": args.checkpoint, "selected_cumulative_epoch": offset + selected_epoch, "metrics": selected_metrics, "manual": manual_totals, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
