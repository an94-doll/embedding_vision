#!/usr/bin/env python3
"""Predict one image with the cumulative epoch 100, 34 and first-run epoch 7 models."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    out = args.output or source.parent / "picture/work/predictions" / f"{source.stem[:8]}_{timestamp}"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(out / "cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(out / "cache/matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    assert torch.cuda.is_available(), "CUDA is required"
    first_job = ROOT / "training/cassette_front_2pt_data405_20261024_215915"
    later_job = ROOT / "training/cassette_front_2pt_data405_total100_20261024_224433"
    first = json.loads((first_job / "training_summary.json").read_text())
    later = json.loads((later_job / "training_summary.json").read_text())
    assert first["best_epoch"] == 7 and later["selected_cumulative_epoch"] == 34
    specs = [(100, 81, "last", later), (34, 15, "best", later), (7, 7, "best", first)]
    picture = Image.open(source).convert("RGB")
    width, height = picture.size
    report = {"source": str(source), "source_sha256": digest(source), "width": width, "height": height,
              "output_dir": str(out), "kpt_shape": [2, 3], "class_name": "cassette_front",
              "settings": {"imgsz": 1280, "rect": True, "detection_conf": 0.25, "keypoint_conf": 0.5, "iou": 0.7,
                           "max_det": 300, "device": 0, "quantize": 32, "half": False},
              "ground_truth_used": False, "manual_corrections": False, "models": []}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    title = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
    subtitle = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    panels = []
    for epoch, local_epoch, tag, summary in specs:
        weights = Path(summary[f"{tag}_weights"])
        assert digest(weights) == summary[f"{tag}_weights_sha256"]
        model = YOLO(str(weights))
        assert model.task == "pose" and list(model.model.model[-1].kpt_shape) == [2, 3]
        assert model.names == {0: "cassette_front"}
        rows = list(csv.DictReader((weights.parents[1] / "results.csv").open()))
        row = rows[local_epoch-1]
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)"):
            assert abs(float(model.ckpt["train_metrics"][key])-float(row[key])) < 1e-5
        result = model.predict(str(source), imgsz=1280, rect=True, device=0, quantize=32, half=False,
                               conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)[0]
        assert result.orig_shape == (height, width)
        assert result.keypoints.data.shape[1:] == (2, 3) and result.boxes.data.device.type == "cuda"
        boxes, points, scores = (v.cpu().numpy() for v in (result.boxes.xyxy, result.keypoints.data, result.boxes.conf))
        assert all(np.isfinite(v).all() for v in (boxes, points, scores))
        visible = points[:, :, 2] >= 0.5
        # Stable display order only; retain every prediction above the same threshold.
        order = sorted(range(len(boxes)), key=lambda i: (float(boxes[i, 1]), float(boxes[i, 0])))
        overlay = picture.copy()
        draw = ImageDraw.Draw(overlay)
        detections = []
        for number, i in enumerate(order, 1):
            box, kp, vis = boxes[i], points[i], visible[i]
            draw.rectangle(tuple(float(v) for v in box), outline="#44d4e8" if vis.all() else "#ffca28", width=1)
            if vis.all():
                draw.line([tuple(kp[0, :2]), tuple(kp[1, :2])], fill="#ff4c62", width=2)
            for side, (x, y, confidence) in enumerate(kp):
                if vis[side]:
                    draw.ellipse((x-3, y-3, x+3, y+3), fill="#55ff8b" if side == 0 else "#59aaff", outline="black", width=1)
            text = f"{number}:{scores[i]:.2f}"
            text_width = draw.textlength(text, font=font)
            pos = (min(max(0, float(box[0])), width-text_width-1), min(max(0, float(box[1])-14), height-15))
            draw.text(pos, text, font=font, fill="white", stroke_width=1, stroke_fill="black")
            detections.append({"display_id": number, "confidence": float(scores[i]), "bbox_xyxy": box.tolist(),
                               "keypoints_xy_confidence": kp.tolist(), "keypoints_displayed": vis.tolist()})
        name = f"{source.stem}_epoch{epoch:03d}_annotated.png"
        overlay.save(out / name)
        model_record = {"cumulative_epoch": epoch, "phase_epoch": local_epoch, "weights": str(weights),
                        "weights_sha256": digest(weights), "objects": len(boxes), "displayed_endpoints": int(visible.sum()),
                        "low_confidence_endpoints": int((~visible).sum()), "complete_edges": int(visible.all(1).sum()),
                        "left_right_order_violations": int((visible.all(1) & (points[:, 0, 0] >= points[:, 1, 0])).sum()),
                        "annotated_image": name, "detections": detections}
        report["models"].append(model_record)
        panel = Image.new("RGB", (width, height+88), "#132231")
        panel.paste(overlay, (0, 88))
        d = ImageDraw.Draw(panel)
        d.text((16, 9), f"累计第 {epoch} 轮 · 模型预测 {len(boxes)} 个目标", font=title, fill="white")
        d.text((16, 49), "绿点：左端  蓝点：右端  红线：长边  黄框：端点置信度不足 | 阈值 0.25 / 0.5", font=subtitle, fill="#cbd8e6")
        panels.append(panel)
        del model, result
        torch.cuda.empty_cache()
    comparison = Image.new("RGB", (width*3, height+88), "#132231")
    for index, panel in enumerate(panels):
        comparison.paste(panel, (index*width, 0))
    comparison.save(out / "comparison_100_34_7.png")
    shutil.copy2(source, out / "original.png")
    assert digest(source) == report["source_sha256"] == digest(out / "original.png")
    (out / "predictions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    cards = "".join(f'<figure><figcaption>累计第 {m["cumulative_epoch"]} 轮：预测 {m["objects"]} 个目标</figcaption><a href="{m["annotated_image"]}"><img src="{m["annotated_image"]}"></a></figure>' for m in report["models"])
    (out / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO 两端点 · 100 / 34 / 7 轮对比</title><style>body{background:#111c29;color:#edf3fa;margin:24px;font:16px system-ui;line-height:1.8}a{color:#78dbfa}img{max-width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:22px}.note{color:#ffd17c}</style><h1>同一张原图 · 100 / 34 / 7 轮模型预测</h1><p>所有模型使用相同参数：1280 输入，检测阈值 0.25，端点阈值 0.5。绿点为左端、蓝点为右端、红线为长边，黄框表示某个预测端点置信度不足。文字为目标编号和检测置信度。</p><p class="note">这些是模型直接预测，未进行人工修正；预测数量不等于正确目标数量。点击图片可查看原始分辨率。</p><p><a href="comparison_100_34_7.png">打开三模型并排对比大图</a> · <a href="original.png">查看未标注原图</a></p>'+cards+'</html>')
    (out / "README.txt").write_text('同一张原始图片的三模型预测，未人工修正，也未回写训练标签。\n检测阈值=0.25，端点置信度阈值=0.5，输入尺寸=1280，FP32 CUDA。\n三张 annotated.png 保留原图分辨率；comparison_100_34_7.png 从左到右为100轮、34轮、第一次7轮。\n绿点=左端，蓝点=右端，红线=长边，黄框=某端点置信度不足。\n模型权重、哈希及所有原始预测坐标见 predictions.json。\n')
    (ROOT / "reports/data405_image_comparison_latest.txt").write_text(str(out) + "\n")
    print(json.dumps({"output_dir": str(out), "models": [{k: m[k] for k in ("cumulative_epoch", "objects", "displayed_endpoints", "low_confidence_endpoints", "annotated_image")} for m in report["models"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
