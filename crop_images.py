#!/usr/bin/env python3
"""将横向图片旋转为竖向（720×1080 方向）后居中裁剪，不缩放。

同步转换已有的五点 YOLO Pose 标签。

默认运行：python3 crop_images.py
依赖：Pillow。输出标签只针对已有标注，不将未标注图片当作负样本。
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_labels(text, width, height, left, top, size, rotate=False):
    result = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 20:
            raise ValueError(f"标签第 {line_number} 行应有 20 列（5 个关键点）")
        values = list(map(float, fields))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"标签第 {line_number} 行存在非有限数值")
        cls, cx, cy, bw, bh = values[:5]
        if cls < 0 or not cls.is_integer() or bw <= 0 or bh <= 0:
            raise ValueError(f"标签第 {line_number} 行类别或检测框无效")
        if not all(0 <= v <= 1 for v in values[1:5]):
            raise ValueError(f"标签第 {line_number} 行检测框必须归一化")
        if rotate:
            cx, cy, bw, bh = 1 - cy, cx, bh, bw
        x1 = max(0.0, (cx - bw / 2) * width - left)
        y1 = max(0.0, (cy - bh / 2) * height - top)
        x2 = min(float(size), (cx + bw / 2) * width - left)
        y2 = min(float(size), (cy + bh / 2) * height - top)
        if x2 <= x1 or y2 <= y1:
            continue
        row = [str(int(cls))]
        row.extend(f"{v:.6f}" for v in (
            (x1 + x2) / (2 * size), (y1 + y2) / (2 * size),
            (x2 - x1) / size, (y2 - y1) / size,
        ))
        for index in range(5):
            x, y, visible = values[5 + 3 * index:8 + 3 * index]
            if visible not in (0, 1, 2) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"标签第 {line_number} 行关键点无效")
            if rotate:
                x, y = 1 - y, x
            x, y = x * width - left, y * height - top
            if visible == 0 or not (0 <= x < size and 0 <= y < size):
                row.extend(["0", "0", "0"])
            else:
                row.extend([f"{x / size:.6f}", f"{y / size:.6f}", str(int(visible))])
        result.append(" ".join(row))
    return "\n".join(result) + ("\n" if result else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "slice/10")
    parser.add_argument("--imgsz", type=int, default=720)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=ROOT / "labels/02")
    parser.add_argument("--output-labels", type=Path, default=None)
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("imgsz 必须大于 0")
    args.output = args.output or ROOT / f"images/10"
    args.output_labels = args.output_labels or ROOT / f"labels/11"
    if not args.input.is_dir():
        parser.error(f"图片目录不存在：{args.input}")
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        parser.error("未找到图片")
    if len({p.stem for p in files}) != len(files):
        parser.error("图片存在同名不同扩展名，标签文件名会冲突")
    if args.input.resolve() == args.output.resolve():
        parser.error("输出目录不能与原图目录相同")

    # 先检查整个批次和标签，避免尺寸或标签错误导致只处理部分图片。
    plans = []
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
        original_size = (width, height)
        rotate = width > height
        if rotate:
            width, height = height, width
        if min(width, height) < args.imgsz:
            parser.error(f"{path.name} 尺寸 {width}×{height} 小于裁剪尺寸，未执行缩放")
        left, top = (width - args.imgsz) // 2, (height - args.imgsz) // 2
        label = args.labels / f"{path.stem}.txt"
        if not label.exists():
            label = path.with_suffix(".txt")
        label_text = None
        if label.exists():
            try:
                label_text = crop_labels(
                    label.read_text(), width, height, left, top, args.imgsz, rotate
                )
            except ValueError as error:
                parser.error(f"{label}: {error}")
        image_out = args.output / path.name
        label_out = args.output_labels / f"{path.stem}.txt"
        if image_out.exists() or label_out.exists():
            parser.error(f"输出已存在：{path.stem}，请指定新的输出目录")
        plans.append((path, image_out, label_out, label, label_text,
                      (left, top, left + args.imgsz, top + args.imgsz), original_size, rotate))

    args.output.mkdir(parents=True, exist_ok=True)
    args.output_labels.mkdir(parents=True, exist_ok=True)
    report = []
    for path, image_out, label_out, label, label_text, box, original_size, rotate in plans:
        with Image.open(path) as image:
            if rotate:
                image = image.transpose(getattr(Image, "Transpose", Image).ROTATE_270)
            cropped = image.crop(box)
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                cropped.convert("RGB").save(image_out, quality=95, subsampling=0)
            else:
                cropped.save(image_out)
        if label_text is not None:
            label_out.write_text(label_text)
        report.append({
            "image": path.name, "original_size": original_size, "crop_box": box,
            "rotation_clockwise_degrees": 90 if rotate else 0,
            "output_size": [args.imgsz, args.imgsz],
            "source_label": str(label) if label_text is not None else None,
            "output_label": str(label_out) if label_text is not None else None,
        })
    (args.output / "crop_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    labelled = sum(item[4] is not None for item in plans)
    print(f"完成：{len(files)} 张图片，尺寸 {args.imgsz}×{args.imgsz}，未缩放。")
    print(f"图片：{args.output}")
    print(f"标签：{args.output_labels}；已转换 {labelled} 个，缺少原标签 {len(files) - labelled} 个。")


if __name__ == "__main__":
    main()
