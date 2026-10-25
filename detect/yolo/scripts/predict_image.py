#!/usr/bin/env python3
"""Annotate one image with the trained four-corner model, honoring EXIF orientation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--weights', default='runs/pose/cassette_top_yolo26s_20261019_195156/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()
    if not 0.25 <= args.conf <= 1:
        raise ValueError('Display confidence must be between 0.25 and 1.')
    source = project_path(args.source)
    weights = project_path(args.weights)
    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError('Source image and trained weights must exist locally.')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    out = project_path(args.output_dir or f'test/image_{source.stem[:8]}_{stamp}')
    out.mkdir(parents=True, exist_ok=False)
    configure_local_cache()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    with Image.open(source) as image:
        raw_size = image.size
        orientation = int(image.getexif().get(274, 1))
        upright = ImageOps.exif_transpose(image).convert('RGB')
    width, height = upright.size
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback.')
    model = YOLO(str(weights)).to('cuda:0')
    if model.task != 'pose' or list(model.model.kpt_shape) != [4, 3]:
        raise RuntimeError('Expected a four-keypoint pose model.')
    if next(model.model.parameters()).device.type != 'cuda':
        raise RuntimeError('Model parameters are not on CUDA.')
    # EXIF has already been applied. Predict on this same upright RGB image that we draw on.
    result = model.predict(upright, imgsz=1280, rect=False, device=0, quantize=32,
                           conf=0.25, iou=0.7, max_det=300, save=False, save_txt=False,
                           save_crop=False, show=False, verbose=False)[0]
    torch.cuda.synchronize(0)
    if result.boxes.data.device.type != 'cuda' or result.keypoints.data.device.type != 'cuda':
        raise RuntimeError('Results are not on CUDA.')
    boxes = result.boxes.xyxy.cpu().numpy()
    points = result.keypoints.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    if points.shape != (len(boxes), 4, 3) or not all(np.isfinite(x).all() for x in (boxes, points, scores)):
        raise RuntimeError('Invalid model output.')
    order = sorted(range(len(boxes)), key=lambda i: (float(boxes[i, 1]), float(boxes[i, 0])))
    detections = []
    shown = 0
    for idx in order:
        display = bool(scores[idx] >= args.conf)
        shown += int(display)
        xy = points[idx, :, :2]
        edges = np.roll(xy, -1, axis=0) - xy
        nxt = np.roll(edges, -1, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        detections.append({'display_id': shown if display else None, 'displayed': display,
                           'confidence': float(scores[idx]), 'bbox_xyxy': boxes[idx].tolist(),
                           'keypoints_xy_confidence': points[idx].tolist(),
                           'clockwise_convex': bool((edges[:, 0]*nxt[:, 1]-edges[:, 1]*nxt[:, 0] > 0).all()),
                           'edge_12_is_long': bool(lengths[0] > max(lengths[1], lengths[3]))})
    annotated = upright.copy()
    draw = ImageDraw.Draw(annotated)
    scale = max(1.0, min(width, height) / 900)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', round(15*scale))
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', round(12*scale))
    stroke, radius = max(2, round(2*scale)), max(3, round(3*scale))
    for det in detections:
        if not det['displayed']:
            continue
        xy = [tuple(p[:2]) for p in det['keypoints_xy_confidence']]
        draw.line(xy + xy[:1], fill='#25e5a0', width=stroke)
        draw.line(xy[:2], fill='#ffb326', width=stroke+1)
        for i, (x, y) in enumerate(xy, 1):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='white', outline='black', width=max(1, round(scale)))
            tx = max(0, min(width-18*scale, x+4*scale))
            ty = max(0, min(height-18*scale, y-16*scale))
            draw.text((tx, ty), str(i), font=small, fill='white', stroke_width=max(1, round(scale)), stroke_fill='black')
        label = f"#{det['display_id']} {det['confidence']:.2f}"
        cx, cy = np.mean(xy, axis=0)
        bounds = draw.textbbox((0, 0), label, font=font)
        tw, th = bounds[2], round(21*scale)
        tx, ty = max(0, min(width-tw-6, cx-tw/2)), max(0, min(height-th, cy-th/2))
        draw.rectangle((tx-3, ty, tx+tw+3, ty+th), fill='#17222b')
        draw.text((tx, ty), label, font=font, fill='white')
    annotated.save(out/'annotated.jpg', quality=97, subsampling=0)
    preview = ImageOps.contain(annotated, (1600, 1600), Image.Resampling.LANCZOS)
    preview.save(out/'preview.jpg', quality=95, subsampling=0)
    report = {
        'status': 'completed', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'source_image': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_raw_size_wh': list(raw_size), 'source_exif_orientation': orientation,
        'output_size_wh': [width, height], 'orientation_applied': True,
        'coordinate_system': 'EXIF-oriented full-resolution image pixels; x right, y down. No annotation header offset.',
        'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'device': str(result.boxes.data.device), 'gpu': torch.cuda.get_device_name(0),
        'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'settings': {'imgsz': 1280, 'rect': False, 'quantize': 32, 'prediction_conf': 0.25, 'display_conf': args.conf, 'iou': 0.7, 'max_det': 300},
        'displayed_count': shown, 'counts_by_confidence': {str(c): int((scores >= c).sum()) for c in (.25, .5, .75, .85)},
        'detections': detections, 'annotated_image': str(out/'annotated.jpg'), 'preview_image': str(out/'preview.jpg'),
        'ground_truth_available': False, 'note': 'Automatic model predictions, not manually verified training labels.'
    }
    write_report(out/'predictions.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'output_size_wh', 'displayed_count', 'counts_by_confidence', 'annotated_image', 'preview_image')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
