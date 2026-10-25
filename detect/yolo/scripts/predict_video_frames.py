#!/usr/bin/env python3
"""Run the trained four-corner model on selected video frames without changing training data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, help='JSON containing extracted frame paths and timestamps')
    parser.add_argument('--weights', default='runs/pose/cassette_top_yolo26s_20261019_195156/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()
    if not 0.25 <= args.conf <= 1:
        raise ValueError('Display confidence must be between 0.25 and 1.')
    selection_path = project_path(args.selection)
    out = selection_path.parent
    selection = json.loads(selection_path.read_text())
    weights = project_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    configure_local_cache()
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. CPU fallback is not permitted.')
    model = YOLO(str(weights)).to('cuda:0')
    if list(model.model.kpt_shape) != [4, 3] or model.task != 'pose':
        raise RuntimeError('Expected a four-keypoint pose model.')
    if next(model.model.parameters()).device.type != 'cuda':
        raise RuntimeError('Model is not on CUDA.')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts = {size: ImageFont.truetype(font_path, size) for size in (12, 14, 18, 22)}
    (out / 'predictions').mkdir(exist_ok=True)
    report = {
        'status': 'running', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'source_video': selection['source_video'], 'source_sha256': selection['source_sha256'],
        'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'torch_cuda': torch.version.cuda},
        'gpu': torch.cuda.get_device_name(0), 'device': 'cuda:0', 'model_task': model.task,
        'kpt_shape': list(model.model.kpt_shape), 'model_end2end': bool(getattr(model.model, 'end2end', False)),
        'settings': {'imgsz': 1280, 'rect': False, 'quantize': 32, 'prediction_conf': 0.25,
                     'display_conf': args.conf, 'iou_argument': 0.7, 'max_det': 300},
        'coordinate_system': 'Original decoded image pixels, x rightward, y downward; keypoints stay in original model order.',
        'ground_truth_available': False,
        'limitations': ['Model confidence is not measured correctness.',
                       'No manual ground-truth corners: no pixel error, precision, recall or mAP can be established.',
                       'Six frames were selected for viewpoint coverage and sharpness before inference; not an evaluation of every video frame.',
                       'A larger object image does not establish better accuracy under a new viewpoint or appearance.'],
        'frames': [],
    }
    for info in selection['frames']:
        source = Path(info['original_image'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != info['original_sha256']:
            raise RuntimeError(f'Frame changed: {source}')
        started = time.perf_counter()
        result = model.predict(str(source), imgsz=1280, rect=False, device=0, quantize=32,
                               conf=0.25, iou=0.7, max_det=300, save=False, save_txt=False,
                               save_crop=False, show=False, verbose=False)[0]
        torch.cuda.synchronize(0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if result.boxes.data.device.type != 'cuda' or result.keypoints.data.device.type != 'cuda':
            raise RuntimeError('Prediction outputs are not on CUDA.')
        if result.keypoints.data.shape[1:] != (4, 3) or len(result.boxes) != len(result.keypoints):
            raise RuntimeError('Invalid four-keypoint output.')
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        if not all(np.isfinite(x).all() for x in (boxes, points, scores)) or not (classes == 0).all():
            raise RuntimeError('Nonfinite predictions or an unexpected class.')
        detections = []
        order = sorted(range(len(boxes)), key=lambda i: (float(boxes[i, 1]), float(boxes[i, 0])))
        visible_number = 0
        for idx in order:
            xy = points[idx, :, :2]
            edges = np.roll(xy, -1, axis=0) - xy
            following = np.roll(edges, -1, axis=0)
            cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            lengths = np.linalg.norm(edges, axis=1)
            shown = bool(scores[idx] >= args.conf)
            if shown:
                visible_number += 1
            edge = bool((xy[:, 0] <= 2).any() or (xy[:, 0] >= info['width'] - 3).any()
                        or (xy[:, 1] <= 2).any() or (xy[:, 1] >= info['height'] - 3).any())
            detections.append({
                'display_id': visible_number if shown else None, 'displayed': shown,
                'confidence': float(scores[idx]), 'class_id': int(classes[idx]),
                'bbox_xyxy': boxes[idx].tolist(), 'keypoints_xy_confidence': points[idx].tolist(),
                'clockwise_convex': bool((cross > 0).all()),
                'edge_12_is_long': bool(lengths[0] > max(lengths[1], lengths[3])),
                'near_image_boundary': edge, 'bbox_width_px': float(boxes[idx, 2] - boxes[idx, 0]),
                'bbox_height_px': float(boxes[idx, 3] - boxes[idx, 1]),
            })
        displayed = [d for d in detections if d['displayed']]
        annotated = Image.open(source).convert('RGB')
        draw = ImageDraw.Draw(annotated)
        for d in displayed:
            xy = np.array(d['keypoints_xy_confidence'])[:, :2]
            coords = [tuple(map(float, p)) for p in xy]
            draw.line(coords + coords[:1], fill='#25e5a0', width=2)
            draw.line(coords[:2], fill='#ffb326', width=3)
            for i, (x, y) in enumerate(coords, 1):
                draw.ellipse((x-3, y-3, x+3, y+3), fill='white', outline='black', width=1)
                tx = min(max(x + 4, 1), info['width'] - 14)
                ty = min(max(y - 14, 0), info['height'] - 15)
                draw.text((tx, ty), str(i), font=fonts[12], fill='white', stroke_width=1, stroke_fill='black')
            label = f"#{d['display_id']} {d['confidence']:.2f}" + (' edge' if d['near_image_boundary'] else '')
            cx, cy = xy.mean(axis=0)
            text_box = draw.textbbox((0, 0), label, font=fonts[12])
            tw, th = text_box[2], 16
            tx = max(0, min(info['width']-tw-5, float(cx)-tw/2))
            ty = max(0, min(info['height']-th-3, float(cy)-th/2))
            draw.rectangle((tx-2, ty, tx+tw+2, ty+th), fill='#17222b')
            draw.text((tx, ty-1), label, font=fonts[12], fill='white')
        canvas = Image.new('RGB', (info['width'], info['height'] + 88), '#17222b')
        canvas.paste(annotated, (0, 88))
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 8), f"Frame {info['number']:02d} | {info['timestamp_seconds']:.2f} s | {len(displayed)} predictions", font=fonts[22], fill='white')
        draw.text((15, 38), f"YOLO26s-pose | conf >= {args.conf:.2f} | input 1280 | CUDA FP32", font=fonts[18], fill='#bdd0dd')
        draw.text((15, 63), 'Orange: edge 1-2; green: other edges; numbers: corner order', font=fonts[14], fill='#ffce79')
        prediction_path = out / 'predictions' / (info['stem'] + '_pred.png')
        canvas.save(prediction_path)
        frame_report = {**info, 'prediction_image': str(prediction_path),
                        'prediction_image_header_height_px': 88, 'output_device': str(result.boxes.data.device),
                        'displayed_count': len(displayed),
                        'counts_by_confidence': {str(c): int((scores >= c).sum()) for c in (.25, .5, .75, .85)},
                        'displayed_confidence': {'min': min((d['confidence'] for d in displayed), default=None),
                                                 'max': max((d['confidence'] for d in displayed), default=None)},
                        'displayed_near_boundary': sum(d['near_image_boundary'] for d in displayed),
                        'displayed_geometry_valid': sum(d['clockwise_convex'] and d['edge_12_is_long'] for d in displayed),
                        'predict_call_wall_ms_including_first_call_warmup': elapsed_ms,
                        'ultralytics_speed_ms': result.speed, 'detections': detections}
        report['frames'].append(frame_report)
        print(json.dumps({k: frame_report[k] for k in ('number', 'timestamp_seconds', 'displayed_count', 'counts_by_confidence', 'displayed_near_boundary', 'displayed_geometry_valid')}, ensure_ascii=False), flush=True)
    report['status'] = 'completed'
    report['total_displayed_instances_across_frames'] = sum(f['displayed_count'] for f in report['frames'])
    report['total_note'] = 'Repeated views of the same physical objects; do not interpret as unique cassette count.'
    write_report(out / 'predictions.json', report)
    columns = 3
    thumb_width = 360
    thumb_height = int((selection['frames'][0]['height'] + 88) * thumb_width / selection['frames'][0]['width'])
    rows = (len(report['frames']) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * thumb_width, rows * thumb_height), '#17222b')
    for i, frame in enumerate(report['frames']):
        picture = Image.open(frame['prediction_image']).convert('RGB')
        picture = ImageOps.contain(picture, (thumb_width, thumb_height))
        sheet.paste(picture, ((i % columns) * thumb_width, (i // columns) * thumb_height))
    sheet.save(out / 'recognition_overview.jpg', quality=95)


if __name__ == '__main__':
    main()
