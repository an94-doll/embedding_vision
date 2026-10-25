#!/usr/bin/env python3
"""Render both trained checkpoints on exactly the six original video frames."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import zipfile
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='test/big_video_yolo26s_20261019_212516/selection.json')
    parser.add_argument('--training-summary', default='training/cassette_partial_yolo26s_20261021_025446/training_summary.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    configure_local_cache()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont

    selection_path = project_path(args.selection)
    summary_path = project_path(args.training_summary)
    selection = json.loads(selection_path.read_text())
    summary = json.loads(summary_path.read_text())
    if len(selection['frames']) != 6:
        raise ValueError('Expected exactly the six original frames.')
    out = project_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'running', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'selection': str(selection_path), 'training_summary': str(summary_path),
        'source_video': selection['source_video'], 'source_video_sha256': selection['source_sha256'],
        'settings': {'imgsz': 1280, 'rect': False, 'quantize': 32, 'half': False,
                     'prediction_conf': 0.25, 'display_conf': 0.5, 'iou': 0.7, 'max_det': 300},
        'coordinate_system': 'Original frame pixels, x right, y down; original model keypoint order.',
        'image_header_height_px': 112,
        'legend': {'cassette': 'green polygon, orange edge 1-2; C identifiers',
                   'occupied_group': 'magenta polygon, cyan edge 1-2; G identifiers'},
        'ground_truth_available': False,
        'limitations': ['Counts are model predictions, not verified cassette counts.',
                       'Both classes are retained; no geometric or manual detection filtering.',
                       'No ground truth: no accuracy, recall or pixel-error measurements.'],
        'checkpoints': [],
    }
    write_report(out / 'predictions.json', report)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden.')
    report['versions'] = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                          'torch_cuda': torch.version.cuda}
    report['gpu'] = torch.cuda.get_device_name(0)
    fonts = {n: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', n)
             for n in (12, 14, 16, 22)}
    with (Path(summary['run_dir']) / 'results.csv').open() as stream:
        history = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    output_files = []
    sources = {f['original_image']: f['original_sha256'] for f in selection['frames']}
    for source, expected in sources.items():
        if sha256(source) != expected:
            raise RuntimeError(f'Original frame hash mismatch: {source}')

    for tag, epoch in (('best', summary['best_epoch']), ('last', summary['epochs_completed'])):
        weights = Path(summary[f'{tag}_weights'])
        expected_hash = summary[f'{tag}_weights_sha256']
        if sha256(weights) != expected_hash:
            raise RuntimeError(f'Checkpoint hash mismatch: {weights}')
        model = YOLO(str(weights)).to('cuda:0')
        if model.task != 'pose' or list(model.model.kpt_shape) != [4, 3]:
            raise RuntimeError('Expected four-keypoint pose model.')
        if model.names != {0: 'cassette', 1: 'occupied_group'}:
            raise RuntimeError(f'Unexpected classes: {model.names}')
        # Stripping checkpoints resets epoch to -1. Validate the human-numbered
        # epoch using its saved validation metrics, the CSV and recorded SHA-256.
        csv_row = next(row for row in history if int(row['epoch']) == epoch)
        for key in ('metrics/mAP50-95(B)', 'metrics/mAP50-95(P)', 'val/pose_loss'):
            if abs(float(model.ckpt['train_metrics'][key]) - csv_row[key]) > 1e-5:
                raise RuntimeError(f'Checkpoint does not match epoch {epoch}: {key}')
        checkpoint = {'epoch': epoch, 'checkpoint': tag, 'weights': str(weights),
                      'sha256': expected_hash, 'saved_at': model.ckpt.get('date'),
                      'epoch_verified_by': 'training summary, SHA-256 and checkpoint metrics matched to results.csv',
                      'names': model.names, 'kpt_shape': [4, 3], 'frames': []}
        observations = []

        def record_input(module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                tensor = inputs[0]
                if tensor.device.type != 'cuda' or tensor.dtype != torch.float32:
                    raise RuntimeError(f'Unexpected forward input: {tensor.device}, {tensor.dtype}')
                observations.append({'device': str(tensor.device), 'shape': list(tensor.shape),
                                     'dtype': str(tensor.dtype)})

        hook = model.model.register_forward_pre_hook(record_input)
        for frame in selection['frames']:
            before = len(observations)
            start = time.perf_counter()
            result = model.predict(frame['original_image'], imgsz=1280, rect=False,
                                   device=0, quantize=32, half=False, conf=0.25,
                                   iou=0.7, max_det=300, save=False, save_txt=False,
                                   save_crop=False, show=False, verbose=False)[0]
            torch.cuda.synchronize(0)
            elapsed = (time.perf_counter() - start) * 1000
            if len(observations) <= before:
                raise RuntimeError('No actual CUDA forward input was recorded.')
            if next(model.model.parameters()).device.type != 'cuda':
                raise RuntimeError('Model is not on CUDA.')
            if result.boxes.data.device.type != 'cuda' or result.keypoints.data.device.type != 'cuda':
                raise RuntimeError('Outputs are not on CUDA.')
            if result.keypoints.data.shape[1:] != (4, 3) or len(result.boxes) != len(result.keypoints):
                raise RuntimeError('Invalid four-point output.')
            boxes = result.boxes.xyxy.cpu().numpy()
            points = result.keypoints.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            if not all(np.isfinite(a).all() for a in (boxes, points, scores)) or not np.isin(classes, [0, 1]).all():
                raise RuntimeError('Invalid prediction values.')
            counters = {0: 0, 1: 0}
            detections = []
            for i in sorted(range(len(boxes)), key=lambda i: (classes[i], boxes[i, 1], boxes[i, 0])):
                cls = int(classes[i])
                displayed = bool(scores[i] >= 0.5)
                if displayed:
                    counters[cls] += 1
                detections.append({'class_id': cls, 'class_name': model.names[cls],
                                   'confidence': float(scores[i]), 'displayed': displayed,
                                   'display_id': f"{'C' if cls == 0 else 'G'}{counters[cls]}" if displayed else None,
                                   'bbox_xyxy': boxes[i].tolist(),
                                   'keypoints_xy_confidence': points[i].tolist()})
            picture = Image.open(frame['original_image']).convert('RGB')
            if picture.size != (frame['width'], frame['height']):
                raise RuntimeError('Source frame dimensions changed.')
            draw = ImageDraw.Draw(picture)
            # Draw groups first so their outlines do not obscure the single-box points.
            for det in sorted((d for d in detections if d['displayed']), key=lambda d: -d['class_id']):
                cls = det['class_id']
                color, long_color = ('#25e5a0', '#ffb326') if cls == 0 else ('#ef71ff', '#55d7ff')
                xy = [tuple(p[:2]) for p in det['keypoints_xy_confidence']]
                draw.line(xy + xy[:1], fill=color, width=2)
                draw.line(xy[:2], fill=long_color, width=3)
                for number, (x, y) in enumerate(xy, 1):
                    draw.ellipse((x-3, y-3, x+3, y+3), fill='white' if cls == 0 else color,
                                 outline='black', width=1)
                    pos = (min(max(x+4, 1), picture.width-15), min(max(y-14, 1), picture.height-16))
                    draw.text(pos, str(number), font=fonts[12], fill='white' if cls == 0 else color,
                              stroke_width=1, stroke_fill='black')
                label = f"{det['display_id']} {det['confidence']:.2f}"
                text_width = draw.textbbox((0, 0), label, font=fonts[12])[2]
                cx = sum(p[0] for p in xy) / 4
                cy = sum(p[1] for p in xy) / 4 if cls == 0 else min(p[1] for p in xy) - 18
                tx = max(0, min(picture.width-text_width-5, cx-text_width/2))
                ty = max(0, min(picture.height-18, cy-8))
                draw.rectangle((tx-2, ty, tx+text_width+2, ty+16), fill='#17222b')
                draw.text((tx, ty-1), label, font=fonts[12], fill='white' if cls == 0 else color)
            canvas = Image.new('RGB', (picture.width, picture.height + 112), '#17222b')
            canvas.paste(picture, (0, 112))
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 7), f"Epoch {epoch} ({tag}.pt) | Frame {frame['number']:02d} | {frame['timestamp_seconds']:.2f} s",
                      font=fonts[22], fill='white')
            draw.text((14, 37), f"Predictions: {counters[0]} cassette (C) + {counters[1]} occupied group (G)",
                      font=fonts[16], fill='white')
            draw.text((14, 61), 'Same original frame | conf >= 0.50 | input 1280 | CUDA FP32',
                      font=fonts[16], fill='#bdd0dd')
            draw.text((14, 85), 'C: green / orange 1-2     G: magenta / cyan 1-2     Corners: 1, 2, 3, 4',
                      font=fonts[14], fill='#bdd0dd')
            filename = f"epoch{epoch:03d}_{frame['stem']}.png"
            destination = out / filename
            canvas.save(destination)
            output_files.append(destination)
            frame_result = {'number': frame['number'], 'timestamp_seconds': frame['timestamp_seconds'],
                            'source_image': frame['original_image'], 'source_sha256': frame['original_sha256'],
                            'prediction_image': str(destination), 'prediction_image_sha256': sha256(destination),
                            'displayed_counts': {model.names[k]: v for k, v in counters.items()},
                            'counts_at_conf_025': {model.names[k]: int((classes == k).sum()) for k in (0, 1)},
                            'forward_inputs': observations[before:], 'output_device': str(result.boxes.data.device),
                            'keypoint_shape': list(result.keypoints.data.shape),
                            'predict_wall_ms_including_first_warmup': elapsed,
                            'ultralytics_speed_ms': result.speed, 'detections': detections}
            checkpoint['frames'].append(frame_result)
            print(json.dumps({'epoch': epoch, 'frame': frame['number'], 'displayed': frame_result['displayed_counts']}), flush=True)
        hook.remove()
        report['checkpoints'].append(checkpoint)
        write_report(out / 'predictions.json', report)
        del model
        torch.cuda.empty_cache()
    if len(output_files) != 12 or any(sha256(p) != h for p, h in sources.items()):
        raise RuntimeError('Output count or source preservation verification failed.')
    for item in report['checkpoints']:
        if sha256(item['weights']) != item['sha256']:
            raise RuntimeError('Checkpoint changed during inference.')
    for path in output_files:
        with Image.open(path) as img:
            img.verify()
    with zipfile.ZipFile(out / '12_detection_photos.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in output_files:
            archive.write(path, path.name)
    report['status'] = 'completed'
    report['completed_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    report['output_image_count'] = len(output_files)
    report['original_frames_and_weights_unchanged'] = True
    write_report(out / 'predictions.json', report)
    print(json.dumps({'status': 'completed', 'output': str(out), 'images': len(output_files)}), flush=True)


if __name__ == '__main__':
    main()
