#!/usr/bin/env python3
"""Compare three cassette checkpoints on real close-up frames and fixed digital crops."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import zipfile
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    configure_local_cache()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont

    out = project_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    (out / 'predictions').mkdir()
    old_report = json.loads(project_path('test/big_video_yolo26s_20261019_212516/predictions.json').read_text())
    selection = json.loads(project_path('test/big_video_yolo26s_20261019_212516/selection.json').read_text())
    summary = json.loads(project_path('training/cassette_partial_yolo26s_20261021_025446/training_summary.json').read_text())
    models = [
        {'id': 'v1', 'label': 'V1', 'weights': old_report['weights'], 'sha256': old_report['weights_sha256']},
        {'id': 'v2_e21', 'label': 'V2 epoch 21', 'weights': summary['best_weights'], 'sha256': summary['best_weights_sha256']},
        {'id': 'v2_e51', 'label': 'V2 epoch 51', 'weights': summary['last_weights'], 'sha256': summary['last_weights_sha256']},
    ]
    # Chosen by inspecting original images, before this test's inference.
    # Every crop is exactly 384 x 384; inference rescales it to 1280 x 1280.
    crop_rectangles = {1: [160, 260, 544, 644], 2: [130, 380, 514, 764], 3: [50, 430, 434, 814]}
    report = {
        'status': 'running', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'question': 'Can the three checkpoints detect single cassette tops at close range?',
        'selection_reason': 'The first three original selected frames show large cassette tops at close range.',
        'source_video': selection['source_video'], 'source_video_currently_exists': Path(selection['source_video']).exists(),
        'source_frames_reused_and_sha256_verified': True,
        'settings': {'imgsz': 1280, 'rect': False, 'device': 'cuda:0', 'quantize': 32,
                     'half': False, 'prediction_conf': 0.25, 'display_conf': 0.5,
                     'iou': 0.7, 'max_det': 300, 'compared_class_id': 0},
        'conditions': {'full': 'Real close-up original 720x1280 frame.',
                       'crop': '384x384 digital crop, scaled to 1280 during inference (3.333x). Not a new camera capture.'},
        'limitations': ['No ground-truth labels or measured camera distance.',
                       'Detection counts are not correct-object counts or recall.',
                       'Digital cropping removes context and changes scale together; it does not add image detail.',
                       'Only single-cassette class 0 is drawn and compared; group class 1 is recorded separately.',
                       'The three frames are correlated views from one video, not a general deployment benchmark.'],
        'inputs': [], 'models': [], 'comparisons': [],
    }
    cases = []
    for frame in selection['frames'][:3]:
        if digest(frame['original_image']) != frame['original_sha256']:
            raise RuntimeError('Source frame hash mismatch.')
        image = Image.open(frame['original_image']).convert('RGB')
        rectangle = crop_rectangles[frame['number']]
        crop = image.crop(rectangle)
        crop_path = out / 'inputs' / f"frame{frame['number']:02d}_crop384.png"
        crop.save(crop_path)
        for condition, path in [('full', Path(frame['original_image'])), ('crop', crop_path)]:
            case = {'frame': frame['number'], 'timestamp_seconds': frame['timestamp_seconds'],
                    'condition': condition, 'path': str(path), 'sha256': digest(path),
                    'original_image': frame['original_image'], 'original_sha256': frame['original_sha256'],
                    'crop_xyxy_in_original': rectangle,
                    'inference_scale_vs_original': 1.0 if condition == 'full' else 1280/384}
            cases.append(case)
    report['inputs'] = cases
    write_report(out / 'predictions.json', report)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required, no CPU fallback.')
    report['gpu'] = torch.cuda.get_device_name(0)
    report['versions'] = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'cuda': torch.version.cuda}
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts = {size: ImageFont.truetype(fontpath, size) for size in (12, 14, 16, 20)}
    for entry in models:
        if digest(entry['weights']) != entry['sha256']:
            raise RuntimeError('Checkpoint hash mismatch.')
        model = YOLO(entry['weights']).to('cuda:0')
        if model.task != 'pose' or list(model.model.kpt_shape) != [4, 3] or 0 not in model.names:
            raise RuntimeError('Unexpected model task or keypoints.')
        observed = []

        def observe(module, inputs):
            tensor = inputs[0]
            if tensor.device.type != 'cuda' or tensor.dtype != torch.float32:
                raise RuntimeError('Expected actual CUDA FP32 model input.')
            observed.append({'device': str(tensor.device), 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)})

        handle = model.model.register_forward_pre_hook(observe)
        model_report = {**entry, 'names': model.names, 'cases': []}
        for case in cases:
            before = len(observed)
            start = time.perf_counter()
            result = model.predict(case['path'], imgsz=1280, rect=False, device=0, quantize=32,
                                   half=False, conf=0.25, iou=0.7, max_det=300,
                                   save=False, save_txt=False, save_crop=False, show=False, verbose=False)[0]
            torch.cuda.synchronize(0)
            elapsed = (time.perf_counter() - start)*1000
            if len(observed) <= before or next(model.model.parameters()).device.type != 'cuda':
                raise RuntimeError('Model forward device check failed.')
            if result.boxes.data.device.type != 'cuda' or result.keypoints.data.device.type != 'cuda':
                raise RuntimeError('Model output device check failed.')
            boxes = result.boxes.xyxy.cpu().numpy()
            points = result.keypoints.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            if points.shape != (len(boxes), 4, 3) or not all(np.isfinite(a).all() for a in (boxes, points, scores)):
                raise RuntimeError('Invalid pose result.')
            rectangle = case['crop_xyxy_in_original']
            count = 0
            detections = []
            for i in sorted(range(len(boxes)), key=lambda i: (classes[i], boxes[i, 1], boxes[i, 0])):
                shown = bool(classes[i] == 0 and scores[i] >= .5)
                count += int(shown)
                mapped_box = boxes[i].copy()
                mapped_points = points[i].copy()
                if case['condition'] == 'crop':
                    mapped_box += np.array([rectangle[0], rectangle[1], rectangle[0], rectangle[1]])
                    mapped_points[:, 0] += rectangle[0]
                    mapped_points[:, 1] += rectangle[1]
                center = (mapped_box[:2] + mapped_box[2:]) / 2
                in_roi = bool(rectangle[0] <= center[0] < rectangle[2] and rectangle[1] <= center[1] < rectangle[3])
                detections.append({'class_id': int(classes[i]), 'class_name': model.names[int(classes[i])],
                                   'confidence': float(scores[i]), 'displayed': shown,
                                   'display_id': count if shown else None,
                                   'bbox_xyxy': boxes[i].tolist(), 'keypoints_xy_confidence': points[i].tolist(),
                                   'bbox_in_original': mapped_box.tolist(),
                                   'keypoints_in_original': mapped_points.tolist(), 'box_center_in_crop_roi': in_roi})
            picture = Image.open(case['path']).convert('RGB')
            # Render digital crops larger for legibility. Predictions remain in input pixels in JSON.
            display_scale = 1.0 if case['condition'] == 'full' else 720/384
            if display_scale != 1:
                picture = picture.resize((720,720), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(picture)
            if case['condition'] == 'full':
                draw.rectangle(rectangle, outline='#ffdf64', width=3)
                draw.text((rectangle[0]+3, rectangle[1]-20), 'Digital crop ROI', font=fonts[14],
                          fill='#ffdf64', stroke_width=1, stroke_fill='black')
            for det in detections:
                if not det['displayed']:
                    continue
                xy = [(p[0]*display_scale, p[1]*display_scale) for p in det['keypoints_xy_confidence']]
                draw.line(xy+xy[:1], fill='#25e5a0', width=2)
                draw.line(xy[:2], fill='#ffb326', width=3)
                for k, (x,y) in enumerate(xy, 1):
                    draw.ellipse((x-3,y-3,x+3,y+3), fill='white', outline='black')
                    draw.text((min(max(x+4,1),picture.width-15), min(max(y-14,1),picture.height-16)),
                              str(k), font=fonts[12], fill='white', stroke_width=1, stroke_fill='black')
                label = f"C{det['display_id']} {det['confidence']:.2f}"
                width = draw.textbbox((0,0), label, font=fonts[12])[2]
                x = max(1, min(picture.width-width-3, sum(p[0] for p in xy)/4-width/2))
                y = max(1, min(picture.height-18, sum(p[1] for p in xy)/4-8))
                draw.rectangle((x-2,y,x+width+2,y+16), fill='#17222b')
                draw.text((x,y-1), label, font=fonts[12], fill='white')
            roi_count = sum(d['displayed'] and d['box_center_in_crop_roi'] for d in detections)
            canvas = Image.new('RGB', (picture.width, picture.height+96), '#17222b')
            canvas.paste(picture, (0,96)); draw = ImageDraw.Draw(canvas)
            draw.text((12,6), f"{entry['label']} | {case['timestamp_seconds']:.2f} s | {case['condition'].upper()}", font=fonts[20], fill='white')
            draw.text((12,34), f"Cassette candidates: {count} | ROI centers: {roi_count} | conf >= 0.50", font=fonts[16], fill='white')
            detail = 'Real close-up frame; yellow rectangle = digital crop ROI' if case['condition']=='full' else 'DIGITAL CROP: input scale 3.33x; not a new camera capture'
            draw.text((12,57), detail, font=fonts[16], fill='#bdd0dd')
            draw.text((12,78), 'Class 0 only | CUDA FP32 | imgsz 1280 | Counts are NOT accuracy', font=fonts[12], fill='#bdd0dd')
            output = out/'predictions'/f"frame{case['frame']:02d}_{case['condition']}_{entry['id']}.png"
            canvas.save(output)
            item = {'frame': case['frame'], 'condition': case['condition'], 'image': str(output),
                    'displayed_single_count': count, 'single_centers_in_crop_roi': roi_count,
                    'single_counts_by_confidence': {str(c): int(((classes==0)&(scores>=c)).sum()) for c in (.25,.5,.75,.85)},
                    'group_count_at_conf_05_not_drawn': int(((classes==1)&(scores>=.5)).sum()),
                    'forward_inputs': observed[before:], 'output_device': str(result.boxes.data.device),
                    'wall_ms_including_first_warmup': elapsed, 'speed_ms': result.speed, 'detections': detections}
            model_report['cases'].append(item)
            print(json.dumps({'model': entry['id'], 'frame': case['frame'], 'condition': case['condition'],
                              'single_count': count, 'in_roi': roi_count}), flush=True)
        handle.remove()
        report['models'].append(model_report)
        write_report(out/'predictions.json', report)
        del model
        torch.cuda.empty_cache()
    for case in cases:
        pictures = []
        for model in report['models']:
            record = next(c for c in model['cases'] if c['frame']==case['frame'] and c['condition']==case['condition'])
            pictures.append(Image.open(record['image']).convert('RGB'))
        sheet = Image.new('RGB',(sum(p.width for p in pictures),pictures[0].height),'#17222b')
        offset=0
        for picture in pictures:
            sheet.paste(picture,(offset,0)); offset += picture.width
        path=out/f"frame{case['frame']:02d}_{case['condition']}_comparison.jpg"
        sheet.save(path,quality=96,subsampling=0)
        report['comparisons'].append({'frame':case['frame'],'condition':case['condition'],'image':str(path),
                                       'left_to_right':['v1','v2_e21','v2_e51']})
    for case in cases:
        if digest(case['path']) != case['sha256'] or digest(case['original_image']) != case['original_sha256']:
            raise RuntimeError('Input changed during testing.')
    for model in models:
        if digest(model['weights']) != model['sha256']:
            raise RuntimeError('Checkpoint changed during testing.')
    report['status']='completed'
    report['completed_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    report['inference_runs']=sum(len(m['cases']) for m in report['models'])
    report['sources_and_weights_unchanged']=True
    write_report(out/'predictions.json',report)
    with zipfile.ZipFile(out/'closeup_comparison_images.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for path in sorted((out/'predictions').glob('*.png'))+sorted(out.glob('*_comparison.jpg')):
            archive.write(path,str(path.relative_to(out)))
    print(json.dumps({'status':'completed','output':str(out),'inference_runs':report['inference_runs']}),flush=True)


if __name__=='__main__':
    main()
