"""Offline A/B: python evaluate_basket_refinement.py --split val --device cpu.

Uses evaluate.py's model, dataset and matching settings. Never changes raw
predictions, labels or the live pipeline. Refused refinements count as failures.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from time import perf_counter

import evaluate as ev
from scripts.basket_refinement import Config, refine_basket

import cv2
import numpy as np


def compare(truth, predictions, refinements, width, height, image):
    matches, _ = ev.match_instances(truth, predictions)
    rows = []
    for g, gt in enumerate(truth):
        p = matches.get(g)
        pred = predictions[p] if p is not None else None
        result = refinements[p] if p is not None else None
        known = gt.points[:, 2] > 0
        raw = np.linalg.norm(pred.points[:, :2]-gt.points[:, :2], axis=1) if pred is not None else None
        refined = np.linalg.norm(result.corners-gt.points[:, :2], axis=1) if result is not None and result.valid else None
        raw_available = bool(pred is not None and np.all(pred.points[:, 2] >= ev.POINT_CONF)
                             and np.all(pred.points[:, :2] >= 0) and np.all(pred.points[:, :2] < [width, height]))
        rows.append(dict(image=image, gt_index=g, pred_index=p, complete_gt=bool(known.all()),
            reason=result.reason if result is not None else 'missed',
            raw_errors=raw[known].tolist() if raw is not None else [],
            refined_errors=refined[known].tolist() if refined is not None else [],
            raw_qualified=bool(known.all() and raw_available and np.all(raw <= ev.CORNER_ERROR_PX)),
            refined_qualified=bool(known.all() and refined is not None and np.all(refined <= ev.CORNER_ERROR_PX))))
    return rows


def summarize(rows, timings, reasons):
    accepted = [r for r in rows if r['reason'] == 'accepted']
    def errors(subset, key):
        return [e for row in subset for e in row[key]]
    paired = [r for r in accepted if r['raw_errors']]
    complete = [r for r in rows if r['complete_gt']]
    return dict(gt_baskets=len(rows), complete_gt_baskets=len(complete),
        prediction_refinement_reasons=dict(reasons),
        matched_gt_acceptance_rate=ev.ratio(len(accepted), sum(r['pred_index'] is not None for r in rows)),
        raw_error_px=ev.distribution(errors(rows, 'raw_errors')),
        accepted_raw_error_px=ev.distribution(errors(accepted, 'raw_errors')),
        accepted_refined_error_px=ev.distribution(errors(accepted, 'refined_errors')),
        worsened_accepted_instances=sum(bool(np.mean(r['refined_errors']) > np.mean(r['raw_errors'])) for r in paired),
        compared_accepted_instances=len(paired),
        raw_qualified=sum(r['raw_qualified'] for r in complete),
        refined_qualified=sum(r['refined_qualified'] for r in complete),
        raw_qualified_rate=ev.ratio(sum(r['raw_qualified'] for r in complete), len(complete)),
        refined_qualified_rate=ev.ratio(sum(r['refined_qualified'] for r in complete), len(complete)),
        refinement_ms=ev.distribution(timings))


def visualize(image, truth, predictions, results):
    panels = [image.copy() for _ in range(3)]
    def draw(canvas, points, visible, color):
        for i, point in enumerate(points):
            j = (i+1) % 4
            if visible[i]:
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(canvas, xy, 4, color, -1)
                cv2.putText(canvas, str(i), (xy[0]+5, xy[1]+15), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                if visible[j]:
                    cv2.line(canvas, xy, tuple(np.rint(points[j]).astype(int)), color, 2)
    for gt in truth:
        draw(panels[0], gt.points[:, :2], gt.points[:, 2] > 0, (255, 255, 0))
    for pred, result in zip(predictions, results):
        draw(panels[1], pred.points[:, :2], np.isfinite(pred.points[:, :2]).all(1), (0, 200, 255))
        # Supported candidate lines remain visible on rejection; never call them refined corners.
        for side, line in enumerate(result.lines):
            if line is not None:
                endpoints = pred.points[[side, (side+1) % 4], :2]
                endpoints = endpoints - (endpoints @ line[:2] + line[2])[:, None] * line[:2]
                cv2.line(panels[2], tuple(np.rint(endpoints[0]).astype(int)),
                         tuple(np.rint(endpoints[1]).astype(int)), (150, 150, 150), 2)
        if result.valid:
            draw(panels[2], result.corners, [True]*4, (0, 220, 0))
        cv2.putText(panels[2], result.reason, tuple(np.maximum(pred.box[:2], [5, 55]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
    for panel, title in zip(panels, ['GT (known points only)', 'RAW POSE', 'REFINEMENT (gray: candidate lines)']):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    return np.concatenate(panels, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path
    torch.set_num_threads(4)
    ev.SPLIT = args.split
    config, images = ev.load_dataset()
    model = YOLO(str(ev.WEIGHTS))
    if model.names != {0: 'white_box', 1: 'basket'} or list(model.model.model[-1].kpt_shape) != [4, 3]:
        raise ValueError('Requires the current white_box/basket four-point model')
    if config['kpt_shape'] != [4, 3]:
        raise ValueError('Requires four-point labels')
    destination = increment_path(ev.PROJECT / f'basket_refinement_{args.split}', mkdir=True)
    (destination/'visualizations').mkdir()
    settings = Config(min_confidence=ev.POINT_CONF)
    rows, timings, reasons, records = [], [], Counter(), []
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f'Cannot read {path}')
        h, w = image.shape[:2]
        truth = [g for g in ev.read_truth(path, w, h, [4, 3]) if g.class_id == 1]
        result = model.predict(image, imgsz=ev.IMGSZ, device=args.device, conf=ev.CONF, iou=ev.NMS_IOU,
                               max_det=ev.MAX_DET, rect=True, augment=False, quantize=32, nms=True, verbose=False)[0]
        predictions = [ev.Instance(b, k, float(c), int(cls)) for b, k, c, cls in zip(
            result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy(),
            result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()) if int(cls) == 1]
        results = []
        for pred in predictions:
            start = perf_counter(); refined = refine_basket(image, pred.points, settings)
            timings.append((perf_counter()-start)*1000); reasons[refined.reason] += 1
            results.append(refined)
            records.append(dict(image=str(path), raw_corners=pred.points.tolist(),
                refined_corners=refined.corners.tolist() if refined.valid else None,
                valid=refined.valid, reason=refined.reason, coverage=refined.coverage,
                lines=[line.tolist() if line is not None else None for line in refined.lines]))
        rows.extend(compare(truth, predictions, results, w, h, str(path)))
        if not cv2.imwrite(str(destination/'visualizations'/f'{i:04d}_{path.stem}.jpg'), visualize(image, truth, predictions, results)):
            raise OSError('Cannot save visualization')
        print(f'[{i}/{len(images)}] {path.name}', flush=True)
    report = dict(split=args.split, weights=str(ev.WEIGHTS), data=str(ev.DATA), config=asdict(settings),
        inference=dict(imgsz=ev.IMGSZ, conf=ev.CONF, iou=ev.NMS_IOU, device=args.device),
        corner_error_px=ev.CORNER_ERROR_PX, match_iou=ev.MATCH_IOU,
        policy='Same class/box matching before refinement. Unknown GT excluded; misses/rejections fail full-quad rate. No raw fallback counted as refined.',
        **summarize(rows, timings, reasons))
    for name, value in [('metrics', report), ('per_instance', rows), ('predictions', records)]:
        (destination/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2)); print(destination)


if __name__ == '__main__':
    main()
