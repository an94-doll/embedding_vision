#!/usr/bin/env python3
"""Check fixed conservative deduplication on cached validation predictions only."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate import (ANGLE_ERROR_DEG, CENTER_ERROR_PX, MATCH_IOU, PAIR_ERROR_PX,
                      POINT_CONF, Instance, analyze_image, box_ious, match_instances)
from scripts.postprocess import DedupConfig, deduplicate_detections

DIRECTORY = Path(__file__).resolve().parent
CACHE = DIRECTORY / "validation_predictions.json"
REPORT = DIRECTORY / "dedup_validation_report.json"
WARMUP_ROUNDS = 5
MEASURED_ROUNDS = 20


def instance(data):
    return Instance(np.asarray(data["box"], dtype=float), np.asarray(data["points"], dtype=float),
                    data["confidence"], data["class_id"])


def arrays(predictions):
    return (np.asarray([p.box for p in predictions], dtype=float).reshape(-1, 4),
            np.asarray([p.points for p in predictions], dtype=float).reshape(-1, 2, 3),
            np.asarray([p.confidence for p in predictions], dtype=float),
            np.asarray([p.class_id for p in predictions], dtype=int))


def summarize(rows):
    gt = sum(r["gt_count"] for r in rows)
    before = sum(r["before_matched"] for r in rows)
    after = sum(r["after_matched"] for r in rows)
    return dict(images=len(rows), gt=gt,
                predictions_before=sum(r["predictions_before"] for r in rows),
                predictions_after=sum(r["predictions_after"] for r in rows),
                removed=sum(r["removed_count"] for r in rows),
                removed_previously_matched=sum(r["removed_previously_matched"] for r in rows),
                ambiguous_predictions=sum(len(r["ambiguous_indices"]) for r in rows),
                before_matched=before, after_matched=after,
                before_gt_recall=before / gt if gt else None,
                after_gt_recall=after / gt if gt else None,
                before_pair_qualified=sum(r["before_pair_qualified"] for r in rows),
                after_pair_qualified=sum(r["after_pair_qualified"] for r in rows),
                before_localization_qualified=sum(r["before_localization_qualified"] for r in rows),
                after_localization_qualified=sum(r["after_localization_qualified"] for r in rows),
                lost_matched_gt=sum(len(r["lost_matched_gt"]) for r in rows),
                gained_matched_gt=sum(len(r["gained_matched_gt"]) for r in rows),
                lost_pair_qualified_gt=sum(len(r["lost_pair_qualified_gt"]) for r in rows),
                gained_pair_qualified_gt=sum(len(r["gained_pair_qualified_gt"]) for r in rows),
                lost_localization_qualified_gt=sum(len(r["lost_localization_qualified_gt"]) for r in rows),
                gained_localization_qualified_gt=sum(len(r["gained_localization_qualified_gt"]) for r in rows))


def main():
    source = ROOT / "scripts/postprocess.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = json.loads(CACHE.read_text())
    config = DedupConfig()
    rows, removed_details, workloads = [], [], []
    for data in cache["images"]:
        predictions = [instance(p) for p in data["predictions"]]
        truth = [instance(g) for g in data["truth"]]
        boxes, points, scores, classes = arrays(predictions)
        shape = (data["height"], data["width"])
        workloads.append((boxes, points, scores, shape, config, classes))
        result = deduplicate_detections(*workloads[-1])
        keep = list(result.keep_indices)
        assert set(keep).isdisjoint(s.removed_index for s in result.suppressed)
        assert sorted(keep + [s.removed_index for s in result.suppressed]) == list(range(len(predictions)))
        retained = [predictions[p] for p in keep]
        before, _ = match_instances(truth, predictions, MATCH_IOU)
        after_local, _ = match_instances(truth, retained, MATCH_IOU)
        after = {g: keep[p] for g, p in after_local.items()}
        before_rows, _, _ = analyze_image(truth, predictions, *shape[::-1], data["image"])
        after_rows, _, _ = analyze_image(truth, retained, *shape[::-1], data["image"])
        before_reverse = {p: g for g, p in before.items()}
        after_reverse = {p: g for g, p in after.items()}
        ious = box_ious(truth, retained)
        for suppression in result.suppressed:
            removed_gt = before_reverse.get(suppression.removed_index)
            covered = []
            if removed_gt is not None:
                covered = [keep[p] for p, pred in enumerate(retained)
                           if ious[removed_gt, p] >= MATCH_IOU and pred.class_id == truth[removed_gt].class_id]
            removed_details.append(dict(image=data["image"], **asdict(suppression),
                                        removed_before_gt_index=removed_gt,
                                        kept_before_gt_index=before_reverse.get(suppression.kept_index),
                                        kept_after_gt_index=after_reverse.get(suppression.kept_index),
                                        removed_gt_covered_by_kept_indices=covered,
                                        removed_gt_after_prediction_index=after.get(removed_gt) if removed_gt is not None else None,
                                        removed_gt_before_metrics=before_rows[removed_gt] if removed_gt is not None else None,
                                        removed_gt_after_metrics=after_rows[removed_gt] if removed_gt is not None else None))
        row = dict(image=data["image"], gt_count=len(truth), predictions_before=len(predictions),
                   predictions_after=len(retained), removed_count=len(result.suppressed),
                   removed_previously_matched=sum(s.removed_index in before_reverse for s in result.suppressed),
                   ambiguous_indices=list(result.ambiguous_indices),
                   before_matched=len(before), after_matched=len(after),
                   lost_matched_gt=sorted(set(before) - set(after)),
                   gained_matched_gt=sorted(set(after) - set(before)))
        for metric in ("pair_qualified", "localization_qualified"):
            before_good = {r["gt_index"] for r in before_rows if r[metric]}
            after_good = {r["gt_index"] for r in after_rows if r[metric]}
            row.update({f"before_{metric}": len(before_good), f"after_{metric}": len(after_good),
                        f"lost_{metric}_gt": sorted(before_good - after_good),
                        f"gained_{metric}_gt": sorted(after_good - before_good)})
        rows.append(row)
    # Warm and measure only NumPy postprocessing, excluding matching, file reads and model work.
    for _ in range(WARMUP_ROUNDS):
        for workload in workloads:
            deduplicate_detections(*workload)
    timings = []
    per_frame = [[] for _ in workloads]
    for _ in range(MEASURED_ROUNDS):
        for index, workload in enumerate(workloads):
            start = perf_counter_ns()
            deduplicate_detections(*workload)
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            timings.append(elapsed_ms)
            per_frame[index].append(elapsed_ms)
    if source_hash != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Postprocess source changed during validation; rerun on one fixed source revision")
    summary = summarize(rows)
    report = dict(
        metadata=dict(cache=str(CACHE.relative_to(ROOT)), cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                      weights_sha256=cache["metadata"]["cache_key"]["weights_sha256"],
                      postprocess_sha256=source_hash, config=asdict(config),
                      split="val", thresholds=dict(match_iou=MATCH_IOU, point_confidence=POINT_CONF,
                                                    pair_error_px=PAIR_ERROR_PX, center_error_px=CENTER_ERROR_PX,
                                                    angle_error_deg=ANGLE_ERROR_DEG),
                      matching="evaluate.match_instances: maximum valid match cardinality, then total box IoU; independent global assignment before/after",
                      correctness_scope="Cached 95-image validation set only; no test threshold tuning and no real-camera acceptance claim"),
        summary=summary,
        groups={group: summarize([r for r in rows if Path(r["image"]).name.startswith(group)])
                for group in ("old_", "new_", "frame_")},
        suppression_details=removed_details,
        frames=rows,
        timing=dict(definition="perf_counter_ns around deduplicate_detections only on cached original-coordinate arrays; no image/JSON read, matching or model inference",
                    python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
                    warmup_rounds=WARMUP_ROUNDS, measured_rounds=MEASURED_ROUNDS, sample_count=len(timings),
                    distinct_frames=len(workloads), max_detections=max(len(w[2]) for w in workloads),
                    median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                    mean_ms=float(np.mean(timings)), maximum_ms=float(np.max(timings)),
                    frame_timing=[dict(image=row["image"], detections=len(workload[2]), samples=len(values),
                                       median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
                                  for row, workload, values in zip(rows, workloads, per_frame)]))
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(REPORT), summary=summary,
                          timing={k: report["timing"][k] for k in ("sample_count", "max_detections", "median_ms", "p95_ms", "maximum_ms")}),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
