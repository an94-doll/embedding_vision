"""Conservative duplicate suppression for two-endpoint detections. Never moves points."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DedupConfig:
    point_confidence: float = 0.50
    endpoint_distance_px: float = 1.5
    endpoint_distance_ratio: float = 0.015
    min_box_iou: float = 0.45
    min_box_size_ratio: float = 0.75
    max_box_center_ratio: float = 0.20
    max_angle_deg: float = 2.0
    max_length_ratio_error: float = 0.03
    border_margin_px: float = 3.0
    min_edge_length_px: float = 12.0


@dataclass(frozen=True)
class Suppression:
    removed_index: int
    kept_index: int
    endpoint_distance_px: float
    box_iou: float


@dataclass(frozen=True)
class DedupResult:
    keep_indices: tuple[int, ...]
    suppressed: tuple[Suppression, ...]
    ambiguous_indices: tuple[int, ...]


def deduplicate_detections(boxes, keypoints, scores, image_shape, config=DedupConfig(), class_ids=None):
    """Return original indices. Non-clique chains, border cases and uncertain points are kept.

    Input coordinates are original image pixels. image_shape is (height, width).
    Pairwise tests are vectorized; only small connected components use Python.
    """
    boxes = np.asarray(boxes, dtype=float)
    points = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if boxes.shape != (n, 4) or points.shape != (n, 2, 3) or scores.shape != (n,):
        raise ValueError("Expected boxes(N,4), keypoints(N,2,3), scores(N)")
    classes = np.zeros(n, dtype=int) if class_ids is None else np.asarray(class_ids)
    if classes.shape != (n,):
        raise ValueError("Expected class_ids(N)")
    if not 0 <= config.point_confidence <= 1 or not 0 < config.min_box_iou <= 1:
        raise ValueError("Invalid duplicate confidence/IoU threshold")
    if (min(config.endpoint_distance_px, config.endpoint_distance_ratio, config.min_box_size_ratio,
            config.max_box_center_ratio, config.max_angle_deg, config.max_length_ratio_error,
            config.min_edge_length_px) <= 0 or config.min_box_size_ratio > 1 or config.border_margin_px < 0):
        raise ValueError("Invalid duplicate geometry threshold")
    if n < 2:
        return DedupResult(tuple(range(n)), (), ())
    height, width = image_shape[:2]
    sizes = boxes[:, 2:] - boxes[:, :2]
    centers = (boxes[:, 2:] + boxes[:, :2]) / 2
    vectors = points[:, 1, :2] - points[:, 0, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    margin = config.border_margin_px
    eligible = (np.isfinite(boxes).all(1) & np.isfinite(points).all((1, 2)) & np.isfinite(scores)
                & (sizes > 0).all(1) & (points[:, :, 2] >= config.point_confidence).all(1)
                & (points[:, :, 0] >= 0).all(1) & (points[:, :, 0] < width).all(1)
                & (points[:, :, 1] >= 0).all(1) & (points[:, :, 1] < height).all(1)
                & (points[:, 0, 0] <= points[:, 1, 0]) & (lengths >= config.min_edge_length_px)
                & (boxes[:, 0] > margin) & (boxes[:, 1] > margin)
                & (boxes[:, 2] < width - margin) & (boxes[:, 3] < height - margin))
    candidates = np.flatnonzero(eligible)
    if len(candidates) < 2:
        return DedupResult(tuple(range(n)), (), ())

    b, p, size, center, vec, length = (a[candidates] for a in (boxes, points, sizes, centers, vectors, lengths))
    intersection = np.maximum(0, np.minimum(b[:, None, 2:], b[None, :, 2:])
                              - np.maximum(b[:, None, :2], b[None, :, :2])).prod(2)
    areas = size.prod(1)
    ious = intersection / (areas[:, None] + areas[None, :] - intersection)
    endpoint_distance = np.linalg.norm(p[:, None, :, :2] - p[None, :, :, :2], axis=3).max(2)
    min_length = np.minimum(length[:, None], length[None, :])
    tolerance = np.minimum(config.endpoint_distance_px, config.endpoint_distance_ratio * min_length)
    size_ratio = (np.minimum(size[:, None], size[None, :]) / np.maximum(size[:, None], size[None, :])).min(2)
    center_ratio = (np.abs(center[:, None] - center[None, :]) / np.minimum(size[:, None], size[None, :])).max(2)
    cross = np.abs(vec[:, None, 0] * vec[None, :, 1] - vec[:, None, 1] * vec[None, :, 0])
    dot = (vec[:, None] * vec[None, :]).sum(2)
    angle = np.degrees(np.arctan2(cross, dot))
    length_ratio_error = np.abs(length[:, None] - length[None, :]) / min_length
    compatible = ((ious >= config.min_box_iou) & (endpoint_distance <= tolerance)
                  & (size_ratio >= config.min_box_size_ratio) & (center_ratio <= config.max_box_center_ratio)
                  & (angle <= config.max_angle_deg) & (length_ratio_error <= config.max_length_ratio_error)
                  & (classes[candidates, None] == classes[None, candidates]))
    np.fill_diagonal(compatible, False)

    seen, removed, ambiguous, suppressed = set(), set(), set(), []
    for start in range(len(candidates)):
        if start in seen:
            continue
        component, pending = [], [start]
        seen.add(start)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in np.flatnonzero(compatible[current]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        if len(component) < 2:
            continue
        # A~B and B~C does not prove A~C. Preserve the entire ambiguous chain.
        if compatible[np.ix_(component, component)].sum() != len(component) * (len(component) - 1):
            ambiguous.update(int(candidates[i]) for i in component)
            continue
        winner = max(component, key=lambda i: (scores[candidates[i]], points[candidates[i], :, 2].min(), -int(candidates[i])))
        kept = int(candidates[winner])
        for i in component:
            if i != winner:
                original = int(candidates[i])
                removed.add(original)
                suppressed.append(Suppression(original, kept, float(endpoint_distance[i, winner]), float(ious[i, winner])))
    return DedupResult(tuple(i for i in range(n) if i not in removed),
                       tuple(sorted(suppressed, key=lambda r: r.removed_index)), tuple(sorted(ambiguous)))
