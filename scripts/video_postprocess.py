"""Stateful two-endpoint video processing, independent of cameras and drawing.

Every current detection survives temporal processing. Lost tracks only provide a
short association memory; their old coordinates are never returned as detections.
"""

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from time import perf_counter

import numpy as np

from .postprocess import DedupConfig, Suppression, deduplicate_detections


@dataclass(frozen=True)
class TemporalConfig:
    tracking_enabled: bool = True
    smoothing_enabled: bool = True
    counting_enabled: bool = True
    point_confidence: float = 0.50
    max_gap_seconds: float = 0.35
    max_center_distance_ratio: float = 0.50
    min_box_size_ratio: float = 0.50
    max_endpoint_distance_px: float = 35.0
    max_endpoint_distance_ratio: float = 0.30
    max_angle_deg: float = 25.0
    max_length_ratio_error: float = 0.35
    fallback_min_iou: float = 0.35
    fallback_max_center_ratio: float = 0.25
    ambiguity_margin: float = 0.015
    ambiguity_ratio: float = 0.80
    smooth_min_cutoff: float = 2.0
    smooth_beta: float = 0.02
    smooth_derivative_cutoff: float = 1.0
    smooth_reset_gap_seconds: float = 0.20
    count_rise_seconds: float = 0.15
    count_fall_seconds: float = 0.30


@dataclass(frozen=True)
class VideoResult:
    keep_indices: tuple[int, ...]
    points: np.ndarray
    track_ids: np.ndarray
    suppressed: tuple[Suppression, ...]
    stable_count: int
    lost_count: int
    dedup_ms: float
    temporal_ms: float


class _OneEuro:
    """Small vector One Euro filter. Coordinates are in pixels, angle in radians."""

    def __init__(self, value, timestamp):
        self.raw = value.copy()
        self.value = value.copy()
        self.derivative = np.zeros_like(value)
        self.timestamp = timestamp

    def update(self, value, timestamp, config):
        dt = timestamp - self.timestamp
        if dt <= 0:
            return self.value.copy()
        derivative = (value - self.raw) / dt
        derivative_alpha = 1.0 / (1.0 + 1.0 / (2 * pi * config.smooth_derivative_cutoff * dt))
        self.derivative += derivative_alpha * (derivative - self.derivative)
        cutoff = config.smooth_min_cutoff + config.smooth_beta * np.abs(self.derivative)
        alpha = 1.0 / (1.0 + 1.0 / (2 * pi * cutoff * dt))
        self.value += alpha * (value - self.value)
        self.raw = value.copy()
        self.timestamp = timestamp
        return self.value.copy()


@dataclass
class _Track:
    identity: int
    box: np.ndarray
    points: np.ndarray
    class_id: int
    timestamp: float
    sequence: int
    filter: _OneEuro | None = None


def _valid_points(points, image_shape, threshold):
    height, width = image_shape[:2]
    return bool(np.isfinite(points).all() and (points[:, 2] >= threshold).all()
                and (points[:, 0] >= 0).all() and (points[:, 0] < width).all()
                and (points[:, 1] >= 0).all() and (points[:, 1] < height).all()
                and np.linalg.norm(points[1, :2] - points[0, :2]) > 1e-6)


def _line_state(points):
    vector = points[1, :2] - points[0, :2]
    center = points[:, :2].mean(axis=0)
    return np.array([*center, atan2(vector[1], vector[0]), np.linalg.norm(vector)])


def _association_costs(tracks, boxes, points, classes, image_shape, config):
    """Build a small vectorized cost matrix from original, unsmoothed measurements."""
    shape = (len(tracks), len(boxes))
    if not all(shape):
        return np.full(shape, np.inf)
    old_boxes = np.array([track.box for track in tracks])
    old_points = np.array([track.points for track in tracks])
    old_classes = np.array([track.class_id for track in tracks])
    old_size = old_boxes[:, 2:] - old_boxes[:, :2]
    new_size = boxes[:, 2:] - boxes[:, :2]
    old_center = (old_boxes[:, :2] + old_boxes[:, 2:]) / 2
    new_center = (boxes[:, :2] + boxes[:, 2:]) / 2
    old_vector = old_points[:, 1, :2] - old_points[:, 0, :2]
    new_vector = points[:, 1, :2] - points[:, 0, :2]
    old_length = np.linalg.norm(old_vector, axis=1)
    new_length = np.linalg.norm(new_vector, axis=1)
    old_valid = np.array([_valid_points(p, image_shape, config.point_confidence) for p in old_points])
    new_valid = np.array([_valid_points(p, image_shape, config.point_confidence) for p in points])
    full_points = old_valid[:, None] & new_valid[None, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        min_size = np.minimum(old_size[:, None], new_size[None, :])
        size_ratio = (min_size / np.maximum(old_size[:, None], new_size[None, :])).min(2)
        center_ratio = (np.abs(old_center[:, None] - new_center[None, :]) / min_size).max(2)
        shorter = np.minimum(old_length[:, None], new_length[None, :])
        endpoint_distance = np.linalg.norm(old_points[:, None, :, :2] - points[None, :, :, :2], axis=3).max(2)
        cross = old_vector[:, None, 0] * new_vector[None, :, 1] - old_vector[:, None, 1] * new_vector[None, :, 0]
        dot = (old_vector[:, None] * new_vector[None, :]).sum(2)
        angle = np.abs(np.arctan2(cross, dot))
        length_error = np.abs(old_length[:, None] - new_length[None, :]) / shorter
        intersection = np.maximum(0, np.minimum(old_boxes[:, None, 2:], boxes[None, :, 2:])
                                  - np.maximum(old_boxes[:, None, :2], boxes[None, :, :2])).prod(2)
        iou = intersection / (old_size.prod(1)[:, None] + new_size.prod(1)[None, :] - intersection)
        line_cost = 0.55 * endpoint_distance / shorter + 0.25 * center_ratio + 0.1 * angle / pi + 0.1 * length_error
        # Missing/low confidence endpoints must not contribute zero-valued geometry.
        box_cost = 0.4 * center_ratio + 0.6 * (1 - iou)
        costs = np.where(full_points, line_cost, box_cost)

    box_valid = ((old_classes[:, None] == classes[None, :])
                 & np.isfinite(old_boxes).all(1)[:, None] & np.isfinite(boxes).all(1)[None, :]
                 & (old_size > 0).all(1)[:, None] & (new_size > 0).all(1)[None, :]
                 & (size_ratio >= config.min_box_size_ratio) & (center_ratio <= config.max_center_distance_ratio))
    line_valid = ((endpoint_distance <= np.minimum(config.max_endpoint_distance_px, shorter * config.max_endpoint_distance_ratio))
                  & (angle <= config.max_angle_deg * pi / 180) & (length_error <= config.max_length_ratio_error))
    fallback_valid = (iou >= config.fallback_min_iou) & (center_ratio <= config.fallback_max_center_ratio)
    valid = box_valid & np.where(full_points, line_valid, fallback_valid) & np.isfinite(costs)
    return np.where(valid, costs, np.inf)


def _unambiguous_best(values, config):
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return None
    order = finite[np.argsort(values[finite], kind="stable")]
    if len(order) > 1:
        first, second = values[order[:2]]
        if second - first < config.ambiguity_margin or first > second * config.ambiguity_ratio:
            return None
    return int(order[0])


class VideoPostprocessor:
    """Conservative dedup + IDs + optional smoothing + scalar count debounce.

    update() uses monotonic reception timestamps, not a nominal camera frame rate.
    Results cover current detections only. reset() never reuses an allocated ID.
    """

    def __init__(self, dedup_config=DedupConfig(), temporal_config=TemporalConfig(), dedup_enabled=True):
        self.dedup_config = dedup_config
        self.config = temporal_config
        self.dedup_enabled = bool(dedup_enabled)
        config = self.config
        positive = (config.max_gap_seconds, config.max_center_distance_ratio, config.min_box_size_ratio,
                    config.max_endpoint_distance_px, config.max_endpoint_distance_ratio, config.max_angle_deg,
                    config.max_length_ratio_error, config.fallback_max_center_ratio, config.smooth_min_cutoff,
                    config.smooth_derivative_cutoff, config.smooth_reset_gap_seconds)
        nonnegative = (config.ambiguity_margin, config.smooth_beta, config.count_rise_seconds, config.count_fall_seconds)
        if (not 0 <= config.point_confidence <= 1 or not 0 < config.fallback_min_iou <= 1
                or not 0 < config.ambiguity_ratio <= 1 or config.min_box_size_ratio > 1
                or any(not np.isfinite(v) or v <= 0 for v in positive)
                or any(not np.isfinite(v) or v < 0 for v in nonnegative)):
            raise ValueError("Invalid temporal configuration")
        self._next_id = 1
        self.reset()

    def reset(self):
        self._tracks = {}
        self._last_sequence = None
        self._last_timestamp = None
        self._image_shape = None
        self._cached = None
        self._stable_count = None
        self._pending_count = None
        self._pending_since = None

    def _count(self, current, timestamp):
        if not self.config.counting_enabled or self._stable_count is None:
            self._stable_count = current
            self._pending_count = None
        elif current == self._stable_count:
            self._pending_count = None
        else:
            if self._pending_count != current:
                self._pending_count = current
                self._pending_since = timestamp
            delay = (self.config.count_rise_seconds if current > self._stable_count
                     else self.config.count_fall_seconds)
            if timestamp - self._pending_since >= delay:
                self._stable_count = current
                self._pending_count = None
        return self._stable_count

    def _smooth(self, track, points, timestamp, sequence):
        config = self.config
        valid = _valid_points(points, self._image_shape, config.point_confidence)
        if not config.smoothing_enabled or not valid:
            track.filter = None
            return points.copy()
        state = _line_state(points)
        continuous = (track.sequence == self._last_sequence and timestamp > track.timestamp
                      and timestamp - track.timestamp <= config.smooth_reset_gap_seconds)
        if track.filter is None or not continuous:
            track.filter = _OneEuro(state, timestamp)
            return points.copy()
        # Unwrap against the previous raw angle before smoothing a line direction.
        state[2] = track.filter.raw[2] + atan2(sin(state[2] - track.filter.raw[2]), cos(state[2] - track.filter.raw[2]))
        state = track.filter.update(state, timestamp, config)
        offset = state[3] * np.array([cos(state[2]), sin(state[2])]) / 2
        result = points.copy()
        result[0, :2], result[1, :2] = state[:2] - offset, state[:2] + offset
        return result

    def update(self, boxes, points, scores, class_ids, image_shape, *, sequence, timestamp, dedup_enabled=None):
        if self._last_sequence is not None and sequence == self._last_sequence:
            return self._cached
        boxes = np.asarray(boxes, dtype=float)
        points = np.asarray(points, dtype=float)
        scores = np.asarray(scores, dtype=float)
        classes = np.asarray(class_ids)
        n = len(scores)
        if boxes.shape != (n, 4) or points.shape != (n, 2, 3) or scores.shape != (n,) or classes.shape != (n,):
            raise ValueError("Expected boxes(N,4), points(N,2,3), scores(N), class_ids(N)")
        image_shape = tuple(image_shape[:2])
        if len(image_shape) != 2 or min(image_shape) <= 0 or not np.isfinite(timestamp):
            raise ValueError("Expected positive image dimensions and a finite monotonic timestamp")
        timestamp = float(timestamp)
        if (self._last_sequence is not None and (sequence < self._last_sequence or timestamp < self._last_timestamp
                                                or image_shape != self._image_shape)):
            self.reset()
        self._image_shape = image_shape
        start = perf_counter()
        enabled = self.dedup_enabled if dedup_enabled is None else bool(dedup_enabled)
        if enabled:
            dedup = deduplicate_detections(boxes, points, scores, image_shape, self.dedup_config, classes)
            keep, suppressed = dedup.keep_indices, dedup.suppressed
        else:
            keep, suppressed = tuple(range(n)), ()
        dedup_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        output = points.copy()
        identities = np.full(n, -1, dtype=np.int64)
        current_tracks = set()
        if self.config.tracking_enabled:
            self._tracks = {key: track for key, track in self._tracks.items()
                            if timestamp - track.timestamp <= self.config.max_gap_seconds}
            old = list(self._tracks.values())
            indices = list(keep)
            costs = _association_costs(old, boxes[indices], points[indices], classes[indices], image_shape, self.config)
            row_best = [_unambiguous_best(row, self.config) for row in costs]
            col_best = [_unambiguous_best(costs[:, column], self.config) for column in range(len(keep))]
            matches = {column: old[row] for column, row in enumerate(col_best)
                       if row is not None and row_best[row] == column}
            for column, index in enumerate(keep):
                track = matches.get(column)
                if track is None:
                    track = _Track(self._next_id, boxes[index].copy(), points[index].copy(),
                                   int(classes[index]), timestamp, sequence)
                    self._next_id += 1
                output[index] = self._smooth(track, points[index], timestamp, sequence)
                track.box, track.points = boxes[index].copy(), points[index].copy()
                track.timestamp, track.sequence = timestamp, sequence
                self._tracks[track.identity] = track
                identities[index] = track.identity
                current_tracks.add(track.identity)
            lost_count = len(self._tracks) - len(current_tracks)
        else:
            self._tracks.clear()
            identities[list(keep)] = np.asarray(keep, dtype=int) + 1
            lost_count = 0
        stable_count = self._count(len(keep), timestamp)
        output.setflags(write=False)
        identities.setflags(write=False)
        result = VideoResult(keep, output, identities, suppressed, stable_count, lost_count,
                             dedup_ms, (perf_counter() - start) * 1000)
        self._last_sequence, self._last_timestamp, self._cached = sequence, timestamp, result
        return result
