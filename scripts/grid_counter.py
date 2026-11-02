"""Image-space stack grouping: gaps first, capacity second, no detections dropped.

Rows and columns describe observed stacks, not calibrated physical basket slots.
Only continuously observed targets establish cells. Short-lived templates support
reassociation; never-seen empty slots cannot be inferred from endpoints alone.
"""

from dataclasses import dataclass
from math import cos, radians, sin
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    enabled: bool = True
    max_per_cell: int = 10
    point_confidence: float = 0.50
    # Both conditions are required; unavailable/ambiguous track IDs stay unassigned.
    confirmation_frames: int = 3
    confirmation_seconds: float = 0.15
    # History supports reassociation only and never contributes to current counts.
    history_seconds: float = 1.0
    allow_box_fallback: bool = False
    show_empty_cells: bool = False
    # Positive angle tilts the downward stack direction toward image right.
    stack_angle_deg: float = 0.0
    column_distance_ratio: float = 0.55
    column_match_ratio: float = 0.65
    layer_pitch_ratio: float = 0.22
    gap_factor: float = 2.2
    gap_confirm_seconds: float = 0.10
    row_match_ratio: float = 0.65
    row_track_overlap: float = 0.60
    row_track_span_factor: float = 1.25
    anchor_alpha: float = 0.15
    track_memory_seconds: float = 1.0
    min_scale_px: float = 4.0


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    indices: tuple[int, ...]
    bounds: tuple[float, float, float, float]
    source: str

    @property
    def count(self):
        return len(self.indices)


@dataclass(frozen=True)
class GridResult:
    cells: tuple[GridCell, ...]
    assignments: tuple[tuple[int, int, int], ...]
    unassigned_indices: tuple[int, ...]
    rows: int
    cols: int
    processing_ms: float


@dataclass
class _Anchor:
    position: float
    scale: float


@dataclass
class _Template:
    bounds: tuple[float, float, float, float]
    last_seen: float


@dataclass
class _Confirmation:
    since: float
    last_seen: float
    sequence: int
    hits: int
    position: tuple[float, float]
    scale: float


@dataclass
class _Gap:
    position: float
    since: float
    confirmed: bool
    adjacent: tuple[int, int]
    last_seen: float
    missing_since: float | None = None


class GridCounter:
    """Group current detections while retaining observed row/column anchors.

    A current box belongs to exactly one cell or unassigned_indices. Confirmation
    controls grouping only; it never removes a detection from the result. History
    never contributes old detections or an assumed count.
    """

    def __init__(self, config=GridConfig()):
        self.config = config
        positive = (config.column_distance_ratio, config.column_match_ratio,
                    config.layer_pitch_ratio, config.gap_factor, config.row_match_ratio,
                    config.track_memory_seconds, config.min_scale_px, config.row_track_span_factor,
                    config.history_seconds)
        if (not isinstance(config.max_per_cell, int) or config.max_per_cell < 1
                or not isinstance(config.confirmation_frames, int) or config.confirmation_frames < 1
                or not 0 <= config.point_confidence <= 1
                or not np.isfinite(config.stack_angle_deg)
                or not 0 < config.anchor_alpha <= 1
                or not 0 < config.row_track_overlap <= 1
                or not np.isfinite(config.gap_confirm_seconds) or config.gap_confirm_seconds < 0
                or not np.isfinite(config.confirmation_seconds) or config.confirmation_seconds < 0
                or any(not np.isfinite(value) or value <= 0 for value in positive)):
            raise ValueError("Invalid grid configuration")
        angle = radians(config.stack_angle_deg)
        self._across = np.array([cos(angle), -sin(angle)])
        self._down = np.array([sin(angle), cos(angle)])
        self.reset()

    def reset(self):
        self._columns = {}
        self._rows = {}
        self._templates = {}
        self._bottom_tracks = {}
        self._gaps = {}
        self._track_cells = {}
        self._confirmations = {}
        self._next_column = 0
        self._next_row = 0
        self._last_sequence = None
        self._last_timestamp = None
        self._shape = None
        self._cached = None

    def _measurements(self, boxes, points, indices, shape):
        """Collect usable geometry; other detections remain explicitly unassigned."""
        height, width = shape
        measurements, unassigned = {}, []
        for index in indices:
            box, pair = boxes[index], points[index]
            point_valid = (np.isfinite(pair).all()
                           and (pair[:, 2] >= self.config.point_confidence).all()
                           and (pair[:, :2] >= 0).all()
                           and (pair[:, 0] < width).all() and (pair[:, 1] < height).all())
            length = float(np.linalg.norm(pair[1, :2] - pair[0, :2])) if point_valid else 0.
            box_valid = bool(np.isfinite(box).all() and (box[2:] > box[:2]).all())
            if point_valid and length > 1e-6:
                center = pair[:, :2].mean(0)
                scale = length
            elif self.config.allow_box_fallback and box_valid:
                center = (box[:2] + box[2:]) / 2
                scale = float(np.abs(self._across) @ (box[2:] - box[:2]))
            else:
                unassigned.append(index)
                continue
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                unassigned.append(index)
                continue
            if box_valid:
                bounds = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            else:
                bounds = np.r_[pair[:, :2].min(0), pair[:, :2].max(0)]
            measurements[index] = (float(center @ self._across), float(center @ self._down),
                                   max(scale, self.config.min_scale_px), tuple(float(v) for v in bounds))
        return measurements, unassigned

    def _confirm(self, measurements, track_ids, sequence, timestamp):
        """Require consecutive fresh observations of one spatially continuous ID."""
        confirmed, current = {}, {}
        ids = [int(track_ids[index]) for index in measurements]
        unique, counts = np.unique(ids, return_counts=True)
        repeated = set(unique[counts > 1])
        for index, measurement in measurements.items():
            identity = int(track_ids[index])
            if identity < 0 or identity in repeated:
                # Explicit single-frame mode remains useful for offline grouping.
                if self.config.confirmation_frames == 1 and self.config.confirmation_seconds == 0:
                    confirmed[index] = measurement
                continue
            position, scale = measurement[:2], measurement[2]
            previous = self._confirmations.get(identity)
            continuous = (previous is not None and previous.sequence == self._last_sequence
                          and 0 <= timestamp - previous.last_seen <= 0.25
                          and np.linalg.norm(np.subtract(position, previous.position))
                          <= 0.75 * min(scale, previous.scale))
            observation = _Confirmation(
                previous.since if continuous else timestamp, timestamp, sequence,
                previous.hits + 1 if continuous else 1, position, scale)
            current[identity] = observation
            if (observation.hits >= self.config.confirmation_frames
                    and timestamp - observation.since + 1e-9 >= self.config.confirmation_seconds):
                confirmed[index] = measurement
        # Missing or invalid observations break confirmation immediately.
        self._confirmations = current
        return confirmed

    def _expire_history(self, timestamp):
        """Retire templates and every spatial reference no longer used by them."""
        self._templates = {key: value for key, value in self._templates.items()
                           if timestamp - value.last_seen <= self.config.history_seconds}
        columns = {column for column, _ in self._templates}
        rows = {row for _, row in self._templates}
        self._columns = {key: value for key, value in self._columns.items() if key in columns}
        self._rows = {key: value for key, value in self._rows.items() if key in rows}
        self._gaps = {key: value for key, value in self._gaps.items() if key in columns}
        self._bottom_tracks = {key: value for key, value in self._bottom_tracks.items()
                               if key in self._templates}
        self._track_cells = {identity: value for identity, value in self._track_cells.items()
                             if value[:2] in self._templates
                             and timestamp - value[2] <= self.config.track_memory_seconds}

    def _new_columns(self, indices, measurements):
        """Full-span constraint avoids joining two columns through a bridge box."""
        groups = []
        for index in sorted(indices, key=lambda i: measurements[i][0]):
            if groups:
                group = groups[-1]
                values = [measurements[i] for i in (*group, index)]
                span = values[-1][0] - values[0][0]
                if span <= self.config.column_distance_ratio * np.median([v[2] for v in values]):
                    group.append(index)
                    continue
            groups.append([index])
        return groups

    def _assign_columns(self, measurements, track_ids):
        columns = {key: [] for key in self._columns}
        unmatched = []
        for index, (u, _, scale, _) in measurements.items():
            prior = self._track_cells.get(int(track_ids[index]))
            candidates = []
            for key, anchor in self._columns.items():
                gate = self.config.column_match_ratio * min(scale, anchor.scale)
                distance = abs(u - anchor.position)
                if distance <= gate:
                    cost = distance / gate - (0.15 if prior and prior[0] == key else 0.)
                    candidates.append((cost, key))
            if candidates:
                columns[min(candidates)[1]].append(index)
            else:
                unmatched.append(index)
        for group in self._new_columns(unmatched, measurements):
            key = self._next_column
            self._next_column += 1
            self._columns[key] = _Anchor(float(np.median([measurements[i][0] for i in group])),
                                         float(np.median([measurements[i][2] for i in group])))
            columns[key] = group
        for key, indices in columns.items():
            if indices:
                alpha, anchor = self.config.anchor_alpha, self._columns[key]
                anchor.position += alpha * (float(np.median([measurements[i][0] for i in indices])) - anchor.position)
                anchor.scale += alpha * (float(np.median([measurements[i][2] for i in indices])) - anchor.scale)
        return columns

    def _groups(self, column, indices, measurements, track_ids, timestamp):
        ordered = sorted(indices, key=lambda i: (-measurements[i][1], measurements[i][0], i))
        if not ordered:
            self._gaps[column] = []
            return []
        values = np.array([measurements[i][:3] for i in ordered])
        gaps = values[:-1, 1] - values[1:, 1]
        scales = (values[:-1, 2] + values[1:, 2]) / 2
        ratios = gaps / scales
        prior = self.config.layer_pitch_ratio
        # A lone large gap must not become the inferred ordinary layer pitch.
        plausible = ratios[(ratios >= prior * 0.5) & (ratios <= prior * 1.75)]
        pitch_ratio = float(np.clip(np.median(plausible), prior * 0.6, prior * 1.4)) if len(plausible) else prior
        candidates = np.flatnonzero(ratios > pitch_ratio * self.config.gap_factor)
        adjacent = [tuple(int(track_ids[i]) for i in ordered[j:j + 2]) for j in range(len(gaps))]
        positions = (values[:-1, 1] + values[1:, 1]) / 2
        old_gaps = self._gaps.get(column)
        first_observation = old_gaps is None
        updated_gaps, confirmed = [], set()
        used_old = set()
        for boundary in candidates:
            position = float(positions[boundary])
            nearest = None
            for old_index, old in enumerate(old_gaps or ()):
                distance = abs(position - old.position)
                same_pair = min(adjacent[boundary]) >= 0 and old.adjacent == adjacent[boundary]
                if old_index not in used_old and (same_pair or distance <= scales[boundary] * pitch_ratio):
                    cost = (not same_pair, distance)
                    if nearest is None or cost < nearest[0]:
                        nearest = (cost, old_index, old)
            if nearest is None:
                gap = _Gap(position, timestamp, first_observation, adjacent[boundary], timestamp)
            else:
                _, old_index, old = nearest
                used_old.add(old_index)
                gap = _Gap(position, old.since, old.confirmed, adjacent[boundary], timestamp)
            gap.confirmed |= timestamp - gap.since >= self.config.gap_confirm_seconds
            if gap.confirmed:
                confirmed.add(int(boundary))
            updated_gaps.append(gap)
        # A confirmed boundary disappears only after a short continuous absence.
        # Reuse a boundary between current points; history never adds detections.
        for old_index, old in enumerate(old_gaps or ()):
            missing_since = old.missing_since if old.missing_since is not None else timestamp
            if (old_index in used_old or not old.confirmed
                    or timestamp - missing_since >= self.config.gap_confirm_seconds):
                continue
            choices = []
            for boundary, pair in enumerate(adjacent):
                same_pair = min(pair) >= 0 and pair == old.adjacent
                contains = values[boundary + 1, 1] <= old.position <= values[boundary, 1]
                if (same_pair or contains) and boundary not in confirmed:
                    choices.append((not same_pair, abs(float(positions[boundary]) - old.position), boundary))
            if choices:
                boundary = min(choices)[2]
                confirmed.add(boundary)
                updated_gaps.append(_Gap(float(positions[boundary]), old.since, True,
                                         adjacent[boundary], old.last_seen, missing_since))
        self._gaps[column] = updated_gaps
        groups, current, source = [], [], "start"
        for position, index in enumerate(ordered):
            if current and position - 1 in confirmed:
                groups.append((current, source))
                current, source = [], "gap"
            elif len(current) >= self.config.max_per_cell:
                groups.append((current, source))
                current, source = [], "count"
            current.append(index)
        if current:
            groups.append((current, source))
        return groups

    def _assign_row(self, column, indices, measurements, track_ids, used_rows):
        bottom_index = max(indices, key=lambda index: measurements[index][1])
        bottom = measurements[bottom_index][1]
        bottom_identity = int(track_ids[bottom_index])
        scale = float(np.median([measurements[index][2] for index in indices]))
        candidates, tracked = [], []
        for key, anchor in self._rows.items():
            gate = self.config.row_match_ratio * min(scale, anchor.scale)
            distance = abs(bottom - anchor.position)
            if key in used_rows:
                continue
            overlap = sum(bool((prior := self._track_cells.get(int(track_ids[index])))
                               and prior[:2] == (column, key)) for index in indices) / len(indices)
            track_gate = max(gate, min(scale, anchor.scale) * self.config.layer_pitch_ratio
                             * (self.config.max_per_cell - 1) * self.config.row_track_span_factor)
            if overlap >= self.config.row_track_overlap and distance <= track_gate:
                tracked.append((-overlap, distance / track_gate, key))
            if distance <= gate:
                candidates.append((distance / gate - 0.30 * overlap, key))
        if tracked:
            key = min(tracked)[2]
            anchor = self._rows[key]
            # Losing bottom layers must not pull a stable row anchor upward.
            if (bottom_identity != self._bottom_tracks.get((column, key))
                    and bottom < anchor.position - scale * self.config.layer_pitch_ratio * 0.5):
                bottom = anchor.position
                bottom_identity = -1
        elif candidates:
            key = min(candidates)[1]
        else:
            key = self._next_row
            self._next_row += 1
            self._rows[key] = _Anchor(bottom, scale)
        return key, bottom, scale, bottom_identity

    def update(self, boxes, points, keep_indices, track_ids, image_shape, *, sequence, timestamp):
        started = perf_counter()
        boxes, points = np.asarray(boxes), np.asarray(points)
        indices = tuple(int(index) for index in keep_indices)
        n = len(boxes)
        track_ids = np.full(n, -1, dtype=int) if track_ids is None else np.asarray(track_ids)
        shape = tuple(int(value) for value in image_shape[:2])
        if (boxes.shape != (n, 4) or points.shape != (n, 2, 3) or track_ids.shape != (n,)
                or len(set(indices)) != len(indices) or any(i < 0 or i >= n for i in indices)
                or len(shape) != 2 or min(shape) <= 0 or not np.isfinite(timestamp)
                or not np.isfinite(track_ids).all()):
            raise ValueError("Invalid grid inputs")
        if (self._shape is not None and (shape != self._shape or sequence < self._last_sequence
                                        or timestamp < self._last_timestamp)):
            self.reset()
        if sequence == self._last_sequence:
            return self._cached
        self._shape = shape
        self._expire_history(timestamp)
        if not self.config.enabled:
            result = GridResult((), (), indices, 0, 0, (perf_counter() - started) * 1000)
        else:
            measurements, _ = self._measurements(boxes, points, indices, shape)
            measurements = self._confirm(measurements, track_ids, sequence, timestamp)
            unassigned = [index for index in indices if index not in measurements]
            columns = self._assign_columns(measurements, track_ids)
            current, row_observations = {}, {}
            for column in sorted(columns, key=lambda key: self._columns[key].position):
                used_rows = set()
                for group, source in self._groups(column, columns[column], measurements, track_ids, timestamp):
                    row, bottom, scale, bottom_identity = self._assign_row(
                        column, group, measurements, track_ids, used_rows)
                    used_rows.add(row)
                    if bottom_identity >= 0:
                        self._bottom_tracks[column, row] = bottom_identity
                    row_observations.setdefault(row, []).append((bottom, scale))
                    bounds = np.array([measurements[index][3] for index in group])
                    envelope = (*bounds[:, :2].min(0), *bounds[:, 2:].max(0))
                    self._templates[column, row] = _Template(
                        tuple(float(value) for value in envelope), timestamp)
                    current[column, row] = (tuple(group), source)
                    for index in group:
                        identity = int(track_ids[index])
                        if identity >= 0:
                            self._track_cells[identity] = (column, row, timestamp)
            for row, values in row_observations.items():
                anchor, alpha = self._rows[row], self.config.anchor_alpha
                bottom, scale = np.median(values, axis=0)
                anchor.position += alpha * (float(bottom) - anchor.position)
                anchor.scale += alpha * (float(scale) - anchor.scale)
            row_numbers = {key: i for i, key in enumerate(sorted(self._rows, key=lambda k: self._rows[k].position))}
            col_numbers = {key: i for i, key in enumerate(sorted(self._columns, key=lambda k: self._columns[k].position))}
            cells, assignments = [], []
            for (column, row), template in self._templates.items():
                if (column, row) not in current and not self.config.show_empty_cells:
                    continue
                group, source = current.get((column, row), ((), "history"))
                cells.append(GridCell(row_numbers[row], col_numbers[column], group, template.bounds, source))
                assignments.extend((index, row_numbers[row], col_numbers[column]) for index in group)
            result = GridResult(tuple(sorted(cells, key=lambda cell: (cell.row, cell.col))),
                                tuple(sorted(assignments)), tuple(unassigned),
                                len(self._rows) if cells else 0,
                                len(self._columns) if cells else 0, (perf_counter() - started) * 1000)
        self._last_sequence, self._last_timestamp, self._cached = sequence, timestamp, result
        return result
