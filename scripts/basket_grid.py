"""Fixed equal 3x5 basket cells, defined by TL/TR/BR/BL image corners."""
from dataclasses import dataclass, field
from time import perf_counter
import cv2
import numpy as np


@dataclass(frozen=True)
class BasketGridConfig:
    enabled: bool = True
    rows: int = 3
    cols: int = 5
    point_confidence: float = .5
    confirmation_frames: int = 3
    confirmation_seconds: float = .15
    corner_alpha: float = .35
    hold_seconds: float = .5
    track_memory_seconds: float = .35


@dataclass(frozen=True)
class BasketCell:
    row: int
    col: int
    indices: tuple
    polygon: tuple
    source: str = 'basket'

    @property
    def count(self): return len(self.indices)

    @property
    def bounds(self):
        p = np.asarray(self.polygon)
        return (*p.min(0), *p.max(0))


@dataclass
class BasketGridResult:
    cells: tuple
    assignments: tuple
    unassigned_indices: tuple
    rows: int
    cols: int
    processing_ms: float
    status: str
    reason: str
    homography: object = None
    corners: object = None
    positions: dict = field(default_factory=dict)
    assignment_states: dict = field(default_factory=dict)
    age_seconds: float | None = None
    raw_corners: object = None


def project_points(points, matrix):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ matrix.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[:, :2] / q[:, 2:]


class BasketGridCounter:
    def __init__(self, config=BasketGridConfig()):
        self.config = config
        if (config.rows < 1 or config.cols < 1 or config.confirmation_frames < 1
                or config.confirmation_seconds < 0
                or not 0 < config.corner_alpha <= 1 or config.hold_seconds < 0
                or config.track_memory_seconds <= 0 or not 0 <= config.point_confidence <= 1):
            raise ValueError('Invalid basket grid configuration')
        self.reset()

    def reset(self):
        self.corners = None
        self.seen_at = None
        self.tracks = {}
        self.last_sequence = None
        self.last_timestamp = None
        self.shape = None
        self.cached = None

    def _corners(self, baskets, shape):
        h, w = shape[:2]
        valid = []
        for p in baskets:
            p = np.asarray(p)
            if p.shape != (4, 3) or not np.isfinite(p).all():
                continue
            if (p[:, 2] < self.config.point_confidence).any():
                continue
            xy = p[:, :2].astype(np.float32)
            if (xy < 0).any() or (xy[:, 0] >= w).any() or (xy[:, 1] >= h).any():
                continue
            if not cv2.isContourConvex(xy) or cv2.contourArea(xy, oriented=True) < 100:
                continue
            valid.append(xy)
        if len(valid) == 1:
            return valid[0], 'observed'
        return None, 'multiple_baskets' if len(valid) > 1 else 'corners_unavailable'

    def update(self, boxes, points, keep_indices, track_ids, image_shape, *,
               sequence, timestamp, basket_points=()):
        started = perf_counter()
        if (self.shape != tuple(image_shape[:2]) or
                self.last_sequence is not None and sequence < self.last_sequence or
                self.last_timestamp is not None and timestamp < self.last_timestamp):
            self.reset()
        if sequence == self.last_sequence and self.cached is not None:
            return self.cached
        previous_sequence = self.last_sequence
        self.last_sequence, self.last_timestamp = sequence, timestamp
        self.shape = tuple(image_shape[:2])
        cfg = self.config
        indices = tuple(int(i) for i in keep_indices)
        observed, reason = self._corners(basket_points, image_shape)
        status = 'disabled'
        if cfg.enabled:
            if observed is not None:
                if self.seen_at is None or timestamp - self.seen_at > cfg.hold_seconds:
                    self.corners = observed
                    self.tracks.clear()
                else:
                    self.corners = (cfg.corner_alpha * observed + (1 - cfg.corner_alpha) * self.corners)
                self.seen_at = timestamp
                status = 'current'
            elif self.seen_at is not None and timestamp - self.seen_at <= cfg.hold_seconds:
                status = 'held'
            else:
                status = 'missing'
                self.corners = None
                self.tracks.clear()
        H = None
        if status in ('current', 'held'):
            H = cv2.getPerspectiveTransform(self.corners.astype(np.float32),
                np.array([[0, 0], [cfg.cols, 0], [cfg.cols, cfg.rows], [0, cfg.rows]], np.float32))
        positions, states, assigned = {}, {}, {}
        self.tracks = {k: v for k, v in self.tracks.items()
                       if timestamp - v['last_seen'] <= cfg.track_memory_seconds}
        if status == 'current':
            h, w = image_shape[:2]
            for i in indices:
                p = points[i]
                good = (np.isfinite(p).all() and (p[:, 2] >= cfg.point_confidence).all()
                        and (p[:, :2] >= 0).all() and (p[:, 0] < w).all() and (p[:, 1] < h).all()
                        and np.linalg.norm(p[1, :2] - p[0, :2]) > 1e-6)
                if not good:
                    states[i] = 'invalid_endpoints'
                    continue
                uv = project_points(p[:, :2], H)
                center = uv.mean(0)
                if not np.isfinite(uv).all():
                    states[i] = 'invalid_projection'
                    continue
                positions[i] = tuple(float(x) for x in center)
                u, v = center
                if not (0 <= u < cfg.cols and 0 <= v < cfg.rows):
                    states[i] = 'outside'
                    continue
                target = (int(np.floor(v)), int(np.floor(u)))
                # Cell geometry depends only on the projected endpoint midpoint.
                track_id = int(track_ids[i])
                if track_id < 0:
                    states[i] = 'untracked'
                    continue
                old = self.tracks.get(track_id)
                if old is None:
                    old = dict(cell=None, pending=None, hits=0, since=timestamp,
                               last_seen=timestamp, sequence=None)
                    self.tracks[track_id] = old
                consecutive = old['sequence'] == previous_sequence
                old.update(last_seen=timestamp, sequence=sequence)
                if old['cell'] == target:
                    old.update(pending=None, hits=0)
                    states[i] = 'assigned'
                else:
                    if old['pending'] != target or not consecutive:
                        old.update(pending=target, hits=0, since=timestamp)
                    old['hits'] += 1
                    if (old['hits'] >= cfg.confirmation_frames
                            and timestamp - old['since'] >= cfg.confirmation_seconds):
                        old.update(cell=target, pending=None, hits=0)
                        states[i] = 'assigned'
                    else:
                        states[i] = 'switch_pending' if old['cell'] is not None else 'confirming'
                if old['cell'] is not None:
                    assigned[i] = old['cell']
        else:
            states = {i: 'grid_' + status for i in indices}
        cells = []
        if H is not None:
            inverse = np.linalg.inv(H)
            for row in range(cfg.rows):
                for col in range(cfg.cols):
                    polygon = project_points([[col, row], [col + 1, row],
                                              [col + 1, row + 1], [col, row + 1]], inverse)
                    members = tuple(i for i, rc in assigned.items() if rc == (row, col))
                    cells.append(BasketCell(row, col, members, tuple(map(tuple, polygon))))
        result = BasketGridResult(tuple(cells), tuple((i, *rc) for i, rc in assigned.items()),
            tuple(i for i in indices if i not in assigned), cfg.rows, cfg.cols,
            (perf_counter() - started) * 1000, status, reason, H,
            self.corners.copy() if H is not None else None, positions, states,
            None if self.seen_at is None else timestamp - self.seen_at,
            None if observed is None else observed.copy())
        self.cached = result
        return result


def grid_report(grid):
    return dict(mode='basket_fixed', status=grid.status, reason=grid.reason,
        rows=grid.rows, cols=grid.cols, age_seconds=grid.age_seconds,
        corners_px=None if grid.corners is None else grid.corners.tolist(),
        raw_corners_px=None if grid.raw_corners is None else grid.raw_corners.tolist(),
        assignments=[dict(index=i, row=r, col=c, state=grid.assignment_states[i])
                     for i, r, c in grid.assignments],
        unassigned_indices=list(grid.unassigned_indices), assignment_states=grid.assignment_states,
        cells=[dict(row=c.row, col=c.col, count=c.count if grid.status == 'current' else None,
                    indices=list(c.indices), polygon_px=c.polygon) for c in grid.cells])


def draw_basket_grid(frame, grid, y_offset=0):
    color = (90, 210, 150) if grid.status == 'current' else (0, 180, 255)
    for cell in grid.cells:
        p = np.rint(cell.polygon).astype(np.int32)
        p[:, 1] += y_offset
        cv2.polylines(frame, [p], True, color, 1, cv2.LINE_AA)
        # Label inside the projected top-left corner, following both cell axes.
        anchor = np.rint(.82 * p[0] + .09 * p[1] + .09 * p[3]).astype(int)
        label = f'({cell.row},{cell.col}) n={cell.count}' if grid.status == 'current' else f'({cell.row},{cell.col}) --'
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0]
        x = int(np.clip(anchor[0], 0, max(0, frame.shape[1] - size[0] - 4)))
        y = int(np.clip(anchor[1], y_offset + 15, frame.shape[0] - 3))
        cv2.rectangle(frame, (x, y - 13), (x + size[0] + 4, y + 3), (25, 30, 28), -1)
        cv2.putText(frame, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
