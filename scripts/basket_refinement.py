"""Pose-guided image-space line refinement. No model, labels, or file I/O."""
from dataclasses import dataclass, field
from itertools import product

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    band_fraction: float = .04
    max_band_px: float = 32
    min_confidence: float = .5
    angle_degrees: float = 15
    min_coverage: float = .45
    ambiguity_margin: float = .04


@dataclass
class Refinement:
    corners: np.ndarray | None
    reason: str
    # Ax + By + C = 0; None means no supported line for that side.
    lines: list = field(default_factory=list)
    coverage: list = field(default_factory=list)

    @property
    def valid(self):
        return self.corners is not None


def _quad_ok(points):
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool(np.all(cross > 1) and np.min(np.linalg.norm(edges, axis=1)) > 10)


def _candidates(edge_points, segments, a, b, band, config):
    direction = b - a
    length = np.linalg.norm(direction)
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    along = (edge_points - a) @ direction
    near = (np.abs((edge_points - a) @ normal) <= band) & (along >= 0) & (along <= length)
    points = edge_points[near]
    if len(points) < 12:
        return []
    options = []
    for segment in segments:
        start, end = segment.reshape(2, 2)
        delta = end - start
        norm = np.linalg.norm(delta)
        if norm < max(12, length * .15) or abs(delta @ direction) / norm < np.cos(np.deg2rad(config.angle_degrees)):
            continue
        if max(abs((start-a) @ normal), abs((end-a) @ normal)) > band:
            continue
        if max((start-a) @ direction, (end-a) @ direction) < 0 or min((start-a) @ direction, (end-a) @ direction) > length:
            continue
        n = np.array([-delta[1], delta[0]]) / norm
        support = points[np.abs((points-start) @ n) <= 2.0]
        if len(support) < 12:
            continue
        vx, vy, x, y = cv2.fitLine(support.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        n = np.array([-vy, vx], dtype=float)
        if n @ normal < 0:
            n = -n
        if abs(np.array([vx, vy]) @ direction) < np.cos(np.deg2rad(config.angle_degrees)):
            continue
        c = -n @ [x, y]
        supported = points[np.abs(points @ n + c) <= 2.0]
        bins = np.clip(((supported-a) @ direction / length * 20).astype(int), 0, 19)
        coverage = len(np.unique(bins)) / 20
        offset = abs(n @ ((a+b)/2) + c)
        if coverage < config.min_coverage or offset > band:
            continue
        score = coverage - .25 * offset / band
        line = np.r_[n, c]
        options.append((score, line, coverage))
    # Keep the strongest representative of each thin-line family, not Hough order.
    unique = []
    midpoint = np.r_[(a+b)/2, 1]
    for candidate in sorted(options, key=lambda item: item[0], reverse=True):
        line = candidate[1]
        if any(abs((line-old[1]) @ midpoint) < 4 and np.dot(line[:2], old[1][:2]) > .99 for old in unique):
            continue
        unique.append(candidate)
        if len(unique) == 2:
            break
    return unique


def refine_basket(image, points, config=Config()):
    """Input four xy-confidence rows in TL, TR, BR, BL order; never reorder IDs.

    The minimal version accepts a complete quad or rejects it as a whole.
    Rejection returns no refined corners: callers retain raw results separately.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        return Refinement(None, 'invalid_input')
    corners = points[:, :2]
    h, w = image.shape[:2]
    if np.any(points[:, 2] < config.min_confidence):
        return Refinement(None, 'low_confidence')
    if np.any(corners < 0) or np.any(corners >= [w, h]) or not _quad_ok(corners):
        return Refinement(None, 'invalid_coarse_quad')
    band = np.clip(np.min(np.ptp(corners, axis=0)) * config.band_fraction, 6, config.max_band_px)
    lo = np.maximum(np.floor(corners.min(0)-band-3).astype(int), 0)
    hi = np.minimum(np.ceil(corners.max(0)+band+3).astype(int), [w, h])
    gray = cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    ys, xs = np.nonzero(edges)
    edge_points = np.column_stack([xs, ys]).astype(float) + lo
    segments = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=15, maxLineGap=8)
    if segments is None:
        return Refinement(None, 'no_edges')
    segments = segments.reshape(-1, 4).astype(float) + np.tile(lo, 2)
    candidates = [_candidates(edge_points, segments, corners[i], corners[(i+1) % 4], band, config)
                  for i in range(4)]
    lines = [c[0][1] if c else None for c in candidates]
    coverage = [c[0][2] if c else 0 for c in candidates]
    if not all(candidates):
        return Refinement(None, 'insufficient_edge_support', lines, coverage)
    solutions = []
    for combination in product(*candidates):
        fitted = [item[1] for item in combination]
        intersections = [np.cross(fitted[(i-1) % 4], fitted[i]) for i in range(4)]
        if any(abs(p[2]) < .15 for p in intersections):
            continue
        quad = np.array([p[:2]/p[2] for p in intersections])
        if (not _quad_ok(quad) or np.any(quad < 0) or np.any(quad >= [w, h])
                or np.any(np.linalg.norm(quad-corners, axis=1) > band * 1.5)):
            continue
        solutions.append((sum(c[0] for c in combination)/4, quad, fitted, [c[2] for c in combination]))
    solutions.sort(key=lambda s: s[0], reverse=True)
    if not solutions:
        return Refinement(None, 'invalid_intersections', lines, coverage)
    best = solutions[0]
    if len(solutions) > 1 and best[0]-solutions[1][0] < config.ambiguity_margin:
        return Refinement(None, 'ambiguous_edges', best[2], best[3])
    return Refinement(best[1], 'accepted', best[2], best[3])
