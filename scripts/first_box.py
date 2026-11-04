"""First observed box per inferred cell. Raw pixels in, calibrated rays out.

No depth stream. Optional plane: normal dot X_mm = offset_mm in color-camera
coordinates (x right, y down, z forward). Image midpoint is display-only.
"""
import math
import numpy as np


def endpoint_geometry(pixels, intrinsics, image_shape, plane=None):
    import cv2
    h, w = image_shape[:2]
    c = intrinsics
    if (c['width'], c['height']) != (w, h):
        raise ValueError('intrinsics_resolution_mismatch')
    fx, fy, cx, cy = (float(c[k]) for k in ('fx', 'fy', 'ppx', 'ppy'))
    if not np.isfinite([fx, fy, cx, cy, *c['coeffs']]).all() or min(fx, fy) <= 0:
        raise ValueError('invalid_intrinsics')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    model = c['model'].split('.')[-1]
    pixels = np.asarray(pixels, dtype=float).reshape(2, 2)
    if model == 'none':
        xy = (pixels - [cx, cy]) / [fx, fy]
    elif model == 'brown_conrady':
        xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, np.array(c['coeffs']),
                                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-12)).reshape(2, 2)
    elif model in ('inverse_brown_conrady', 'kannala_brandt4', 'ftheta'):
        import pyrealsense2 as rs
        intr = rs.intrinsics()
        for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
            setattr(intr, key, c[key])
        intr.model = getattr(rs.distortion, model)
        rays = np.array([rs.rs2_deproject_pixel_to_point(intr, p.tolist(), 1.) for p in pixels])
        xy = rays[:, :2] / rays[:, 2:]
    else:
        # Modified Brown is not OpenCV's standard Brown; don't silently substitute.
        raise ValueError('unsupported_distortion_model:' + model)
    rays = np.column_stack((xy, np.ones(2)))
    if not np.isfinite(rays).all():
        raise ValueError('invalid_rays')
    result = dict(corrected_endpoints_px=(xy * [fx, fy] + [cx, cy]).tolist(),
                  rays_camera=rays.tolist(), midpoint_camera_mm=None,
                  status='rays_only', width_mm=None, width_error_mm=None)
    if plane is None:
        return result
    normal = np.asarray(plane['normal'], dtype=float)
    offset = float(plane['offset_mm'])
    if normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(offset) or np.linalg.norm(normal) == 0:
        raise ValueError('invalid_plane')
    denom = rays @ (normal / np.linalg.norm(normal))
    if np.any(np.abs(denom) < 1e-8):
        raise ValueError('ray_parallel_to_plane')
    xyz = rays * ((offset / np.linalg.norm(normal)) / denom)[:, None]
    if np.any(xyz[:, 2] <= 0) or not np.isfinite(xyz).all():
        raise ValueError('plane_not_in_front_of_camera')
    width = float(np.linalg.norm(xyz[1] - xyz[0]))
    result.update(status='plane_localized', endpoints_camera_mm=xyz.tolist(),
                  midpoint_camera_mm=xyz.mean(0).tolist(), width_mm=width,
                  width_error_mm=width - 30.)
    return result


def locate_first_boxes(grid, boxes, points, intrinsics, image_shape, *, angle_deg=0.,
                       point_conf=.5, plane=None, lost_count=0):
    """Select before quality filtering. Never substitute a lower valid box.

    Group IDs are image-space IDs, not persistent physical slot IDs. Incomplete
    grouping is reported separately from geometric conversion.
    """
    fixed = hasattr(grid, 'homography')
    if fixed and grid.status != 'current':
        return []
    if fixed:
        from scripts.basket_grid import project_points
    down = np.array([math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))])
    h, w = image_shape[:2]
    rows = []
    for cell in grid.cells:
        row = dict(row=cell.row, col=cell.col, detected_count=cell.count,
                   status='empty', detection_index=None, midpoint_camera_mm=None)
        rows.append(row)
        if not cell.indices:
            continue
        def position(i):
            if fixed and i in grid.positions:
                return grid.positions[i][1]
            p = points[i]
            center = p[:, :2].mean(0) if np.isfinite(p[:, :2]).all() else (boxes[i, :2] + boxes[i, 2:]) / 2
            return float(center @ down)
        i = min(cell.indices, key=position)
        p = np.asarray(points[i])
        row.update(detection_index=int(i))
        valid = (np.isfinite(p).all() and np.all(p[:, 2] >= point_conf)
                 and np.all(p[:, :2] >= 0) and np.all(p[:, 0] < w) and np.all(p[:, 1] < h)
                 and np.linalg.norm(p[1, :2] - p[0, :2]) > 1e-6)
        if not valid:
            row['status'] = 'invalid_first_endpoints'
            continue
        row.update(raw_endpoints_px=p[:, :2].tolist(), endpoint_confidence=p[:, 2].tolist(),
                   display_midpoint_px=p[:, :2].mean(0).tolist())
        # Only unassigned observations above this candidate in the same column
        # can change its ordering. Never treat an unrelated partial box as a veto.
        across = np.array([down[1], -down[0]])
        bounds = cell.bounds
        corners = np.array([[bounds[0], bounds[1]], [bounds[2], bounds[1]],
                            [bounds[0], bounds[3]], [bounds[2], bounds[3]]])
        lo, hi = (corners @ across).min(), (corners @ across).max()
        blockers = []
        for j in grid.unassigned_indices:
            center = (boxes[j, :2] + boxes[j, 2:]) / 2
            if fixed:
                uv = grid.positions.get(j)
                if uv is None:
                    uv = project_points([center], grid.homography)[0]
                if (cell.col <= uv[0] < cell.col + 1
                        and cell.row <= uv[1] <= position(i)):
                    blockers.append(int(j))
            elif lo <= center @ across <= hi and center @ down <= position(i):
                blockers.append(int(j))
        row.update(selection_status='incomplete_observation' if blockers else 'observed_candidate',
                   unassigned_above=blockers, temporarily_lost_tracks=int(lost_count))
        if fixed:
            row['assignment_state'] = grid.assignment_states[i]
            if row['assignment_state'] != 'assigned':
                row['selection_status'] = 'uncertain_assignment'
        if intrinsics is None:
            row['status'] = 'missing_intrinsics'
            continue
        try:
            row.update(endpoint_geometry(p[:, :2], intrinsics, image_shape, plane))
        except (ValueError, ImportError, RuntimeError) as error:
            row.update(status='geometry_unavailable', reason=str(error))
    return rows


def draw_first_boxes(canvas, rows, y_offset=0):
    import cv2
    for row in rows:
        if 'display_midpoint_px' not in row:
            continue
        x, y = np.rint(row['display_midpoint_px']).astype(int)
        y += y_offset
        color = (255, 80, 255) if row['status'] in ('rays_only', 'plane_localized') and row.get('selection_status') == 'observed_candidate' else (0, 180, 255)
        cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(canvas, f"F{row['row']},{row['col']}",
                    (max(0, x - 18), max(15, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
