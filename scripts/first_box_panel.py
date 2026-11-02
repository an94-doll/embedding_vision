"""Side-by-side endpoint readout. Coordinates always refer to original camera pixels."""
import math
import numpy as np

PANEL_WIDTH = 420
ROW_HEIGHT = 108
TOP = 112


def append_first_box_panel(canvas, rows, page=0, state='LIVE'):
    import cv2
    height, width = canvas.shape[:2]
    capacity = max(1, (height - TOP - 36) // ROW_HEIGHT)
    pages = max(1, math.ceil(len(rows) / capacity))
    page = max(0, min(page, pages - 1))
    out = np.full((height, width + PANEL_WIDTH, 3), (32, 29, 25), np.uint8)
    out[:, :width] = canvas
    def text(label, y, color=(225, 229, 233), scale=.49):
        cv2.putText(out, label, (width + 18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text('FIRST BOX / ENDPOINTS', 30, (100, 220, 240), .65)
    text(f'{state}  |  {len(rows)} cells', 56)
    text('Pixels (x, y) | raw / undistorted', 79, (165, 175, 185), .45)
    text('Cell IDs match Frow,col in the image', 99, (165, 175, 185), .43)
    if not rows:
        text('Waiting for confirmed first boxes' if state == 'LIVE' else 'No current endpoint positions', 150)
    def pair(value):
        return '--' if value is None else f'({value[0]:.1f}, {value[1]:.1f})'
    labels = {'rays_only':'2D', 'plane_localized':'3D', 'missing_intrinsics':'No intrinsics',
              'geometry_unavailable':'Geometry unavailable', 'empty':'Empty',
              'invalid_first_endpoints':'Endpoints unavailable'}
    shown = rows[page * capacity:(page + 1) * capacity]
    for index, row in enumerate(shown):
        y = TOP + index * ROW_HEIGHT
        status = labels.get(row['status'], row['status'])
        if row.get('selection_status') == 'incomplete_observation':
            status = 'Check upper box'
        text(f"F{row['row']},{row['col']}  |  n={row['detected_count']}  |  {status}", y + 18, (100, 220, 240))
        raw = row.get('raw_endpoints_px', [None, None])
        corrected = row.get('corrected_endpoints_px', [None, None])
        text(f'L raw {pair(raw[0])} / und {pair(corrected[0])}', y + 41, (150, 230, 155), .43)
        text(f'R raw {pair(raw[1])} / und {pair(corrected[1])}', y + 64, (235, 198, 130), .43)
        xyz = row.get('midpoint_camera_mm')
        text('Mid mm: ' + (', '.join(f'{v:.2f}' for v in xyz) if xyz is not None else '-- (no plane position)'),
             y + 86, (170, 180, 190), .43)
        cv2.line(out, (width + 18, y + 99), (width + PANEL_WIDTH - 18, y + 99), (62, 58, 52), 1)
    text(f'P previous / N next    {page + 1}/{pages}', height - 15, (175, 190, 200), .46)
    return out, page
