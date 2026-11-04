"""Compact, always-visible 3x5 count map for the live viewer side panel."""
import cv2


def draw_count_map(canvas, grid, left, state='LIVE'):
    valid = grid is not None and grid.status == 'current' and state not in ('STALE', 'YOLO OFF')
    cells = {(c.row, c.col): c.count for c in grid.cells} if valid else {}
    x0, y0, cw, ch, gap = left + 18, 104, 72, 62, 6
    uncertain = set()
    if valid:
        uncertain = {(r, c) for i, r, c in grid.assignments if grid.assignment_states[i] != 'assigned'}
    for row in range(3):
        for col in range(5):
            count = cells.get((row, col))
            x, y = x0 + col * (cw + gap), y0 + row * (ch + gap)
            pending = (row, col) in uncertain
            fill = (46, 74, 46) if count else (41, 44, 43)
            color = (130, 240, 170) if count else (145, 155, 160)
            if pending:
                color, fill = (70, 200, 255), (40, 62, 76)
            cv2.rectangle(canvas, (x, y), (x + cw - 1, y + ch - 1), fill, -1)
            cv2.putText(canvas, f'{row},{col}', (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (160, 177, 182), 1, cv2.LINE_AA)
            label = '--' if count is None else str(count)
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .85, 2)[0]
            cv2.putText(canvas, label, (x + (cw - size[0]) // 2, y + 47),
                        cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2, cv2.LINE_AA)
    summary = (f'Assigned {len(grid.assignments)}  |  Unassigned {len(grid.unassigned_indices)}'
               if valid else 'Counts unavailable - waiting for current grid')
    cv2.putText(canvas, summary, (left + 18, 326), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (210, 220, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Amber: cell assignment awaiting confirmation', (left + 18, 347),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (155, 180, 195), 1, cv2.LINE_AA)
