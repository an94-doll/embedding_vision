import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import cv2
from scripts.basket_grid import (BasketGridConfig, BasketGridCounter, project_points,
                                 grid_report, draw_basket_grid)
from scripts.first_box import locate_first_boxes
from scripts.first_box_panel import append_first_box_panel


CORNERS = np.array([[100, 80, .95], [1100, 150, .95], [1000, 650, .95], [180, 600, .95]])
UNIT = np.array([[0, 0], [5, 0], [5, 3], [0, 3]], np.float32)
INV = cv2.getPerspectiveTransform(UNIT, CORNERS[:, :2].astype(np.float32))
SHAPE = (720, 1280, 3)


def observations(centers):
    endpoints = [project_points([[u - .15, v], [u + .15, v]], INV) for u, v in centers]
    p = np.array([np.column_stack((xy, [.95, .95])) for xy in endpoints]).reshape(-1, 2, 3)
    b = np.array([[*xy.min(0), *xy.max(0)] for xy in endpoints]).reshape(-1, 4)
    return b, p


def update(counter, centers, seq=1, time=0., baskets=None, ids=None):
    b, p = observations(centers)
    r = counter.update(b, p, range(len(p)), np.arange(len(p)) if ids is None else ids, SHAPE,
                       sequence=seq, timestamp=time, basket_points=[CORNERS] if baskets is None else baskets)
    return r, b, p


def instant(**kwargs):
    return BasketGridCounter(BasketGridConfig(confirmation_frames=1, confirmation_seconds=0., **kwargs))


class BasketGridTests(unittest.TestCase):
    def test_raw_corners_are_not_smoothed_or_retained_when_missing(self):
        counter = instant()
        update(counter, [])
        moved = CORNERS.copy()
        moved[:, 0] += 20
        grid, _, _ = update(counter, [], 2, .1, baskets=[moved])
        np.testing.assert_allclose(grid.raw_corners, moved[:, :2])
        np.testing.assert_allclose(grid.corners, CORNERS[:, :2] + [7, 0], atol=1e-4)
        self.assertIsNotNone(grid_report(grid)['raw_corners_px'])
        frame = np.zeros(SHAPE, np.uint8)
        with patch.object(cv2, 'putText', wraps=cv2.putText) as put:
            draw_basket_grid(frame, grid)
        labels = [c.args[1] for c in put.call_args_list]
        self.assertNotIn('R0', labels)
        self.assertNotIn('S3', labels)
        held, _, _ = update(counter, [], 3, .2, baskets=[])
        self.assertIsNone(held.raw_corners)
        self.assertIsNotNone(held.corners)

    def test_all_fifteen_cells_perspective_assignment(self):
        centers = [(c + .5, r + .5) for r in range(3) for c in range(5)]
        grid, _, _ = update(instant(), centers)
        self.assertEqual(grid.status, 'current')
        self.assertEqual(len(grid.cells), 15)
        self.assertEqual([c.count for c in grid.cells], [1] * 15)
        self.assertEqual(grid.assignments[-1], (14, 2, 4))
        np.testing.assert_allclose(project_points(CORNERS[:, :2], grid.homography), UNIT, atol=1e-6)

    def test_empty_cells_and_more_than_ten_never_change_layout(self):
        counter = instant()
        grid, _, _ = update(counter, [(1.5, 1.5)] * 12)
        self.assertEqual(grid.cells[6].count, 12)
        empty, _, _ = update(counter, [], 2, .1)
        self.assertEqual(len(empty.cells), 15)
        self.assertEqual(sum(c.count for c in empty.cells), 0)

    def test_confirmation_needs_frames_and_time_and_no_duplicate_advancement(self):
        counter = BasketGridCounter()
        a, _, _ = update(counter, [(.5, .5)], 1, 0)
        repeated, _, _ = update(counter, [(.5, .5)], 1, 0)
        self.assertIs(a, repeated)
        b, _, _ = update(counter, [(.5, .5)], 2, .1)
        self.assertFalse(b.assignments)
        c, _, _ = update(counter, [(.5, .5)], 3, .2)
        self.assertEqual(c.assignments, ((0, 0, 0),))

    def test_missing_observation_breaks_new_confirmation(self):
        counter = BasketGridCounter()
        update(counter, [(.5, .5)], 1, 0)
        update(counter, [], 2, .1)
        c, _, _ = update(counter, [(.5, .5)], 3, .2)
        self.assertFalse(c.assignments)

    def test_midpoint_near_divider_assigned_but_outside_rejected(self):
        grid, _, _ = update(instant(), [(1.02, .5), (-.3, .5), (2.5, 1.5)])
        self.assertEqual(grid.unassigned_indices, (1,))
        self.assertEqual(grid.assignment_states[0], 'assigned')
        self.assertEqual(grid.assignment_states[1], 'outside')
        self.assertEqual(grid.assignments, ((0, 0, 1), (2, 1, 2)))

    def test_boundary_hysteresis_then_confirmed_switch(self):
        counter = BasketGridCounter(BasketGridConfig(confirmation_frames=2, confirmation_seconds=.05))
        update(counter, [(.5, .5)], 1, 0)
        update(counter, [(.5, .5)], 2, .1)
        boundary, _, _ = update(counter, [(1.02, .5)], 3, .2)
        self.assertEqual(boundary.assignments, ((0, 0, 0),))
        self.assertEqual(boundary.assignment_states[0], 'switch_pending')
        pending, _, _ = update(counter, [(1.5, .5)], 4, .3)
        self.assertEqual(pending.assignment_states[0], 'assigned')
        switched, _, _ = update(counter, [(1.5, .5)], 5, .4)
        self.assertEqual(switched.assignments, ((0, 0, 1),))

    def test_lost_basket_keeps_lines_but_not_counts_then_expires(self):
        counter = instant()
        update(counter, [(.5, .5)])
        held, b, p = update(counter, [(.5, .5)], 2, .2, baskets=[])
        self.assertEqual(held.status, 'held')
        self.assertEqual(len(held.cells), 15)
        self.assertFalse(held.assignments)
        self.assertIsNone(grid_report(held)['cells'][0]['count'])
        self.assertEqual(locate_first_boxes(held, b, p, None, SHAPE), [])
        missing, _, _ = update(counter, [(.5, .5)], 3, .6, baskets=[])
        self.assertEqual(missing.status, 'missing')
        self.assertFalse(missing.cells)

    def test_invalid_or_multiple_baskets_do_not_establish_grid(self):
        low = CORNERS.copy(); low[0, 2] = .1
        crossed = CORNERS[[0, 2, 1, 3]]
        for baskets in ([low], [crossed], [CORNERS, CORNERS]):
            grid, _, _ = update(instant(), [(.5, .5)], baskets=baskets)
            self.assertEqual(grid.status, 'missing')
            self.assertFalse(grid.assignments)

    def test_invalid_endpoints_and_unknown_track_stay_unassigned(self):
        counter = instant()
        b, p = observations([(.5, .5), (1.5, .5)])
        p[0, 0, 2] = .1
        grid = counter.update(b, p, [0, 1], [0, -1], SHAPE, sequence=1, timestamp=0,
                              basket_points=[CORNERS])
        self.assertEqual(grid.assignment_states, {0: 'invalid_endpoints', 1: 'untracked'})

    def test_endpoint_spanning_does_not_veto_midpoint_assignment(self):
        for left, right, expected in [(-.02, 1.02, 'assigned'), (-.2, 1.2, 'assigned')]:
            xy = project_points([[left, .5], [right, .5]], INV)
            p = np.column_stack((xy, [.9, .9]))[None]
            b = np.array([[*xy.min(0), *xy.max(0)]])
            grid = instant().update(b, p, [0], [0], SHAPE, sequence=1, timestamp=0,
                                    basket_points=[CORNERS])
            self.assertEqual(grid.assignment_states[0], expected)

    def test_reset_clears_geometry(self):
        counter = instant()
        update(counter, [(.5, .5)])
        counter.reset()
        grid, _, _ = update(counter, [(.5, .5)], baskets=[])
        self.assertEqual(grid.status, 'missing')

    def test_first_box_uses_basket_vertical_coordinate(self):
        grid, b, p = update(instant(), [(.5, .8), (.5, .3), (.5, 1.3)])
        rows = locate_first_boxes(grid, b, p, None, SHAPE)
        self.assertEqual(rows[0]['detection_index'], 1)
        self.assertEqual(rows[5]['detection_index'], 2)
        self.assertEqual(len(rows), 15)

    def test_draw_and_count_map_zero_unknown_and_stale(self):
        grid, _, _ = update(instant(), [(.5, .5)] * 3)
        frame = np.zeros(SHAPE, np.uint8)
        draw_basket_grid(frame, grid)
        self.assertTrue(frame.any())
        for state, expected in [('LIVE', '3'), ('STALE', '--'), ('YOLO OFF', '--')]:
            with patch.object(cv2, 'putText', wraps=cv2.putText) as put:
                out, page = append_first_box_panel(frame, [], state=state, grid=grid)
            labels = [c.args[1] for c in put.call_args_list]
            self.assertIn(expected, labels)
            self.assertEqual(out.shape, (720, 1700, 3))
            if state == 'LIVE': self.assertEqual(labels.count('0'), 14)
            else: self.assertEqual(labels.count('--'), 15)


if __name__ == '__main__':
    unittest.main()
