"""Stack grouping preserves detections and never turns capacity into filtering."""

from dataclasses import replace
import unittest

import numpy as np

from scripts.grid_counter import GridConfig, GridCounter


SHAPE = (1000, 1200)
ALGORITHM_CONFIG = replace(GridConfig(), confirmation_frames=1, confirmation_seconds=0.)


def detections(centers, *, width=100., confidence=.9, angle=0.):
    centers = np.array(centers, dtype=float).reshape(-1, 2)
    across = np.array([np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))])
    points = np.empty((len(centers), 2, 3))
    points[:, :, :2] = centers[:, None] + np.array([-.5, .5])[None, :, None] * width * across
    points[:, :, 2] = confidence
    boxes = np.c_[centers - [width / 2, 8], centers + [width / 2, 8]]
    return boxes, points


def update(counter, centers, *, sequence=1, timestamp=1., track_ids=None, keep=None, **kwargs):
    boxes, points = detections(centers, **kwargs)
    keep = tuple(range(len(boxes))) if keep is None else keep
    track_ids = np.arange(len(boxes)) if track_ids is None else track_ids
    return counter.update(boxes, points, keep, track_ids, SHAPE,
                          sequence=sequence, timestamp=timestamp)


def count_bottom_up(result, col=0):
    return [cell.count for cell in sorted(result.cells, key=lambda cell: -cell.row)
            if cell.col == col and cell.count]


class GridCounterTests(unittest.TestCase):
    def assert_conservation(self, result, expected):
        assigned = [item[0] for item in result.assignments]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned).union(result.unassigned_indices), set(expected))
        self.assertTrue(set(assigned).isdisjoint(result.unassigned_indices))
        self.assertTrue(all(cell.count <= 10 for cell in result.cells))
        self.assertEqual(sum(cell.count for cell in result.cells) + len(result.unassigned_indices), len(expected))

    def test_gap_then_capacity_six_ten_ten(self):
        ys = list(range(900, 799, -20)) + list(range(680, 299, -20))
        result = update(GridCounter(ALGORITHM_CONFIG), [(200, y) for y in ys])
        self.assertEqual(count_bottom_up(result), [6, 10, 10])
        self.assertEqual([cell.source for cell in reversed(result.cells)], ["start", "gap", "count"])
        self.assert_conservation(result, range(26))

    def test_twenty_continuous_and_nineteen_use_hard_capacity(self):
        for count, expected in ((20, [10, 10]), (19, [10, 9]), (10, [10]), (1, [1])):
            with self.subTest(count=count):
                result = update(GridCounter(ALGORITHM_CONFIG), [(200, 900 - i * 20) for i in range(count)])
                self.assertEqual(count_bottom_up(result), expected)
                self.assertEqual(len(result.cells), len(expected))
                self.assert_conservation(result, range(count))

    def test_gap_wins_at_capacity_boundary_only_one_split(self):
        ys = list(range(900, 719, -20)) + [600, 580]
        result = update(GridCounter(ALGORITHM_CONFIG), [(200, y) for y in ys])
        self.assertEqual(count_bottom_up(result), [10, 2])
        self.assertEqual(result.cells[0].source, "gap")
        self.assertEqual(len(result.cells), 2)

    def test_only_large_gap_does_not_become_layer_pitch(self):
        result = update(GridCounter(ALGORITHM_CONFIG), [(200, 700), (200, 300)])
        self.assertEqual(count_bottom_up(result), [1, 1])

    def test_rows_are_shared_across_columns_when_top_is_missing_in_one(self):
        centers = [(200, 700), (200, 680), (200, 400), (200, 380), (500, 700), (500, 680)]
        result = update(GridCounter(ALGORITHM_CONFIG), centers)
        self.assertEqual((result.rows, result.cols), (2, 2))
        cells = {(cell.row, cell.col): cell.count for cell in result.cells}
        self.assertEqual(cells, {(0, 0): 2, (1, 0): 2, (1, 1): 2})
        self.assert_conservation(result, range(6))

    def test_history_retains_empty_cells_and_column_indices(self):
        counter = GridCounter(replace(ALGORITHM_CONFIG, show_empty_cells=True))
        first = update(counter, [(200, 700), (200, 400), (500, 700), (500, 400)])
        second = update(counter, [(500, 400)], sequence=2, timestamp=1.04, track_ids=[3])
        self.assertEqual((second.rows, second.cols), (2, 2))
        self.assertEqual(len(second.cells), len(first.cells))
        cells = {(cell.row, cell.col): cell for cell in second.cells}
        self.assertEqual(cells[0, 1].indices, (0,))
        self.assertEqual(cells[1, 0].source, "history")
        self.assertEqual(cells[1, 0].count, 0)
        empty = update(counter, [], sequence=3, timestamp=1.08)
        self.assertEqual(sum(cell.count for cell in empty.cells), 0)
        self.assertEqual((empty.rows, empty.cols), (2, 2))
        self.assertTrue(all(cell.source == "history" for cell in empty.cells))
        self.assert_conservation(second, [0])

    def test_gap_requires_continuous_evidence_after_first_frame(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        update(counter, [(200, 700), (200, 680), (200, 660)])
        centers = [(200, 700), (200, 680), (200, 550)]
        pending = update(counter, centers, sequence=2, timestamp=1.03)
        self.assertEqual(count_bottom_up(pending), [3])
        confirmed = update(counter, centers, sequence=3, timestamp=1.14)
        self.assertEqual(count_bottom_up(confirmed), [2, 1])
        self.assert_conservation(confirmed, range(3))

    def test_transient_gap_does_not_confirm(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        normal = [(200, 700), (200, 680), (200, 660)]
        gap = [(200, 700), (200, 680), (200, 550)]
        update(counter, normal)
        update(counter, gap, sequence=2, timestamp=1.03)
        update(counter, normal, sequence=3, timestamp=1.2)
        result = update(counter, gap, sequence=4, timestamp=1.3)
        self.assertEqual(count_bottom_up(result), [3])

    def test_confirmed_gap_disappearance_is_debounced_using_neighbor_identity(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        first = update(counter, [(200, 700), (200, 680), (200, 550)])
        self.assertEqual(count_bottom_up(first), [2, 1])
        compressed = [(200, 700), (200, 680), (200, 660)]
        transient = update(counter, compressed, sequence=2, timestamp=1.04)
        self.assertEqual(count_bottom_up(transient), [2, 1])
        self.assertEqual(transient.cells[0].source, "gap")
        persistent = update(counter, compressed, sequence=3, timestamp=1.15)
        self.assertEqual(count_bottom_up(persistent), [3])
        self.assert_conservation(transient, range(3))
        self.assert_conservation(persistent, range(3))

    def test_gap_threshold_jitter_recovers_without_waiting_for_confirmation_again(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        # 20 px normal layer pitch makes the split threshold 44 px.
        update(counter, [(200, 700), (200, 680), (200, 635)])
        below = update(counter, [(200, 700), (200, 680), (200, 637)], sequence=2, timestamp=1.03)
        above = update(counter, [(200, 700), (200, 680), (200, 635)], sequence=3, timestamp=1.06)
        self.assertEqual(count_bottom_up(below), [2, 1])
        self.assertEqual(count_bottom_up(above), [2, 1])

    def test_gap_short_hold_can_use_position_when_identity_is_missing(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        update(counter, [(200, 700), (200, 680), (200, 635)], track_ids=[-1, -1, -1])
        result = update(counter, [(200, 700), (200, 680), (200, 637)],
                        sequence=2, timestamp=1.03, track_ids=[-1, -1, -1])
        self.assertEqual(count_bottom_up(result), [2, 1])
        self.assert_conservation(result, range(3))

    def test_losing_four_bottom_layers_keeps_original_row_and_anchor(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        centers = [(200, 900 - 20 * i) for i in range(10)]
        first = update(counter, centers)
        first_anchor = counter._rows[0].position
        for sequence in range(2, 12):
            result = update(counter, centers[4:], sequence=sequence, timestamp=1. + sequence * .03,
                            track_ids=np.arange(4, 10))
            self.assertEqual((result.rows, result.cols), (1, 1))
            self.assertEqual(result.cells[0].count, 6)
            self.assertEqual(result.cells[0].row, first.cells[0].row)
        self.assertEqual(counter._rows[0].position, first_anchor)
        self.assert_conservation(result, range(6))

    def test_track_row_evidence_precedes_closer_neighbor_row(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        lower = [(200, 900 - 20 * i) for i in range(10)]
        upper = [(200, 650), (200, 630)]
        update(counter, lower + upper)
        result = update(counter, lower[8:] + upper, sequence=2, timestamp=1.04,
                        track_ids=[8, 9, 10, 11])
        cells = {(cell.row, cell.col): cell.indices for cell in result.cells}
        self.assertEqual((result.rows, result.cols), (2, 1))
        self.assertEqual(cells[1, 0], (0, 1))
        self.assertEqual(cells[0, 0], (2, 3))
        self.assert_conservation(result, range(4))

    def test_track_inheritance_has_a_finite_stack_span(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        update(counter, [(200, 900), (200, 880)])
        result = update(counter, [(200, 500), (200, 480)], sequence=2, timestamp=1.04)
        self.assertEqual(result.rows, 2)
        self.assertEqual(sum(cell.count > 0 for cell in result.cells), 1)

    def test_track_history_never_changes_forced_ten_nine_partition(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        centers = [(200, 900 - i * 20) for i in range(20)]
        update(counter, centers)
        result = update(counter, centers[1:], sequence=2, timestamp=1.04, track_ids=np.arange(1, 20))
        self.assertEqual(count_bottom_up(result), [10, 9])
        self.assertEqual(result.rows, 2)
        self.assert_conservation(result, range(19))

    def test_same_sequence_returns_cache_without_confirming_gap(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        first = update(counter, [(200, 700), (200, 680)])
        again = update(counter, [(200, 700), (200, 300)], timestamp=2.)
        self.assertIs(first, again)

    def test_low_confidence_and_invalid_points_use_box_center(self):
        boxes, points = detections([(200, 700), (200, 680)], confidence=.1)
        points[0, :, :2] = np.nan
        result = GridCounter(replace(ALGORITHM_CONFIG, allow_box_fallback=True)).update(
            boxes, points, (0, 1), [10, 11], SHAPE, sequence=1, timestamp=1.)
        self.assertEqual(count_bottom_up(result), [2])
        self.assert_conservation(result, range(2))

    def test_unusable_geometry_is_unassigned_not_removed(self):
        boxes, points = detections([(200, 700), (200, 680)])
        boxes[0] = np.nan
        points[0] = np.nan
        result = GridCounter(ALGORITHM_CONFIG).update(boxes, points, (0, 1), [10, 11], SHAPE, sequence=1, timestamp=1.)
        self.assertEqual(result.unassigned_indices, (0,))
        self.assertEqual(result.assignments, ((1, 0, 0),))
        self.assert_conservation(result, range(2))

    def test_inputs_unchanged_and_only_keep_indices_count(self):
        boxes, points = detections([(200, 700), (200, 680), (200, 660)])
        before_boxes, before_points = boxes.copy(), points.copy()
        result = GridCounter(ALGORITHM_CONFIG).update(boxes, points, (2, 0), [1, 2, 3], SHAPE, sequence=1, timestamp=1.)
        self.assert_conservation(result, [2, 0])
        np.testing.assert_array_equal(boxes, before_boxes)
        np.testing.assert_array_equal(points, before_points)

    def test_column_bridge_cannot_join_distant_columns(self):
        result = update(GridCounter(ALGORITHM_CONFIG), [(200, 700), (250, 700), (300, 700)])
        self.assertEqual(result.cols, 2)
        self.assert_conservation(result, range(3))

    def test_configurable_stack_angle_handles_slanted_column(self):
        angle = 30.
        down = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
        centers = [np.array([350., 300.]) + down * i * 20 for i in range(20)]
        counter = GridCounter(replace(ALGORITHM_CONFIG, stack_angle_deg=angle))
        result = update(counter, centers, angle=angle)
        self.assertEqual(result.cols, 1)
        self.assertEqual(count_bottom_up(result), [10, 10])

    def test_rollback_shape_change_and_reset_clear_history(self):
        for cause in ("time", "sequence", "shape", "reset"):
            with self.subTest(cause=cause):
                counter = GridCounter(ALGORITHM_CONFIG)
                update(counter, [(200, 700), (500, 700)], sequence=5, timestamp=5.)
                boxes, points = detections([(500, 700)])
                if cause == "reset":
                    counter.reset()
                result = counter.update(boxes, points, (0,), [2], (1100, 1200) if cause == "shape" else SHAPE,
                                        sequence=4 if cause == "sequence" else 6,
                                        timestamp=4. if cause == "time" else 6.)
                self.assertEqual((result.rows, result.cols), (1, 1))

    def test_disabled_preserves_all_detections_as_unassigned(self):
        result = update(GridCounter(replace(ALGORITHM_CONFIG, enabled=False)), [(200, 700), (200, 680)])
        self.assertEqual((result.rows, result.cols), (0, 0))
        self.assert_conservation(result, [0, 1])

    def test_default_requires_both_three_frames_and_elapsed_time(self):
        counter = GridCounter()
        centers = [(200, 700), (200, 680)]
        for sequence, timestamp in ((1, 1.), (2, 1.03), (3, 1.08)):
            result = update(counter, centers, sequence=sequence, timestamp=timestamp)
            self.assertEqual(result.cells, ())
            self.assertEqual((result.rows, result.cols), (0, 0))
            self.assertEqual(result.unassigned_indices, (0, 1))
            self.assertEqual(counter._templates, {})
            self.assert_conservation(result, [0, 1])
        confirmed = update(counter, centers, sequence=4, timestamp=1.15)
        self.assertEqual(count_bottom_up(confirmed), [2])
        self.assertEqual(confirmed.unassigned_indices, ())
        self.assert_conservation(confirmed, [0, 1])

    def test_default_low_confidence_never_creates_grid_or_ghosts(self):
        counter = GridCounter()
        for sequence, center in enumerate(((200, 700), (500, 400), (800, 200)), 1):
            result = update(counter, [center], sequence=sequence,
                            timestamp=float(sequence), confidence=.01)
            self.assertEqual(result.unassigned_indices, (0,))
            self.assertEqual(result.cells, ())
            self.assert_conservation(result, [0])
        empty = update(counter, [], sequence=4, timestamp=100.)
        self.assertEqual((empty.rows, empty.cols, empty.cells), (0, 0, ()))
        for state in (counter._columns, counter._rows, counter._templates,
                      counter._gaps, counter._bottom_tracks, counter._track_cells):
            self.assertFalse(state)

    def test_single_high_confidence_false_detection_does_not_establish_template(self):
        counter = GridCounter()
        first = update(counter, [(200, 700)])
        self.assertEqual(first.unassigned_indices, (0,))
        self.assertFalse(counter._templates)
        empty = update(counter, [], sequence=2, timestamp=98.)
        self.assertEqual((empty.rows, empty.cols, empty.cells), (0, 0, ()))
        self.assertFalse(counter._confirmations)

    def test_explicit_box_fallback_still_requires_temporal_confirmation(self):
        counter = GridCounter(replace(GridConfig(), allow_box_fallback=True))
        for sequence in (1, 2):
            pending = update(counter, [(200, 700)], sequence=sequence,
                             timestamp=1. + (sequence - 1) * .08, confidence=.01)
            self.assertEqual(pending.cells, ())
            self.assertEqual(pending.unassigned_indices, (0,))
            self.assertFalse(counter._templates)
        confirmed = update(counter, [(200, 700)], sequence=3,
                           timestamp=1.16, confidence=.01)
        self.assertEqual(count_bottom_up(confirmed), [1])
        self.assert_conservation(confirmed, [0])

    def test_position_jumps_cannot_confirm_a_reused_track_id(self):
        counter = GridCounter()
        for sequence, center in enumerate(((200, 700), (500, 400), (800, 200)), 1):
            result = update(counter, [center], track_ids=[5], sequence=sequence,
                            timestamp=1. + (sequence - 1) * .08)
            self.assertEqual(result.cells, ())
            self.assertEqual(result.unassigned_indices, (0,))
        self.assertEqual(counter._confirmations[5].hits, 1)
        self.assertFalse(counter._templates)

    def test_missing_or_invalid_detection_resets_continuous_confirmation(self):
        for invalid in (False, True):
            with self.subTest(invalid_points=invalid):
                counter = GridCounter()
                update(counter, [(200, 700)], sequence=1, timestamp=1.)
                update(counter, [(200, 700)], sequence=2, timestamp=1.08)
                update(counter, [(200, 700)] if invalid else [], sequence=3,
                       timestamp=1.16, confidence=.01 if invalid else .9)
                for sequence, timestamp in ((4, 1.24), (5, 1.32)):
                    pending = update(counter, [(200, 700)], sequence=sequence, timestamp=timestamp)
                    self.assertEqual(pending.cells, ())
                confirmed = update(counter, [(200, 700)], sequence=6, timestamp=1.40)
                self.assertEqual(count_bottom_up(confirmed), [1])

    def test_long_frame_gap_breaks_confirmation(self):
        counter = GridCounter()
        update(counter, [(200, 700)], sequence=1, timestamp=1.)
        update(counter, [(200, 700)], sequence=2, timestamp=1.08)
        result = update(counter, [(200, 700)], sequence=3, timestamp=1.40)
        self.assertEqual(result.cells, ())
        self.assertEqual(counter._confirmations[0].hits, 1)

    def test_default_requires_usable_unique_track_ids(self):
        for identities in ([-1, -1], [4, 4]):
            with self.subTest(identities=identities):
                counter = GridCounter()
                for sequence in range(1, 4):
                    result = update(counter, [(200, 700), (200, 680)],
                                    track_ids=identities, sequence=sequence,
                                    timestamp=1. + (sequence - 1) * .08)
                self.assertEqual(result.cells, ())
                self.assert_conservation(result, [0, 1])
        boxes, points = detections([(200, 700)])
        result = GridCounter().update(boxes, points, (0,), None, SHAPE,
                                      sequence=1, timestamp=1.)
        self.assertEqual(result.unassigned_indices, (0,))

    def test_same_sequence_does_not_advance_confirmation(self):
        counter = GridCounter()
        first = update(counter, [(200, 700)], sequence=1, timestamp=1.)
        duplicate = update(counter, [(200, 700)], sequence=1, timestamp=1.2)
        self.assertIs(first, duplicate)
        self.assertEqual(counter._confirmations[0].hits, 1)
        second = update(counter, [(200, 700)], sequence=2, timestamp=1.20)
        self.assertEqual(second.cells, ())
        third = update(counter, [(200, 700)], sequence=3, timestamp=1.21)
        self.assertEqual(count_bottom_up(third), [1])

    def test_empty_cells_hidden_by_default_but_short_history_can_reassociate(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        update(counter, [(200, 700)], sequence=1, timestamp=1.)
        original_keys = set(counter._templates)
        empty = update(counter, [], sequence=2, timestamp=1.1)
        self.assertEqual((empty.rows, empty.cols, empty.cells), (0, 0, ()))
        self.assertEqual(set(counter._templates), original_keys)
        again = update(counter, [(200, 700)], sequence=3, timestamp=1.2)
        self.assertEqual(set(counter._templates), original_keys)
        self.assertEqual(count_bottom_up(again), [1])
        self.assert_conservation(again, [0])

    def test_expired_history_cleans_all_references_and_new_keys_do_not_collide(self):
        counter = GridCounter(ALGORITHM_CONFIG)
        update(counter, [(200, 700), (500, 400)], sequence=1, timestamp=1., track_ids=[10, 11])
        update(counter, [(500, 400)], sequence=2, timestamp=1.5, track_ids=[11])
        update(counter, [(500, 400)], sequence=3, timestamp=2.1, track_ids=[11])
        self.assertEqual(set(counter._columns), {1})
        self.assertEqual(set(counter._rows), {1})
        self.assertEqual(set(counter._templates), {(1, 1)})
        self.assertNotIn(10, counter._track_cells)
        result = update(counter, [(500, 400), (800, 200)], sequence=4,
                        timestamp=2.2, track_ids=[11, 12])
        self.assertEqual(set(counter._columns), {1, 2})
        self.assertEqual(set(counter._rows), {1, 2})
        self.assertEqual(set(counter._templates), {(1, 1), (2, 2)})
        self.assertEqual((result.rows, result.cols), (2, 2))
        self.assert_conservation(result, [0, 1])
        empty = update(counter, [], sequence=5, timestamp=4.)
        self.assertEqual((empty.rows, empty.cols, empty.cells), (0, 0, ()))
        for state in (counter._columns, counter._rows, counter._templates, counter._gaps,
                      counter._bottom_tracks, counter._track_cells, counter._confirmations):
            self.assertFalse(state)

    def test_explicit_empty_history_display_is_still_time_limited(self):
        counter = GridCounter(replace(ALGORITHM_CONFIG, show_empty_cells=True))
        update(counter, [(200, 700)], sequence=1, timestamp=1.)
        held = update(counter, [], sequence=2, timestamp=1.5)
        self.assertEqual((held.rows, held.cols), (1, 1))
        self.assertEqual(held.cells[0].source, "history")
        self.assertEqual(held.cells[0].count, 0)
        expired = update(counter, [], sequence=3, timestamp=2.01)
        self.assertEqual((expired.rows, expired.cols, expired.cells), (0, 0, ()))

    def test_reset_clears_pending_confirmation(self):
        counter = GridCounter()
        update(counter, [(200, 700)], sequence=1, timestamp=1.)
        update(counter, [(200, 700)], sequence=2, timestamp=1.08)
        counter.reset()
        pending = update(counter, [(200, 700)], sequence=3, timestamp=1.16)
        self.assertEqual(pending.cells, ())
        self.assertEqual(counter._confirmations[0].hits, 1)

    def test_invalid_config_and_duplicate_keep_rejected(self):
        for values in ({"max_per_cell": 0}, {"gap_factor": 0}, {"gap_confirm_seconds": -1},
                       {"confirmation_frames": 0}, {"confirmation_seconds": -1},
                       {"history_seconds": 0}):
            with self.assertRaises(ValueError):
                GridCounter(replace(ALGORITHM_CONFIG, **values))
        with self.assertRaises(ValueError):
            update(GridCounter(ALGORITHM_CONFIG), [(200, 700)], keep=(0, 0))


if __name__ == "__main__":
    unittest.main()
