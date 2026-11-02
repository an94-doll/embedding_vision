"""Temporal processing must preserve all current boxes and avoid identity mixing."""

from dataclasses import replace
import unittest

import numpy as np

from scripts.video_postprocess import TemporalConfig, VideoPostprocessor


def detections(shifts=(0,), *, confidence=.9):
    boxes = np.array([[100., 100. + shift, 240., 140. + shift] for shift in shifts]).reshape(-1, 4)
    points = np.array([[[110., 130. + shift, confidence], [210., 130. + shift, confidence]]
                       for shift in shifts]).reshape(-1, 2, 3)
    return boxes, points, np.full(len(shifts), .9), np.zeros(len(shifts), dtype=int)


def update(processor, shifts, sequence, timestamp, **kwargs):
    return processor.update(*detections(shifts, **kwargs), (720, 1280), sequence=sequence, timestamp=timestamp)


class VideoPostprocessTests(unittest.TestCase):
    def processor(self, **kwargs):
        return VideoPostprocessor(temporal_config=replace(TemporalConfig(), **kwargs), dedup_enabled=False)

    def test_neighbors_keep_identity_when_order_and_confidence_reverse(self):
        processor = self.processor()
        first = update(processor, (0, 4), 1, 10.)
        boxes, points, scores, classes = detections((4.2, .2))
        scores[:] = [.99, .1]
        current = processor.update(boxes, points, scores, classes, (720, 1280), sequence=2, timestamp=10.05)
        self.assertEqual(tuple(current.track_ids), tuple(first.track_ids[::-1]))
        self.assertEqual(current.keep_indices, (0, 1))

    def test_ambiguous_neighbors_get_new_ids_and_raw_points(self):
        processor = self.processor()
        first = update(processor, (0, 4), 1, 1.)
        current = update(processor, (2, 2), 2, 1.05)
        self.assertTrue(set(first.track_ids).isdisjoint(current.track_ids))
        np.testing.assert_array_equal(current.points, detections((2, 2))[1])
        self.assertEqual(current.lost_count, 2)
        self.assertEqual(len(current.keep_indices), 2)

    def test_short_loss_reuses_identity_but_not_old_drawing_or_filter(self):
        processor = self.processor()
        first = update(processor, (0,), 1, 1.)
        empty = update(processor, (), 2, 1.05)
        self.assertEqual(empty.points.shape, (0, 2, 3))
        self.assertEqual(empty.keep_indices, ())
        self.assertEqual(empty.lost_count, 1)
        returned = update(processor, (2,), 3, 1.10)
        self.assertEqual(tuple(returned.track_ids), tuple(first.track_ids))
        np.testing.assert_array_equal(returned.points, detections((2,))[1])
        expired = update(processor, (2,), 4, 1.50)
        self.assertNotEqual(expired.track_ids[0], first.track_ids[0])
        self.assertEqual(expired.lost_count, 0)

    def test_new_boxes_all_retained_and_low_confidence_points_not_filtered(self):
        processor = self.processor()
        first = update(processor, (0,), 1, 1.)
        boxes, points, scores, classes = detections((1, 20), confidence=.1)
        points[0, 0, :2] = [0., 0.]
        current = processor.update(boxes, points, scores, classes, (720, 1280), sequence=2, timestamp=1.05)
        self.assertEqual(current.keep_indices, (0, 1))
        self.assertEqual(current.track_ids[0], first.track_ids[0])
        np.testing.assert_array_equal(current.points, points)
        recovered = update(processor, (2,), 3, 1.10)
        np.testing.assert_array_equal(recovered.points, detections((2,))[1])

    def test_filter_uses_actual_dt_and_preserves_line_length(self):
        fast, slow = self.processor(), self.processor()
        update(fast, (0,), 1, 1.)
        update(slow, (0,), 1, 1.)
        a = update(fast, (2,), 2, 1.02)
        b = update(slow, (2,), 2, 1.10)
        self.assertGreater(b.points[0, 0, 1], a.points[0, 0, 1])
        self.assertGreater(a.points[0, 0, 1], 130.)
        self.assertLess(b.points[0, 0, 1], 132.)
        self.assertAlmostEqual(np.linalg.norm(a.points[0, 1, :2] - a.points[0, 0, :2]), 100.)
        np.testing.assert_array_equal(a.points[:, :, 2], .9)

    def test_same_sequence_returns_cached_without_advancing_count(self):
        processor = self.processor()
        first = update(processor, (0,), 1, 1.)
        again = update(processor, (0, 20), 1, 10.)
        self.assertIs(first, again)
        current = update(processor, (0, 20), 2, 1.1)
        self.assertEqual(current.stable_count, 1)

    def test_reset_or_time_and_sequence_rollback_never_reuse_ids(self):
        processor = self.processor()
        identity = update(processor, (0,), 5, 5.).track_ids[0]
        for sequence, timestamp in ((4, 6.), (6, 5.), (7, 7.)):
            if sequence == 7:
                processor.reset()
            current = update(processor, (0,), sequence, timestamp)
            self.assertGreater(current.track_ids[0], identity)
            identity = current.track_ids[0]

    def test_count_debounce_independent_of_track_creation_and_loss(self):
        processor = self.processor()
        self.assertEqual(update(processor, (0,), 1, 0.).stable_count, 1)
        self.assertEqual(update(processor, (0, 20), 2, .05).stable_count, 1)
        self.assertEqual(update(processor, (0, 20), 3, .21).stable_count, 2)
        self.assertEqual(update(processor, (0,), 4, .25).stable_count, 2)
        self.assertEqual(update(processor, (0, 20), 5, .4).stable_count, 2)
        self.assertEqual(update(processor, (), 6, .5).stable_count, 2)
        self.assertEqual(update(processor, (), 7, .81).stable_count, 0)
        self.assertEqual(update(processor, (), 8, 1.).lost_count, 0)

    def test_empty_first_frame_and_zero_delay_count(self):
        processor = self.processor(count_rise_seconds=0., count_fall_seconds=0.)
        self.assertEqual(update(processor, (), 1, 0.).stable_count, 0)
        self.assertEqual(update(processor, (0,), 2, .1).stable_count, 1)
        self.assertEqual(update(processor, (), 3, .2).stable_count, 0)

    def test_all_switches_are_independent_and_input_arrays_unchanged(self):
        processor = self.processor(tracking_enabled=False, smoothing_enabled=True, counting_enabled=False)
        original = detections((0, 4))
        snapshots = [array.copy() for array in original]
        current = processor.update(*original, (720, 1280), sequence=1, timestamp=1.)
        self.assertEqual(tuple(current.track_ids), (1, 2))
        self.assertEqual(current.stable_count, 2)
        self.assertEqual(current.lost_count, 0)
        for actual, snapshot in zip(original, snapshots):
            np.testing.assert_array_equal(actual, snapshot)
        np.testing.assert_array_equal(current.points, original[1])
        self.assertFalse(current.points.flags.writeable)
        tracking = self.processor(smoothing_enabled=False, counting_enabled=False)
        first = update(tracking, (0,), 1, 1.)
        second = update(tracking, (2,), 2, 1.05)
        self.assertEqual(tuple(first.track_ids), tuple(second.track_ids))
        np.testing.assert_array_equal(second.points, detections((2,))[1])

    def test_existing_dedup_is_wrapped_and_suppressed_id_minus_one(self):
        processor = VideoPostprocessor()
        result = update(processor, (0, 0), 1, 1.)
        self.assertEqual(result.keep_indices, (0,))
        self.assertEqual(result.track_ids[1], -1)
        self.assertEqual(len(result.suppressed), 1)
        self.assertEqual(result.stable_count, 1)
        self.assertEqual(update(processor, (0, 0), 2, 1.05).keep_indices, (0,))
        boxes, points, scores, classes = detections((0, 0))
        enabled = processor.update(boxes, points, scores, classes, (720, 1280), sequence=3,
                                   timestamp=1.1, dedup_enabled=False)
        self.assertEqual(enabled.keep_indices, (0, 1))

    def test_classes_cannot_share_id_and_invalid_shape_rejected(self):
        processor = self.processor()
        first = update(processor, (0,), 1, 1.)
        boxes, points, scores, classes = detections((0,))
        current = processor.update(boxes, points, scores, classes + 1, (720, 1280), sequence=2, timestamp=1.1)
        self.assertNotEqual(first.track_ids[0], current.track_ids[0])
        with self.assertRaises(ValueError):
            processor.update(boxes, points[:, :, :2], scores, classes, (720, 1280), sequence=3, timestamp=1.2)
        with self.assertRaises(ValueError):
            self.processor(max_gap_seconds=-1.)


if __name__ == "__main__":
    unittest.main()
