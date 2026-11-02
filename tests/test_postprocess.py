"""Protect neighboring cassettes; suppression must leave original coordinates intact."""

import unittest

import numpy as np

from scripts.postprocess import DedupConfig, deduplicate_detections


def inputs(shifts=(0, 0)):
    n = len(shifts)
    boxes = np.tile([100., 100., 240., 140.], (n, 1))
    points = np.array([[[110. + shift, 130., .9], [210. + shift, 130., .9]] for shift in shifts])
    scores = np.linspace(.9, .7, n)
    return boxes, points, scores


def run(boxes, points, scores, **kwargs):
    return deduplicate_detections(boxes, points, scores, (720, 1280), **kwargs)


class DedupTests(unittest.TestCase):
    def test_exact_duplicate_keeps_higher_box_score_without_moving_points(self):
        boxes, points, scores = inputs()
        originals = [a.copy() for a in (boxes, points, scores)]
        result = run(boxes, points, scores)
        self.assertEqual(result.keep_indices, (0,))
        self.assertEqual((result.suppressed[0].removed_index, result.suppressed[0].kept_index), (1, 0))
        for array, original in zip((boxes, points, scores), originals):
            np.testing.assert_array_equal(array, original)

    def test_rule_can_trigger_below_normal_nms_iou(self):
        boxes = np.array([[100., 100., 200., 130.], [100., 94., 200., 124.]])
        points = np.array([[[110., 120., .9], [190., 120., .9]]] * 2)
        result = run(boxes, points, np.array([.9, .8]))
        self.assertEqual(result.keep_indices, (0,))
        self.assertLess(result.suppressed[0].box_iou, .7)

    def test_parallel_adjacent_boxes_are_kept(self):
        boxes, points, scores = inputs()
        points[1, :, 1] += 4
        self.assertEqual(run(boxes, points, scores).keep_indices, (0, 1))

    def test_shared_single_endpoint_is_not_duplicate(self):
        boxes, points, scores = inputs()
        points[1, 1, 0] += 40
        self.assertEqual(run(boxes, points, scores).keep_indices, (0, 1))

    def test_low_confidence_or_reversed_endpoints_are_kept(self):
        for change in ("confidence", "order", "nonfinite"):
            boxes, points, scores = inputs()
            if change == "confidence":points[1, 0, 2] = .49
            elif change == "order":points[1] = points[1, ::-1].copy()
            else:points[1, 0, 0] = np.nan
            with self.subTest(change=change):
                self.assertEqual(run(boxes, points, scores).keep_indices, (0, 1))

    def test_truncated_real_neighbor_counterexample_is_kept_even_with_lax_distance(self):
        # Two actual detections matched different GTs in new_000082, not duplicates.
        boxes = np.array([[647.580, 0., 747.146, 21.171], [642.094, 0., 739.816, 27.885]])
        points = np.array([[[650.988, 14.975, .980], [742.804, 19.480, .956]],
                           [[648.503, 13.849, .996], [746.037, 17.381, .992]]])
        config = DedupConfig(endpoint_distance_px=5., endpoint_distance_ratio=.1,
                             max_length_ratio_error=.1, max_angle_deg=10.)
        self.assertEqual(run(boxes, points, np.array([.672, .547]), config=config).keep_indices, (0, 1))

    def test_short_edge_uses_stricter_relative_tolerance(self):
        boxes, points, scores = inputs((0, .5))
        points[:, 1, 0] = points[:, 0, 0] + 20
        self.assertEqual(run(boxes, points, scores).keep_indices, (0, 1))

    def test_different_box_geometry_or_class_is_kept(self):
        boxes, points, scores = inputs()
        self.assertEqual(run(boxes, points, scores, class_ids=[0, 1]).keep_indices, (0, 1))
        boxes[1, 1] -= 40
        self.assertEqual(run(boxes, points, scores).keep_indices, (0, 1))

    def test_non_clique_chain_is_kept_including_high_confidence_middle(self):
        boxes, points, _ = inputs((0, 1, 2))
        result = run(boxes, points, np.array([.8, .99, .7]))
        self.assertEqual(result.keep_indices, (0, 1, 2))
        self.assertEqual(result.ambiguous_indices, (0, 1, 2))
        self.assertFalse(result.suppressed)

    def test_clique_ties_are_deterministic_and_idempotent(self):
        boxes, points, _ = inputs((0, .5, 1))
        scores = np.array([.9, .9, .9])
        result = run(boxes, points, scores)
        self.assertEqual(result.keep_indices, (0,))
        indices = list(result.keep_indices)
        self.assertEqual(run(boxes[indices], points[indices], scores[indices]).keep_indices, (0,))

    def test_empty_single_and_invalid_shapes(self):
        result = run(np.empty((0, 4)), np.empty((0, 2, 3)), np.empty(0))
        self.assertEqual(result.keep_indices, ())
        boxes, points, scores = inputs((0,))
        self.assertEqual(run(boxes, points, scores).keep_indices, (0,))
        with self.assertRaises(ValueError):run(boxes, points[:, :, :2], scores)


if __name__ == "__main__":
    unittest.main()
