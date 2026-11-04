import unittest
from unittest.mock import patch
import json
import cv2
import numpy as np
from scripts.basket_refinement import refine_basket
from evaluate_basket_refinement import compare, summarize
from evaluate import Instance
from collections import Counter


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((240, 300, 3), np.uint8)
        self.xy = np.array([[40, 40], [240, 40], [240, 200], [40, 200]], float)
        self.points = np.column_stack([self.xy + [3, 2], np.ones(4)])

    def test_clean_rectangle(self):
        cv2.polylines(self.image, [self.xy.astype(np.int32)], True, (255, 255, 255), 2)
        old = self.points.copy()
        result = refine_basket(self.image, self.points)
        self.assertTrue(result.valid, result.reason)
        self.assertLess(np.linalg.norm(result.corners-self.xy, axis=1).max(), 4)
        np.testing.assert_array_equal(self.points, old)

    def test_blank_low_confidence_and_invalid_order(self):
        self.assertFalse(refine_basket(self.image, self.points).valid)
        self.points[0, 2] = .1
        self.assertEqual(refine_basket(self.image, self.points).reason, 'low_confidence')
        self.points[0, 2] = 1
        self.assertEqual(refine_basket(self.image, self.points[[1, 0, 2, 3]]).reason, 'invalid_coarse_quad')

    def test_rejection_and_miss_count_as_failure(self):
        gt = Instance(np.array([40, 40, 240, 200]), np.column_stack([self.xy, [2]*4]), class_id=1)
        pred = Instance(gt.box, np.column_stack([self.xy, [1]*4]), class_id=1)
        result = refine_basket(self.image, pred.points)
        rows = compare([gt], [pred], [result], 300, 240, 'fixture')
        report = summarize(rows, [1], Counter([result.reason]))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report['raw_qualified_rate'], 1)
        self.assertEqual(report['refined_qualified_rate'], 0)
        missed = compare([gt], [], [], 300, 240, 'fixture')
        self.assertEqual(summarize(missed, [], Counter())['refined_qualified_rate'], 0)

    def test_accepted_report_serializes_and_partial_denominator(self):
        cv2.polylines(self.image, [self.xy.astype(np.int32)], True, (255, 255, 255), 2)
        gt = Instance(np.array([40, 40, 240, 200]), np.column_stack([self.xy, [2, 2, 0, 0]]), class_id=1)
        pred = Instance(gt.box, self.points, class_id=1)
        result = refine_basket(self.image, pred.points)
        rows = compare([gt], [pred], [result], 300, 240, 'fixture')
        report = summarize(rows, [1], Counter([result.reason]))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report['complete_gt_baskets'], 0)
        self.assertIsNone(report['refined_qualified_rate'])

    def test_perspective_quad(self):
        xy = np.array([[50, 40], [230, 55], [250, 200], [30, 190]], float)
        cv2.polylines(self.image, [xy.astype(np.int32)], True, (255, 255, 255), 2)
        result = refine_basket(self.image, np.column_stack([xy + [3, 2], [1]*4]))
        self.assertTrue(result.valid, result.reason)
        self.assertLess(np.linalg.norm(result.corners-xy, axis=1).max(), 4)

    def test_ambiguous_lines_are_rejected(self):
        cv2.polylines(self.image, [self.xy.astype(np.int32)], True, (255, 255, 255), 2)
        candidates = [
            [(1., np.array([0., 1., -40.]), 1.), (1., np.array([0., 1., -43.]), 1.)],
            [(1., np.array([1., 0., -240.]), 1.)],
            [(1., np.array([0., 1., -200.]), 1.)],
            [(1., np.array([1., 0., -40.]), 1.)],
        ]
        with patch('scripts.basket_refinement._candidates', side_effect=candidates):
            result = refine_basket(self.image, self.points)
        self.assertEqual(result.reason, 'ambiguous_edges')
        self.assertIsNone(result.corners)

    def test_nonfinite_input(self):
        self.points[0, 0] = float('nan')
        self.assertEqual(refine_basket(self.image, self.points).reason, 'invalid_input')
