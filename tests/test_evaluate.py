"""评估口径回归：匹配、分母、置信度、几何和无目标边界。无需模型推理。"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluate as ev


def instance(box=(0, 0, 10, 10), xy=((2, 8), (8, 8)), scores=(2, 2), class_id=0):
    return ev.Instance(np.asarray(box, dtype=float), np.column_stack([xy, scores]).astype(float), .9, class_id)


def analyze(truth, predictions):
    rows = ev.analyze_image(truth, predictions, 200, 100, "test.jpg")
    return rows, ev.summarize(*rows)


class EvaluationTests(unittest.TestCase):
    def test_dense_matching_recovers_two_valid_pairs(self):
        truth = [instance((0, 0, 10, 10)), instance((2, 0, 12, 10))]
        pred = [instance((.5, 0, 10.5, 10)), instance((-2, 0, 8, 10))]
        matches, _ = ev.match_instances(truth, pred)
        self.assertEqual(matches, {0: 1, 1: 0})

    def test_matching_maximizes_count_before_total_iou(self):
        ious = np.array([[.99, 0, .51], [.51, .99, 0], [0, .51, 0]])
        with patch.object(ev, "box_ious", return_value=ious):
            matches, _ = ev.match_instances([instance()] * 3, [instance()] * 3)
        self.assertEqual(matches, {0: 2, 1: 0, 2: 1})

    def test_wrong_class_is_not_matched(self):
        matches, _ = ev.match_instances([instance()], [instance(class_id=1)])
        self.assertEqual(matches, {})

    def test_low_confidence_is_retained_raw_but_not_qualified(self):
        _, summary = analyze([instance()], [instance(scores=(.001, .9))])
        pck = summary["endpoints"]["all"]["pck_pixels"]["5.0"]
        self.assertEqual(pck["raw_all_gt"], 1)
        self.assertEqual(pck["available_all_gt"], .5)
        self.assertEqual(summary["pairs"]["qualified_pair_rate_all_evaluable_gt"], 0)

    def test_misses_count_in_all_gt_denominators(self):
        _, summary = analyze([instance(), instance()], [instance(scores=(.9, .9))])
        pck = summary["endpoints"]["all"]["pck_pixels"]["5.0"]
        self.assertEqual(pck["raw_matched"], 1)
        self.assertEqual(pck["raw_all_gt"], .5)
        self.assertEqual(summary["pairs"]["qualified_pair_rate_all_evaluable_gt"], .5)
        self.assertEqual(summary["detection"]["recall"], .5)

    def test_partial_gt_only_evaluates_known_endpoint(self):
        rows, summary = analyze([instance(scores=(2, 0))], [instance(scores=(.9, .9))])
        self.assertEqual(len(rows[1]), 1)
        self.assertEqual(summary["pairs"]["evaluable_gt_pairs"], 0)
        self.assertIsNone(summary["geometry"]["qualified_localization_rate_all_evaluable_gt"])
        json.dumps(summary, allow_nan=False)

    def test_translation_geometry_and_tolerance_boundary(self):
        gt = instance(xy=((10, 10), (30, 10)))
        pred = instance(xy=((13, 14), (33, 14)), scores=(.9, .9))
        rows, _ = analyze([gt], [pred])
        row = rows[0][0]
        self.assertEqual(row["center_error_px"], 5)
        self.assertEqual(row["left_error_px"], 5)
        self.assertEqual(row["angle_error_deg"], 0)
        self.assertEqual(row["length_error_px"], 0)
        self.assertTrue(row["localization_qualified"])

    def test_angle_wrap_is_two_not_178_degrees(self):
        def endpoints(angle):
            r = math.radians(angle)
            v = np.array([math.cos(r), math.sin(r)]) * 10
            return np.array([50, 50]) + np.array([-v, v])
        rows, _ = analyze([instance(xy=endpoints(89))], [instance(xy=endpoints(-89), scores=(.9, .9))])
        self.assertAlmostEqual(rows[0][0]["angle_error_deg"], 2)

    def test_endpoint_ids_are_not_silently_reordered(self):
        rows, _ = analyze([instance()], [instance(xy=((8, 8), (2, 8)), scores=(.9, .9))])
        row = rows[0][0]
        self.assertTrue(row["reversed_order"])
        self.assertEqual(row["left_error_px"], 6)
        self.assertEqual(row["angle_error_deg"], 0)
        self.assertFalse(row["pair_qualified"])

    def test_nonfinite_and_out_of_image_points_are_unavailable(self):
        for xy in [((float("nan"), 8), (8, 8)), ((-1, 8), (8, 8))]:
            with self.subTest(xy=xy):
                rows, summary = analyze([instance()], [instance(xy=xy, scores=(.9, .9))])
                self.assertFalse(rows[1][0]["available"])
                self.assertFalse(rows[0][0]["pair_qualified"])
                json.dumps(summary, allow_nan=False)
                image = ev.draw_comparison(np.zeros((100, 200, 3), dtype=np.uint8),
                                           [instance()], [instance(xy=xy, scores=(.9, .9))], rows[0], rows[2])
                self.assertEqual(image.shape, (128, 400, 3))

    def test_collapsed_prediction_does_not_qualify(self):
        rows, _ = analyze([instance()], [instance(xy=((5, 8), (5, 8)), scores=(.9, .9))])
        self.assertFalse(rows[0][0]["pair_qualified"])
        self.assertEqual(rows[0][0]["reason"], "degenerate_pred_edge")

    def test_duplicate_is_a_subset_of_false_positives(self):
        pred = [instance(scores=(.9, .9)), instance(scores=(.9, .9)), instance((100, 0, 110, 10))]
        _, summary = analyze([instance()], pred)
        self.assertEqual(summary["detection"]["false_positives_including_duplicates"], 2)
        self.assertEqual(summary["detection"]["duplicate_candidates"], 1)
        self.assertAlmostEqual(summary["detection"]["precision"], 1 / 3)

    def test_background_false_positive_and_all_empty(self):
        _, summary = analyze([], [instance(scores=(.9, .9))])
        self.assertEqual(summary["detection"]["false_positives_including_duplicates"], 1)
        self.assertIsNone(summary["detection"]["recall"])
        for truth, pred in [([], []), ([instance()], []), ([instance(xy=((2, 8), (2, 8)))], [])]:
            _, summary = analyze(truth, pred)
            json.dumps(summary, allow_nan=False)

    def test_labels_scale_to_original_pixels_and_empty_files_are_background(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "images/test/sample.jpg"
            label = root / "labels/test/sample.txt"
            label.parent.mkdir(parents=True)
            label.write_text("0 .5 .5 .5 .4 .25 .7 2 .75 .7 2\n")
            truth = ev.read_truth(image, 200, 100)
            np.testing.assert_allclose(truth[0].box, [50, 30, 150, 70])
            np.testing.assert_allclose(truth[0].points, [[50, 70, 2], [150, 70, 2]])
            label.write_text("")
            self.assertEqual(ev.read_truth(image, 200, 100), [])
            output = root / "points.csv"
            ev.save_csv(output, [], ev.POINT_FIELDS)
            with output.open(encoding="utf-8-sig") as handle:
                self.assertEqual(next(csv.reader(handle)), ev.POINT_FIELDS)


class BasketEvaluationTests(unittest.TestCase):
    def basket(self, scores=(2, 2, 2, 2), xy=None):
        return instance(box=(0, 0, 100, 80), xy=xy or ((10, 10), (90, 10), (90, 70), (10, 70)),
                        scores=scores, class_id=1)

    def test_mixed_classes_ignore_white_padding(self):
        white = instance()
        white.points = np.vstack([white.points, [[0, 0, 0], [0, 0, 0]]])
        rows, summary = analyze([white, self.basket()], [white, self.basket()])
        self.assertEqual(len(rows[1]), 6)
        self.assertEqual(summary['endpoints']['all']['evaluable_gt_points'], 2)
        self.assertEqual(summary['geometry']['raw_matched']['center_error_px']['mean'], 0)
        self.assertEqual(summary['by_class']['basket']['quadrilaterals']['qualified_baskets'], 1)
        with tempfile.TemporaryDirectory() as d:
            for filename, data, fields in [('instances', rows[0], ev.INSTANCE_FIELDS),
                                            ('points', rows[1], ev.POINT_FIELDS),
                                            ('predictions', rows[2], ev.PREDICTION_FIELDS)]:
                ev.save_csv(Path(d)/filename, data, fields)
        json.dumps(summary, allow_nan=False)

    def test_missing_basket_and_partial_corners_denominators(self):
        _, summary = analyze([self.basket(), self.basket(scores=(2, 2, 0, 0))], [])
        basket = summary['by_class']['basket']
        self.assertEqual(basket['corners']['all']['evaluable_gt_points'], 6)
        self.assertEqual(basket['quadrilaterals']['evaluable_gt_baskets'], 1)
        self.assertEqual(basket['quadrilaterals']['qualified_rate_all_evaluable_gt'], 0)
        self.assertEqual(basket['corners']['all']['pck_pixels']['5.0']['available_all_gt'], 0)

    def test_low_confidence_and_fixed_corner_ids(self):
        for pred in [self.basket(scores=(.9, .9, .9, .1)),
                     self.basket(xy=((90, 10), (10, 10), (10, 70), (90, 70)))]:
            rows, summary = analyze([self.basket()], [pred])
            self.assertFalse(rows[0][0]['corners_qualified'])
        self.assertEqual(summary['by_class']['basket']['corners']['top_left']['pixel_error_raw_matched']['mean'], 80)

    def test_wrong_class_overlap_is_not_duplicate(self):
        _, summary = analyze([instance()], [instance(class_id=1)])
        self.assertEqual(summary['detection']['duplicate_candidates'], 0)
        self.assertEqual(summary['by_class']['basket']['detection']['false_positives_including_duplicates'], 1)

    def test_four_point_label_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            label = root/'labels/test/a.txt'; label.parent.mkdir(parents=True)
            label.write_text('0 .5 .5 .5 .4 .25 .7 2 .75 .7 2 0 0 0 0 0 0\n'
                             '1 .5 .5 1 1 0 0 2 1 0 2 1 1 2 0 1 2\n')
            truth = ev.read_truth(root/'images/test/a.jpg', 200, 100, [4, 3])
            self.assertEqual([g.class_id for g in truth], [0, 1])
            np.testing.assert_allclose(truth[1].points[2], [200, 100, 2])

    def test_corner_tolerance_boundary_and_out_of_image(self):
        shifted = ((13, 14), (93, 14), (93, 74), (13, 74))
        rows, _ = analyze([self.basket()], [self.basket(xy=shifted)])
        self.assertTrue(rows[0][0]['corners_qualified'])
        self.assertEqual(rows[0][0]['max_corner_error_px'], 5)
        rows, _ = analyze([self.basket()], [self.basket(xy=((-1, 10), (90, 10), (90, 70), (10, 70)))])
        self.assertFalse(rows[1][0]['available'])

    def test_single_basket_model_maps_to_semantic_basket(self):
        names, shape = ev.dataset_layout({'names': {0: 'basket'}, 'kpt_shape': [4, 3]})
        local = self.basket()
        local.class_id = 0
        truth = ev.semantic_instances([local], names)
        self.assertEqual(truth[0].class_id, 1)
        self.assertEqual(local.class_id, 0)
        _, summary = analyze(truth, truth)
        self.assertEqual(summary['by_class']['basket']['quadrilaterals']['qualified_baskets'], 1)
        self.assertEqual(summary['pairs']['evaluable_gt_pairs'], 0)

    def test_dataset_layout_rejects_wrong_class_semantics(self):
        ev.dataset_layout({'names': {0: 'white_box', 1: 'basket'}, 'kpt_shape': [4, 3]})
        with self.assertRaises(ValueError):
            ev.dataset_layout({'names': {0: 'basket', 1: 'white_box'}, 'kpt_shape': [4, 3]})

    def test_basket_draw_handles_missing_and_nonfinite(self):
        pred = self.basket(xy=((float('nan'), 10), (90, 10), (90, 70), (10, 70)))
        rows, summary = analyze([self.basket()], [pred])
        json.dumps(summary, allow_nan=False)
        canvas = ev.draw_comparison(np.zeros((100, 200, 3), np.uint8),
                                    [self.basket()], [pred], rows[0], rows[2])
        self.assertEqual(canvas.shape, (128, 400, 3))


if __name__ == "__main__":
    unittest.main()
