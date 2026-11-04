import unittest
from types import SimpleNamespace as NS
import numpy as np
from scripts.pose_schema import validate_model, prediction_arrays, draw_baskets


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    def cpu(self): return self
    def numpy(self): return self.data


def result(points, classes):
    n = len(classes)
    return NS(boxes=NS(xyxy=Tensor(np.zeros((n, 4))), conf=Tensor(np.ones(n)), cls=Tensor(classes)),
              keypoints=NS(data=Tensor(points)))


class PoseSchemaTests(unittest.TestCase):
    def test_dedicated_model_roles_and_class_zero(self):
        for role, n in [('white_box', 2), ('basket', 4)]:
            model = NS(task='pose', names={0: role}, model=NS(model=[NS(kpt_shape=[n, 3])]))
            validate_model(model, role=role)
            with self.assertRaises(ValueError):
                validate_model(model, role='basket' if role == 'white_box' else 'white_box')
            arrays = prediction_arrays(result(np.ones((1, n, 3)), [0]), class_id=0, point_count=n)
            self.assertEqual(arrays[1].shape, (1, n, 3))
        mixed = NS(task='pose', names={0: 'white_box', 1: 'basket'}, model=NS(model=[NS(kpt_shape=[4, 3])]))
        with self.assertRaises(ValueError):
            validate_model(mixed, role='basket')

    def test_white_boxes_exclude_baskets_and_placeholder_points(self):
        points = np.array([[[10, 20, .9], [40, 20, .9], [0, 0, 0], [0, 0, 0]],
                           [[5, 5, .9], [60, 5, .9], [60, 60, .9], [5, 60, .9]]])
        r = result(points, [0, 1])
        boxes, endpoints, scores, classes = prediction_arrays(r)
        self.assertEqual(endpoints.shape, (1, 2, 3))
        np.testing.assert_array_equal(endpoints[0], points[0, :2])
        np.testing.assert_array_equal(classes, [0])
        _, corners, _, _ = prediction_arrays(r, 1, 4)
        np.testing.assert_array_equal(corners, points[1:])

    def test_legacy_and_empty_outputs(self):
        r = result(np.ones((1, 2, 3)), [0])
        self.assertEqual(prediction_arrays(r)[1].shape, (1, 2, 3))
        self.assertEqual(prediction_arrays(r, 1, 4)[1].shape, (0, 4, 3))
        r = result(np.empty((0, 4, 3)), [])
        self.assertEqual(prediction_arrays(r)[1].shape, (0, 2, 3))

    def test_supported_models(self):
        for shape, names in [([2, 3], {0: 'white_box'}), ([4, 3], {0: 'white_box', 1: 'basket'})]:
            validate_model(NS(task='pose', names=names, model=NS(model=[NS(kpt_shape=shape)])))
        with self.assertRaises(ValueError):
            validate_model(NS(task='pose', names={0: 'basket', 1: 'white_box'}, model=NS(model=[NS(kpt_shape=[4, 3])])) )

    def test_basket_overlay_respects_header_and_missing_corners(self):
        canvas = np.zeros((100, 80, 3), dtype=np.uint8)
        corners = np.array([[[10, 10, .9], [60, 10, .9], [60, 60, 0], [10, 60, .9]]])
        draw_baskets(canvas, corners, (80, 80), y_offset=20)
        self.assertFalse(canvas[:20].any())
        self.assertTrue(canvas[30, 30].any())
        self.assertFalse(canvas[60, 60].any())


if __name__ == '__main__':
    unittest.main()
