"""Adapt supported pose models to white-box endpoints and basket corners."""
import numpy as np

DEFAULT_BOX_WEIGHTS = 'runs/pose/basket_white_box_2pt/weights/best.pt'
DEFAULT_BASKET_WEIGHTS = 'runs/pose/basket_4pt/weights/best.pt'


def validate_model(model, role=None):
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    shape = list(model.model.model[-1].kpt_shape)
    if role is not None:
        expected = {'white_box': [2, 3], 'basket': [4, 3]}[role]
        if model.task != 'pose' or names != {0: role} or shape != expected:
            raise ValueError(f'Expected dedicated {role} pose model: names={{0: {role!r}}}, kpt_shape={expected}')
        return
    legacy = shape == [2, 3] and len(names) == 1
    basket = shape == [4, 3] and names == {0: 'white_box', 1: 'basket'}
    if model.task != 'pose' or not (legacy or basket):
        raise ValueError('Expected a single-class two-endpoint model or white_box/basket four-keypoint model.')


def prediction_arrays(result, class_id=0, point_count=2):
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    if not len(boxes):
        return boxes, np.empty((0, point_count, 3)), scores, classes
    if result.keypoints is None:
        raise ValueError('Model output has no keypoints.')
    points = result.keypoints.data.cpu().numpy()
    if (points.shape not in ((len(boxes), 2, 3), (len(boxes), 4, 3))
            or not all(np.isfinite(a).all() for a in (boxes, scores, points))):
        raise ValueError('Invalid pose output.')
    selected = classes == class_id
    if selected.any() and points.shape[1] < point_count:
        raise ValueError('Not enough keypoints for selected class.')
    return boxes[selected], points[selected, :point_count].reshape(-1, point_count, 3), scores[selected], classes[selected]


def draw_baskets(canvas, points, image_shape, point_conf=.5, y_offset=0):
    """Draw only observed corners/edges; never feed baskets to box counting."""
    import cv2
    h, w = image_shape[:2]
    color = (230, 170, 60)
    for corners in points:
        valid = (np.isfinite(corners).all(1) & (corners[:, 2] >= point_conf)
                 & (corners[:, 0] >= 0) & (corners[:, 0] < w)
                 & (corners[:, 1] >= 0) & (corners[:, 1] < h))
        xy = np.rint(corners[:, :2]).astype(int)
        xy[:, 1] += y_offset
        for i in range(4):
            j = (i + 1) % 4
            if valid[i] and valid[j]:
                cv2.line(canvas, tuple(xy[i]), tuple(xy[j]), color, 2, cv2.LINE_AA)
            if valid[i]:
                cv2.circle(canvas, tuple(xy[i]), 4, color, -1, cv2.LINE_AA)
