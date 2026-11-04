#!/usr/bin/env python3
"""离线检查一张图片：检测、分格、首盒和坐标侧栏；不打开相机，不发送运动指令。"""

import argparse
from dataclasses import replace
import json
from scripts.pose_schema import DEFAULT_BOX_WEIGHTS, DEFAULT_BASKET_WEIGHTS
from scripts.check_env import configure_local_cache, project_path, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="图片路径；相对路径以本项目为基准")
    parser.add_argument("--weights", default=DEFAULT_BOX_WEIGHTS)
    parser.add_argument("--basket-weights", default=DEFAULT_BASKET_WEIGHTS)
    parser.add_argument("--device", default="cpu", help="cpu 或 CUDA 设备编号，例如 0；不自动回退")
    parser.add_argument("--intrinsics", help="仅用于与该图片匹配的彩色流内参 JSON")
    parser.add_argument("--plane", help="已测定平面 JSON，含 normal 和 offset_mm；需要 --intrinsics")
    parser.add_argument("--output", default="reports/offline_preview", help="新输出目录，拒绝覆盖")
    args = parser.parse_args(argv)
    if args.plane and not args.intrinsics:
        parser.error("--plane 需要 --intrinsics")
    configure_local_cache()

    import cv2
    import torch
    from ultralytics import YOLO
    import detect_d405 as live
    from scripts.first_box import locate_first_boxes, draw_first_boxes
    from scripts.first_box_panel import append_first_box_panel
    from scripts.basket_grid import BasketGridCounter as GridCounter, grid_report
    from scripts.video_postprocess import VideoPostprocessor

    source, weights, output = map(project_path, (args.source, args.weights, args.output))
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请另选 --output：{output}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"图片无法读取：{source}")
    device = "cpu" if args.device == "cpu" else int(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前不可用。离线检查可显式使用 --device cpu。")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = YOLO(str(weights))
    live.validate_model(model, role="white_box")
    basket_weights = project_path(args.basket_weights)
    if not basket_weights.is_file():
        raise FileNotFoundError(str(basket_weights))
    basket_model = YOLO(str(basket_weights))
    live.validate_model(basket_model, role="basket")
    options = dict(imgsz=live.IMGSZ, device=device, conf=live.CONF, iou=live.NMS_IOU,
                   max_det=live.MAX_DET, quantize=32, rect=True, nms=True,
                   augment=False, save=False, verbose=False)
    result = model.predict(image, **options)[0]
    basket_result = basket_model.predict(image, **options)[0]
    boxes, points, scores, classes = live.prediction_arrays(result)
    filtered = VideoPostprocessor(dedup_config=live.DEDUP, temporal_config=live.TEMPORAL,
                                 dedup_enabled=live.DEDUP_ENABLED).update(
        boxes, points, scores, classes, image.shape, sequence=1, timestamp=1.)
    # 静态图片没有跨帧证据：只为检查布局将确认门槛改为单帧，不修改实时配置。
    grid_config = replace(live.GRID, confirmation_frames=1, confirmation_seconds=0.)
    _, basket_points, _, _ = live.prediction_arrays(basket_result, class_id=0, point_count=4)
    grid = GridCounter(grid_config).update(boxes, points, filtered.keep_indices, filtered.track_ids,
                                         image.shape, sequence=1, timestamp=1., basket_points=basket_points)
    intrinsics = json.loads(project_path(args.intrinsics).read_text()) if args.intrinsics else None
    plane = json.loads(project_path(args.plane).read_text()) if args.plane else None
    first_boxes = locate_first_boxes(grid, boxes, points, intrinsics, image.shape,
        point_conf=live.POINT_CONF, plane=plane)
    canvas, counts = live.render_frame(image, boxes, filtered.points, filtered.keep_indices,
                                      grid=grid, removed=len(filtered.suppressed))
    cv2.rectangle(canvas, (0, 115), (canvas.shape[1], live.HEADER_HEIGHT), (30, 37, 42), -1)
    cv2.putText(canvas, "OFFLINE IMAGE | single-frame grouping, not live confirmation", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (90, 200, 255), 1, cv2.LINE_AA)
    draw_first_boxes(canvas, first_boxes, live.HEADER_HEIGHT)
    canvas, _ = append_first_box_panel(canvas, first_boxes, state="OFFLINE", grid=grid, show_counts=True)
    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "preview.jpg"), canvas):
        raise OSError("无法保存离线可视化")
    report = dict(status="offline_image", source=str(source), weights=str(weights), basket_weights=str(basket_weights), device=device,
        coordinate_system="original_image_pixels; camera geometry only with supplied matching intrinsics",
        single_frame_grouping=True, live_confirmation_tested=False, robot_commands_sent=False,
        plane=plane, intrinsics=intrinsics, counts=counts, grid=grid_report(grid), cells=first_boxes,
        boxes_xyxy=boxes.tolist(), keypoints_xyc=points.tolist(), confidence=scores.tolist(),
        kept_indices=[int(i) for i in filtered.keep_indices])
    write_report(output / "predictions.json", report)
    print(json.dumps(dict(status=report["status"], output=str(output), counts=counts), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
