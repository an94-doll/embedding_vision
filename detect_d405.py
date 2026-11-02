#!/usr/bin/env python3
"""D405白盒端点与自动分格。顶部配置；Q/Esc退出，D切换去重，R重建格位。"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.check_env import configure_local_cache, write_report
from scripts.d405_camera import CaptureConfig, D405Capture, Rate
from scripts.grid_counter import GridConfig, GridCounter
from scripts.first_box_panel import append_first_box_panel, PANEL_WIDTH
from scripts.first_box import locate_first_boxes, draw_first_boxes
from scripts.postprocess import DedupConfig
from scripts.video_postprocess import TemporalConfig, VideoPostprocessor


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/pose/basket_white_box_2pt/weights/best.pt"
DEVICE = 0                             # 默认GPU 0；显式改成 "cpu" 可做CPU调试。
IMGSZ = 1280
CONF = 0.25
POINT_CONF = 0.50
NMS_IOU = 0.70
MAX_DET = 300
CAMERA = CaptureConfig(width=1280, height=720, fps=30, serial=None, timeout=5.0)
DEDUP_ENABLED = True
DEDUP = DedupConfig(
    point_confidence=POINT_CONF,
    endpoint_distance_px=1.5,            # 取像素与相对长度阈值中更严格的一个。
    endpoint_distance_ratio=0.015,
    min_box_iou=0.45,                   # 已经过模型NMS；额外判据不是再提高NMS的IoU。
    min_box_size_ratio=0.75,
    max_box_center_ratio=0.20,
    max_angle_deg=2.0,
    max_length_ratio_error=0.03,
    border_margin_px=3.0,               # 接近画面边缘的目标不做额外抑制。
    min_edge_length_px=12.0,
)
TEMPORAL = TemporalConfig(
    tracking_enabled=True,
    smoothing_enabled=True,
    counting_enabled=True,
    point_confidence=POINT_CONF,
    max_gap_seconds=0.35,               # 短暂漏检只保留身份，不绘制旧端点。
    count_rise_seconds=0.15,
    count_fall_seconds=0.30,            # 防抖计数独立显示，不影响当前检测数。
)
GRID = GridConfig(
    enabled=True,
    max_per_cell=10,
    point_confidence=POINT_CONF,
    confirmation_frames=3,            # 仅连续确认的盒子参与分格，其他检测仍计入Current。
    confirmation_seconds=0.15,
    allow_box_fallback=False,          # 双端点不完整时不凭检测框中心建立格子。
    history_seconds=1.0,               # 失去观测的格位短时保留关联，随后自动清理。
    show_empty_cells=False,
    stack_angle_deg=0.0,               # 0=画面竖直堆叠；正角向画面右侧倾斜。
    column_distance_ratio=0.55,
    layer_pitch_ratio=0.22,            # 典型层距/底边长度的初始参考，可按实际摆放调整。
    gap_factor=2.2,                     # 相邻间距 > 典型层距×系数时，优先按间隔切分。
    gap_confirm_seconds=0.10,          # 新间隔短时确认；满十个的切分立即执行。
)
# Optional measured plane in the COLOR camera frame: normal dot X_mm = offset_mm.
# Leave None until measured; never substitute the approximate 150 mm viewing distance.
FIRST_BOX_PLANE = None
FIRST_BOX_PATH = ROOT / "reports/d405_first_boxes_latest.json"
SHOW_BOXES = False
WINDOW = "D405 | box grid | Q/Esc quit | D dedup | R reset grid"
REPORT_PATH = ROOT / "reports/d405_detection_latest.json"
HEADER_HEIGHT = 152
STALE_AFTER_SECONDS = 0.50              # 仅表示采集停顿，不筛选任何盒子。


def check_display():
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("需要在有桌面显示的本机终端运行此窗口程序。")
    if any(line.strip().startswith("GUI:") and line.split(":", 1)[1].strip() == "NONE"
           for line in cv2.getBuildInformation().splitlines()):
        raise RuntimeError("当前OpenCV没有GUI支持，请使用带窗口支持的opencv-python。")
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if not Path(os.environ.get("QT_QPA_FONTDIR", "/nonexistent")).is_dir() and font_dir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(font_dir)


def load_model():
    import torch
    from ultralytics import YOLO

    if not WEIGHTS.is_file():
        raise FileNotFoundError("权重不存在：" + str(WEIGHTS))
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；请检查GPU环境，或在顶部显式设置 DEVICE="cpu"。')
    model = YOLO(str(WEIGHTS.resolve()))
    if model.task != "pose" or list(model.model.model[-1].kpt_shape) != [2, 3] or len(model.names) != 1:
        raise ValueError("需要单类别、两个端点的Pose权重。")
    options = dict(imgsz=IMGSZ, device=DEVICE, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                   quantize=32, rect=True, nms=True, augment=False, save=False, verbose=False)
    model.predict(np.zeros((CAMERA.height, CAMERA.width, 3), dtype=np.uint8), **options)
    if DEVICE != "cpu":
        torch.cuda.synchronize(DEVICE)
    return model, options


def prediction_arrays(result):
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    if len(boxes) and result.keypoints is None:
        raise ValueError("模型输出缺少端点。")
    points = np.empty((0, 2, 3), dtype=float) if not len(boxes) else result.keypoints.data.cpu().numpy()
    if points.shape != (len(boxes), 2, 3) or not all(np.isfinite(a).all() for a in (boxes, scores, points)):
        raise ValueError("模型输出包含非有限值或不符合两端点格式。")
    return boxes, points, scores, classes


def draw_grid(frame, grid):
    """Cell coordinates/counts only; individual tracking identities stay internal."""
    height, width = frame.shape[:2]
    colors = {"gap": (100, 225, 100), "count": (0, 175, 255),
              "start": (245, 195, 100), "history": (140, 140, 140)}
    codes = {"gap": "G", "count": "C", "start": "S", "history": "E"}
    for cell in grid.cells:
        if cell.count == 0 and not GRID.show_empty_cells:
            continue
        x1, y1, x2, y2 = np.rint(cell.bounds).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        color = colors.get(cell.source, colors["start"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if cell.count == 0 else 2)
        lines = (f"({cell.row},{cell.col})", f"{cell.count}/{GRID.max_per_cell} {codes.get(cell.source, 'S')}")
        tx, ty = max(0, min(width - 80, x1 + 3)), max(13, min(height - 22, y1 + 13))
        tw = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0] for line in lines)
        cv2.rectangle(frame, (tx - 1, ty - 12), (tx + tw + 3, ty + 19), (25, 25, 25), -1)
        for line_index, line in enumerate(lines):
            cv2.putText(frame, line, (tx + 1, ty + line_index * 16), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, color, 1, cv2.LINE_AA)


def render_frame(image, boxes, points, keep_indices, *, grid=None, stable_count=None, lost_count=0,
                 removed=0, enabled=True, metrics=None, show_boxes=SHOW_BOXES):
    """Draw only current detections; stable count never creates or hides an observation."""
    metrics = metrics or {}
    frame = image.copy()
    height, width = frame.shape[:2]
    order = tuple(keep_indices)
    edges = 0
    if grid is not None:
        draw_grid(frame, grid)
    for i in order:
        keypoints = points[i]
        valid = (np.isfinite(keypoints).all(1) & (keypoints[:, 2] >= POINT_CONF)
                 & (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width)
                 & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height))
        both = bool(valid.all()) and float(np.linalg.norm(keypoints[1, :2] - keypoints[0, :2])) > 1e-6
        color = (235, 235, 235) if both else (0, 210, 255)
        if both:
            cv2.line(frame, tuple(np.rint(keypoints[0, :2]).astype(int)),
                     tuple(np.rint(keypoints[1, :2]).astype(int)), (255, 220, 70), 2, cv2.LINE_AA)
            edges += 1
        for side, good in enumerate(valid):
            if good:
                xy = tuple(np.rint(keypoints[side, :2]).astype(int))
                cv2.circle(frame, xy, 4, ((80, 230, 100), (0, 170, 255))[side], -1, cv2.LINE_AA)
                cv2.circle(frame, xy, 4, (20, 20, 20), 1, cv2.LINE_AA)
        if show_boxes:
            cv2.rectangle(frame, tuple(np.rint(boxes[i, :2]).astype(int)),
                          tuple(np.rint(boxes[i, 2:]).astype(int)), color, 1)
    canvas = cv2.copyMakeBorder(frame, HEADER_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 37, 42))
    counts = dict(boxes=len(order), stable=len(order) if stable_count is None else int(stable_count),
                  lost_tracks=int(lost_count), edges=edges, removed=removed, partial=len(order) - edges)
    if grid is not None:
        occupied = any(cell.count > 0 for cell in grid.cells)
        show_grid = occupied or (GRID.show_empty_cells and bool(grid.cells))
        counts.update(rows=grid.rows if show_grid else 0, cols=grid.cols if show_grid else 0,
                      cells=sum(cell.count > 0 or GRID.show_empty_cells for cell in grid.cells),
                      assigned=len(grid.assignments), unassigned=len(grid.unassigned_indices))
    if grid is None:
        grid_status = "Grid: OFF"
    elif not show_grid:
        grid_status = f"Grid: waiting for confirmed boxes | Not grouped: {len(grid.unassigned_indices)}"
    else:
        grid_status = f"Grid: {grid.rows} rows x {grid.cols} cols | Confirmed/grouped: {len(grid.assignments)} | Not grouped: {len(grid.unassigned_indices)}"
    lines = [f"Current: {counts['boxes']} | Stable: {counts['stable']} | Edges: {edges} | Partial: {counts['partial']} | Removed: {removed}",
             grid_status,
             f"Camera {metrics.get('camera_fps', 0):.1f} FPS | Inference {metrics.get('inference_fps', 0):.1f} FPS | Predict {metrics.get('predict_ms', 0):.1f} ms | Post: {metrics.get('post_ms', 0):.2f} ms",
             f"Frame age (host RX to drawing): {metrics.get('frame_age_ms', 0):.1f} ms | Skipped capture frames: {metrics.get('skipped', 0)}",
             "Cell (row,col) starts at top-left (0,0) | G=gap, C=count, S=start, E=empty history",
             f"Q/Esc: quit | D: extra dedup {'ON' if enabled else 'OFF'} (next frame) | R: reset inferred grid | Stable=count estimate"]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 21 + index * 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas, counts


def render_waiting_frame(image, age_ms):
    """A stalled camera has no new observations; don't redraw historical endpoints."""
    canvas = cv2.copyMakeBorder(image, HEADER_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 37, 42))
    for index, line in enumerate(("STALE CAMERA FRAME | Waiting for a new frame",
                                  f"Last frame received {age_ms:.0f} ms ago | No current endpoint overlay",
                                  "Tracking, smoothing and count state have not advanced | Q/Esc: quit")):
        cv2.putText(canvas, line, (10, 25 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (90, 150, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    configure_local_cache()
    capture = None
    window = False
    report = dict(status="starting", weights=str(WEIGHTS.resolve()), device=DEVICE, imgsz=IMGSZ,
                  confidence=CONF, point_confidence=POINT_CONF, nms_iou=NMS_IOU,
                  dedup_settings=vars(DEDUP), temporal_settings=vars(TEMPORAL), grid_settings=vars(GRID),
                  processed_frames=0, removed_detections=0, skipped_capture_frames=0,
                  ids="conservative cross-frame IDs; ambiguous associations start new IDs",
                  stable_count="time-debounced current count; not a physical ground-truth count")
    enabled = DEDUP_ENABLED
    packet, waiting_redraw = None, 0.0
    panel_page = 0
    panel_base, panel_rows = None, []
    postprocessor = VideoPostprocessor(dedup_config=DEDUP, temporal_config=TEMPORAL,
                                      dedup_enabled=DEDUP_ENABLED)
    grid_counter = GridCounter(GRID)
    sequence, skipped, processed, removed_total = 0, 0, 0, 0
    inference_rate = Rate()
    predict_ms = 0.0
    try:
        check_display()
        model, options = load_model()
        capture = D405Capture(CAMERA)
        capture.start()
        report.update(status="running", camera=capture.info)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window = True
        cv2.resizeWindow(WINDOW, CAMERA.width + PANEL_WIDTH, CAMERA.height + HEADER_HEIGHT)
        print("D405检测已启动：仅对连续确认且双端点可用的盒子分格，间隔优先、十个兜底。Q/Esc退出，D切换额外去重，R重建格位。", flush=True)
        while True:
            fresh = capture.latest(sequence)
            if fresh is not None:
                skipped += max(0, fresh["sequence"] - sequence - 1)
                sequence, packet = fresh["sequence"], fresh
                started = time.perf_counter()
                result = model.predict(packet["image"], **options)[0]
                boxes, points, scores, classes = prediction_arrays(result)  # Waits for CPU retrieval.
                predict_ms = (time.perf_counter() - started) * 1000
                inference_rate.add()
                processed += 1
                filtered = postprocessor.update(boxes, points, scores, classes, packet["image"].shape,
                                                sequence=sequence, timestamp=packet["received_at"],
                                                dedup_enabled=enabled)
                grid = grid_counter.update(boxes, points, filtered.keep_indices, filtered.track_ids,
                                           packet["image"].shape, sequence=sequence, timestamp=packet["received_at"])
                removed_total += len(filtered.suppressed)
                metrics = dict(camera_fps=capture.rate.fps(), inference_fps=inference_rate.fps(),
                               predict_ms=predict_ms, dedup_ms=filtered.dedup_ms,
                               grid_ms=grid.processing_ms,
                               post_ms=filtered.dedup_ms + filtered.temporal_ms + grid.processing_ms, skipped=skipped,
                               frame_age_ms=(time.monotonic() - packet["received_at"]) * 1000)
                canvas, counts = render_frame(packet["image"], boxes, filtered.points, filtered.keep_indices,
                                               grid=grid if GRID.enabled else None, stable_count=filtered.stable_count,
                                               lost_count=filtered.lost_count, removed=len(filtered.suppressed),
                                               enabled=enabled, metrics=metrics)
                first_boxes = locate_first_boxes(
                    grid, boxes, points, capture.info.get("intrinsics"), packet["image"].shape,
                    angle_deg=GRID.stack_angle_deg, point_conf=POINT_CONF,
                    plane=FIRST_BOX_PLANE, lost_count=filtered.lost_count)
                draw_first_boxes(canvas, first_boxes, HEADER_HEIGHT)
                panel_base, panel_rows = canvas, first_boxes
                canvas, panel_page = append_first_box_panel(panel_base, panel_rows, panel_page)
                first_report = dict(status="current", frame_number=packet["frame_number"],
                    sequence=sequence, received_at_monotonic=packet["received_at"],
                    device_timestamp_ms=packet["device_timestamp_ms"], camera=capture.info,
                    coordinate_system="color_camera_x_right_y_down_z_forward",
                    expected_edge_width_mm=30., nominal_pitch_mm=7., plane=FIRST_BOX_PLANE,
                    cells=first_boxes)
                write_report(FIRST_BOX_PATH, first_report)
                report["last_first_boxes"] = first_report
                cv2.imshow(WINDOW, canvas)                # Always this prediction's original packet.
                report.update(last_counts=counts, last_frame_number=packet["frame_number"],
                              last_metrics=metrics,
                              last_grid=dict(rows=grid.rows, cols=grid.cols,
                                             unassigned_indices=list(grid.unassigned_indices),
                                             cells=[dict(row=cell.row, col=cell.col, count=cell.count,
                                                         indices=list(cell.indices), source=cell.source,
                                                         bounds=[float(v) for v in cell.bounds])
                                                    for cell in grid.cells]))
            elif packet is not None:
                now = time.monotonic()
                age = now - packet["received_at"]
                if age > STALE_AFTER_SECONDS and now - waiting_redraw >= 0.10:
                    panel_base = render_waiting_frame(packet["image"], age * 1000)
                    panel_rows = []
                    canvas, panel_page = append_first_box_panel(panel_base, [], panel_page, 'STALE')
                    cv2.imshow(WINDOW, canvas)
                    write_report(FIRST_BOX_PATH, dict(status="stale", cells=[]))
                    waiting_redraw = now
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:            # A closed native window may already have been destroyed.
                break
            if key in (ord("d"), ord("D")):
                # Process each camera packet exactly once. No state update on a redraw.
                enabled = not enabled
            if key in (ord('n'), ord('p')) and panel_base is not None:
                panel_page += 1 if key == ord('n') else -1
                canvas, panel_page = append_first_box_panel(panel_base, panel_rows, panel_page,
                    'STALE' if packet is not None and time.monotonic() - packet['received_at'] > STALE_AFTER_SECONDS else 'LIVE')
                cv2.imshow(WINDOW, canvas)
            if key in (ord("r"), ord("R")):
                grid_counter.reset()
        report["status"] = "stopped"
    except KeyboardInterrupt:
        report["status"] = "stopped"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print("检测程序停止：" + report["error"], file=sys.stderr, flush=True)
    finally:
        if capture is not None:
            try:
                capture.stop()
            except Exception as error:
                report.update(status="failed", camera_stop_error=str(error))
        if window:
            try:
                cv2.destroyAllWindows()
            except cv2.error as error:
                report.update(status="failed", window_close_error=str(error))
        report.update(processed_frames=processed, removed_detections=removed_total,
                      skipped_capture_frames=skipped, dedup_enabled_at_exit=enabled)
        write_report(FIRST_BOX_PATH, dict(status=report["status"], cells=[]))
        write_report(REPORT_PATH, report)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
