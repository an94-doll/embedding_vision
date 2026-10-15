#!/usr/bin/env python3
"""Export front/back detections, P1-P5 coordinates and image/video overlays."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from pose_common import (ROOT, DEFAULT_WEIGHTS, IMAGE_SUFFIXES, VIDEO_SUFFIXES, check_imgsz,
                         detections_from_result, draw_detections, save_json,
                         select_device, validate_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="图片、图片目录或视频文件")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/predict/front_back_pose")
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--point-conf", type=float, default=0.25, help="仅控制绘图，JSON 保留所有五点")
    parser.add_argument("--iou", type=float, default=0.7, help="检测框 NMS 阈值")
    parser.add_argument("--fps", type=float, default=None, help="视频输出 FPS；默认读取原视频，无法读取时用 25")
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    writers, video_shapes, frame_counts = {}, {}, Counter()
    try:
        check_imgsz(args.imgsz)
        if not (0 <= args.conf <= 1 and 0 <= args.point_conf <= 1 and 0 < args.iou <= 1):
            raise ValueError("置信度或 IoU 阈值无效")
        if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
            raise ValueError("fps 必须是正的有限数值")
        source, output = args.source.resolve(), args.output.resolve()
        if not source.exists() or not args.weights.is_file():
            raise ValueError("输入路径或模型权重不存在")
        if source.is_file() and source.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise ValueError("不支持的输入文件类型")
        if output.exists():
            raise ValueError("输出目录已存在，请指定新的 --output：" + str(output))
        if source.is_dir() and (output == source or source in output.parents):
            raise ValueError("输出目录不能位于输入图片目录内")
        import cv2
        from ultralytics import YOLO

        device = select_device(args.device)
        model = YOLO(str(args.weights.resolve()))
        validate_model(model)
        output.mkdir(parents=True)
        if not args.no_overlays:
            (output / "overlays").mkdir()
        stream = model.predict(source=str(source), imgsz=args.imgsz, device=device,
                               conf=args.conf, iou=args.iou, stream=True, save=False,
                               augment=False, verbose=False)
        total_frames = total_detections = no_detection_frames = 0
        class_counts = Counter()
        output_names = {}
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in stream:
                path = Path(result.path).resolve()
                is_video = path.suffix.lower() in VIDEO_SUFFIXES
                frame_index = frame_counts[str(path)]
                frame_counts[str(path)] += 1
                detections = detections_from_result(result)
                height, width = result.orig_img.shape[:2]
                entry = {"source": str(path), "type": "video" if is_video else "image",
                         "frame_index": frame_index if is_video else None,
                         "width": width, "height": height, "coordinate_system": "original_pixels",
                         "detections": detections}
                handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                total_frames += 1
                total_detections += len(detections)
                no_detection_frames += int(not detections)
                class_counts.update(d["class_name"] for d in detections)
                if args.no_overlays:
                    continue
                canvas = draw_detections(result.orig_img, detections, args.point_conf)
                if str(path) not in output_names:
                    # Full original filename + suffix avoids same-stem image collisions.
                    candidate = path.name + (".mp4" if is_video else ".jpg")
                    if candidate in output_names.values():
                        candidate = "{:04d}_{}".format(len(output_names), candidate)
                    output_names[str(path)] = candidate
                destination = output / "overlays" / output_names[str(path)]
                if is_video:
                    if str(path) not in writers:
                        fps = args.fps
                        if fps is None:
                            capture = cv2.VideoCapture(str(path))
                            fps = capture.get(cv2.CAP_PROP_FPS)
                            capture.release()
                            if not math.isfinite(fps) or fps <= 0:
                                fps = 25.0
                        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (width, height))
                        if not writer.isOpened():
                            writer.release()
                            raise OSError("无法创建视频输出：" + str(destination))
                        writers[str(path)] = writer
                        video_shapes[str(path)] = (width, height)
                    if video_shapes[str(path)] != (width, height):
                        raise ValueError("视频帧尺寸发生变化")
                    writers[str(path)].write(canvas)
                elif not cv2.imwrite(str(destination), canvas):
                    raise OSError("无法写入图片：" + str(destination))
        if total_frames == 0:
            raise ValueError("输入中没有可推理的图片或视频帧")
        save_json(output / "prediction_summary.json", {
            "weights": str(args.weights.resolve()), "source": str(source), "device": device,
            "imgsz": args.imgsz, "confidence_threshold": args.conf,
            "overlay_point_confidence_threshold": args.point_conf,
            "frames_or_images": total_frames, "detections": total_detections,
            "no_detection_frames_or_images": no_detection_frames,
            "class_counts": {name: class_counts[name] for name in ("front", "back")},
            "keypoint_names": ["P1", "P2", "P3", "P4", "P5"],
            "coordinate_system": "original_pixels", "video_audio_preserved": False,
        })
        print("完成 {} 张图片/帧，{} 个目标；输出：{}".format(total_frames, total_detections, output))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    finally:
        for writer in writers.values():
            writer.release()


if __name__ == "__main__":
    main()
