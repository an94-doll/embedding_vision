"""训练正反面分类与五点 Pose，参数直接在下方修改。"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    model = YOLO("/home/wangjr/projects/yolo_pose/weights/yolov8n-pose.pt")

    model.train(
        data=str(ROOT / "dataset/data.yaml"),
        epochs=100,
        patience=50,
        batch=8,
        imgsz=736,
        device=0,  # 使用 GPU 0；使用 CPU 时改为 "cpu"。
        workers=2,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs/pose"),
        name="front_back_pose",
        exist_ok=False,
        degrees=20.0,
        translate=0.05,
        scale=0.2,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        val=True,
        plots=True,
    )
