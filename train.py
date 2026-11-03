"""训练篮内白盒底边两端点；在 Conda 环境中运行本文件，参数直接在下方修改。"""

from pathlib import Path
from scripts.check_env import configure_local_cache

configure_local_cache()
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    # 使用 COCO 预训练权重开始新实验，数据配置负责指定单类和两个关键点。
    model = YOLO(str(ROOT / "weights/yolov8s-pose.pt"))

    model.train(
        data=str(ROOT / "box_in_basket/dataset/data.yaml"),
        epochs=250,
        patience=80,
        batch=8,
        imgsz=1280,
        device=0,  # 使用 GPU 0；使用 CPU 时改为 "cpu"。
        workers=2,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs/pose"),
        name="white_box_basket_4pt",
        exist_ok=False,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,

        # 保留1280输入的小目标细节；rect=False 使下方轻量 Mosaic 生效。
        rect=False,
        multi_scale=0.0,
        degrees=10.0,
        translate=0.05,
        scale=0.25,
        shear=0.0,
        perspective=0.0,

        # data.yaml 的 flip_idx=[1, 0] 会在水平翻转时交换左右端点。
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        close_mosaic=100,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,

        # 适度模拟白盒与金属篮的颜色、曝光变化。
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.25,
        val=True,
        plots=True,
    )
