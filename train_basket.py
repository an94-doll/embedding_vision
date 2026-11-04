#!/usr/bin/env python3
"""单类别铁篮四角训练；在 yolo-pose 环境运行 python train_basket.py。"""
from pathlib import Path

import yaml

from scripts.check_env import configure_local_cache

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'box_in_basket/basket_dataset/data.yaml'
WEIGHTS = ROOT / 'weights/yolov8s-pose.pt'


def main():
    # 只检查任务配置，不重复检查或改写标签。
    schema = yaml.safe_load(DATA.read_text(encoding='utf-8-sig'))
    names = schema.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (names != {0: 'basket'} or schema.get('kpt_shape') != [4, 3]
            or schema.get('flip_idx') != [1, 0, 3, 2]):
        raise ValueError('需要单类 basket=0、四点 [4,3]、flip_idx=[1,0,3,2] 的数据配置')
    if not schema.get('train') or not schema.get('val'):
        raise ValueError('数据配置必须包含 train 和 val')
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    configure_local_cache()
    from ultralytics import YOLO

    # 从 COCO 预训练权重开始新实验，不从联合模型续训。
    model = YOLO(str(WEIGHTS))
    return model.train(
        data=str(DATA),
        epochs=250,
        patience=80,
        batch=8,
        imgsz=1280,
        device=0,  # GPU 0；不自动回退到 CPU。
        workers=2,
        seed=42,
        deterministic=True,
        project=str(ROOT / 'runs/pose'),
        name='basket_4pt',
        exist_ok=False,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        rect=False,
        multi_scale=0.0,
        degrees=10.0,
        translate=0.05,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,  # YAML 将左上/右上、右下/左下成对交换。
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.25,
        val=True,
        plots=True,
    )


if __name__ == '__main__':
    main()
