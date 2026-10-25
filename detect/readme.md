# 包埋盒长边两点检测

只保留最新一批 **277 张**人工修正数据。默认模型为 **第 127 轮 base**，标注为同一长边的左、右两个明显转角。

```text
detect/
├── data/data_last/
│   ├── color.mp4                  原始视频
│   ├── metadata.json、frames.csv  采集信息
│   └── picture/
│       ├── raw/                   每秒 2 张的原始图片
│       └── work/
│           ├── images/train/      未画线图片，与标签一一对应
│           ├── labels/train/      最新 YOLO Pose 两点标签
│           ├── previews/          当前标注预览图
│           ├── batch/             标注 JSON、人工复核状态、修改记录与编辑服务
│           ├── annotation_editor.html  浏览器标注页面
│           └── start_annotation_editor.sh  编辑器启动脚本
└── yolo/
    ├── weights/base.pt            默认权重：第 127 轮
    ├── weights/base.json          权重来源、校验值和参数
    ├── scripts/                   训练、检测、检查工具
    ├── training/                  最新批次的训练快照、loss、报告及测试图
    ├── runs/pose/                 最新 200 轮结果，保留 epoch200.pt
    ├── reports/                   环境检查和清理记录
    └── environment.yml、requirements.lock.txt  Python 环境与依赖
```

**查看/修改标注：** 在本目录运行 `bash data/data_last/picture/work/start_annotation_editor.sh`，浏览器打开 `http://127.0.0.1:8766/`。默认只显示两点及连线，可选择显示框。

**看模型效果：** 打开 [`yolo/training/cassette_reviewed277_total200_20261025_194050/test_predictions/index.html`](yolo/training/cassette_reviewed277_total200_20261025_194050/test_predictions/index.html)，比较第 127、200 轮对五张独立测试图的预测。

**训练数据划分：** 192 张训练、50 张验证、5 张独立测试、30 张间隔帧。五张测试图（31、83、134、190、259）不参与训练；划分快照位于 `yolo/training/cassette_reviewed277_100_20261025_192844/dataset/`，100/200 轮共用。

**下载：** 图片、视频和权重使用 Git LFS；克隆后运行 `git lfs pull`。部分历史配置含原机器绝对路径，新训练使用 `yolo/scripts/train_reviewed_two_points.py --prepare` 生成当前路径配置。
