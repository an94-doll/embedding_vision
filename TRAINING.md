# 正反面与五点 Pose 联合训练

这套脚本使用一个两类别 YOLO Pose 模型。每个目标输出检测框、类别及检测分数，以及 P1–P5 的坐标和关键点置信度。

- 类别：`0=front`（正面）、`1=back`（反面）。
- 点序：以物体顶部朝上观察，P1 左上、P2 左下、P3 底边中点、P4 右下、P5 右上。
- 数据：`dataset/data.yaml`，`kpt_shape: [5, 3]`。
- 环境：现有 Conda 环境名为 `yolo-pose`。

**本次只编写脚本并做静态检查，未启动训练、评估或推理。以下命令供之后手动运行。**

## 训练

`train.py` 只保留模型加载和 `model.train(...)`，所有训练参数直接在文件中修改。

```bash
conda activate yolo-pose
cd /home/wangjr/projects/yolo_pose_box
python train.py
```

脚本默认使用 GPU 0（`device=0`）；使用 CPU 时，将代码中的参数改为 `device="cpu"`。
预训练权重直接填写为已有文件 `/home/wangjr/projects/yolo_pose/weights/yolov8n-pose.pt`。
数据配置为 `dataset/data.yaml`，包含两个类别和五个关键点。

默认 100 轮、早停耐心值 30、batch 8、随机种子 42。采用适度旋转（±20°）、平移、缩放及颜色增强，关闭水平/垂直翻转、Mosaic 和 MixUp。
原图仍为 720×720，模型输入尺寸为 736（32 的倍数），框架会同步变换训练图片和标签。
训练使用验证集，不自动评估测试集。

首次训练输出为 `runs/pose/front_back_pose/`，包含 `weights/best.pt`、`weights/last.pt`、`args.yaml`、`results.csv` 和训练曲线。
同名目录已存在时，Ultralytics 会创建带编号的新目录；评估和推理需指定实际的 `best.pt`。

如需从未完成训练的检查点续训，将脚本中的模型加载和训练调用替换为：

```python
model = YOLO(str(ROOT / "runs/pose/front_back_pose/weights/last.pt"))
model.train(resume=True, device=0)
```

续训使用检查点中的训练配置。正常结束后被清理的权重可能不再包含优化器状态，不适合断点续训。

## 评估

先使用验证集检查效果：

```bash
python evaluate.py \
  --weights runs/pose/front_back_pose/weights/best.pt \
  --data dataset/data.yaml --split val --device 0
```

模型及参数确定后再使用测试集：

```bash
python evaluate.py \
  --weights runs/pose/front_back_pose/weights/best.pt \
  --data dataset/data.yaml --split test --device 0
```

默认 `--split val`，不会隐式回退或自动使用 test。输出位于 `runs/evaluate/val_front_back/` 或 `test_front_back/`，同名目录存在时递增名称。

输出包括标准 YOLO 检测和 Pose 指标、曲线及混淆图，并另存：

- `joint_metrics.json`：补充的正反面和点位指标。
- `per_image.csv`：每图目标数、匹配数、识别正确数和点位误差。
- `per_instance.csv`：每个真实目标的类别预测、匹配 IoU 和置信度。
- `per_keypoint.csv`：每个有效真实点的像素误差与归一化误差。

标准 mAP 使用较低的候选框阈值（`--ap-conf 0.001`），补充识别/点位指标使用实际检测阈值（`--conf 0.25`）。
补充指标按检测框 IoU≥0.5 降序进行一对一匹配，**匹配不要求类别相同**，因此正反面预测错误不会被排除。可通过 `--match-iou` 修改阈值。

指标含义：

- `front_back_accuracy_matched`：匹配目标中正反面正确的比例，同时报告漏检数。
- `correct_face_detection_rate_all_gt`：以所有真实目标为分母，漏检视为失败。
- 平均/中位数像素误差：匹配目标中真实标注可见点的误差，坐标按原图尺寸计算；漏检不纳入均值。
- PCK@0.05：误差≤真实框对角线的 5%。分别提供仅匹配点和全部真实点两个分母；全部真实点指标将漏检视为失败。
- 点位误差不按预测关键点置信度筛选，并包含类别预测错误的匹配目标，避免只保留容易样本。

五点为自定义点，标准 Pose mAP/OKS 的解释依赖框架对自定义点使用的尺度设置，建议同时查看原图像素误差和 PCK。
当前数据是连续帧随机划分，相邻帧可能跨集合；独立拍摄序列更适合评估泛化表现。

## 图片或视频推理

```bash
# 单张图片
python predict.py --source images/06/frame_0031.jpg --device 0 --output runs/predict/example

# 图片目录（框架读取该目录中的图片/视频）
python predict.py --source images/01 --device 0 --output runs/predict/group01

# 视频
python predict.py --source /absolute/path/video.mp4 --device 0 --output runs/predict/video01
```

默认权重为 `runs/pose/front_back_pose/weights/best.pt`；可通过 `--weights` 指定其他训练结果。
输出目录已存在时脚本会停止，请为新运行指定不同的 `--output`。

输出文件：

- `predictions.jsonl`：每张图片/每个视频帧一行，含目标类别、检测分数、检测框、五点坐标及置信度。
- `prediction_summary.json`：处理数量、无目标帧数、每类目标数及运行参数。
- `overlays/`：绘制正反面类别、检测框和编号 P1–P5 的图片或视频。

所有导出坐标均为原图/原视频帧的像素坐标。`--point-conf` 仅控制绘图显示，JSON 保留所有五点，可供后续筛选。
`--no-overlays` 只导出数据，不生成叠加图片/视频。
视频保留原帧尺寸，FPS 默认从源视频读取，读取失败时使用 25；可用 `--fps` 指定。叠加视频使用 MP4，不包含原视频音轨。

## 文件关系

`pose_common.py` 为评估和推理脚本提供配置检查、标签解析、设备选择、模型一致性检查、结果导出和绘图。`train.py` 独立，只包含训练代码。
模型训练、评估和推理仅在直接运行相应入口脚本时启动。
