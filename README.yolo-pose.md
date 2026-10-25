# 白色盒子正反面与五点关键点预测

使用现有模型对图片或视频中的白色盒子进行检测，输出正反面类别、检测框以及 P1–P5 的二维坐标，并生成带标注的图片或视频。

## 1. 运行所需文件

在当前项目目录中，预测使用以下文件：

```text
yolo_pose_box/
├── predict.py                         # 图片、目录、视频预测入口
├── pose_common.py                     # 模型检查、结果导出和绘图函数
└── runs/
    └── pose/
        └── front_back_pose/
            └── weights/
                └── best.pt           # 默认模型权重
```

`predict.py` 和 `pose_common.py` 需放在同一目录。模型文件可放在其他位置，通过 `--weights` 指定。

Python 环境需要 PyTorch、Ultralytics 和 OpenCV。当前机器已有 Conda 环境 `yolo-pose`，其中的 Ultralytics 为本机安装的版本，直接使用该环境即可。

依赖版本见 [requirements.txt](requirements.txt)；Python、CPU/GPU 环境要求及新环境准备步骤见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 2. 快速开始

打开终端，激活环境并进入项目目录：

```bash
conda activate yolo-pose
cd /home/wangjr/projects/yolo_pose_box
```

预测一张图片：

```bash
python predict.py \
  --source images/06/frame_0031.jpg \
  --weights runs/pose/front_back_pose/weights/best.pt \
  --device auto \
  --output runs/predict/frame0031
```

运行完成后，查看：

```text
runs/predict/frame0031/
├── predictions.jsonl
├── prediction_summary.json
└── overlays/
    └── frame_0031.jpg.jpg
```

叠加图片保留输入文件的完整名称，再追加 `.jpg`，因此可能出现 `.jpg.jpg`。

每次运行应使用尚不存在的输出目录。已存在的目录会触发报错，以保护已有结果；可将 `--output` 改成新的名称。

## 3. 常用预测命令

### 单张图片

```bash
python predict.py \
  --source /home/wangjr/projects/yolo_pose_box/images/test/01.jpeg \
  --weights //home/wangjr/projects/yolo_pose_box/runs/pose/front_back_pose-2/weights/best.pt \
  --device 0 \
  --output runs/predict/image01
```

将 `/absolute/path/example.jpg` 替换为实际图片路径。

### 图片目录

```bash
python predict.py \
  --source images/05 \
  --weights runs/pose/front_back_pose/weights/best.pt \
  --device 0 \
  --conf 0.25 \
  --point-conf 0.25 \
  --output runs/predict/group05
```

将待处理图片放入同一个目录，或直接指定现有的某个图片组。输出目录应位于输入目录之外。

### 视频

```bash
python predict.py \
  --source /absolute/path/video.mp4 \
  --weights runs/pose/front_back_pose/weights/best.pt \
  --device 0 \
  --output runs/predict/video01
```

脚本按帧导出结果，并在 `overlays/` 中生成带检测框、类别和点编号的 MP4 视频。

- 输出视频保持原视频帧尺寸。
- 默认读取原视频 FPS，读取失败时使用 25 FPS。
- 可用 `--fps 30` 指定输出视频 FPS；该参数只改变输出播放帧率。
- 输出叠加视频不包含原视频音轨。
- 视频输出文件名为原视频完整名称追加 `.mp4`，例如 `video.mp4.mp4`。

### 使用 CPU

```bash
python predict.py \
  --source images/06/frame_0031.jpg \
  --device cpu \
  --output runs/predict/cpu01
```

`--device auto` 在 CUDA 可用时使用 GPU 0，否则使用 CPU。显式指定 `--device 0` 时，如果该 GPU 不可用，脚本会报错。

### 只导出坐标和类别

```bash
python predict.py \
  --source images/06 \
  --device auto \
  --no-overlays \
  --output runs/predict/json_only01
```

此模式生成 `predictions.jsonl` 和 `prediction_summary.json`，不生成叠加图片或视频。

## 4. 参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--source` | 必填 | 图片文件、图片目录或视频文件路径 |
| `--weights` | `runs/pose/front_back_pose/weights/best.pt` | 模型权重；省略时相对于脚本目录查找 |
| `--output` | `runs/predict/front_back_pose` | 输出目录；省略时位于脚本目录内，且必须尚不存在 |
| `--device` | `auto` | 自动选择设备，或指定 `0`、`cpu`；支持单 GPU 编号 |
| `--imgsz` | `736` | 模型输入尺寸，须为正的 32 倍数 |
| `--conf` | `0.25` | 目标检测置信度下限，范围 `[0, 1]` |
| `--point-conf` | `0.25` | 绘图时显示关键点的置信度下限，范围 `[0, 1]` |
| `--iou` | `0.7` | 检测框 NMS 的 IoU 阈值，范围 `(0, 1]` |
| `--fps` | 自动读取 | 视频输出 FPS；读取失败时默认 25，手动值须大于 0 |
| `--no-overlays` | 关闭 | 开启后只导出数据文件 |

传入的相对路径按当前终端所在目录解析。路径包含空格时，请使用引号。

无需手动将输入图片裁剪或缩放到 736；框架处理模型输入尺寸，导出的坐标仍对应原图片或原视频帧。

`--conf` 控制保留哪些目标；`--point-conf` 仅控制叠加图中显示哪些点。即使某个点低于绘图阈值，JSON 中仍会保留该点的坐标和置信度。

## 5. 类别与点序

| 类别编号 | 类别名称 | 含义 |
|---|---|---|
| `0` | `front` | 正面 |
| `1` | `back` | 反面 |

关键点按物体自身方向定义。将设定的顶部朝上观察，从左上角开始逆时针编号：

```text
P1 ─────────── P5
│               │
│               │
P2 ──── P3 ──── P4
```

- 正面：白色矩形斜面所在侧为底部。
- 反面：三个矩形凹槽所在侧为底部。
- P1/P5 为顶部两个角，P2/P4 为底部两个角，P3 为底边中点。
- 左右按物体方向定义，物体旋转后编号不按画面位置重新排列。
- 标注基准为栅格所在面；正面底边位于栅格面与白色斜面的交界，圆角或斜切角按主面直边延伸近似定位。

叠加图会标出 `front/back`、检测分数及 P1–P5。低于 `--point-conf` 的点不绘制。

## 6. 输出数据的读取

### `predictions.jsonl`

每行是一个 JSON 对象，对应一张图片或一个视频帧。没有检测到目标时，该行的 `detections` 为 `[]`。

主要字段如下：

| 字段 | 说明 |
|---|---|
| `source` | 输入文件路径 |
| `type` | `image` 或 `video` |
| `frame_index` | 视频帧编号，从 0 开始；图片为 `null` |
| `width` / `height` | 原图片或原视频帧的宽高 |
| `coordinate_system` | `original_pixels`，原图像素坐标 |
| `detections` | 当前图片/帧内的目标列表 |
| `detections[].class_id` / `class_name` | 正反面类别编号与名称 |
| `detections[].confidence` | 该类别的目标检测分数 |
| `detections[].bbox_xyxy` | `[左上x, 左上y, 右下x, 右下y]` |
| `detections[].keypoints` | 按 P1–P5 顺序排列的五个点，含 `name`、`x`、`y`、`confidence` |

坐标原点在图像左上角，x 向右、y 向下，单位为像素。

读取示例：

```python
import json

with open("runs/predict/frame0031/predictions.jsonl", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        for target in record["detections"]:
            print(record["source"], record["frame_index"])
            print(target["class_name"], target["confidence"])
            for point in target["keypoints"]:
                print(point["name"], point["x"], point["y"], point["confidence"])
```

### `prediction_summary.json`

保存权重路径、设备、阈值、图片/帧数量、目标总数、无目标图片/帧数及正反面计数。视频中的类别计数按逐帧检测累加，不能作为去重后的盒子数量。

## 7. 常见问题

| 提示或现象 | 处理方式 |
|---|---|
| `输入路径或模型权重不存在` | 核对 `--source`、`--weights`，必要时使用绝对路径 |
| `输出目录已存在` | 将 `--output` 改为尚不存在的新目录 |
| `输出目录不能位于输入图片目录内` | 将输出目录放在输入目录之外 |
| `请求的 CUDA 设备不可用` | 改用 `--device auto` 或 `--device cpu`，或检查 GPU 环境 |
| 提示缺少 `torch`、`ultralytics` 或 `cv2` | 确认已激活 `yolo-pose` 环境 |
| `需要训练后的两类别五点 Pose 权重` | 使用本项目的 `best.pt`，其类别应为 `front/back`、关键点维度应为 `[5, 3]` |
| 检测框出现，但部分点未显示 | 检查 `--point-conf`，并查看 JSON 中对应点的置信度 |
| `detections` 为空 | 当前帧没有达到 `--conf` 的目标；可检查图像清晰度、遮挡和阈值 |
| 无法创建视频输出 | 检查输出目录权限和 OpenCV 的 MP4 编码支持，或用 `--no-overlays` 导出坐标 |

支持的图片扩展名为 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`；视频扩展名为 `.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v`、`.webm`、`.mpg`、`.mpeg`。具体视频能否解码取决于本机 OpenCV 支持。
