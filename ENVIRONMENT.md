# 预测运行环境与依赖

本文仅说明 `predict.py` 的运行环境。Python 包版本根据当前机器的 `yolo-pose` 环境记录。

## 当前环境

| 项目 | 当前值 |
|---|---|
| 系统 | Linux x86_64，glibc 2.35 |
| Conda 环境名 | `yolo-pose` |
| Python | `3.12.14` |
| Python 路径 | `/home/wangjr/miniconda3/envs/yolo-pose/bin/python` |
| PyTorch | `2.11.0+cu128` |
| Torchvision | `0.26.0+cu128` |
| Ultralytics | `8.4.143`，本地源码可编辑安装 |
| OpenCV | `opencv-python 5.0.0.93` |
| NumPy | `2.5.2` |
| PyYAML | `6.0.3` |

建议使用 Python 3.12，便于与现有环境一致。其他 Python、系统或依赖版本未在本项目中验证。

`requirements.txt` 固定上述直接依赖的基础版本；其他间接依赖由这些包的安装元数据解析。
它不是完整的离线环境包，也未锁定所有间接依赖。

## 使用现有环境

当前机器已经安装所需依赖，无需为本项目重新安装：

```bash
conda activate yolo-pose
cd /home/wangjr/projects/yolo_pose_box
python predict.py --help
```

之后按 [README.md](README.md) 的图片或视频命令进行预测。

## CPU 与 GPU

- CPU 预测：使用 `--device cpu`。
- 自动选择：使用 `--device auto`；CUDA 可用时使用 GPU 0，否则使用 CPU。
- NVIDIA GPU 预测：使用 `--device 0`，需要该设备对当前环境可见，且驱动兼容所用 PyTorch CUDA 构建。

当前 PyTorch/Torchvision 的 `+cu128` 表示 CUDA 12.8 构建，并不表示当前会话一定能访问 GPU。
GPU 驱动、容器/会话的 GPU 访问权限不由 `requirements.txt` 管理。
依赖文件采用 `torch==2.11.0`、`torchvision==0.26.0` 的基础版本约束，允许对应的 CPU/CUDA 构建；新机器应根据设备选择合适的 PyTorch wheel 来源。

## 在新环境中准备依赖

以下是准备步骤说明，本次未执行这些安装命令。

### 1. 准备 Python 环境

```bash
conda create -n yolo-pose-predict python=3.12
conda activate yolo-pose-predict
```

### 2. 准备相同版本的 Ultralytics 源码

当前环境实际使用的源码目录为：

```text
/home/wangjr/projects/yolo_pose/ultralytics
```

该目录通过可编辑安装提供 `ultralytics` 包。`requirements.txt` 中的版本号不能证明公开包与这份本地源码完全相同。
为尽量复现当前行为，应保留或复制相同源码目录，再安装它。
在当前机器可使用：

```bash
python -m pip install -e /home/wangjr/projects/yolo_pose/ultralytics
```

在其他机器请把路径替换为复制后的源码目录。
安装前如需使用特定 CPU/CUDA 版 PyTorch，请先按目标平台配置相应安装来源和版本；安装本地源码时也可能解析并安装其依赖。

### 3. 安装或核对项目依赖

进入本项目目录后执行：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

先安装本地 Ultralytics 源码后，如果其版本为 `8.4.143`，通常可满足依赖文件中的同版本约束。
各固定版本能否在目标平台安装，取决于相应软件源是否提供适用 wheel；这里记录的是当前已安装版本，不保证所有平台或公开软件源均提供同一构建。

不要同时安装多个提供 `cv2` 的 OpenCV 变体。当前记录使用 `opencv-python`。
在精简 Linux 系统中，OpenCV 还可能需要系统图形/视频运行库；若出现 `libGL.so.1`、`libgthread-2.0.so.0` 等缺失提示，需要通过目标系统的软件包管理器补齐。

## 环境检查命令

```bash
python -c "import sys, torch, torchvision, ultralytics, cv2; print('Python:', sys.version); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Ultralytics:', ultralytics.__version__); print('Ultralytics source:', ultralytics.__file__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

检查 `Ultralytics source` 是否指向预期源码。若显式使用 `--device 0`，还应确认 `CUDA available` 为 `True`。

## 预测所需文件

Python 环境准备好后，还需要：

```text
predict.py
pose_common.py
runs/pose/front_back_pose/weights/best.pt
```

`predict.py` 与 `pose_common.py` 放在同一目录。模型权重不在 `requirements.txt` 中，需单独携带；若放在其他位置，使用 `--weights` 指定。
视频叠加输出还依赖本机 OpenCV 的输入解码和 MP4 编码支持；只需要坐标结果时可使用 `--no-overlays`。
