# D405 包埋盒两点实时检测与录像

使用第 **173 轮 YOLO26s-Pose** 模型，在 RealSense D405 彩色画面中识别白色蜡块正面**下沿的左、右两个实际角点**。目标不是上方唇缘、底托或篮架角点。两个点按图像 X 方向排列；只有达到显示置信度的点才绘制，两点都可见时连线。

仓库包含运行脚本、原始完整模型权重、模型结构和少量训练配置／指标，不包含训练图片、视频、标签或数据集。

## 安装

适用环境：Linux x86_64、Python 3.11、NVIDIA CUDA GPU，以及具备桌面显示的本机终端。D405 需要 RealSense 设备访问权限；建议通过 USB 3.x 直连。

先安装 Git LFS，再克隆指定分支并拉取模型文件（已克隆时直接进入仓库执行`git lfs pull`）：

```bash
git lfs install
git clone --single-branch --branch baomai https://github.com/an94-doll/embedding_vision.git
cd embedding_vision
git lfs pull
bash scripts/setup_env.sh
conda activate yolo26
python scripts/check_env.py
```

安装脚本默认创建或复用专用 Conda 环境 `yolo26`，不修改系统 Python、驱动或系统 CUDA。已有环境不兼容时，可改用独立名称：

```bash
YOLO_ENV=yolo26_cassette bash scripts/setup_env.sh
conda activate yolo26_cassette
```

也可用 `conda env create -f environment.yml` 恢复导出的 Linux 环境。依赖锁定版本包括 PyTorch `2.10.0+cu128`、torchvision `0.25.0+cu128`、Ultralytics `8.4.182`、带 GUI 的 OpenCV 和 `pyrealsense2`。模型文件通过 Git LFS 保存；若下载后只有几行 LFS 指针文本，请先完成 `git lfs pull`。

环境文件来自实际运行环境，已检查版本和脚本语法；导出时没有在另一台机器重新安装验证。CUDA 张量检查结果写入 `reports/env_check.json`。运行需要 CUDA，不会自动退回 CPU。

## 打开实时窗口

在已激活的环境中运行：

```bash
./start_live.sh
```

默认使用第173轮权重、1280×720相机画面、30 FPS采集请求、30 FPS显示上限，以及 **ANNOTATED 带标注录制模式**。启动只打开实时画面，按 R 后才开始录像。

脚本使用当前环境的 Python，也可显式指定解释器：

```bash
YOLO_PYTHON=/path/to/environment/bin/python ./start_live.sh
```

所有参数都可以在启动命令后覆盖。例如 USB 2.x 连接下，D405 的1280×720配置可能只支持15 FPS，此时使用：

```bash
./start_live.sh --fps 15
./start_live.sh --fps 15 --record-mode raw
./start_live.sh --conf 0.25 --endpoint-conf 0.5 --show-boxes
./start_live.sh --serial CAMERA_SERIAL
```

程序使用当前桌面的 `DISPLAY`／`WAYLAND_DISPLAY`，不会硬编码显示服务器或认证文件。请在主机桌面终端运行；没有桌面时可做有限帧验证：

```bash
./start_live.sh --headless --max-frames 60 --fps 15
```

无人值守验证不会自动录像。出现取帧超时或配置不支持时，会记录具体错误并退出；请检查 USB 连接、选择的 FPS 和是否有其他程序占用相机。

## 按键和录像

先点击实时窗口，使其获得键盘焦点。

| 按键 | 功能 |
| --- | --- |
| R | 开始／停止当前模式的录像 |
| A | 空闲时切换 RAW 原始画面与 ANNOTATED 带标注画面；录制中锁定 |
| Q 或 Esc | 停止录像并退出 |
| B | 显示／隐藏检测框 |
| D | 显示／隐藏置信度、遮挡点与重叠诊断 |
| Y | 开关 YOLO；带标注录制中禁用此操作 |
| 空格 | 暂停／恢复预览；带标注录制中禁用此操作 |
| S | 保存当前原图、标注图和检测 JSON |
| 1 / 2 / 3 | 检测置信度设为0.10／0.25／0.50 |
| [ / ] | 调低／调高检测置信度 |
| , / . | 调低／调高端点显示置信度 |
| - / = | 调整检测框 NMS IoU |

**RAW** 保存相机采集的原始画面，不烧入检测结果；暂停预览或关闭 YOLO 时，原始录像仍继续。**ANNOTATED** 保存已完成检测的新帧，只烧入画面上的端点、连线及当前启用的框／诊断，不包含顶部状态栏。1280×720采集时，两种视频都保持1280×720。带标注录制开始时会自动开启 YOLO 并恢复实时预览。

每次按 R 开始都会创建独立目录，不覆盖已有录像：

```text
test/d405_recordings/recording_日期_时间/
  raw.mp4 或 annotated.mp4
  frames.csv
  metadata.json
```

`metadata.json` 记录模式、尺寸、相机 FPS、视频容器 FPS、写入／丢弃帧数，以及带标注录像的模型哈希、轮次和初始参数。标注录像跟随当前显示设置，录制期间仍可切换 B／D 或调整置信度。`frames.csv` 保留每个视频帧对应的相机帧号和原始采集时间。S键照片保存在 `test/d405_live_日期_时间/`。

**帧率与时长：**采集请求、实际收到帧率、YOLO推理帧率和显示更新帧率是不同指标。30 FPS设置不保证实际达到30 FPS。MP4为固定帧率：RAW使用相机配置帧率，ANNOTATED使用相机帧率与显示上限中的较小值。推理较慢或队列丢帧时不会复制旧帧，播放时长可能短于真实录制时间；需要真实时序时请使用 `frames.csv`。

运行状态每5秒更新到 `reports/d405_live_latest.json`；退出后保留最终帧数、性能、录像信息和错误原因。

## 模型与指标

默认权重为 `weights/cassette_lower_epoch173.pt`，原始完整文件大小为 **106,838,542字节**，SHA-256：

```text
e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e
```

模型从官方 COCO YOLO26s-Pose 权重独立微调，没有使用旧127轮包埋盒模型初始化。先训练200轮，再从本次第200轮完整状态延长至300轮；在全部300轮中，按各项 `val/*_loss` 之和选择最低者，最终仍为第173轮，验证总损失 **1.79037**。本次使用137张训练图、20张验证图；最终测试和隔离图没有参与训练或选模。

| 第173轮验证指标 | 检测框 | 关键点 |
| --- | ---: | ---: |
| Precision | 0.96670 | 0.98297 |
| Recall | 0.91810 | 0.93675 |
| mAP50 | 0.93764 | 0.95993 |
| mAP50–95 | 0.58654 | 0.95963 |

这些是选模使用的验证集指标，不等同于独立测试集或所有真实场景的准确率。端点置信度也不代表像素定位误差。

`metadata/model_card.json` 提供模型指标与语义，`metadata/training_config.json` 提供训练参数摘要。`models/cassette_lower_yolo26s_pose.yaml` 是实际模型结构，单类 `cassette_front`、`kpt_shape: [2, 3]`、左右翻转索引 `[1, 0]`。结构文件本身不含训练参数，实时推理加载 `.pt` 权重。

完整 `.pt` 保留优化器、EMA及原训练元信息，文件字节与选定原模型一致；其内部可能含原训练环境路径，实时推理不会访问这些训练路径。旁边的 JSON 和运行入口均采用项目相对路径。

默认推理使用 FP32、`imgsz=1280`、`rect=True`、框置信度0.25、端点显示阈值0.50，并显式启用普通框 NMS，使 IoU 热键有效。可通过 `--weights /path/to/checkpoint.pt` 临时指定其他本地两点模型；若相邻 JSON 存在，会核验哈希和轮次，不会修改默认模型配置。
