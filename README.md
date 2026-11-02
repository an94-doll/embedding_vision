# 铁架内白盒检测与首盒定位

本目录是从 `yolo_pose_box` 拆出的视觉抓取感知工程：保留篮架内白盒底边两端点检测、保守去重、跨帧处理、自动分格、首盒选择、坐标侧栏和 D405 录像，以及对应训练数据、权重、评估和测试。数据源实际名为 `box_in_basket`，不是 `box_in_blanket`。

代码、数据和模型均在本目录内，可以把整个目录搬到其他位置继续开发。环境复用本机已有的 Conda `yolo-pose`，没有创建环境或安装、升级依赖。这里的“独立”指工程文件独立，运行仍需该 Python 环境、RealSense SDK、相机和相应设备权限。

## 先打开实时检测

在主机桌面终端中运行：

```bash
conda activate yolo-pose
cd /home/wangjr/projects/yolo_pose_box/rack_grasp_vision
python detect_d405.py
```

这是视觉抓取开发的主入口：显示底边两端点、连线、推断格位、各格盒数、首盒十字和右侧端点坐标。顶部配置保留原值：1280×720、30 FPS 请求、GPU 0、框置信度 0.25、端点置信度 0.50。USB 配置不支持 30 FPS 时，在 `detect_d405.py` 的 `CAMERA` 中把 `fps` 改为 15。

- `Q` / Esc：退出；`D`：开关额外去重；`R`：重建推断格位。
- `N` / `P`：翻页查看右侧首盒坐标。
- 每格最多十个、连续三帧且持续至少 0.15 秒才确认分格，保留原来的“间隔优先、十个兜底”逻辑。
- 格位是图像中推断的分组，不是已标定的固定铁架槽位；完全漏检的盒子不能由此推断出来。

## 需要录像或截图时

```bash
./start_live.sh
# 连接配置只支持 15 FPS 时：
./start_live.sh --fps 15
```

这个入口使用同一个篮架模型，显示首盒侧栏，并保留原采集界面的录像操作。`R` 开始／停止录像，`A` 在空闲时切换 RAW／ANNOTATED，`S` 保存截图及 JSON，`B` 切换框，`D` 切换诊断，`Q` / Esc 退出。ANNOTATED 录制中不允许关闭检测或暂停；视频不包含顶部信息栏和侧栏。

录像保存在本目录 `test/d405_recordings/recording_日期_时间/`，包括 MP4、`frames.csv` 和 `metadata.json`；截图保存在 `test/d405_live_日期_时间/`。MP4 为固定帧率，丢帧时播放时长可能短于真实时间，真实采集时序以 `frames.csv` 为准。启动不会自动录像。

两种实时入口应分开运行，避免争用相机。对外逐帧首盒接口由 `detect_d405.py` 提供；录像入口侧栏用于查看，不代替此接口。

## 没有相机时检查模型和可视化

```bash
python preview_image.py \
  --source box_in_basket/dataset/images/val/frame_0012.jpg \
  --device cpu --output reports/my_preview
```

输出 `preview.jpg` 和 `predictions.json`，目录必须尚不存在。该离线入口显式使用 CPU，不改变实时入口的 GPU 设置。静态图片仅为预览采用单帧分格，画面标记 OFFLINE，不能据此判断跨帧确认效果。

没有匹配内参时仅显示原图端点，几何状态是 `missing_intrinsics`。只有图片确实匹配该彩色流标定时，才用 `--intrinsics 内参.json`；JSON 包含 `width, height, fx, fy, ppx, ppy, model, coeffs`。测得平面后可加 `--plane 平面.json`，字段为 `normal` 和 `offset_mm`。不从历史日志自动套用标定。

## 数据和权重在哪里

```text
box_in_basket/
  data_origin/            477 张原始图片及配对标签（320 张新增 + 157 张原有）
  white_box_pose/         320 张复核图片、标签、语义和校验记录
  white_box_pose_reviewed_320_images_20261028_145012.zip
  dataset/                实际训练划分：334 train / 95 val / 48 test
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml
    split_manifest.csv
    split_summary.json
runs/pose/
  basket_white_box_2pt/    第一轮训练：best.pt、last.pt、参数、指标和图表
  basket_white_box_2pt-2/  第二轮训练：best.pt、last.pt、参数、指标和图表
runs/evaluate/            两个模型已有的测试集评估及对照可视化
weights/
  yolov8n-pose.pt          新训练使用的 COCO 预训练权重，已复制进来
  base.json               录像界面的本工程默认模型配置
scripts/                   相机、录像、后处理、分格、首盒几何及坐标栏
tests/                     原有 108 个专项测试
diagnostics/               历史去重、时序和分格检查及辅助脚本
provenance/                文件来源清单、原项目文档和历史运行报告
reports/                   本次拆分验证及后续运行输出
```

默认保留 `runs/pose/basket_white_box_2pt/weights/best.pt`，与原首盒主入口一致；`start_live.sh` 和 `weights/base.json` 也已对齐到这个模型，不再误用旧的第 173 轮 YOLO26 模型。第二个实验完整保留，`evaluate.py` 默认仍评估第二个模型，切换时修改顶部 `WEIGHTS`。

数据是单类 `white_box`，`kpt_shape=[2,3]`，两个点为图像左、右底边实际端点；`flip_idx=[1,0]`。没有重新划分数据或改写标签。原随机帧划分可能把相邻帧放入不同集合，最终抓取验收应使用独立采集场景。

历史 48 张测试图上，第一个模型的检测 precision / recall 为 0.9381 / 0.9794；两端点均不超过 5 像素且置信度等条件满足的比例为 1312/1618（81.09%）。原报告位于 `runs/evaluate/basket_white_box_2pt_test/metrics.json`。这是原测试集二维结果，不能换算为机械臂毫米精度。

## 继续训练和评估

沿用现有环境：

```bash
python train.py
python evaluate.py
python -m unittest discover -s tests -v
```

`train.py` 的参数在文件顶部及 `model.train()` 中修改；初始化权重已经改为本目录的 `weights/yolov8n-pose.pt`。默认是从 COCO 预训练模型开始新实验，不是自动续训。若想从现有篮架权重再次微调，可把 `YOLO(...)` 的路径改为本目录中的 `best.pt`。

四个 `best.pt` / `last.pt` 的保存轮次为 `epoch=-1`，属于精简检查点；不要把它们当作完整优化器断点来承诺无缝恢复训练。原轮次指标和配置保留在相应训练目录。

数据 YAML 已移除原机器的绝对 `path`，由自身所在目录解析。重新划分实验数据时，使用规范化后的脚本名，并指定新的输出目录，不能覆盖当前训练划分：

```bash
python split_dataset.py \
  --source box_in_basket/data_origin \
  --schema box_in_basket/white_box_pose/pose_schema.yaml \
  --output box_in_basket/dataset_experiment
```

历史 `args.yaml`、CSV 清单、JSON 和模型内部仍保留原始路径作为来源记录；正常入口不读取这些路径来寻找外部数据。不要直接拿历史 `args.yaml` 作为新机器的运行配置。

## 视觉抓取从这里继续

当前可独立继续的是**感知与定位链路的开发**：当前图像 → 白盒两端点 → 保守去重／跟踪 → 分格 → 每格观测首盒 → 原图坐标／去畸变射线 → 可选已测平面定位。

`detect_d405.py` 每帧原子更新 `reports/d405_first_boxes_latest.json`。消费者应读取：

- 顶层 `status`：新帧为 `current`；停流为 `stale`；退出清空 `cells`。
- `sequence`、`frame_number`、`received_at_monotonic`：识别新帧并判断是否过期；单调时间属于采集主机，不能直接与另一台主机比较。
- `camera.intrinsics`：当前彩色流的实际内参和畸变模型。
- `cells[].raw_endpoints_px`、`corrected_endpoints_px`、`rays_camera`：原图端点、去畸变端点和相机射线。
- `selection_status`：`observed_candidate` 是当前观测的候选；`incomplete_observation` 表示同列上方还有未分组观测。暂失轨迹另由 `temporarily_lost_tracks` 记录。
- `cells[].status`：未给平面为 `rays_only`；求交成功为 `plane_localized`；缺点、缺标定或不支持的畸变模型会明确报告原因，不会换选下面的盒子掩盖失败。

默认 `FIRST_BOX_PLANE=None`，毫米坐标为空。后续按以下顺序接入抓取：

1. **测量盒底边所在平面或接入实测深度。** 现有求交平面方程是 `normal · X_mm = offset_mm`，配置到 `detect_d405.py` 的 `FIRST_BOX_PLANE`；相机坐标 X 向右、Y 向下、Z 向前。不能用约 150 mm 的观看距离替代测量值。
2. **完成相机到机械臂底座的外参／手眼标定。** 当前 `midpoint_camera_mm` 属于彩色相机坐标；需要按实际安装方式转换到机器人坐标。
3. **定义夹爪抓取点、TCP 偏置、姿态和进退路径。** 当前中点是盒底边中点，还不是完整夹爪位姿；图像格号也还不是固定物理槽位号。
4. **独立场景验证定位误差、漏检和碰撞余量，再做机械臂联调。** 30 mm 是名义端点间距，`width_error_mm` 只是几何一致性检查；7 mm 只是名义层距。毫米误差（包括此前讨论的 2 mm 要求）尚需实测闭环验收。

本包没有机器人驱动或运动控制指令。相机／铁架移动后需更新标定；候选状态不能直接作为已允许抓取的信号。

## 环境与验证记录

现有环境实测为 Python 3.12.14、PyTorch 2.11.0+cu128、Ultralytics 8.4.143。`requirements.txt` 只记录关键已安装版本，不要求重装。旧项目 Python 3.11／yolo26 的环境说明已归档到 `provenance/original_project/`，不要照旧说明重建当前环境。

**该 Conda 环境的 Ultralytics 库来自可编辑安装**，模块位于 `/home/wangjr/projects/yolo_pose/ultralytics/ultralytics`。它属于当前环境依赖，仍需保留；本次没有修改或复制这套第三方库。迁往另一台机器时，需要另行提供兼容环境。

本次结果见 `reports/acceptance.json`、`reports/unit_tests.txt`、`reports/environment_observed.json`，离线实测效果见 `reports/offline_preview/preview.jpg`。原目录访问被进程内阻止后，在临时搬迁位置检查数据解析和实际图片推理，完成后将整个工程放回本目录。该检查针对工程外部路径依赖，不替代相机和机械臂实机验证。

当前执行进程看不到 CUDA 设备及 D405，实时窗口、GPU 帧率、相机流和机械臂闭环未实测。回到能访问硬件的主机桌面终端后，先运行 `python detect_d405.py` 检查新帧和坐标；如果出现错误，查看本目录 `reports/d405_detection_latest.json`，不要读取 `provenance/` 中的历史报告作为当前状态。
