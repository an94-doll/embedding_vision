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

这是视觉抓取开发的主入口：显示底边两端点、连线、篮架固定格线、各格盒数、首盒十字，以及右侧 3×5 数量示意图和端点坐标。顶部配置保留原值：1280×720、30 FPS 请求、GPU 0、框置信度 0.25、端点置信度 0.50。USB 配置不支持 30 FPS 时，在 `detect_d405.py` 的 `CAMERA` 中把 `fps` 改为 15。

- `Q` / Esc：退出；`D`：开关额外去重；`R`：清空篮架与归属历史，下一帧重新建立格位。
- `N` / `P`：翻页查看右侧首盒坐标。
- 固定 **3 行 × 5 列**，等间距、忽略窄边框。四角顺序为左上、右上、右下、左下；透视变换后分别转换白盒的两个端点，取中点归格。编号从左上 `(0,0)` 到右下 `(2,4)`，空格也保留。十个不再作为切格条件或计数上限。
- 右侧顶部的 15 格数量图始终可见，下方首盒坐标通过 `N` / `P` 翻页。`0` 是当前无已归属观测；`--` 是计数不可用；黄色格表示有仍保留旧格位的换格待确认目标。未归属数量单独显示，不塞进最近格子。
- 初次归属、换格均需连续至少三帧且持续 0.15 秒。只按透视校正后的两个端点中点确定行列，不再因端点跨格或中点靠近分界线而拒绝归属。换格确认期间暂留旧归属；中点位于篮架外、端点缺失或低置信度的白盒仍不归属。
- 角点采用 0.35 系数平滑。篮架四角不完整或同时出现多个有效篮架时，最多保留旧格线 0.5 秒（`HELD`），暂停归属、计数显示 `--`，不输出首盒；超时清除格线（`MISSING`）。重新看到有效篮架后恢复。相机停流显示 `STALE`，不会显示旧数量为当前数量。
- 计数只来自当前检测，完全漏检的盒子不能由此推断出来。透视格位按篮架角点平面建立，堆叠高度和斜视可能造成归属偏移；毫米定位仍需标定。

分格参数位于 `detect_d405.py` 的 `GRID`；录像窗口使用 `BasketGridConfig` 相同默认值。旧的间距推断算法保留在 `scripts/grid_counter.py`，当前窗口不再调用。独立篮架模型未提供有效四角时不建立固定格位，不回退到混合模型。

## 需要录像或截图时

```bash
./start_live.sh
# 连接配置只支持 15 FPS 时：
./start_live.sh --fps 15
```

这个入口使用同一个篮架模型，显示固定格线、数量示意图和首盒侧栏，并保留原采集界面的录像操作。`R` 开始／停止录像，`A` 在空闲时切换 RAW／ANNOTATED，`S` 保存截图及 JSON，`B` 切换框，`D` 切换诊断，`Q` / Esc 退出。ANNOTATED 录制中不允许关闭检测或暂停；视频不包含顶部信息栏和侧栏。

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

实时窗口和离线预览使用两个独立模型，对同一张原始图像分别推理：

- 白盒：`runs/pose/basket_white_box_2pt/weights/best.pt`，单类 `white_box`、`kpt_shape=[2,3]`。
- 篮架：`runs/pose/basket_4pt/weights/best.pt`，单类 `basket`、`kpt_shape=[4,3]`，按左上、右上、右下、左下输出角点。

两个模型的类别 ID 均为 0，通过各自模型区分，不混入对方计数。主窗口顶部 `WEIGHTS` / `BASKET_WEIGHTS` 分别配置；录像窗口和离线预览用 `--weights` / `--basket-weights` 指定。录像默认白盒模型仍由 `weights/base.json` 记录，`start_live.sh` 同步指定两个模型。置信度、输入尺寸等沿用现有设置，推理耗时包含两次模型预测。

不再使用混合模型进行实时检测。暂时取消原始／平滑四角的双色轮廓和 R/S 标签，保留平滑计算、绿色固定格线、数量示意图和首盒坐标；JSON 的原始角点字段仍保留用于诊断。

历史权重保留；`evaluate.py` 默认评估 `runs/pose/white_box_basket_4pt/weights/best.pt`，切换时修改顶部 `WEIGHTS` 和匹配的 `DATA`。

当前训练数据为 `white_box`、`basket` 两类，`kpt_shape=[4,3]`、`flip_idx=[1,0,3,2]`，语义见 `box_in_basket/basket_pose_schema.yaml`。本次模型切换没有重新划分数据或改写标签。原随机帧划分可能把相邻帧放入不同集合，最终抓取验收应使用独立采集场景。

历史 48 张测试图上，第一个模型的检测 precision / recall 为 0.9381 / 0.9794；两端点均不超过 5 像素且置信度等条件满足的比例为 1312/1618（81.09%）。原报告位于 `runs/evaluate/basket_white_box_2pt_test/metrics.json`。这是原测试集二维结果，不能换算为机械臂毫米精度。

## 继续训练和评估

沿用现有环境：

```bash
python train.py
python evaluate.py
python -m unittest discover -s tests -v
```

`train.py` 的参数在文件顶部及 `model.train()` 中修改；初始化权重已经改为本目录的 `weights/yolov8s-pose.pt`。默认是从 COCO 预训练模型开始新实验，不是自动续训。若想从现有篮架权重再次微调，可把 `YOLO(...)` 的路径改为本目录中的 `best.pt`。

### 白盒与铁篮评估

运行 `python evaluate.py`，默认在 48 张 test 图片上评估新四点模型；调参时将顶部 `SPLIT` 改成 `val`。输出位于 `runs/evaluate/white_box_basket_4pt_test/`，重跑会自动递增目录。

- `metrics.json`：`standard_yolo_metrics_by_class` 分类别报告框与 Pose 的 precision、recall、mAP；`by_class.white_box` 保留端点、底边中点、角度和长度误差；`by_class.basket` 报告四角逐点误差、像素 PCK 和四角同时合格率。
- 四角同时合格：四个 GT 点均已标注，预测均在图内、置信度至少 `POINT_CONF=0.5`，且每角误差不超过 `CORNER_ERROR_PX=5` 像素。完整 GT 的漏检计失败；部分标注只统计已知角点，不进入四角合格率分母。固定角点 ID，不按距离重排。
- 白盒只评估前两个点，后两个占位点不进入误差或中心计算。顶层 `endpoints/pairs/geometry` 仍仅指白盒；顶层 `detection` 包含两类。
- `per_instance.csv`、`per_keypoint.csv`、`per_prediction.csv` 均带类别字段；不适用的指标留空。`visualizations/` 左侧为 GT、右侧为预测，铁篮四边闭合并标角点编号。
- 旧两点模型仍可评估，但必须同时选择对应的单类两点数据配置。

### 铁篮外框精修实验（不改模型与标签）

```bash
python evaluate_basket_refinement.py --split val --device cpu
```

- `scripts/basket_refinement.py` 是纯 OpenCV/NumPy 模块：`refine_basket(image, points)` 接收原图和 `(4,3)` 的原始像素坐标＋置信度，返回 `corners`、`valid`、`reason`、候选边和支持覆盖率。角点顺序固定为左上、右上、右下、左下。
- 算法在粗边附近提取 Canny/Hough 线段，使用 Huber 直线拟合，每边保留最多两个候选，再联合比较四边形。方向、覆盖率、交点、修正幅度或候选歧义不通过时返回 `corners=None`；不会用原始预测冒充精修成功。
- 为保持第一版简洁，只接受或拒绝完整四边形，暂不输出部分有效精修角点；低置信度、粗角点出画等情况直接拒绝。它未接入实时程序。
- `evaluate_basket_refinement.py` 复用 `evaluate.py` 的模型、数据、匹配与阈值；默认只跑验证集，不重复算标准 mAP。输出至自动递增的 `runs/evaluate/basket_refinement_val/`。
- `metrics.json` 报告接受率、拒绝原因、接受样本配对误差、修坏实例数、全部完整 GT 的四角合格率和精修耗时；`per_instance.json` 保留匹配与误差，`predictions.json` 保留原始/精修角点和拟合边。三栏图片依次为 GT、Pose、精修（灰色为候选边，失败时不绘制精修角点）。
- 参数固定后可用 `--split test` 做对照。未知 GT 不参与误差；漏检、拒绝计入整体失败。标签不进入精修模块，已有训练/推理入口保持不变。

首轮验证集结果（`runs/evaluate/basket_refinement_val-2/metrics.json`）：95 个 GT 篮子均匹配，只有 1 个接受精修；该样本四角平均误差从 15.75 降至 14.48 像素，其中两个角点误差反而增加。21 个完整标注篮子的四角 ≤5 像素合格数，原始和精修均为 0。完整样本拒绝原因主要是边缘支持不足（11）和候选歧义（7），另有交点检查失败（2）。当前版本仅用于离线实验，不建议替代 Pose 输出；尚未运行独立测试集或实时视频验收。

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

当前可独立继续的是**感知与定位链路的开发**：当前图像 → 白盒两端点与篮架四角 → 保守去重／跟踪 → 固定 3×5 归属 → 每格观测首盒 → 原图坐标／去畸变射线 → 可选已测平面定位。

`detect_d405.py` 每帧原子更新 `reports/d405_first_boxes_latest.json`。消费者应读取：

- 顶层 `status`：新帧为 `current`；停流为 `stale`；退出清空 `cells`。
- `sequence`、`frame_number`、`received_at_monotonic`：识别新帧并判断是否过期；单调时间属于采集主机，不能直接与另一台主机比较。
- `grid.status`：`current` / `held` / `missing` / `disabled`；顶层 `status=current` 仅表示图像为新帧，消费者还必须检查格位状态。`grid.cells` 提供 15 格多边形、检测索引和数量；保留格线但不可计数时 `count=null`。`grid.assignments` 给出归属与防抖状态，`grid.unassigned_indices` 单独记录未归属目标。
- `camera.intrinsics`：当前彩色流的实际内参和畸变模型。
- `cells[].raw_endpoints_px`、`corrected_endpoints_px`、`rays_camera`：原图端点、去畸变端点和相机射线。
- `selection_status`：`observed_candidate` 是当前观测的候选；`incomplete_observation` 表示同格上方还有未归属观测，`uncertain_assignment` 表示首盒归属还处于换格确认中。暂失轨迹另由 `temporarily_lost_tracks` 记录。
- `cells[].status`：未给平面为 `rays_only`；求交成功为 `plane_localized`；缺点、缺标定或不支持的畸变模型会明确报告原因，不会换选下面的盒子掩盖失败。

默认 `FIRST_BOX_PLANE=None`，毫米坐标为空。后续按以下顺序接入抓取：

1. **测量盒底边所在平面或接入实测深度。** 现有求交平面方程是 `normal · X_mm = offset_mm`，配置到 `detect_d405.py` 的 `FIRST_BOX_PLANE`；相机坐标 X 向右、Y 向下、Z 向前。不能用约 150 mm 的观看距离替代测量值。
2. **完成相机到机械臂底座的外参／手眼标定。** 当前 `midpoint_camera_mm` 属于彩色相机坐标；需要按实际安装方式转换到机器人坐标。
3. **定义夹爪抓取点、TCP 偏置、姿态和进退路径。** 当前中点是盒底边中点，还不是完整夹爪位姿；格号按当前篮架四角顺序定义，篮架旋转或角点语义变化后不能直接作为持久槽位身份。
4. **独立场景验证定位误差、漏检和碰撞余量，再做机械臂联调。** 30 mm 是名义端点间距，`width_error_mm` 只是几何一致性检查；7 mm 只是名义层距。毫米误差（包括此前讨论的 2 mm 要求）尚需实测闭环验收。

本包没有机器人驱动或运动控制指令。相机／铁架移动后需更新标定；候选状态不能直接作为已允许抓取的信号。

## 环境与验证记录

现有环境实测为 Python 3.12.14、PyTorch 2.11.0+cu128、Ultralytics 8.4.143。`requirements.txt` 只记录关键已安装版本，不要求重装。旧项目 Python 3.11／yolo26 的环境说明已归档到 `provenance/original_project/`，不要照旧说明重建当前环境。

**该 Conda 环境的 Ultralytics 库来自可编辑安装**，模块位于 `/home/wangjr/projects/yolo_pose/ultralytics/ultralytics`。它属于当前环境依赖，仍需保留；本次没有修改或复制这套第三方库。迁往另一台机器时，需要另行提供兼容环境。

本次结果见 `reports/acceptance.json`、`reports/unit_tests.txt`、`reports/environment_observed.json`，离线实测效果见 `reports/offline_preview/preview.jpg`。原目录访问被进程内阻止后，在临时搬迁位置检查数据解析和实际图片推理，完成后将整个工程放回本目录。该检查针对工程外部路径依赖，不替代相机和机械臂实机验证。

当前执行进程看不到 CUDA 设备及 D405，实时窗口、GPU 帧率、相机流和机械臂闭环未实测。回到能访问硬件的主机桌面终端后，先运行 `python detect_d405.py` 检查新帧和坐标；如果出现错误，查看本目录 `reports/d405_detection_latest.json`，不要读取 `provenance/` 中的历史报告作为当前状态。
