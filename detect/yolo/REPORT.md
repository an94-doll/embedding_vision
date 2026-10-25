# 包埋盒顶部四点识别：环境与训练记录

日期：2026-10-19（Asia/Shanghai）。状态：第 1 阶段全部验收通过。环境检查于 16:41:51 开始，最终模型检查于 16:42:55 开始；详细时间见 JSON。

## 目录与环境

- 项目：`/home/ubuntu/BL/detect/yolo`
- 专用环境：`yolo26`，前缀 `/home/ubuntu/miniconda3/envs/yolo26`
- 系统：Ubuntu 22.04.5 LTS，x86_64，glibc 2.35。
- Conda：26.3.2；Python：3.11.16。环境激活已实测通过，见 `reports/activation_check.json`。
- 开始时目录无业务文件、同名环境不存在；复查了项目及父目录，未发现适用的额外 AGENTS.md。保留所有其他项目和环境。
- 驱动报告 GPU：NVIDIA GeForce RTX 4090，显存 49,140 MiB，驱动 580.142，计算能力 8.9。显存按本机实测记录。

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /home/ubuntu/BL/detect/yolo
```

## 安装命令与来源

实际执行了以下命令。Conda 输出保存为 `reports/conda_create.json`，pip 安装日志及包含下载 URL、哈希的安装记录分别保存为 `reports/pip_*_install.log` 和 `reports/pip_*_install.json`。

```bash
/home/ubuntu/miniconda3/bin/conda create \
  -p /home/ubuntu/miniconda3/envs/yolo26 \
  --override-channels -c https://repo.anaconda.com/pkgs/main \
  python=3.11 pip -y --json > reports/conda_create.json

set -o pipefail
/home/ubuntu/miniconda3/envs/yolo26/bin/python -m pip --isolated install \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --no-cache-dir --progress-bar off --report reports/pip_torch_install.json \
  2>&1 | tee reports/pip_torch_install.log
```

随后执行：

```bash
set -o pipefail
/home/ubuntu/miniconda3/envs/yolo26/bin/python -m pip --isolated install \
  ultralytics==8.4.182 --constraint constraints.txt \
  --index-url https://pypi.org/simple \
  --no-cache-dir --progress-bar off --report reports/pip_ultralytics_install.json \
  2>&1 | tee reports/pip_ultralytics_install.log
```

`constraints.txt` 固定 torch、torchvision、Ultralytics 三个版本。`--isolated` 避免用户级 pip 设置影响指定安装源，`--no-cache-dir` 避免保留大体积下载缓存。

| 组件 | 实际版本 |
| --- | --- |
| Python | 3.11.16 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| Ultralytics | 8.4.182 |
| PyTorch CUDA 构建版本 `torch.version.cuda` | 12.8 |
| 实际加载的 CUDA Runtime API 版本 | 12.8，`cudaRuntimeGetVersion=12080` |
| CUDA Runtime wheel | nvidia-cuda-runtime-cu12 12.8.90 |
| cuDNN wheel | nvidia-cudnn-cu12 9.10.2.21 |
| NVIDIA 驱动 | 580.142 |
| 驱动报告支持的 CUDA 版本 | 13.0，与本环境实际 Runtime 区分记录 |

其余实际版本见 `requirements.lock.txt`。已通过 `python -m pip check`，无依赖冲突。

选择 PyTorch 稳定版 2.10.0 / torchvision 0.25.0 的 CUDA 12.8 构建；官方源存在 Python 3.11、Linux x86_64 wheel。使用独立环境内的 CUDA 运行库，未安装或修改系统 CUDA，也未修改系统 Python、NVIDIA 驱动或 base 环境。没有使用 nightly。

- [PyTorch 官方版本配对及安装命令](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA 驱动与 CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)：580 系列驱动可向后兼容 CUDA 12.x。
- [Ultralytics 官方安装说明](https://docs.ultralytics.com/quickstart/)
- [Ultralytics 8.4.182](https://pypi.org/project/ultralytics/8.4.182/)：安装前已在内存检查 wheel，包含 Pose26 实现。

## 权重

- 文件：`weights/yolo26s-pose.pt`
- 官方下载 URL：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt`
- 下载方式：专用环境的 Python `urllib.request.urlopen` 流式写入 `.part` 文件，校验大小及 SHA-256 后重命名。
- 实际大小：24,151,790 字节。
- SHA-256：`a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646`
- 哈希与 [Ultralytics 官方 Hugging Face 权重页](https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26s-pose.pt) 一致。来源及下载时间见 `reports/weight_source.json`。
- 保留原始 COCO 权重，不保存经过修改的模型；[官方 Pose 模型原本包含 17 个关键点](https://docs.ultralytics.com/tasks/pose)。

重新下载到空的目标路径可使用：

```bash
curl --fail --location --retry 3 \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt \
  --output weights/yolo26s-pose.pt.part
printf '%s  %s\n' \
  a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646 \
  weights/yolo26s-pose.pt.part | sha256sum --check
# 确认校验通过且目标权重不存在后，将 .part 重命名为 .pt。
```

## 验证与复现

检查脚本默认路径相对于项目根目录解析；支持 `--output`，模型脚本另支持 `--weights`。异常写入 JSON 并返回非零退出码。两个脚本都要求实际 CUDA，不允许 CPU 回退。Ultralytics 和 matplotlib 配置隔离在本项目 `.cache/` 内。

```bash
python scripts/check_env.py
python scripts/check_model.py
python -m pip check
```

以上两个检查脚本由专用环境的 `/home/ubuntu/miniconda3/envs/yolo26/bin/python` 在可访问 GPU 的执行上下文运行，均返回退出码 0。

| 验收项 | 实际结果 |
| --- | --- |
| 激活环境、Python 3.11 | 通过，`reports/activation_check.json` |
| PyTorch / torchvision / Ultralytics 导入 | 通过 |
| `torch.cuda.is_available()` | `true` |
| CUDA 矩阵乘法 | `ones(128,128) @ full((128,64),0.5)`，输入和输出均为 `cuda:0` / FP32，输出全部等于 64，有限值与同步检查通过 |
| CUDA Runtime 来源 | 环境内 `site-packages/nvidia/cuda_runtime/lib/libcudart.so.12`，实际 API 查询为 12.8 |
| 模型加载 | `PoseModel` / `Pose26`，任务 `pose`，模型参数均在 `cuda:0`，FP32 |
| 原始关键点 | 从模型头和 YAML 读取均为 `[17, 3]` |
| 实际前向输入 | Hook 记录到 `[1,3,640,640]`、`cuda:0`、FP32；包括库内部 warmup 和本次图片预测 |
| 返回结构 | 1 个 `Results`，0 个检测，boxes 为 `[0,6]`，keypoints 为 `[0,17,3]`；两个输出张量都在 `cuda:0` |
| 同步及输出检查 | 通过；合成图没有检测目标属于正常结果 |
| 依赖一致性 | `reports/pip_check.json`：`No broken requirements found.` |

环境结果在 `reports/env_check.json`，模型结果在 `reports/model_check.json`。驱动报告总显存 49,140 MiB；PyTorch 设备属性报告可见总显存 48,499.0625 MiB，两种来源的原值分别保留。

本次 Ultralytics 记录的单图 inference 为约 3.52 ms；包含 predictor 初始化和 warmup 的预测调用墙钟时间约 736.14 ms；PyTorch 峰值分配约 164.69 MiB。这是一次前向检查，不代表持续吞吐量或实际业务 FPS。

补充验证了缺失权重和显式设置 `CUDA_VISIBLE_DEVICES=''` 的失败路径：均写入失败 JSON 并返回退出码 1，不回退 CPU；从 `/tmp` 调用也正确解析项目路径。临时错误报告已清理，简要结果见 `reports/script_behavior_checks.json`。

### 导出与恢复

- `environment.yml`：从 `conda env export -p /home/ubuntu/miniconda3/envs/yolo26` 导出，保留 32 个 Conda 包的精确版本/构建及 56 个 pip 管理包；移除绝对 `prefix`，明确官方 Conda 源并添加 `nodefaults`，为 pip 部分补充官方 PyPI 和 PyTorch cu128 源。
- `requirements.lock.txt`：从专用环境 `python -m pip --isolated --no-cache-dir freeze --all` 导出 60 个 Python 分发包的精确版本；将 Conda 的 pip 构建机本地 URL 规范为实际版本 `pip==26.2.1`，避免不可移植路径，并保留 `+cu128` 标记及恢复源。
- `constraints.txt`：保留本阶段三个直接依赖的固定版本。

在 Linux x86_64 上恢复到另一个空环境名称：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
cd /home/ubuntu/BL/detect/yolo
conda env create -n yolo26_repro -f environment.yml
conda activate yolo26_repro
python scripts/check_env.py --output reports/env_check_repro.json
python scripts/check_model.py --output reports/model_check_repro.json
python -m pip check
```

或在已有的独立 Python 3.11 环境中安装锁定的 Python 包：

```bash
python -m pip --isolated install -r requirements.lock.txt
```

已核对导出内容与当前安装版本一致；本阶段没有额外创建第二套环境执行完整恢复测试。

## 已遇到的问题与边界

- 初始沙箱内 `nvidia-smi` 返回 “couldn't communicate with the NVIDIA driver”；沙箱外只读查询成功，确认是设备访问隔离。GPU 验证在具有设备访问权限的执行上下文运行，不修改驱动。
- 初始沙箱内联网返回 `Operation not permitted`；安装及下载使用获准的网络访问。
- GitHub API 元数据查询返回 HTTP 403 rate limit exceeded；官方 release 下载直链正常，已完成权重下载。
- pip 曾遇到 `RemoteDisconnected`，自动重试后下载成功，两个安装命令均返回 0。
- 初次模型检查提示 `half` 参数弃用；已根据安装版本的参数定义改为 `quantize=32`，复验通过且不再出现该提示。
- 当前没有未解决的验收阻塞。重新检查时仍需允许访问 GPU；受限沙箱内失败不能视为模型或驱动故障。
- 本阶段只建立环境并验证原始模型；未启动训练、未读取或修改项目数据集，合成图仅在内存中生成，不保存验证图片。
- 后续阶段才通过数据配置设置 `kpt_shape: [4, 3]`，实现顺时针四角及点 1—2 为长边、不区分正反的任务定义。本阶段未验证包埋盒识别精度。


## 第 2 阶段：数据检查与首轮四点训练（2026-10-19）

用户在本阶段授权检查 `/home/ubuntu/BL/detect/clean`，满足要求后启动训练。结论：`cassette_4videos_2fps_yolo` 满足首轮基线训练要求，已于 2026-10-19T19:52:12.669155+08:00 启动，并于同日 19:58:29 正常完成 100 轮；`cassette_4videos_2fps_review` 的待复核初标未用于训练。完成后的效果评估见下方第 3 阶段。

### 数据检查

- 共 137 张图片、1,654 个目标、6,616 个关键点。训练集 121 张 / 1,484 个目标，验证集 16 张 / 170 个目标。
- 556 个文件的 SHA-256 校验全部通过；图片可解码，图片、TXT、manifest 和 samples.json 一一对应。
- 全部标签为类别 0 `cassette_top`、每行 17 列、四点 visibility=2，坐标有限且归一化有效。四边形顺时针且凸，1—2 为长边，框等于角点外接框。
- 原视频 1、3、4 为训练，视频 2 为验证；无来源视频/原帧跨集合，也无文件哈希或解码像素完全相同的跨集合重复图片。
- 抽查 8 张预览，覆盖全部四个来源视频、完整帧和大小裁剪，目标语义和标注约定一致。没有逐目标重新人工描点；部分画面存在模糊、圆角和窄裁剪边缘，角点应视为近似标注。
- 实际 Ultralytics 加载器检查通过：train 121、val 16，损坏图片 0；四点张量和 batch 拼接正常。
- 结构检查结果为 `reports/dataset_audit.json`，加载检查为 `reports/dataset_loader_check.json`，视觉抽查说明为 `reports/dataset_visual_review.json`，启动时快照也保存在本次任务目录。

原始 clean 目录没有写入或改标注；训练使用校验后复制的 `datasets/cassette_4videos_2fps_yolo`，数据加载缓存只写入该副本。未改变原有划分。

限制：数据来自同一物理摆放场景，137 张导出图实际来自 92 个原帧；没有独立测试集或专门的负样本集，全部采用角点均标为可见。适合首轮拟合及当前视角验证，验证 mAP 不能证明跨场景泛化或部署精度。后续需补充独立场景、遮挡/出画和负样本，并评估像素角点误差。训练沿用 Ultralytics 对自定义四点的默认 OKS 配置，mAP 不是像素精度指标。

### 本次训练配置与运行位置

- 预训练权重：原始 `weights/yolo26s-pose.pt`，SHA-256 保持不变。
- 数据驱动重建 `[4,3]` Pose26 输出头，实际日志为 `nc=1`，成功迁移 795/879 个预训练项。
- 输入 1280；batch 8；最多 100 epochs；patience 30；AdamW，lr0=0.001，cosine schedule，warmup=3，seed=42；GPU `cuda:0`。
- 首轮使用 FP32（amp=False）；workers=4；关闭 flips、mosaic、mixup、cutmix、copy-paste；rotation=5°、translate=0.05、scale=0.2，仅亮度轻微增强。
- 保存 best.pt、last.pt 和逐 epoch CSV；plots=False，不生成训练预览图片。
- 配置：`/home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/train_config.json`
- 日志：`/home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/train.log`
- 动态状态：`/home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/status.json`
- 指标：`/home/ubuntu/BL/detect/yolo/runs/pose/cassette_top_yolo26s_20261019_195156/results.csv`
- 最优权重：`/home/ubuntu/BL/detect/yolo/runs/pose/cassette_top_yolo26s_20261019_195156/weights/best.pt`
- 最新权重：`/home/ubuntu/BL/detect/yolo/runs/pose/cassette_top_yolo26s_20261019_195156/weights/last.pt`

启动验证时已完成 22 轮，loss 均有限、训练/验证均实际执行，best.pt 和 last.pt 均已生成。实时进度以 status.json 和 results.csv 为准。初期 NVIDIA 查询训练进程使用约 23,246 MiB 显存；其他已有进程保留。

训练曾由 tmux 独立会话运行，现已结束；以下为当时的会话和日志查看命令（完成后会话可能已退出）：

```bash
tmux attach -t yolo26-20261019_195156
# 查看后使用 Ctrl-b 再按 d 可离开会话而保留训练。
tail -f /home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/train.log
cat /home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/status.json
```

当时实际启动的 Python 命令如下（仅作记录，无需重复启动）：

```bash
/home/ubuntu/miniconda3/envs/yolo26/bin/python -u \
  /home/ubuntu/BL/detect/yolo/scripts/train_cassette.py \
  --config /home/ubuntu/BL/detect/yolo/training/cassette_top_yolo26s_20261019_195156/train_config.json
```

完整参数见 train_config.json；后续新实验应设置新的 project/name 和任务目录。当前没有启动阻塞。第 1 阶段的 acceptance.json 保留当时的验收快照，本报告现已追加训练记录。

## 第 3 阶段：首轮训练效果评估（2026-10-19）

结论：首轮训练正常完成，当前验证场景的目标召回较高，但四角定位仍有明显误差。若以“目标的四个角点均在标注 5 像素以内”为要求，当前仅 60/170 个验证目标（35.29%）达标；不能仅凭 94.18% 的 pose mAP50–95 判断已满足精密定位要求。

### 完成状态与最佳权重

- 正常完成 100/100 轮，没有触发提前停止；2026-10-19 19:52:12 至 19:58:29，耗时 376.71 秒（约 6 分 17 秒）。
- `best.pt` 对应第 99 轮。安装版本按 box mAP50–95 与 pose mAP50–95 之和选优，已核对 CSV 和权重中保存的指标。单独 pose mAP50–95 最高为第 93 轮的 94.466%，并非最终 best.pt 的选择规则。
- 最优权重：`runs/pose/cassette_top_yolo26s_20261019_195156/weights/best.pt`，大小 21,815,406 字节；SHA-256：`bdf9389fd088680f681891ea03f17db40a29cded58a9774a9185d009a627feb2`。
- 训练 pose loss 从 6.73968 降至 0.16853，验证 pose loss 从 7.06264 降至 0.29831；曲线呈收敛趋势，但同场景验证不足以排除泛化问题。

### Ultralytics 最佳权重验证指标

直接读取训练结束后重新验证 best.pt 的结果，来源为本次任务 `status.json` 和 `train.log`；验证集 16 张图、170 个目标。

| 指标 | 检测框 | 四点姿态 |
|---|---:|---:|
| Precision | 97.65% | 98.24% |
| Recall | 97.97% | 98.56% |
| mAP50 | 99.37% | 99.47% |
| mAP50–95 | 47.22% | 94.18% |

这里的 P/R 是验证器按 F1 曲线选择置信度后的指标，不是下面固定 conf=0.25 的结果。检测框在更严格 IoU 下的表现明显下降。四点姿态采用自定义关键点默认 OKS sigma=1/4=0.25，它按目标尺度评估距离，不能换算为固定像素容差；相关规则见 [Ultralytics 官方 Pose 数据集说明](https://docs.ultralytics.com/datasets/pose/)。

### GPU 复测：像素误差和实际匹配

使用 best.pt 在 `cuda:0` 重新推理全部验证图：FP32，imgsz=1280、rect=False、conf=0.25、iou=0.7、max_det=300。按检测框 IoU≥0.5 从大到小进行一对一匹配。像素误差为原图或未缩放裁剪图坐标中，对应编号角点的欧氏距离，不做旋转、平移或重排对齐；统计的是相对现有近似标注的误差，不是物理测量误差。

- 170 个标注目标，188 个预测，167 个匹配；21 个预测未匹配，3 个目标未匹配（未检出或检测框 IoU 不达标）。固定阈值 Precision=88.83%，Recall=98.24%。
- 对匹配的 167 个目标 / 668 个角点，平均误差 **4.81 px**，中位数 **3.54 px**，P90 **9.37 px**，P95 **12.46 px**，最大 **44.55 px**。未匹配目标未计入这些条件误差统计。
- 点 1、2、3、4 的平均误差分别为 3.42、5.55、5.83、4.45 px。
- 长边 1—2 的方向误差按 180° 等价计算：平均 1.28°，P95 3.09°，最大 5.07°。
- 本次 188 个预测均满足四点顺时针且凸、1—2 比相邻短边长；这只说明当前输出的几何顺序，不代表预测对象或位置全部正确。

| 每个目标四点均满足的容差 | 达标目标 / 全部 170 个目标 | 达标率 |
|---|---:|---:|
| ≤2 px | 1 / 170 | 0.59% |
| ≤3 px | 6 / 170 | 3.53% |
| ≤5 px | 60 / 170 | 35.29% |
| ≤10 px | 130 / 170 | 76.47% |

上表把未匹配目标记为失败，避免只报告匹配目标的平均误差而掩盖漏检。相同位置在不同帧中仍会重复出现，因此 170 个标注实例不是 170 个独立物理对象。

### 置信度阈值对比与失败样例

在同一批预测上增加置信度过滤，并重新匹配；没有更改模型或默认配置。这只是当前验证集的阈值对比，不是独立测试集上的部署结论。

| 置信度 | 匹配目标 | 未匹配预测 | 未匹配目标 | Precision | Recall | 四点均≤5 px |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 167 | 21 | 3 | 88.83% | 98.24% | 60/170 |
| 0.50 | 167 | 13 | 3 | 92.78% | 98.24% | 60/170 |
| 0.75 | 161 | 2 | 9 | 98.77% | 94.71% | 60/170 |
| 0.85 | 140 | 1 | 30 | 99.29% | 82.35% | 58/170 |
| 0.90 | 61 | 0 | 109 | 100.00% | 35.88% | 18/170 |

完整帧 `v2_f0000_t000000` 在 conf=0.25 时有 13 个未匹配预测。已目视检查角点叠加图，可见空白背景及金属架上的误检，也有重复预测。误差较大的 F 组裁剪图中，预测右侧角点明显向内偏，部分下边缘偏高；例如 `v2_f0022_t011000_F/F01` 平均误差 21.51 px、最大 44.55 px。提高置信度会减少误报，也会降低召回；它没有增加本次四点均≤5 px 的目标数。

当前数据来自同一物理摆放场景，验证只覆盖视频 2 的 10 个来源帧，导出为 16 张图（1 张完整帧、15 张裁剪），且参与了 best.pt 的选择。没有独立测试集；背景负样本、遮挡和场景变化覆盖不足。下一轮应优先补充空背景/金属架等误检样本、核查大误差裁剪图角点标注，并增加独立摆放场景测试；本次只评估，未追加训练或修改数据及权重。

### 复现与产物

```bash
cd /home/ubuntu/BL/detect/yolo
/home/ubuntu/miniconda3/envs/yolo26/bin/python scripts/evaluate_cassette.py \
  --job training/cassette_top_yolo26s_20261019_195156
```

- 评估脚本：`scripts/evaluate_cassette.py`。
- 统计 JSON：`training/cassette_top_yolo26s_20261019_195156/evaluation_summary.json`，包含逐图结果、未匹配实例、最差目标、权重哈希和阈值对比。
- 分析曲线：`training/cassette_top_yolo26s_20261019_195156/test/evaluation_curves.png`，包括 AP、pose loss 和角点误差累计分布。
- 临时叠加图仅用于本次目视复核，复核后已清理。若通过 `--review-images` 重新生成，应在复核后清理 `test/temporary_*.png`。
- GPU 复测需要设备访问权限；本次已实际完成 CUDA 推理，没有 CPU 回退。原始 clean 数据、训练副本标签、best.pt 和 last.pt 均未修改。

## 第 4 阶段：big.mp4 近景视频选帧识别（2026-10-19）

对用户提供的 `/home/ubuntu/BL/detect/data/big.mp4` 完成 6 帧 GPU 推理及逐帧视觉检查。视频 720×1280，884 帧，约 29.5 秒；SHA-256 为 `009b790af22e3f741591a582ede632bfea01d102df5a41a83d36dc1970d3f0f5`。

先均匀抽取 15 个候选视角，人工挑选近景斜视、较正视、其他目标组、中景和末尾全景 6 种情况，再在每个候选前后 6 帧中按全图 Laplacian 方差选择较清晰帧。该指标只作清晰度筛选依据，全部选帧在模型推理前确定，未按模型表现挑选图片。

沿用第 99 轮 `best.pt`，权重哈希与第 3 阶段一致。使用 RTX 4090 的 `cuda:0`、FP32、imgsz=1280、rect=False、max_det=300；推理保存 conf≥0.25 候选，识别图统一展示 conf≥0.50 输出。模型参数及结果张量均实际在 CUDA。识别图的橙线为点 1—2，绿线为其他边，数字为点顺序；颜色不代表人工确认正确。

| 时间 | conf≥0.50 预测数 | 视觉判断 |
|---|---:|---|
| 2.30 秒 | 8 | 中央一列部分检出，顶部、底部及右下区域仍有漏检，部分角点不贴边。 |
| 4.43 秒 | 7 | 中央较大的整列目标明显漏检；输出主要落在横梁及盒盖透气孔附近。 |
| 8.70 秒 | 67 | 主要盒顶有较多检出，但右侧盒盖透气孔产生大量误检；误检 #29 置信度约 0.866。 |
| 14.70 秒 | 49 | 右侧和中央几组有检出，左侧及左上组漏检，部分区域重复或偏移。 |
| 23.10 秒 | 13 | 覆盖不全，下方目标组漏检，中部部分角点明显偏内。 |
| 29.23 秒 | 56 | 本次抽查中覆盖相对较好；上方仍有遗漏，左下局部存在重复或边界偏移。 |

结论：当前模型对本视频近景视角适应不足，目标占比变大没有带来稳定改善。外观与尺度变化、倾斜视角和负样本覆盖不足是可能因素，但距离、角度和光照同时变化，本次不能分离其因果贡献。单纯提高置信度无法消除高分透气孔误检。

这里是无标注视频的定性检查，预测数不能视为正确目标数，跨帧还存在同一物体重复出现。不能计算该视频真实 Precision、Recall、mAP 或角点像素误差，也不能沿用之前验证集的 4.81 px 均值。本次未训练、未改标注或权重。

产物统一位于 `test/big_video_yolo26s_20261019_212516/`：

- `index.html`：原图与识别图对照及逐帧判断，可离线查看。
- `recognition_overview.jpg`：六帧识别总览。
- `frames/`、`predictions/`：6 张原始 PNG 与 6 张完整识别 PNG。识别 PNG 上方增加 88 px 标题区；JSON 中角点坐标仍使用原图 720×1280 坐标。
- `predictions.json`：权重哈希、GPU、参数、阈值统计、全部候选框及四点坐标/置信度。
- `review.json`、`selection.json`、`video_metadata.json`：目视判断、选帧依据和输入来源。

复现推理（选出的原图已保留）：

```bash
cd /home/ubuntu/BL/detect/yolo
/home/ubuntu/miniconda3/envs/yolo26/bin/python scripts/predict_video_frames.py \
  --selection test/big_video_yolo26s_20261019_212516/selection.json --conf 0.5
```

完整 6 张识别图和总览已目视检查；临时选帧联系图已删除，用户请求的原图与识别图保留在对应 `test/` 任务目录。

## 第 5 阶段：新数据包独立训练（2026-10-21）

用户提供新数据包 `datasets/cassette_yolo26_training`，要求单独新训练，不续训旧模型。本次从官方 COCO `weights/yolo26s-pose.pt` 初始化，SHA-256 为 `a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646`；不使用之前的 cassette best.pt，`resume=False`。

任务目录：`training/cassette_partial_yolo26s_20261021_025446`。创建了该目录下的独立 `dataset/` 副本，600 个文件逐项与输入核对 SHA-256，原数据包保持不变。训练加载器缓存只进入副本。

新包为两类别四点任务：class 0 `cassette`，class 1 `occupied_group`；`kpt_shape=[4,3]`。单盒点 1—2 为完整长边端点，点 3—4 为小凸角外侧角点；组框沿用包内定义。这与旧单类四角模型的标签语义、数据划分均有变化，不能直接横向比较两次汇总 mAP。

- 训练：147 张，视频 1/2/4，3,882 个单盒与 255 个组框。
- 验证：46 张，视频 3，877 个单盒与 46 个组框。
- 193 张原图哈希、720×1280 尺寸、17 列标签、坐标回转、正样本与黄色互斥、忽略范围覆盖均通过；无来源视频或解码像素重复的跨集合泄漏。
- 所有 5,060 个正样本四边形顺时针且凸。26 个单盒的 1—2 边在图像上的长度略短于某条连接边；新包点 3—4 属于小凸角，不能按投影边长重排语义点。已抽查四个视频及 6 个代表异常实例，保留已审核点位，不修改标签。
- 新包不完整标注的黄色目标不进入正样本，需要包内 `IgnorePoseTrainer` / `IgnorePoseValidator`。直接使用普通 YOLO CLI 会丢失忽略规则，因此本次原样执行副本 `train.py`，外层脚本只增加完整性检查和状态回调。
- RTX 4090 GPU 检查通过：两张真实图片输入 `[2,3,640,384]`，真实前向/反向；one-to-many、one-to-one 两条分支的忽略负样本分类梯度均为 0，正样本梯度非零；无忽略范围时与官方损失等价。该检查没有 optimizer 更新或保存权重。

训练参数按数据包建议：最多 150 轮，patience=30，imgsz=1280，batch=4，workers=4，AdamW lr0=0.001、lrf=0.01，FP32（amp=False），seed=42。关闭所有几何增强和翻转，保留轻微 HSV。rect=True，实际首个训练 batch 为 `[4,3,1280,736]`，按完整图等比例 letterbox。初始可用显存约 38,705 MiB，保留机器上其他进程。

2026-10-21 02:57:58 启动，约 03:01:22 完成，共 51 轮、约 203.57 秒。连续 30 轮验证 fitness 未提升，按 patience=30 正常提前停止，最佳权重来自第 21 轮。实际训练器、验证器和两分支忽略损失均生效，全程损失有限，best.pt / last.pt 均已保存。状态与产物文件：

- `training/cassette_partial_yolo26s_20261021_025446/status.json`
- `training/cassette_partial_yolo26s_20261021_025446/train.log`
- `training/cassette_partial_yolo26s_20261021_025446/train_config.json`
- `training/cassette_partial_yolo26s_20261021_025446/actual_train_args.json`
- `training/cassette_partial_yolo26s_20261021_025446/dataset_check.json`、`supplemental_audit.json`、`gpu_smoke.json`
- `runs/pose/cassette_partial_yolo26s_20261021_025446/results.csv`
- `runs/pose/cassette_partial_yolo26s_20261021_025446/weights/best.pt`、`last.pt`

实际运行命令（记录用途，不要重复启动同一任务）：

```bash
/home/ubuntu/miniconda3/envs/yolo26/bin/python -u \
  /home/ubuntu/BL/detect/yolo/scripts/train_packaged_cassette.py \
  --config /home/ubuntu/BL/detect/yolo/training/cassette_partial_yolo26s_20261021_025446/train_config.json
```

训练通过 tmux 会话 `yolo26-partial-20261021_025446` 独立运行，现已结束。原入口命令及权重来源在 `launch.json`、`train_config.json` 中记录；适配器代码与数据一起留在副本中供复现。

评估限制：训练 88/83 张、验证 38/38 张图分别对单盒/组框停用全图背景分类监督，已审核正样本仍受监督。验证器会忽略不确定区域中与已标注目标无关的预测。因此指标只针对可评价的部分标签，不能作为整图完整检出率、数量准确率或独立部署场景表现；旧模型及旧实验目录保持不变。

### 完成验证与结果

最佳权重重新验证结果（两类别均值）：检测框 Precision 73.16%、Recall 86.61%、mAP50 79.36%、mAP50–95 43.66%；四点 Pose Precision 78.34%、Recall 88.28%、mAP50 86.12%、mAP50–95 75.01%。本次验证计入 46 张图片、877 个单盒和 46 个组框；忽略了 5,921 个落在不确定区域且与已标注目标无关的低阈值预测。

| 类别 | 验证标注数 | 检测框 mAP50 | 检测框 mAP50–95 | Pose Precision | Pose Recall |
|---|---:|---:|---:|---:|---:|
| cassette 单盒 | 877 | 60.14% | 24.33% | 61.10% | 82.67% |
| occupied_group 组框 | 46 | 98.58% | 63.00% | 95.57% | 93.89% |

单盒类别的识别与定位仍明显弱于组框，不能用两类别均值概括单盒效果。完整训练汇总为 `training/cassette_partial_yolo26s_20261021_025446/training_summary.json`；逐轮数据位于本次 `results.csv`。

最佳权重 SHA-256：`da0001e5a7835ad910cf5d79b5a42bd0583b2f04c8a47bd7d5b743c4d7e1dfbb`。普通 Ultralytics 在未导入训练适配器的独立进程中加载 best.pt 成功，在 cuda:0 对一张验证图推理返回 `[35,4,3]` 关键点张量，两类别名称正确；此步骤只验证文件兼容性，不视为精度验收。结果见 `checkpoint_compatibility.json`。

训练及最终验证完成后，首次写入每类统计遇到 `TypeError: Object of type int64 is not JSON serializable`。原因是 Ultralytics summary 的图片/目标计数为 NumPy 整数；已修复状态回调的标量转换，重新验证 best.pt 并恢复 status.json 为 completed，没有重跑或续训。原始错误保留在 train.log，恢复经过记录于 `reporting_recovery`，恢复前状态保存在 `status_before_reporting_recovery.json`。训练结束时间采用日志最后修改时间，统计修复在约 03:03:45 完成。

完成后再次核对 600 个输入与副本文件的哈希、官方预训练权重哈希及旧 best.pt 哈希，均保持不变。没有保留临时视觉检查图片。

### 第 21 / 51 轮在原六帧上的推理对比

按用户要求，对第一版 `big.mp4` 的同六张原始帧分别运行新版 best.pt（第 21 轮）和 last.pt（第 51 轮）。通过已记录的权重 SHA-256 以及检查点验证指标与 results.csv 的对应关系核验轮次；未启动训练。六张原图和两份权重在推理前后哈希不变。

产物保存在 `test/big_video_epochs21_51_20261021/`：12 张独立 PNG、`12_detection_photos.zip` 与完整预测 `predictions.json`。脚本为 `scripts/compare_cassette_checkpoints.py`。两组沿用第一版推理设置：imgsz=1280、rect=False、FP32、候选置信度 0.25、显示置信度 0.50、iou=0.7、max_det=300；实际前向输入为 cuda:0 上的 `[1,3,1280,1280]` FP32，输出也在 CUDA。单盒与组框均保留，不按几何形状或人工观感过滤。

| 帧时间 | 第 21 轮：单盒 / 组框 | 第 51 轮：单盒 / 组框 |
|---|---:|---:|
| 2.30 s | 12 / 2 | 10 / 2 |
| 4.43 s | 20 / 3 | 19 / 2 |
| 8.70 s | 20 / 2 | 14 / 1 |
| 14.70 s | 29 / 7 | 22 / 7 |
| 23.10 s | 34 / 4 | 22 / 4 |
| 29.23 s | 43 / 9 | 40 / 7 |

表中是置信度不低于 0.50 的预测数量，不是真实盒数或正确检测数。两组在近景仍有明显漏检；没有人工真值，不能据此计算精确率、召回率或角点像素误差。12 张图已目视检查，临时检查拼图已移除。图片保留完整 720×1280 原帧，顶部额外增加 112 px 图例区，JSON 坐标仍相对于原图。

复现命令（输出必须指定新的目录，以保留已有图）：

```bash
/home/ubuntu/miniconda3/envs/yolo26/bin/python scripts/compare_cassette_checkpoints.py \
  --output test/big_video_epochs21_51_new
```

### 三个模型的近景测试（2026-10-21）

用户要求测试离得很近时三个模型能否识别。本次运行第一版 best.pt、新版第 21 轮 best.pt、新版第 51 轮 last.pt，各对 2.30、4.43、8.70 秒三个近景原帧及其中央 384×384 裁剪执行推理，共 18 次真实 CUDA FP32 前向。原视频 big.mp4 当前已不在原路径，使用之前保存且 SHA-256 核对通过的原始帧。

产物：`test/closeup_three_models_20261021/`，包含 18 张独立预测图、6 张三模型对比图、原像素裁剪图、`predictions.json`、`review.json`、`artifact_check.json` 和照片压缩包。只比较 class 0 单盒，组框不画在对比图中，但保留在预测 JSON。参数与之前一致：imgsz=1280、rect=False、FP32、conf=0.25，绘制和下表统计阈值为 0.50。原图输入等比例缩放系数为 1；裁剪输入放大到 1280×1280，相对原始像素为 3.333 倍。

| 时间 | 第一版：原图中央区 / 裁剪 | 新版 21 轮：原图中央区 / 裁剪 | 新版 51 轮：原图中央区 / 裁剪 |
|---|---:|---:|---:|
| 2.30 s | 5 / 8 | 2 / 0 | 0 / 1 |
| 4.43 s | 0 / 7 | 1 / 0 | 0 / 0 |
| 8.70 s | 6 / 5 | 1 / 0 | 0 / 0 |

原图中央区统计预测框中心落在黄色裁剪矩形内的候选数；不是人工确认的正确目标数或召回率。第一版裁剪后的四点输出覆盖多个白色盒体区域，但部分薄边与宽面重复出现框，点位/目标面语义仍需核实。新版 21 轮三张裁剪图均无超过 0.50 的输出；新版 51 轮唯一输出位于左侧支架背景，未覆盖盒顶。4.43 s 裁剪两份新版权重在 conf=0.25 时也没有单盒输出。

本组测试中，第一版对放大后的盒体保留较强响应，新版两权重对这些大目标输入明显失效；三个模型均未证明近景四点定位可靠。原始实拍近景也有明显漏检和孔洞误检。裁剪同时改变尺度、上下文和画面比例，不增加图像细节，不能视为真实相机进一步靠近的完整替代，也不能单独分离物理距离的因果作用。未测量相机距离，没有四点真值，不报告厘米范围、准确率或像素误差。

已目视检查全部 18 次输出组成的 6 张对比图，图片及压缩包校验通过，原始帧和三份权重哈希保持不变；未训练或修改数据集。复现脚本：

```bash
/home/ubuntu/miniconda3/envs/yolo26/bin/python scripts/test_cassette_closeup.py \
  --output test/closeup_three_models_new
```


## 2026-10-24：data_405 正面长边两端点试训（已完成）

用户明确要求使用现有候选标注先试训。独立运行 `cassette_front_2pt_data405_20261024_215915`，由官方 YOLO26s-Pose 初始化，类别 `cassette_front`，`kpt_shape: [2, 3]`，没有续训旧四点模型。完整快照 903 张，700 训练 / 183 验证 / 20 时序间隔；没有混入原测试样例，也没有修改人工复核训练清单。

上限 50 轮，patience=12；实际 19 轮，最佳第 7 轮，RTX 4090 上用时 327.9 秒。最终 best.pt 验证 Box mAP50=0.8576、Box mAP50–95=0.4486、Pose mAP50=0.9215、Pose mAP50–95=0.8640。标签多数尚未精修，同视频验证、默认 OKS sigma=[0.5,0.5]，这些分数不能作为真实定位精度承诺。

四张精修验证图：18/20 个目标匹配；预测 24 个、未匹配 6 个；32 个可见端点误差中位数 7.52 像素，5 像素内仅 6/32。模型可用于试验，仍需改进标注和精确定位。

- [详细训练报告](training/cassette_front_2pt_data405_20261024_215915/TRAINING_REPORT.md)
- [预测对比 HTML](training/cassette_front_2pt_data405_20261024_215915/evaluation/index.html)
- [最佳权重](runs/pose/cassette_front_2pt_data405_20261024_215915/weights/best.pt)
- [配置与状态](training/cassette_front_2pt_data405_20261024_215915/train_config.json)


## 2026-10-24：data_405 两点模型累计 100 轮续训完成

从第一版第 19 轮 last.pt 权重再训练 81 轮，累计 100 轮，patience=0，优化器因旧检查点已清除训练状态而重新初始化。独立运行 `cassette_front_2pt_data405_total100_20261024_224433`，本阶段最佳为累计第 34 轮，保留 best.pt、最后第 100 轮 last.pt 和完整 resume.pt。

[详细报告](training/cassette_front_2pt_data405_total100_20261024_224433/TRAINING_REPORT.md) · [浏览器结果](training/cassette_front_2pt_data405_total100_20261024_224433/results.html)。
