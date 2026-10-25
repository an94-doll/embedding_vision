# 包埋盒两点模型

当前默认 base 为用户选定的第 **127** 轮：[`weights/base.pt`](weights/base.pt)。它是原始 `best_loss.pt` 的完整副本，验证 loss 总和为 **0.99246**。来源、校验值及默认推理参数记录在 [`weights/base.json`](weights/base.json)。

后续新训练、检测及自动预标注使用这个 base，直到用户明确更换。当前训练入口为 `scripts/train_reviewed_two_points.py`，从 `../data/data_last/picture/work` 读取最新人工修正标注。新训练设置 `resume=False`，明确续训已有运行时恢复该运行的检查点。

在项目脚本中取得并校验默认权重：

```python
from base_weights import resolve_base
from ultralytics import YOLO

weights, metadata = resolve_base()
model = YOLO(str(weights))
```

直接加载时使用完整路径 `/home/ubuntu/BL/detect/yolo/weights/base.pt`。模型为单类 `cassette_front`、两个关键点，顺序为左、右。预览默认显示两点和连线，隐藏框。

旧批次数据、旧模型及重复缓存已清理，历史脚本与 `REPORT.md` 仅保留参考用途。当前运行的 `best_loss.pt` 以相对链接复用 base，另保留 `epoch200.pt`。100 轮任务目录中的 dataset 是最新训练共用的冻结快照，并非旧批次。

第 127 轮及第 200 轮的报告与五张独立测试图效果位于 [`training/cassette_reviewed277_total200_20261025_194050/`](training/cassette_reviewed277_total200_20261025_194050/)。历史配置记录当时的绝对路径；清理已移除中间恢复权重，因此它们属于运行档案，不应直接重跑。更换电脑后，使用当前脚本重新生成训练配置。
