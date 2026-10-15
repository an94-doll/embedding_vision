# YOLO Pose 数据集划分

脚本：`split_dataset.py`，仅使用 Python 标准库。

默认输入为脚本所在目录的 `images/01–06` 和 `labels/01–06`。
01–03 为反面（类别 1），04–06 为正面（类别 0）。
脚本读取 `annotations/XX_skipped.json`，排除已记录的裁切图片；缺少标签或标签为空的图片也会排除。无效标签、类别不符或重复的图片内容会停止构建。

## 运行

```bash
# 检查输入并预览数量，不创建文件
python3 split_dataset.py --dry-run

# 创建数据集，默认随机种子 42、比例 7:2:1
python3 split_dataset.py

# 创建另一份划分，指定输出目录和种子
python3 split_dataset.py --seed 123 --output dataset_seed123
```

输出目录已存在时脚本会停止，请指定新的 `--output`。默认目录按脚本位置解析；手动传入的相对路径按运行时所在目录解析。

## 划分规则

正面、反面分别按比例计算数量，采用最大余数法取整，保持样本总数准确。
每个类别内部再为各拍摄组分配数量，使各组尽量按比例出现在三份集合中；组内按固定随机种子打乱。
同一输入和种子得到同一划分，不受目录枚举顺序影响。

当前 233 张有效图片的结果：

| 集合 | 正面 | 反面 | 总数 |
|---|---:|---:|---:|
| train | 70 | 93 | 163 |
| val | 20 | 27 | 47 |
| test | 10 | 13 | 23 |

当前裁切图片已从输入图片目录移除，因此本次新增排除记录为 0。

## 输出

```text
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── split_manifest.csv
├── excluded_manifest.csv
└── split_summary.json
```

图片和标签以组号作为文件名前缀，例如 `01_frame_0001.jpg` / `01_frame_0001.txt`。
图片与标签复制到输出目录；原始坐标、关键点顺序和可见性不变。
`split_manifest.csv` 保存来源路径、输出相对路径、集合、类别和图片 SHA-256；`excluded_manifest.csv` 保存排除原因；`split_summary.json` 保存种子、比例及每类/每组数量。

`data.yaml` 配置 `front/back`、`kpt_shape: [5, 3]` 和 `flip_idx: [4, 3, 2, 1, 0]`，可直接作为 YOLO Pose 的数据配置。
配置中的 `path` 是生成目录的绝对路径；若移动整个数据集，需要同步修改此字段。

同一拍摄组的连续帧会随机分散到不同集合，因此相邻图像可能非常相似。这份划分适合当前训练流程，泛化评估应另外使用独立拍摄序列。
