# 第173轮模型对应训练数据

这里保存 `weights/cassette_lower_epoch173.pt` 实际训练、验证和选模使用的数据快照。图片、YOLO标签与当时冻结的文件逐字节一致；标注预览来自用户修正后的编辑器导出，SHA-256也与训练冻结清单一致。预览显示人工标注，不是模型预测。

| 划分 | 原图 | 标注预览 | YOLO标签文件 | 目标数 | 其中空场景图片 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 137 | 137 | 137 | 3772 | 13 |
| val | 20 | 20 | 20 | 664 | 3 |
| 合计 | 157 | 157 | 157 | 4436 | 16 |

训练集由106张新图和31张已确认的旧图组成；验证集为20张新图。保留原有划分，5张最终测试图、19张隔离图及49张未确认的旧图未纳入此包。验证集参与选模，不能当作独立最终测试集。

## 文件对应关系

```text
data/
  images/train/、images/val/          原始PNG图片
  labels/train/、labels/val/          YOLO两点Pose标签
  annotated/train/、annotated/val/    人工标注预览PNG
  data.yaml                         可移植的数据配置
  manifest.json                     逐图对应关系、来源类别、统计与SHA-256
  SHA256SUMS.txt                     本目录文件的校验值（不含本文件自身）
```

例如 `images/train/new_000001.png` 对应 `labels/train/new_000001.txt` 和 `annotated/train/new_000001_annotated.png`。实际划分以目录和清单为准。

原图尺寸为1280×720。标注预览保留编辑器的30像素页脚，尺寸为1280×750；页脚和已绘制的点线仅供人工检查。**训练只读取 `images/` 原图及 `labels/` 标签，不使用 `annotated/` 预览。** `data.yaml` 未写入机器绝对路径，可从仓库使用，也可向Ultralytics传入该配置文件的绝对路径。

## 标注含义

单类 `cassette_front`，`kpt_shape: [2, 3]`，`flip_idx: [1, 0]`。两个点是白色蜡块正面**下沿的左、右两个实际角点**，按图像X方向排序；不是上方唇缘或底托角点。预览用绿色表示左点、蓝色表示右点，两点均可定位时以红线连接。

每个目标的YOLO标签占一行，共11个数：

```text
class cx cy width height left_x left_y left_v right_x right_y right_v
```

坐标与框尺寸按原图宽高归一化。`v=2` 表示可见点；不能可靠定位的点为 `0 0 0`，没有凭遮挡边界猜测坐标。即使另一端不可定位，可见端仍保留。全部标签包含8066个可定位端点、806个不可定位端点。空场景保留原图与空标签文件，用于背景负样本。

## 下载与完整性核验

PNG图片通过Git LFS存储。在仓库根目录执行：

```bash
git lfs install
git lfs pull --include="data/**/*.png"
cd data
sha256sum -c SHA256SUMS.txt
```

若图片文件只有几行以 `version https://git-lfs.github.com/spec/v1` 开头的文本，说明只取得了LFS指针，需要执行上述下载命令。运行模型所需的权重可在仓库根目录用 `git lfs pull` 一并下载。

`manifest.json` 记录对应模型哈希、训练快照时间和每张图片的划分、目标数、端点数及原图／预览／标签哈希，便于核对来源。本次上传未修改原始数据或重新训练模型。
