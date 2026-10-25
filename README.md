# baomai · 包埋盒长边两点检测

本分支在 `detect/` 保存最新一批完整数据、浏览器标注工具和 YOLO26 Pose 模型。

| 位置 | 内容 |
| --- | --- |
| `detect/data/data_last/color.mp4` | 最新原始视频 |
| `detect/data/data_last/picture/raw/` | 每秒 2 张，共 277 张原图 |
| `detect/data/data_last/picture/work/` | 图片、最新标签、预览、人工复核记录及 HTML 编辑器 |
| `detect/yolo/weights/base.pt` | 默认模型：第 127 轮 |
| `detect/yolo/scripts/` | 训练、检测和检查脚本 |
| `detect/yolo/training/` | 最新批次的数据划分、训练报告、loss 及五张测试图效果 |
| `detect/yolo/runs/pose/` | 最新 200 轮结果及第 200 轮权重 |

详细目录和打开方法见 [detect/readme.md](detect/readme.md)。大文件通过 Git LFS 保存，克隆后运行 `git lfs pull`。

仓库原有 `yolo-pose` 分支文件保持保留，原说明见 [README.yolo-pose.md](README.yolo-pose.md)。
