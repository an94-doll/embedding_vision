白盒两端点标注数据集

图片：320 张；标签：320 个；白盒实例：12987 个。
打包时间：2026-10-28T14:50:12+08:00

images/：原始图片。
labels/：对应同名 YOLO Pose TXT，保留最新保存内容。
pose_schema.yaml：类别、关键点及水平翻转配置。
manifest.json：逐图计数、尺寸和图片/TXT 的 SHA-256。
validation_report.json：本次数据格式检查结果。

每行一个白盒，共 11 列：
class_id cx cy width height left_x left_y left_v right_x right_y right_v

class_id 为 0（white_box）。框及关键点坐标均归一化。
关键点 ID 由固定顺序确定：第一个为左端点 0，第二个为右端点 1。
v 是独立的标注状态；可见端点为 2，未知端点为 0 0 0。
kpt_shape: [2, 3]；flip_idx: [1, 0]。

本包保留全部图片，尚未划分训练集和验证集。
