铁篮四角点数据集

125 张图片、125 个同名 TXT、125 个铁篮实例，均已保存复核。
类别 ID 保持为 1（basket）；类别 0 为预留 white_box，本包不含白盒实例。
点位：0 左上、1 右上、2 右下、3 左下。
TXT：class cx cy w h，随后四组 x y visibility，坐标按图片宽高归一化。
visibility=2 为可见；0 为未知或遮挡且未标注，坐标为 0 0。
遮挡点留空不表示漏复核。尚未划分训练集与验证集。
pose_schema.yaml 是配置片段，不是含 train/val 路径的完整训练配置。
images/ 和 labels/ 可用于现有 review_pose.py 复核工具。
