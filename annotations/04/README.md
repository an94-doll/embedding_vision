# 04 组正面标注

图像 images/04 共 43 张 720×720 图片；labels/04 已完成 38 个标签，类别 0（front，正面）。
frame_0025–frame_0029 主体上端被裁切，已记录并跳过，未生成空标签。训练时用 annotations/04_valid_images.txt 构建划分，排除跳过图片。

白色矩形斜面所在侧为底部。以顶部朝上观察，从左上角逆时针编号：P1 左上 → P2 左下 → P3 底边中点 → P4 右下 → P5 右上。
延续共面要求，五点取正面栅格面的边界，斜面用于区分上下。
P2/P4 取栅格面与白色斜面的交界两端，凸舌或缺口处按主栅格面边线延伸近似定位；不取斜面下缘外角或厚度侧面角点。
P3 为 P2/P4 的图像坐标中点，检测框覆盖整个主体（含斜面、侧壁及凸舌）。
所有点可见，visibility=2，每行 20 列，位置按 720×720 归一化。
沿用 annotations/pose_schema.yaml：0 front，1 back，kpt_shape [5,3]，flip_idx [4,3,2,1,0]。

本目录 jpg 是原尺寸标注预览；annotations/04_review_1.jpg 至 04_review_5.jpg 为逐批核查图。
像素记录见 annotations/04_points.json，跳过原因见 04_skipped.csv / 04_skipped.json。
已逐张目测检查方向、完整性、栅格面角点，并通过标签匹配、类别、格式、坐标范围、框内点位、中点与点序校验。
模糊、反光和斜面交界存在人工目测误差；格式检查不等同于三维共面测量。

已进一步复核 frame_0015 起的 24 张有效帧：统一 P5 到主面直边延长交点，并将偏移 P2 调整到主栅格面下边与左边的交点，避免凸舌下缘；联动校正两帧 P4 并重算 P3。原标注备份在 annotations/04_before_p5_p2_refinement，修正记录在 04_p5_p2_corrections.json，集中预览为 04_precision_review_1.jpg 至 04_precision_review_3.jpg。斜面交界缺乏清晰边缘的帧仍为目测近似。
