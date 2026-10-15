# 02 组标注说明

图片目录 `images/02`，共 44 张 720×720 图片；完成 40 张，标签保存到 `labels/02`。
类别 1（back，反面），五点均可见，visibility=2。

以设定的顶部朝上观察目标，从左上角逆时针编号：
P1 左上角 → P2 左下角 → P3 底边中点 → P4 右下角 → P5 右上角。
三个矩形凹槽所在侧是底部。
所有角点取栅格所在面的实体边界，不取厚度侧面的外角。检测框覆盖整个盒子。
P3 取 P2/P4 的图像坐标中点，点位按每张图 720×720 尺寸归一化，标签每行 20 列。

跳过 frame_0010、frame_0011、frame_0012、frame_0013：主体上端超出图像上边缘；frame_0013 的栅格面基本可见，但顶部侧壁少量被裁，按整个主体完整的要求跳过。
跳过图片未生成空标签，具体原因见 `annotations/02_skipped.csv` 或 `02_skipped.json`。
训练时请排除这四张图片；可用 `annotations/02_valid_images.txt` 构建训练/验证划分，避免将未标注的主体当作背景。

本目录 jpg 是原尺寸标注预览；`annotations/02_review_1.jpg` 至 `02_review_5.jpg` 为放大核查图。
`annotations/02_points.json` 保存像素点位与检测框；`02_validation.json` 保存数量与格式检查结果。
所有完整图均已目测核查。模糊、反光边缘仍存在像素定位误差。

已进一步放大复核 frame_0019–frame_0025，修正栅格面上角点位置，并微调 frame_0019 的底角及重算 P3；扩大 frame_0022 检测框左边缘以包含完整轮廓。修正记录见 `annotations/02_nearby_accuracy_corrections.json`，最新集中预览为 `annotations/02_frames_0019_0025_review.jpg`。
