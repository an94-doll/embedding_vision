# 包埋盒 YOLO 默认权重

用户于 2026-10-25 明确指定：后续 YOLO 操作使用第 127 轮作为 base。

- 默认权重：`/home/ubuntu/BL/detect/yolo/weights/base.pt`。
- 来源与 SHA-256：`/home/ubuntu/BL/detect/yolo/weights/base.json`。该文件记录的来源是 `cassette_reviewed277_total200_20261025_194050/weights/best_loss.pt`，第 127 轮，验证 loss 总和 0.99246。
- 新训练、检测、自动预标注均默认加载此 base。Python 脚本可使用 `detect/yolo/scripts/base_weights.py` 中的 `resolve_base()` 校验并取得路径。
- 新训练从此权重初始化，使用 `resume=False`。用户明确要求继续某次运行时，从那次运行的检查点恢复完整训练状态；不要将“继续训练”解释成重开训练。
- 除非用户明确要求更换，不要自动用未来训练的 `last.pt`、`best.pt` 或官方预训练权重替换 base。
- 模型为 `cassette_front` 单类、`kpt_shape: [2, 3]`、`flip_idx: [1, 0]`。两点依次为图像左端、右端；遵循用户修正后的明显长边两角标准。
- 可视化默认只显示两个关键点及其连线，隐藏边界框；框可作为可选显示项。
- 历史脚本仅作参考或提供复用函数；旧数据及旧权重已按用户要求清理。新任务使用以上 base，不要直接继承历史脚本内的旧模型默认值。
- 用户随后要求只保留最新一批数据及有用权重。当前数据在 `detect/data/data_last/`（原目录名为 `20261025T081334_943078Z`）；原视频、277 张原图、完整当前标注和人工修改 JSON 记录必须保留。历史重复预览、旧批次及旧权重已经删除，详见 `yolo/reports/cleanup_report.json`。
- 第 127 轮 base、第 200 轮完整检查点、最新训练报告和数据快照仍保留。运行目录下 `best_loss.pt` 是指向 base 的相对符号链接。
- 最新批次的独立测试图为 31、83、134、190、259，不纳入训练或模型选择。100 轮任务目录中的 dataset 是同一最新批次、供 200 轮训练共用的快照，不能当作旧批次删除。
