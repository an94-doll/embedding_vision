铁篮历史图片小规模诊断测试集：24 张，12 张四角完整、12 张部分标注。
来源：box_in_basket/data_with_basket；只保留 basket，类别 1 映射为 0，坐标/可见性不变。
筛除与当前 basket_pose 数据同名图片；对全部训练/验证图做 32×32 灰度缩略图平均绝对差筛查，阈值 8/255。
选择种子 42，完整选择记录见 selection.json。该筛查不证明采集序列独立；本集是历史诊断样本，不用于调参。
仅 images/test 和 labels/test 为新增副本。data.yaml 的 train/val 指向已有数据，是兼容框架的配置，不是重新划分。
评估模型：runs/pose/basket_4pt/weights/best.pt。
结果：runs/evaluate/basket_4pt_legacy24/metrics.json；visualizations 为 GT/预测对照。
评估配置：CPU、imgsz=1280、CONF=0.25、MATCH_IOU=0.5、POINT_CONF=0.5，标准 AP 使用 conf=0.001。
运行方式（项目根目录，yolo-pose 环境）：
python - <<'RUN'
import evaluate as ev
ev.WEIGHTS = ev.ROOT / 'runs/pose/basket_4pt/weights/best.pt'
ev.DATA = ev.ROOT / 'box_in_basket/basket_test_legacy_24/data.yaml'
ev.SPLIT = 'test'
ev.NAME = 'basket_4pt_legacy24'
ev.DEVICE = 'cpu'
ev.WORKERS = 0
ev.main()
RUN
