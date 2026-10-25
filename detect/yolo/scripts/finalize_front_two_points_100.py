#!/usr/bin/env python3
"""Finalize a 19+81 epoch continuation and compare best/epoch-100 checkpoints."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    root = Path(__file__).resolve().parents[1]
    status_path = job / "finalization_status.json"
    write_json(status_path, {"status": "waiting_for_training", "pid": os.getpid()})
    try:
        while True:
            state = json.loads((job / "status.json").read_text())
            if state["status"] == "completed":
                break
            if state["status"] == "failed":
                raise RuntimeError(f"Training failed: {state.get('error')}")
            os.kill(state["pid"], 0)
            time.sleep(5)
        config = json.loads((job / "train_config.json").read_text())
        prior = Path(config["prior_job"])
        assert state["epochs_completed"] == 81 and state["cumulative_epochs_completed"] == 100
        assert config["train"]["patience"] == 0
        for tag in ("best", "last"):
            write_json(status_path, {"status": f"evaluating_{tag}", "pid": os.getpid()})
            command = [sys.executable, str(root / "scripts/evaluate_front_two_points.py"), "--job", str(job), "--checkpoint", tag]
            with (job / f"evaluation_{tag}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
        write_json(status_path, {"status": "writing_report", "pid": os.getpid()})
        os.environ["YOLO_CONFIG_DIR"] = str(job / "cache/ultralytics")
        os.environ["MPLCONFIGDIR"] = str(job / "cache/matplotlib")
        os.environ["YOLO_OFFLINE"] = "true"
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        best = json.loads((job / "training_summary.json").read_text())
        last = json.loads((job / "last_checkpoint_summary.json").read_text())
        baseline = json.loads((prior / "training_summary.json").read_text())
        run = Path(state["save_dir"])
        old_state = json.loads((prior / "status.json").read_text())
        old_run = Path(old_state["save_dir"])
        history = []
        for phase, directory, offset in ((1, old_run, 0), (2, run, 19)):
            for row in csv.DictReader((directory / "results.csv").open()):
                r = {k: float(v) for k, v in row.items()}
                r.update(phase=phase, phase_epoch=int(r["epoch"]), epoch=offset+int(r["epoch"]))
                history.append(r)
        assert [r["epoch"] for r in history] == list(range(1, 101))
        with (job / "cumulative_results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        resume = torch.load(run / "weights/resume.pt", map_location="cpu", weights_only=False)
        assert resume["epoch"] == 80 and resume["optimizer"] is not None and resume["ema"] is not None
        assert list(resume["ema"].model[-1].kpt_shape) == [2, 3]
        assert resume["train_args"]["epochs"] == 81 and resume["train_args"]["patience"] == 0
        source_weights = Path(config["weights"])
        assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == config["weights_sha256"]
        manifest = list(csv.DictReader((Path(config["snapshot_dataset"]) / "manifest.csv").open()))
        source = Path(config["source_dataset"])
        unchanged = sum(hashlib.sha256((source / "labels/train" / f'{r["id"]}.txt').read_bytes()).hexdigest() == r["label_sha256"] for r in manifest)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        keys = [("train/box_loss", "val/box_loss"), ("train/pose_loss", "val/pose_loss"), ("metrics/mAP50(B)", "metrics/mAP50-95(B)"), ("metrics/mAP50(P)", "metrics/mAP50-95(P)")]
        for ax, names in zip(axes.flat, keys):
            for key in names:
                ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
            ax.axvline(19.5, color="gray", linestyle="--", label="Optimizer restart")
            ax.set_xlabel("Cumulative epoch"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.savefig(job / "cumulative_training_curves.png", dpi=160); plt.close(fig)
        comparisons = []
        for label, data, epoch in (("第一版 best", baseline, baseline["best_epoch"]),
                                    ("续训 best", best, best["selected_cumulative_epoch"]),
                                    ("第 100 轮", last, 100)):
            metrics = data.get("selected_checkpoint_validation_metrics", data["final_best_validation_metrics"])
            manual = data["four_reviewed_validation_frames"]
            comparisons.append({"name": label, "cumulative_epoch": epoch,
                                "box_map50": metrics["metrics/mAP50(B)"], "box_map50_95": metrics["metrics/mAP50-95(B)"],
                                "pose_map50": metrics["metrics/mAP50(P)"], "pose_map50_95": metrics["metrics/mAP50-95(P)"],
                                "matched_objects": manual["matched_objects_iou50"], "unmatched_predictions": manual["unmatched_predictions_iou50"],
                                "endpoint_median_error_px": manual["matched_visible_endpoint_error_median_px"],
                                "endpoints_within_5px": manual["confident_endpoints_within_5px"],
                                "endpoints_within_10px": manual["confident_endpoints_within_10px"],
                                "measured_visible_endpoints": manual["matched_visible_endpoints"]})
        overall = {"status": "completed", "prior_epochs": 19, "additional_epochs": 81, "total_epochs": 100,
                   "early_stopping": False, "continuation_mode": config["continuation_mode"],
                   "additional_training_seconds": state["elapsed_seconds"], "continuation_best_cumulative_epoch": best["selected_cumulative_epoch"],
                   "full_resume_checkpoint": str(run / "weights/resume.pt"), "resume_checkpoint_verified": True,
                   "resume_checkpoint_sha256": hashlib.sha256((run / "weights/resume.pt").read_bytes()).hexdigest(),
                   "source_labels_matching_initial_snapshot": unchanged, "source_label_count": len(manifest),
                   "protected_source_files_unchanged": all(best["source_guard_files_unchanged"].values()),
                   "comparison": comparisons}
        write_json(job / "continuation_summary.json", overall)
        table_header = "| 模型 | 累计轮数 | Box mAP50 | Box mAP50–95 | Pose mAP50–95 | 匹配目标 / 20 | 未匹配预测 | 端点中位误差 px | ≤5px / 32 | ≤10px / 32 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        table_rows = "".join(f'| {c["name"]} | {c["cumulative_epoch"]} | {c["box_map50"]:.4f} | {c["box_map50_95"]:.4f} | {c["pose_map50_95"]:.4f} | {c["matched_objects"]} | {c["unmatched_predictions"]} | {c["endpoint_median_error_px"]:.2f} | {c["endpoints_within_5px"]} | {c["endpoints_within_10px"]} |\n' for c in comparisons)
        report = f'''# 包埋盒两端点：累计 100 轮训练完成

在第一版 19 轮基础上，从第 19 轮 `last.pt` 的模型权重继续训练 81 轮，累计完成 100 轮。`patience=0`，未提前停止。本阶段耗时 {state['elapsed_seconds']/60:.2f} 分钟。保留第一版全部文件，未覆盖旧模型。

上一版 Ultralytics 在训练结束时移除了优化器和 EMA 训练状态，所以本次是继承权重的连续微调，优化器重新初始化，并非完整状态无缝恢复。本阶段日志的第 1—81 轮对应累计第 20—100 轮。起始学习率延续上次最终记录的 0.0006436，以 1 轮预热后逐步衰减。原参数仍为 imgsz=1280、batch=4、AdamW、FP32、seed=42。

## 文件

- [第 100 轮模型 last.pt](../../runs/pose/{run.name}/weights/last.pt)
- [本阶段最佳模型 best.pt](../../runs/pose/{run.name}/weights/best.pt)：累计第 {best['selected_cumulative_epoch']} 轮。
- [完整恢复检查点 resume.pt](../../runs/pose/{run.name}/weights/resume.pt)：包含累计第 100 轮对应的模型、优化器和 EMA 状态。本阶段内部 epoch=80（从 0 开始）。已实际加载验证。
- [结果浏览页面](results.html)，其中可打开最佳和第 100 轮的各 13 张预测对比图。
- [完整 100 轮曲线](cumulative_training_curves.png)、[合并 CSV](cumulative_results.csv)、[日志](train.log)、[参数](train_config.json)、[状态](status.json)。

## 与第一版比较

''' + table_header + table_rows + f'''
上表自动 mAP 来自相同 183 张验证图。端点误差和匹配来自第 766—769 张精修图，置信度为检测 0.25、端点 0.5，采用 IoU≥0.5 一对一匹配。像素中位误差仅统计匹配目标的可见端点；三个模型实际参与统计的点数分别为 {', '.join(str(c['measured_visible_endpoints']) for c in comparisons)}。≤5px、≤10px 以全部 32 个已标注可见端点为分母，未检出或低置信度的点不算通过。

训练和验证来自同一视频，且大多数标签仍是自动候选；四张精修图也参与了验证和模型选择，不能当作独立测试。默认 Pose OKS sigma=[0.5,0.5]，mAP 较高不代表精确到 5 像素。训练到 100 轮是轮数要求，是否改善精度应依据上述比较和预测图判断。

## 数据保护与验收

沿用第一版经过哈希验证的独立快照：700 张训练（24,468 实例）、183 张验证（4,124 实例）、20 张时序间隔。没有混入原测试样例。类别 cassette_front，kpt_shape=[2,3]，左右端点；训练和推理均已检查模型形状。

合并记录逐轮确认累计 epoch 为 1—100，没有缺失或重复。两个阶段均使用原始未画线图片；没有修改任何人工复核状态。交付时原标签与首次快照匹配 {unchanged}/903 张；受保护配置、清单与样例哈希检查结果为 {overall['protected_source_files_unchanged']}。如用户在编辑器里同期修改，原快照仍不受影响。

第 100 轮模型 SHA-256：`{last['last_weights_sha256']}`。

本阶段最佳模型 SHA-256：`{best['best_weights_sha256']}`。

完整恢复检查点 SHA-256：`{overall['resume_checkpoint_sha256']}`。
'''
        (job / "TRAINING_REPORT.md").write_text(report)
        tr = "".join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["cumulative_epoch"]}</td><td>{c["box_map50_95"]:.4f}</td><td>{c["pose_map50_95"]:.4f}</td><td>{c["matched_objects"]}/20</td><td>{c["endpoint_median_error_px"]:.2f} px</td><td>{c["endpoints_within_5px"]}/32</td></tr>' for c in comparisons)
        (job / "results.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>包埋盒 · 100 轮训练结果</title><style>body{font:16px system-ui;background:#101923;color:#edf2f8;margin:32px;line-height:1.8}a{color:#70d9f4}table{border-collapse:collapse;width:100%}th,td{border:1px solid #405367;padding:8px}img{max-width:100%}.note{color:#ffc674}nav{display:flex;gap:24px;flex-wrap:wrap}</style><h1>累计 100 轮训练已完成</h1><p>上一阶段 19 轮 + 本阶段 81 轮；关闭提前停止；继承第 19 轮权重并重新初始化优化器。</p><nav><a href="evaluation/index.html">查看本阶段最佳模型预测</a><a href="evaluation_last/index.html">查看第 100 轮模型预测</a><a href="TRAINING_REPORT.md">训练报告</a></nav><p class="note">大部分标签仍未精修。4 张精修图来自验证集，并非独立测试。Pose 分数不能替代端点像素误差。</p><table><tr><th>模型</th><th>累计轮数</th><th>Box mAP50–95</th><th>Pose mAP50–95</th><th>匹配目标</th><th>端点中位误差</th><th>5px 内端点</th></tr>' + tr + '</table><h2>完整 100 轮曲线</h2><img src="cumulative_training_curves.png" alt="100轮训练曲线"></html>')
        shutil.copy2(root / "scripts/evaluate_front_two_points.py", job / "evaluation_script_snapshot.py")
        with (root / "REPORT.md").open("a") as f:
            f.write(f'\n\n## 2026-10-24：data_405 两点模型累计 100 轮续训完成\n\n从第一版第 19 轮 last.pt 权重再训练 81 轮，累计 100 轮，patience=0，优化器因旧检查点已清除训练状态而重新初始化。独立运行 `{run.name}`，本阶段最佳为累计第 {best["selected_cumulative_epoch"]} 轮，保留 best.pt、最后第 100 轮 last.pt 和完整 resume.pt。\n\n[详细报告](training/{run.name}/TRAINING_REPORT.md) · [浏览器结果](training/{run.name}/results.html)。\n')
        write_json(status_path, {"status": "completed", "pid": os.getpid(), "total_epochs": 100, "report": str(job / "TRAINING_REPORT.md"), "comparison": comparisons})
        print(json.dumps(overall, ensure_ascii=False, indent=2))
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
