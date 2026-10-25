#!/usr/bin/env python3
"""Train the audited four-corner dataset and persist restart-independent status."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

from check_env import ROOT, configure_local_cache, project_path, write_report


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    job_dir = config_path.parent
    status_path = job_dir / "status.json"
    state = {"status": "starting", "started_at": now(), "updated_at": now(), "pid": os.getpid(), "config": str(config_path), "epochs_completed": 0, "epochs_requested": config["train"]["epochs"], "source_dataset": config["source_dataset"], "snapshot_dataset": config["snapshot_dataset"], "task": "pose", "expected_kpt_shape": [4, 3]}
    write_report(status_path, state)
    started = time.perf_counter()
    try:
        configure_local_cache()
        import torch
        from ultralytics import YOLO, settings
        settings.update({"datasets_dir": str(ROOT / "datasets"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs"), "sync": False})
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires CUDA; CPU fallback is disabled.")
        for report_name in ("dataset_audit.json", "dataset_loader_check.json"):
            if json.loads((ROOT / "reports" / report_name).read_text())["status"] != "passed":
                raise RuntimeError(f"Required audit has not passed: {report_name}")
        snapshot = Path(config["snapshot_dataset"])
        if hashlib.sha256((snapshot / "SHA256SUMS.txt").read_bytes()).hexdigest() != config["dataset_sha256_manifest"]:
            raise RuntimeError("Dataset manifest changed since audit.")
        # Recheck immutable training assets immediately before use; caches are separate.
        for line in (snapshot / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip(" *")
            if name.startswith(("images/", "labels/")) or name in ("data.yaml", "samples.json", "manifest.csv"):
                if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Training asset changed since audit: {name}")
        weights = Path(config["weights"])
        if hashlib.sha256(weights.read_bytes()).hexdigest() != config["weights_sha256"]:
            raise RuntimeError("Pretrained checkpoint changed since verification.")

        def persist():
            state["updated_at"] = now()
            state["elapsed_seconds"] = time.perf_counter() - started
            write_report(status_path, state)

        def training_started(trainer):
            net = trainer.model
            if list(net.model[-1].kpt_shape) != [4, 3] or trainer.device.type != "cuda":
                raise RuntimeError("Training model must have four keypoints and run on CUDA.")
            state.update({"status": "running", "device": str(trainer.device), "gpu": torch.cuda.get_device_name(trainer.device), "actual_kpt_shape": list(net.model[-1].kpt_shape), "save_dir": str(trainer.save_dir), "train_images": len(trainer.train_loader.dataset), "val_images": len(trainer.test_loader.dataset), "amp": bool(trainer.amp), "best_checkpoint": str(trainer.best), "last_checkpoint": str(trainer.last)})
            persist()

        def epoch_started(trainer):
            state["current_epoch"] = trainer.epoch + 1
            persist()

        def epoch_finished(trainer):
            state["epochs_completed"] = min(trainer.epoch + 1, trainer.epochs)
            state["metrics"] = {str(k): float(v) for k, v in trainer.metrics.items()}
            losses = {str(k): float(v) for k, v in trainer.label_loss_items(trainer.tloss).items()}
            state["train_losses"] = losses
            state["losses_finite"] = all(torch.isfinite(torch.tensor(v)).item() for v in losses.values())
            state["last_checkpoint_exists"] = trainer.last.exists()
            state["best_checkpoint_exists"] = trainer.best.exists()
            state["epoch_seconds"] = trainer.epoch_time
            state["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
            persist()

        model = YOLO(str(weights))
        model.add_callback("on_train_start", training_started)
        model.add_callback("on_train_epoch_start", epoch_started)
        model.add_callback("on_fit_epoch_end", epoch_finished)
        model.train(**config["train"])
        state.update({"status": "completed", "finished_at": now(), "best_checkpoint_exists": model.trainer.best.exists(), "last_checkpoint_exists": model.trainer.last.exists()})
        persist()
        return 0
    except BaseException as exc:
        state.update({"status": "failed", "updated_at": now(), "elapsed_seconds": time.perf_counter() - started, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=12)}})
        write_report(status_path, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
