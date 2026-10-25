#!/usr/bin/env python3
"""Freeze the data_405 candidate labels and train an isolated two-point trial."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
from zoneinfo import ZoneInfo

from base_weights import resolve_base

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "data/data_405/picture/work"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def check_label(content):
    import numpy as np
    a = np.loadtxt(io.StringIO(content), ndmin=2) if content.strip() else np.empty((0, 11))
    assert a.ndim == 2 and a.shape[1] == 11, "Expected 11 columns"
    assert np.isfinite(a).all() and (a[:, 0] == 0).all(), "Invalid class/number"
    lo, hi = a[:, 1:3] - a[:, 3:5] / 2, a[:, 1:3] + a[:, 3:5] / 2
    assert (a[:, 3:5] > 0).all() and (lo >= -2e-8).all() and (hi <= 1 + 2e-8).all(), "Invalid box"
    kp = a[:, 5:].reshape(-1, 2, 3)
    assert np.isin(kp[:, :, 2], [0, 2]).all(), "Invalid visibility"
    visible = kp[:, :, 2] == 2
    assert (kp[~visible] == 0).all(), "Missing points must be zero"
    for i in range(2):
        xy = kp[visible[:, i], i, :2]
        assert (xy >= lo[visible[:, i]] - 2e-8).all() and (xy <= hi[visible[:, i]] + 2e-8).all(), "Point outside box"
    both = visible.all(1)
    assert (kp[both, 0, 0] < kp[both, 1, 0]).all(), "Point order"
    assert len(np.unique(a[:, :5], axis=0)) == len(a), "Duplicate box"
    return {"instances": len(a), "visible_endpoints": int(visible.sum()), "missing_endpoints": int((~visible).sum())}


def prepare():
    import yaml
    from PIL import Image
    weights, base = resolve_base()
    run_name = "cassette_front_2pt_data405_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    job = ROOT / "training" / run_name
    job.mkdir(parents=True, exist_ok=False)
    dataset = job / "dataset"
    # Detect concurrent browser saves; the editable source is never written here.
    review_bytes = (SOURCE / "batch/frame_review_status.csv").read_bytes()
    reviews = {int(r["frame"]): r for r in csv.DictReader(io.StringIO(review_bytes.decode("utf-8-sig")))}
    original_manifest = json.loads((SOURCE / "batch/manifest.json").read_text())
    originals = {x["id"]: x for x in original_manifest["images"]}
    assert len(reviews) == len(originals) == 903
    guards = ["batch_train.txt", "data.yaml", "sample_train.txt", "data.sample.yaml", "batch/frame_review_status.csv"]
    preserved = {rel: sha((SOURCE / rel).read_bytes()) for rel in guards}
    preserved.update(json.loads((SOURCE / "batch/preserved_sample_hashes.json").read_text()))
    write_json(job / "source_preservation_hashes.json", preserved)
    (job / "source_review_status.csv").write_bytes(review_bytes)
    rows, checksums = [], []
    for i in range(1, 904):
        stem = f"color_{i:06d}"
        split = "train" if i <= 700 else "gap" if i <= 720 else "val"
        if reviews[i]["review_status"] == "rejected":
            split = "rejected"
        for kind in ("images", "labels"):
            (dataset / kind / split).mkdir(parents=True, exist_ok=True)
        image_path = SOURCE / "images/train" / f"{stem}.png"
        label_path = SOURCE / "labels/train" / f"{stem}.txt"
        image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        assert sha(image_bytes) == originals[stem]["sha256"], f"Image changed: {stem}"
        assert sha((SOURCE.parent / "raw" / image_path.name).read_bytes()) == sha(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
            assert im.size == (1280, 720)
        counts = check_label(label_bytes.decode())
        if counts["instances"] == 0:
            rec = json.loads((SOURCE / "batch/annotations" / f"{stem}.json").read_text())
            assert rec.get("confirmed_empty"), f"Unconfirmed empty image: {stem}"
        for kind, suffix, data in (("images", ".png", image_bytes), ("labels", ".txt", label_bytes)):
            dest = dataset / kind / split / (stem + suffix)
            dest.write_bytes(data)
            assert sha(dest.read_bytes()) == sha(data)
            checksums.append(f"{sha(data)}  {dest.relative_to(dataset)}")
        rows.append({"frame": i, "id": stem, "split": split, "source_review_status": reviews[i]["review_status"],
                     "image_sha256": sha(image_bytes), "label_sha256": sha(label_bytes), **counts})
    # This guards against a source edit during the snapshot, including an edit
    # whose aggregate review status remained unchanged.
    assert (SOURCE / "batch/frame_review_status.csv").read_bytes() == review_bytes, "Review changed during snapshot; prepare again"
    for row in rows:
        assert sha((SOURCE / "labels/train" / (row["id"] + ".txt")).read_bytes()) == row["label_sha256"], "Source labels changed during snapshot"
    cfg = {"path": str(dataset), "train": "images/train", "val": "images/val", "names": {0: "cassette_front"},
           "kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["long_edge_left", "long_edge_right"]}}
    (dataset / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    checksums.append(f"{sha((dataset / 'data.yaml').read_bytes())}  data.yaml")
    (dataset / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    with (dataset / "manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"created_at": now(), "source_images": len(rows), "format_check": "passed", "pixel_integrity": "passed",
               "review_counts": dict(Counter(r["source_review_status"] for r in rows)),
               "split": {}, "quality_scope": "User-authorized trial on existing candidate labels; not quality acceptance.",
               "validation_limit": "Same video, later temporal block; 20-frame (5-second) gap. Candidate labels are not reliable ground truth."}
    for split in ("train", "val", "gap", "rejected"):
        chosen = [r for r in rows if r["split"] == split]
        summary["split"][split] = {"images": len(chosen), **{k: sum(r[k] for r in chosen) for k in ("instances", "visible_endpoints", "missing_endpoints")}}
    write_json(job / "dataset_audit.json", summary)
    config = {"created_at": now(), "run_name": run_name, "source_dataset": str(SOURCE), "snapshot_dataset": str(dataset),
              "weights": str(weights), "weights_sha256": sha(weights.read_bytes()),
              "dataset_sha256_manifest": sha((dataset / "SHA256SUMS.txt").read_bytes()),
              "initialization": f'User-selected cassette two-point base, source epoch {base["epoch"]}; new fine-tuning run with resume=False.',
              "base_selection": base,
              "train": {"data": str(dataset / "data.yaml"), "epochs": 50, "patience": 12, "imgsz": 1280, "batch": 4,
                        "device": "0", "workers": 4, "project": str(ROOT / "runs/pose"), "name": run_name,
                        "exist_ok": False, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "amp": False,
                        "seed": 42, "deterministic": True, "rect": True, "cache": False, "resume": False,
                        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
                        "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
                        "flipud": 0.0, "fliplr": 0.0, "hsv_h": 0.01, "hsv_s": 0.2, "hsv_v": 0.2,
                        "save": True, "save_period": -1, "plots": False, "val": True}}
    write_json(job / "train_config.json", config)
    shutil.copy2(__file__, job / "entrypoint_snapshot.py")
    (ROOT / "reports/data405_trial_latest.txt").write_text(str(job) + "\n")
    print(json.dumps({"job": str(job), "summary": summary}, ensure_ascii=False, indent=2))


def train(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    job = config_path.parent
    os.environ["YOLO_CONFIG_DIR"] = str(job / "cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(job / "cache/matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    started = time.perf_counter()
    offset = int(config.get("completed_epochs_before_this_run", 0))
    state = {"status": "starting", "started_at": now(), "pid": os.getpid(), "config": str(config_path),
             "epochs_completed": 0, "epochs_requested": config["train"]["epochs"], "expected_kpt_shape": [2, 3],
             "completed_epochs_before_this_run": offset, "cumulative_epochs_completed": offset,
             "cumulative_epochs_requested": offset + config["train"]["epochs"],
             "continuation_mode": config.get("continuation_mode", "initial_training")}

    def persist():
        state.update(updated_at=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(job / "status.json", state)

    persist()
    try:
        import torch
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        assert torch.cuda.is_available(), "CUDA required; no silent CPU fallback"
        assert sha(Path(config["weights"]).read_bytes()) == config["weights_sha256"]
        dataset = Path(config["snapshot_dataset"])
        sums = dataset / "SHA256SUMS.txt"
        assert sha(sums.read_bytes()) == config["dataset_sha256_manifest"]
        for line in sums.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            assert sha((dataset / rel).read_bytes()) == digest, f"Snapshot changed: {rel}"
        audit = json.loads((job / "dataset_audit.json").read_text())

        def on_start(trainer):
            shape = list(trainer.model.model[-1].kpt_shape)
            assert shape == [2, 3] and trainer.device.type == "cuda"
            assert len(trainer.train_loader.dataset) == audit["split"]["train"]["images"]
            assert len(trainer.test_loader.dataset) == audit["split"]["val"]["images"]
            assert trainer.data["names"] == {0: "cassette_front"}
            assert trainer.epochs == config["train"]["epochs"]
            if config.get("require_all_epochs"):
                assert trainer.args.patience == 0, "Full-length training requires disabled early stopping"
            state.update(status="running", actual_kpt_shape=shape, gpu=torch.cuda.get_device_name(trainer.device),
                         save_dir=str(trainer.save_dir), train_images=len(trainer.train_loader.dataset),
                         val_images=len(trainer.test_loader.dataset), amp=bool(trainer.amp),
                         best_checkpoint=str(trainer.best), last_checkpoint=str(trainer.last))
            write_json(job / "actual_train_args.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(trainer.args).items()})
            persist()

        def epoch_start(trainer):
            state["current_epoch"] = trainer.epoch + 1
            state["current_cumulative_epoch"] = offset + trainer.epoch + 1
            persist()

        def epoch_end(trainer):
            losses = {str(k): float(v) for k, v in trainer.label_loss_items(trainer.tloss).items()}
            assert all(math.isfinite(v) for v in losses.values()), "Nonfinite training loss"
            state.update(epochs_completed=state["current_epoch"],
                         cumulative_epochs_completed=offset + state["current_epoch"],
                         metrics={str(k): float(v) for k, v in trainer.metrics.items()}, train_losses=losses,
                         losses_finite=True, best_epoch=int(trainer.stopper.best_epoch),
                         best_cumulative_epoch=offset + int(trainer.stopper.best_epoch),
                         epoch_seconds=float(trainer.epoch_time), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device),
                         best_checkpoint_exists=trainer.best.exists(), last_checkpoint_exists=trainer.last.exists())
            persist()

        def preserve_resume_checkpoint(trainer):
            # final_eval strips last.pt and best.pt. Keep a separate full state
            # for a real optimizer/EMA resume if the user later extends this run.
            if config.get("preserve_full_checkpoint"):
                target = trainer.wdir / "resume.pt"
                tmp = trainer.wdir / "resume.pt.tmp"
                shutil.copyfile(trainer.last, tmp)
                tmp.replace(target)
                state["full_resume_checkpoint"] = str(target)

        model = YOLO(config["weights"])
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_epoch_start", epoch_start)
        model.add_callback("on_fit_epoch_end", epoch_end)
        model.add_callback("on_model_save", preserve_resume_checkpoint)
        model.train(**config["train"])
        if config.get("require_all_epochs"):
            assert state["epochs_completed"] == config["train"]["epochs"], "Training ended before the requested epoch count"
        state.update(status="completed", finished_at=now())
        persist()
        return 0
    except BaseException as exc:
        state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        persist()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        raise SystemExit(train(args.config))
