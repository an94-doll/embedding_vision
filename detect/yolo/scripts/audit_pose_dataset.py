#!/usr/bin/env python3
"""Read-only audit of the cassette four-corner export before training."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import yaml

from check_env import project_path, write_report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    errors = []
    counts = {split: {"images": 0, "objects": 0, "source_videos": []} for split in ("train", "val")}
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "source_root": str(root), "status": "failed", "errors": errors, "counts": counts}
    config = yaml.safe_load((root / "data.yaml").read_text())
    if config.get("kpt_shape") != [4, 3] or config.get("names") != {0: "cassette_top"}:
        errors.append("Expected one cassette_top class and kpt_shape [4,3].")
    if config.get("train") != "images/train" or config.get("val") != "images/val":
        errors.append("Unexpected train/val paths.")
    if "download" in config:
        errors.append("Unexpected dataset download/execution directive.")
    samples = json.loads((root / "samples.json").read_text())
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    by_id = {s["sample_id"]: s for s in samples}
    if len(by_id) != len(samples) or {s["sample_id"] for s in manifest} != set(by_id) or len(manifest) != len(samples):
        errors.append("Duplicate IDs or manifest/sample ID mismatch.")
    for row in manifest:
        sample = by_id.get(row["sample_id"], {})
        for key in ("source_frame_id", "source_video", "split", "image", "label", "width", "height", "object_count", "sha256"):
            if row[key] != str(sample.get(key)):
                errors.append(f"{row['sample_id']}: manifest mismatch for {key}")
    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"Checksum file path missing or outside dataset: {name}")
            continue
        actual = digest(path)
        checksums[name] = actual
        if actual != expected:
            errors.append(f"Checksum mismatch: {name}")
    for split in counts:
        expected = {s["sample_id"] for s in samples if s["split"] == split}
        images = [p for p in (root / "images" / split).iterdir() if p.is_file()]
        labels = list((root / "labels" / split).glob("*.txt"))
        if not expected or len(images) != len(expected) or {p.stem for p in images} != expected or {p.stem for p in labels} != expected:
            errors.append(f"{split}: missing, duplicate or unpaired image/label files")
    video_splits, frame_splits = defaultdict(set), defaultdict(set)
    file_splits, pixel_splits = defaultdict(set), defaultdict(set)
    dimensions, edge_ratios, box_heights = [], [], []
    review_states = Counter()
    for sample in samples:
        sid, split = sample["sample_id"], sample["split"]
        try:
            image_path, label_path = root / sample["image"], root / sample["label"]
            with Image.open(image_path) as image:
                image.load()
                w, h = image.size
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(str((w, h)).encode() + rgb.tobytes()).hexdigest()
            assert (w, h) == (sample["width"], sample["height"]), "image dimensions differ from metadata"
            dimensions.append((w, h))
            sha = digest(image_path)
            assert sha == sample["sha256"], "image SHA differs from samples.json"
            assert checksums.get(sample["image"]) == sha, "image absent from SHA256SUMS"
            assert checksums.get(sample["label"]) == digest(label_path), "label absent from SHA256SUMS"
            file_splits[sha].add(split)
            pixel_splits[pixel_hash].add(split)
            video_splits[sample["source_video"]].add(split)
            frame_splits[sample["source_frame_id"]].add(split)
            lines = label_path.read_text().splitlines()
            assert len(lines) == sample["object_count"] == len(sample["objects"]), "object count mismatch"
            assert len(set(lines)) == len(lines), "duplicate label rows"
            counts[split]["images"] += 1
            counts[split]["objects"] += len(lines)
            for index, (line, obj) in enumerate(zip(lines, sample["objects"]), 1):
                values = np.asarray([float(x) for x in line.split()], dtype=np.float64)
                assert values.shape == (17,) and np.isfinite(values).all(), f"row {index}: invalid 17-column label"
                assert values[0] == 0, f"row {index}: wrong class"
                cx, cy, bw, bh = values[1:5]
                assert 0 < cx < 1 and 0 < cy < 1 and 0 < bw <= 1 and 0 < bh <= 1, f"row {index}: invalid bbox"
                keypoints = values[5:].reshape(4, 3)
                assert (keypoints[:, 2] == 2).all(), f"row {index}: unexpected visibility"
                xy = keypoints[:, :2]
                assert ((xy >= 0) & (xy <= 1)).all(), f"row {index}: keypoint out of bounds"
                points = xy * [w, h]
                assert np.allclose(points, obj["points"], atol=2e-5, rtol=0), f"row {index}: TXT and sample points differ"
                edges = np.roll(points, -1, axis=0) - points
                following = np.roll(edges, -1, axis=0)
                cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
                assert (cross > 0).all(), f"row {index}: non-convex or not clockwise in image coordinates"
                lengths = np.linalg.norm(edges, axis=1)
                assert lengths[0] > max(lengths[1], lengths[3]), f"row {index}: 1-2 is not a long edge"
                edge_ratios.append(float(lengths[0] / max(lengths[1], lengths[3])))
                minimum, maximum = xy.min(axis=0), xy.max(axis=0)
                expected_box = np.concatenate(((minimum + maximum) / 2, maximum - minimum))
                assert np.allclose(values[1:5], expected_box, atol=2e-7, rtol=0), f"row {index}: bbox is not the corner enclosure"
                box_heights.append(float(bh * h))
                review_states[obj.get("visual_review", "unspecified")] += 1
        except Exception as exc:
            errors.append(f"{sid}: {type(exc).__name__}: {exc}")
    for name, groups in (("video", video_splits), ("source_frame", frame_splits), ("file_hash", file_splits), ("decoded_pixels", pixel_splits)):
        for key, splits in groups.items():
            if len(splits) > 1:
                errors.append(f"Train/val leakage by {name}: {key}")
    for split in counts:
        counts[split]["source_videos"] = sorted(v for v, splits in video_splits.items() if split in splits)
    report.update({
        "status": "passed" if not errors else "failed", "kpt_shape": config.get("kpt_shape"), "names": config.get("names"),
        "verified_checksum_files": len(checksums),
        "source_package_fingerprint": digest(root / "SHA256SUMS.txt"),
        "source_frames_contributing": len(frame_splits),
        "sample_kinds": dict(Counter(s["kind"] for s in samples)), "object_review_states": dict(review_states),
        "image_size_range": {"min_width": min(w for w, h in dimensions), "max_width": max(w for w, h in dimensions), "min_height": min(h for w, h in dimensions), "max_height": max(h for w, h in dimensions)},
        "min_long_to_adjacent_edge_ratio": min(edge_ratios), "min_bbox_height_pixels": min(box_heights),
        "warnings": ["Small initial dataset from the same physical arrangement; validation is not an independent-scene generalization test.", "No independent test split or dedicated negative images; all current annotations use visibility=2.", "The review-only package is excluded; this audit validates structure and geometry, not every pixel-level annotation."],
        "training_started": False,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", default="reports/dataset_audit.json")
    args = parser.parse_args()
    try:
        report = audit(args.root.resolve())
    except Exception as exc:
        report = {"status": "failed", "source_root": str(args.root.resolve()), "errors": [f"{type(exc).__name__}: {exc}"], "training_started": False}
    write_report(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
