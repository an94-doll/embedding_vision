#!/usr/bin/env python3
"""Prepare a reproducible, class-stratified five-keypoint YOLO Pose dataset.

Examples:
    python3 split_dataset.py --dry-run
    python3 split_dataset.py
    python3 split_dataset.py --seed 123 --output dataset_seed123

Uses only the Python standard library. Existing output directories are never
overwritten. Invalid labels stop the build; unlabelled images are excluded.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUP_CLASS = {"01": 1, "02": 1, "03": 1, "04": 0, "05": 0, "06": 0,"09": 0,"10": 1}
CLASS_NAMES = {0: "front", 1: "back"}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sample:
    group: str
    class_id: int
    image: Path
    label: Path
    digest: str

    @property
    def output_name(self):
        return self.group + "_" + self.image.name


def validate_label(path, expected_class):
    """Validate every instance, including visibility and normalized geometry."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    for line_number, line in enumerate(lines, 1):
        try:
            values = list(map(float, line.split()))
        except ValueError:
            raise ValueError("{}:{} contains nonnumeric label fields".format(path, line_number))
        prefix = "{}:{}".format(path, line_number)
        if len(values) != 20 or not all(math.isfinite(v) for v in values):
            raise ValueError(prefix + " must have 20 finite fields (five keypoints)")
        if values[0] != expected_class:
            raise ValueError(prefix + " has wrong class; expected {}".format(expected_class))
        cx, cy, width, height = values[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(prefix + " has invalid normalized bounding box")
        xmin, xmax = cx - width / 2, cx + width / 2
        ymin, ymax = cy - height / 2, cy + height / 2
        if xmin < -TOLERANCE or ymin < -TOLERANCE or xmax > 1 + TOLERANCE or ymax > 1 + TOLERANCE:
            raise ValueError(prefix + " has a bounding box outside the image")
        for index in range(5):
            x, y, visibility = values[5 + index * 3:8 + index * 3]
            if not (0 <= x <= 1 and 0 <= y <= 1 and visibility in (0, 1, 2)):
                raise ValueError(prefix + " has invalid P{}".format(index + 1))
            if visibility and not (xmin - TOLERANCE <= x <= xmax + TOLERANCE and
                                   ymin - TOLERANCE <= y <= ymax + TOLERANCE):
                print("警告：{} 的 P{} 位于边界框外，已保留该标注".format(
                        prefix, index + 1
                ))
    return True


def collect_samples(images_dir, labels_dir, annotations_dir):
    samples, excluded = [], []
    for group, class_id in GROUP_CLASS.items():
        image_group, label_group = images_dir / group, labels_dir / group
        if not image_group.is_dir() or not label_group.is_dir():
            raise ValueError("Missing image or label directory for group " + group)
        skip_file = annotations_dir / (group + "_skipped.json")
        skipped = {}
        if skip_file.is_file():
            skipped = {r["image"]: r.get("reason", "recorded_skip")
                       for r in json.loads(skip_file.read_text(encoding="utf-8"))}
        images = sorted(p for p in image_group.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            raise ValueError("Same-stem images would share a label in group " + group)
        for image in images:
            label = label_group / (image.stem + ".txt")
            reason = skipped.get(image.name)
            if reason is None and not label.is_file():
                reason = "missing_label"
            if reason is None and not validate_label(label, class_id):
                reason = "empty_label"
            if reason is not None:
                excluded.append({"group": group, "image": str(image), "label": str(label),
                                 "reason": reason})
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            samples.append(Sample(group, class_id, image, label, digest))
        for label in sorted(label_group.glob("*.txt")):
            if label.stem not in stems:
                excluded.append({"group": group, "image": "", "label": str(label),
                                 "reason": "orphan_label"})
    if not samples:
        raise ValueError("No valid labelled images found")
    # Exact copies cannot safely be assigned independently to different splits.
    # Ask the operator to resolve duplicates instead of silently dropping data.
    by_digest = {}
    for sample in samples:
        if sample.digest in by_digest:
            raise ValueError("Identical source images: {} and {}; resolve before splitting".format(
                by_digest[sample.digest], sample.image))
        by_digest[sample.digest] = sample.image
    return samples, excluded


def split_counts(total, ratios):
    """Largest-remainder rounding: totals are exact, ties follow split order."""
    denominator = sum(ratios)
    counts = [total * r // denominator for r in ratios]
    remainders = [total * r % denominator for r in ratios]
    order = sorted(range(3), key=lambda i: (-remainders[i], i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def group_quotas(group_sizes, targets):
    """Preserve group representation while meeting exact class split totals."""
    total = sum(group_sizes.values())
    if not total:
        return {}
    groups = sorted(group_sizes)
    quotas = {g: [group_sizes[g] * t // total for t in targets] for g in groups}
    row_gaps = {g: group_sizes[g] - sum(quotas[g]) for g in groups}
    col_gaps = [targets[i] - sum(quotas[g][i] for g in groups) for i in range(3)]
    while sum(col_gaps):
        candidates = [(g, i) for g in groups for i in range(3)
                      if row_gaps[g] and col_gaps[i]]
        # Integer arithmetic avoids floating-point tie instability.
        g, i = max(candidates, key=lambda item: (
            group_sizes[item[0]] * targets[item[1]] - quotas[item[0]][item[1]] * total,
            -groups.index(item[0]), -item[1]))
        quotas[g][i] += 1
        row_gaps[g] -= 1
        col_gaps[i] -= 1
    assert all(sum(quotas[g]) == group_sizes[g] for g in groups)
    assert [sum(quotas[g][i] for g in groups) for i in range(3)] == targets
    return quotas


def make_plan(samples, ratios, seed):
    plan, quotas = [], {}
    for class_id in sorted(CLASS_NAMES):
        class_samples = [s for s in samples if s.class_id == class_id]
        sizes = Counter(s.group for s in class_samples)
        targets = split_counts(len(class_samples), ratios)
        class_quotas = group_quotas(sizes, targets)
        quotas.update(class_quotas)
        for group in sorted(sizes):
            shuffled = sorted((s for s in class_samples if s.group == group),
                              key=lambda s: s.image.name)
            random.Random(seed * 100 + int(group)).shuffle(shuffled)
            offset = 0
            for split, count in zip(SPLITS, class_quotas[group]):
                plan.extend((sample, split) for sample in shuffled[offset:offset + count])
                offset += count
    assert len(plan) == len(samples)
    assert len({s.image for s, _ in plan}) == len(samples)
    return sorted(plan, key=lambda item: (SPLITS.index(item[1]), item[0].group, item[0].image.name)), quotas


def summarize(plan, quotas, ratios, seed, excluded):
    counts = {split: {name: 0 for name in CLASS_NAMES.values()} for split in SPLITS}
    for sample, split in plan:
        counts[split][CLASS_NAMES[sample.class_id]] += 1
    return {"seed": seed, "ratios": ratios, "valid_images": len(plan),
            "excluded_count": len(excluded), "counts": counts,
            "group_counts": {g: dict(zip(SPLITS, q)) for g, q in sorted(quotas.items())},
            "strategy": "class-stratified random split with balanced group quotas",
            "sequence_note": "Adjacent frames may appear in different splits; this is not an independent-sequence evaluation."}


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(output, plan, summary, excluded):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-build-", dir=str(output.parent)))
    try:
        for kind in ("images", "labels"):
            for split in SPLITS:
                (temporary / kind / split).mkdir(parents=True)
        manifest = []
        for sample, split in plan:
            image_relative = Path("images") / split / sample.output_name
            label_relative = Path("labels") / split / (Path(sample.output_name).stem + ".txt")
            image_out, label_out = temporary / image_relative, temporary / label_relative
            if image_out.exists() or label_out.exists():
                raise ValueError("Output filename collision: " + sample.output_name)
            shutil.copy2(sample.image, image_out)
            shutil.copy2(sample.label, label_out)
            if hashlib.sha256(image_out.read_bytes()).hexdigest() != sample.digest:
                raise ValueError("Copied image checksum mismatch: " + sample.output_name)
            if label_out.read_bytes() != sample.label.read_bytes():
                raise ValueError("Copied label mismatch: " + sample.output_name)
            manifest.append({"split": split, "group": sample.group, "class_id": sample.class_id,
                             "class_name": CLASS_NAMES[sample.class_id],
                             "source_image": str(sample.image), "source_label": str(sample.label),
                             "output_image": image_relative.as_posix(),
                             "output_label": label_relative.as_posix(), "sha256": sample.digest})
        # JSON strings are valid YAML quoted scalars, including paths with spaces.
        yaml_text = ("path: " + json.dumps(str(output), ensure_ascii=False) + "\n"
                     "train: images/train\nval: images/val\ntest: images/test\n"
                     "names:\n  0: front\n  1: back\n"
                     "kpt_shape: [5, 3]\nflip_idx: [4, 3, 2, 1, 0]\n")
        (temporary / "data.yaml").write_text(yaml_text, encoding="utf-8")
        write_csv(temporary / "split_manifest.csv", list(manifest[0]), manifest)
        write_csv(temporary / "excluded_manifest.csv", ["group", "image", "label", "reason"], excluded)
        (temporary / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for split in SPLITS:
            rows = [row for row in manifest if row["split"] == split]
            expected = len(rows)
            assert len(list((temporary / "images" / split).iterdir())) == expected
            assert len(list((temporary / "labels" / split).iterdir())) == expected
        if output.exists():
            raise ValueError("Output appeared during build; refusing to overwrite: " + str(output))
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "annotations")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--ratios", type=int, nargs=3, default=[7, 2, 1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files")
    args = parser.parse_args()
    if any(r <= 0 for r in args.ratios):
        parser.error("All three ratio weights must be positive integers")
    images_dir, labels_dir = args.images_dir.resolve(), args.labels_dir.resolve()
    output = args.output.resolve()
    if any(output == source or source in output.parents or output in source.parents
           for source in (images_dir, labels_dir)):
        parser.error("Output must be separate from source image and label directories")
    if output.exists() and not args.dry_run:
        parser.error("Output already exists; select a new --output directory: " + str(output))
    try:
        samples, excluded = collect_samples(images_dir, labels_dir, args.annotations_dir.resolve())
        plan, quotas = make_plan(samples, args.ratios, args.seed)
        summary = summarize(plan, quotas, args.ratios, args.seed, excluded)
        print("有效样本：{}；排除记录：{}；种子：{}".format(len(plan), len(excluded), args.seed))
        print("集合      正面      反面      合计")
        for split in SPLITS:
            counts = summary["counts"][split]
            print("{:<8} {:>6} {:>9} {:>9}".format(split, counts["front"], counts["back"], sum(counts.values())))
        if args.dry_run:
            print("预览完成，未写入文件。")
        else:
            build_dataset(output, plan, summary, excluded)
            print("数据集：" + str(output))
            print("训练配置：" + str(output / "data.yaml"))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
