#!/usr/bin/env python3
"""Prepare a reproducible YOLO Pose dataset without inspecting label contents.

Examples:
    python3 split_dataset.py --dry-run
    python3 split_dataset.py
    python3 split_dataset.py --seed 123 --output dataset_seed123
    python3 split_dataset.py --source box_in_basket/data_origin \\
        --schema box_in_basket/white_box_pose/pose_schema.yaml

Accepts the original numbered groups or a source folder containing images/
and labels/. Outputs JPG images without resizing or rotating them. Pillow is
required for image export; a supplied schema also requires PyYAML. Existing
output directories are never overwritten. Labels, including empty background
labels, are copied unchanged; unlabelled images are excluded.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUP_CLASS = {"01": 1, "02": 1, "03": 1, "04": 0, "05": 0, "06": 0,"08": None,"09": 0,"10": 1}
# Folder categories are only used for split quotas, never to validate labels.
CLASS_NAMES = {0: "front", 1: "back", None: "mixed"}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    group: str
    class_id: int | None
    image: Path
    label: Path
    digest: str

    @property
    def output_name(self):
        prefix = "" if self.group == "all" else self.group + "_"
        return prefix + self.image.stem + ".jpg"


def collect_samples(images_dir, labels_dir, annotations_dir, grouped=True):
    samples, excluded = [], []
    groups = GROUP_CLASS.items() if grouped else [("all", None)]
    for group, class_id in groups:
        image_group = images_dir / group if grouped else images_dir
        label_group = labels_dir / group if grouped else labels_dir
        if not image_group.is_dir() or not label_group.is_dir():
            raise ValueError("Missing image or label directory for group " + group)
        skip_file = annotations_dir / (group + "_skipped.json")
        skipped = {}
        if grouped and skip_file.is_file():
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
    for class_id in CLASS_NAMES:
        class_samples = [s for s in samples if s.class_id == class_id]
        sizes = Counter(s.group for s in class_samples)
        targets = split_counts(len(class_samples), ratios)
        class_quotas = group_quotas(sizes, targets)
        quotas.update(class_quotas)
        for group in sorted(sizes):
            shuffled = sorted((s for s in class_samples if s.group == group),
                              key=lambda s: s.image.name)
            group_seed = seed * 100 + int(group) if group.isdigit() else f"{seed}:{group}"
            random.Random(group_seed).shuffle(shuffled)
            offset = 0
            for split, count in zip(SPLITS, class_quotas[group]):
                plan.extend((sample, split) for sample in shuffled[offset:offset + count])
                offset += count
    assert len(plan) == len(samples)
    assert len({s.image for s, _ in plan}) == len(samples)
    return sorted(plan, key=lambda item: (SPLITS.index(item[1]), item[0].group, item[0].image.name)), quotas


def summarize(plan, quotas, ratios, seed, excluded, grouped=True):
    counts = {split: {name: 0 for name in CLASS_NAMES.values()} for split in SPLITS}
    for sample, split in plan:
        counts[split][CLASS_NAMES[sample.class_id]] += 1
    return {"seed": seed, "ratios": ratios, "valid_images": len(plan),
            "excluded_count": len(excluded), "counts": counts,
            "group_counts": {g: dict(zip(SPLITS, q)) for g, q in sorted(quotas.items())},
            "strategy": ("folder-category-stratified random split with balanced group quotas"
                         if grouped else "random image split"),
            "counts_note": "Image counts by folder category, not object counts. Label contents are not inspected.",
            "image_format": "jpg", "jpeg_quality": 95,
            "sequence_note": "Adjacent frames may appear in different splits; this is not an independent-sequence evaluation."}


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_jpg(source, target):
    """Keep pixel geometry unchanged so normalized labels remain applicable."""
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Image export requires Pillow; run: python -m pip install Pillow") from error
    with source.open("rb") as handle:
        if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                "Git LFS pointer instead of image pixels: " + str(source)
                + "; download the actual image in its original repository before splitting."
            )
    with Image.open(source) as image:
        # Avoid recompressing existing RGB JPEGs. Do not apply EXIF transpose:
        # annotations refer to the stored pixel coordinates.
        if image.format == "JPEG" and image.mode == "RGB":
            shutil.copy2(source, target)
            return
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", image.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = image.convert("RGB")
        rgb.save(target, format="JPEG", quality=95, subsampling=0)


def load_schema(path):
    """Read dataset metadata without examining any annotation files."""
    if path is None:
        return ("names:\n  0: front\n  1: back\n"
                "kpt_shape: [5, 3]\nflip_idx: [4, 3, 2, 1, 0]\n")
    try:
        import yaml
    except ImportError as error:
        raise ValueError("Reading --schema requires PyYAML; run: python -m pip install PyYAML") from error
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    fields = {key: config[key] for key in ("names", "kpt_shape", "flip_idx", "kpt_names")
              if key in config}
    if "names" not in fields:
        raise ValueError("Dataset schema must include names: " + str(path))
    return yaml.safe_dump(fields, allow_unicode=True, sort_keys=False)


def build_dataset(output, plan, summary, excluded, schema=None):
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
            export_jpg(sample.image, image_out)
            shutil.copy2(sample.label, label_out)
            if label_out.read_bytes() != sample.label.read_bytes():
                raise ValueError("Copied label mismatch: " + sample.output_name)
            manifest.append({"split": split, "group": sample.group, "class_id": sample.class_id,
                             "class_name": CLASS_NAMES[sample.class_id],
                             "source_image": str(sample.image), "source_label": str(sample.label),
                             "output_image": image_relative.as_posix(),
                             "output_label": label_relative.as_posix(), "sha256": sample.digest,
                             "output_sha256": hashlib.sha256(image_out.read_bytes()).hexdigest()})
        # JSON strings are valid YAML quoted scalars, including paths with spaces.
        yaml_text = ("train: images/train\nval: images/val\ntest: images/test\n"
                     + (schema if schema is not None else load_schema(None)))
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
    parser.add_argument("--source", type=Path, default=ROOT / "box_in_basket/data_origin",
                        help="Flat dataset root containing images/ and labels/; omit for numbered groups")
    parser.add_argument("--schema", type=Path, default=ROOT / "box_in_basket/white_box_pose/pose_schema.yaml",
                        help="YAML with names and optional keypoint metadata; required with --source")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "annotations")
    parser.add_argument("--output", type=Path,
                        help="New output directory; default: dataset beside --source, or project/dataset")
    parser.add_argument("--ratios", type=int, nargs=3, default=[7, 2, 1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Preview split counts without inspecting labels or writing files")
    args = parser.parse_args()
    if any(r <= 0 for r in args.ratios):
        parser.error("All three ratio weights must be positive integers")
    if args.source and args.schema is None:
        parser.error("--source requires --schema to select the correct classes and keypoint layout")
    grouped = args.source is None
    images_dir = (args.source / "images" if args.source else args.images_dir).resolve()
    labels_dir = (args.source / "labels" if args.source else args.labels_dir).resolve()
    default_output = args.source.resolve().parent / "dataset" if args.source else ROOT / "dataset"
    output = (args.output or default_output).resolve()
    if any(output == source or source in output.parents or output in source.parents
           for source in (images_dir, labels_dir)):
        parser.error("Output must be separate from source image and label directories")
    if output.exists() and not args.dry_run:
        parser.error("Output already exists; select a new --output directory: " + str(output))
    try:
        schema = load_schema(args.schema)
        samples, excluded = collect_samples(images_dir, labels_dir, args.annotations_dir.resolve(), grouped)
        plan, quotas = make_plan(samples, args.ratios, args.seed)
        summary = summarize(plan, quotas, args.ratios, args.seed, excluded, grouped)
        print("有效样本：{}；排除记录：{}；种子：{}".format(len(plan), len(excluded), args.seed))
        if grouped:
            print("集合      正面组    反面组    混合组    合计（图片数）")
        else:
            print("集合      图片数")
        for split in SPLITS:
            counts = summary["counts"][split]
            if grouped:
                print("{:<8} {:>6} {:>9} {:>9} {:>9}".format(
                    split, counts["front"], counts["back"], counts["mixed"], sum(counts.values())))
            else:
                print("{:<8} {:>6}".format(split, sum(counts.values())))
        if args.dry_run:
            print("预览完成，未写入文件。")
        else:
            build_dataset(output, plan, summary, excluded, schema)
            print("数据集：" + str(output))
            print("训练配置：" + str(output / "data.yaml"))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
