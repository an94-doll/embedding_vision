#!/usr/bin/env python3
"""Split basket-only data into train/val (8:2, seed 42), without a test split.

python split_basket_dataset.py --dry-run
python split_basket_dataset.py

Images are copied unchanged. Only the class token is remapped to 0 in exported
labels; coordinates/visibility and source files are preserved. No geometry checks.
"""
import argparse
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

import yaml

from split_dataset import IMAGE_EXTENSIONS, write_csv

ROOT = Path(__file__).resolve().parent


def remap_label(text, source_id):
    lines = []
    for line in text.splitlines(keepends=True):
        if line.strip():
            match = re.match(r'(\s*)(\S+)(.*)', line, re.DOTALL)
            if match[2] != str(source_id):
                raise ValueError(f'Expected basket class {source_id}, found {match[2]}')
            line = match[1] + '0' + match[3]
        lines.append(line)
    return ''.join(lines)


def prepare(source, ratios=(8, 2), seed=42):
    source = Path(source).resolve()
    if len(ratios) != 2 or any(r <= 0 for r in ratios):
        raise ValueError('Provide two positive train/val ratio weights')
    schema = yaml.safe_load((source/'pose_schema.yaml').read_text(encoding='utf-8-sig'))
    names = schema['names']
    names = dict(enumerate(names)) if isinstance(names, list) else names
    basket_ids = [int(k) for k, name in names.items() if name == 'basket']
    if len(basket_ids) != 1:
        raise ValueError('Schema must identify exactly one basket class')
    source_id = basket_ids[0]
    if schema.get('kpt_shape') != [4, 3]:
        raise ValueError('Expected basket kpt_shape: [4, 3]')
    output_schema = {k: schema[k] for k in ('kpt_shape', 'flip_idx', 'kpt_oks_sigmas') if k in schema}
    output_schema['names'] = {0: 'basket'}
    if source_id in schema.get('kpt_names', {}):
        output_schema['kpt_names'] = {0: schema['kpt_names'][source_id]}
    images_dir, labels_dir = source/'images', source/'labels'
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) < 2:
        raise ValueError('At least two labelled images are required')
    stems = {p.stem for p in images}
    if len(stems) != len(images):
        raise ValueError('Same-stem images would share an output label')
    labels = {p.stem for p in labels_dir.glob('*.txt')}
    if stems != labels:
        raise ValueError(f'Image/label mismatch: {len(stems-labels)} missing labels; {len(labels-stems)} orphan labels')
    samples = []
    for image in images:
        label = labels_dir/(image.stem+'.txt')
        # Decode bytes directly to retain original newline style during remapping.
        text = label.read_bytes().decode('utf-8-sig')
        try:
            mapped = remap_label(text, source_id)
        except ValueError as error:
            raise ValueError(f'{label}: {error}') from error
        samples.append((image, label, mapped))
    random.Random(seed).shuffle(samples)
    train_count = (len(samples)*ratios[0]*2 + sum(ratios)) // (2*sum(ratios))
    if train_count == 0 or train_count == len(samples):
        raise ValueError('These ratios leave one split empty; choose less extreme ratios')
    plan = [(image, label, text, 'train' if i < train_count else 'val')
            for i, (image, label, text) in enumerate(samples)]
    summary = dict(source=str(source), seed=seed, ratios=list(ratios), images=len(images),
                   counts=dict(train=train_count, val=len(images)-train_count),
                   class_mapping={str(source_id): 0},
                   strategy='seeded random image split',
                   sequence_note='Adjacent frames may cross splits; this is not an independent-sequence evaluation.')
    return plan, output_schema, summary


def build(output, plan, schema, summary):
    output = Path(output).resolve()
    source = Path(summary['source']).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from the source tree')
    if output.exists():
        raise ValueError(f'Output already exists; choose a new --output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary)/'dataset'
        for kind in ('images', 'labels'):
            for split in ('train', 'val'):
                (stage/kind/split).mkdir(parents=True)
        rows = []
        for image, label, mapped, split in plan:
            image_relative = Path('images')/split/image.name
            label_relative = Path('labels')/split/label.name
            shutil.copy2(image, stage/image_relative)
            (stage/label_relative).write_bytes(mapped.encode('utf-8'))
            rows.append(dict(split=split, source_image=str(image), source_label=str(label),
                             output_image=str(image_relative), output_label=str(label_relative)))
        config = dict(train='images/train', val='images/val', **schema)
        (stage/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        write_csv(stage/'split_manifest.csv', list(rows[0]), rows)
        (stage/'split_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        if output.exists():
            raise ValueError(f'Output appeared during export: {output}')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'box_in_basket/basket_pose')
    parser.add_argument('--output', type=Path, default=ROOT/'box_in_basket/basket_dataset')
    parser.add_argument('--ratios', type=int, nargs=2, default=[8, 2], metavar=('TRAIN', 'VAL'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        plan, schema, summary = prepare(args.source, args.ratios, args.seed)
        if not args.dry_run:
            build(args.output, plan, schema, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('预览完成，未写入文件。' if args.dry_run else f'训练配置：{args.output.resolve() / "data.yaml"}')
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
