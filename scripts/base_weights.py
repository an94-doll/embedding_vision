"""Resolve the user-selected base for new cassette two-point YOLO tasks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'weights/base.json'


def resolve_base():
    """Return (checkpoint, metadata), failing if the selected file has changed."""
    metadata = json.loads(MANIFEST.read_text())
    weights = (MANIFEST.parent / metadata['weights']).resolve()
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != metadata['sha256']:
        raise ValueError(f'Selected base checksum mismatch: {weights}')
    legacy = metadata['kpt_shape'] == [2, 3]
    basket = metadata['kpt_shape'] == [4, 3] and metadata['names'] == {'0': 'white_box', '1': 'basket'}
    if metadata['task'] != 'pose' or not (legacy or basket):
        raise ValueError('Unsupported selected pose schema.')
    return weights, metadata


if __name__ == '__main__':
    print(resolve_base()[0])
