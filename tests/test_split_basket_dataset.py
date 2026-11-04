import tempfile
import unittest
from pathlib import Path

import yaml
from split_basket_dataset import prepare, build, remap_label


class BasketSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        for kind in ('images', 'labels'):
            (self.source/kind).mkdir(parents=True)
        (self.source/'pose_schema.yaml').write_text('names: {0: white_box, 1: basket}\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nkpt_names: {1: [tl, tr, br, bl]}\n')
        for i in range(10):
            (self.source/'images'/f'{i}.jpg').write_bytes(bytes([i]))
            (self.source/'labels'/f'{i}.txt').write_bytes(b'1 .5 .5 .8 .8 0 0 0\r\n')

    def test_export_has_only_train_val_and_single_basket_class(self):
        plan, schema, summary = prepare(self.source)
        self.assertEqual(summary['counts'], {'train': 8, 'val': 2})
        self.assertEqual(plan, prepare(self.source)[0])
        output = self.root/'output'
        build(output, plan, schema, summary)
        config = yaml.safe_load((output/'data.yaml').read_text())
        self.assertNotIn('test', config)
        self.assertEqual(config['names'], {0: 'basket'})
        self.assertEqual(config['kpt_names'], {0: ['tl', 'tr', 'br', 'bl']})
        self.assertEqual({p.name for p in (output/'images').iterdir()}, {'train', 'val'})
        self.assertEqual(len(list((output/'labels').rglob('*.txt'))), 10)
        for image, label, text, split in plan:
            self.assertEqual((output/'images'/split/image.name).read_bytes(), image.read_bytes())
            self.assertEqual((output/'labels'/split/label.name).read_bytes(), b'0 .5 .5 .8 .8 0 0 0\r\n')
            self.assertTrue(label.read_bytes().startswith(b'1 '))
        with self.assertRaises(ValueError):
            build(output, plan, schema, summary)
        with self.assertRaises(ValueError):
            build(self.source/'output', plan, schema, summary)

    def test_missing_pair_and_bad_ratios_fail(self):
        with self.assertRaises(ValueError):
            prepare(self.source, [1, 0])
        (self.source/'labels/0.txt').unlink()
        with self.assertRaises(ValueError):
            prepare(self.source)

    def test_remap_preserves_rest_and_background(self):
        self.assertEqual(remap_label('  1\tcontent\r\n\n1 more\n', 1), '  0\tcontent\r\n\n0 more\n')
        self.assertEqual(remap_label('', 1), '')
        with self.assertRaises(ValueError):
            remap_label('0 unexpected class\n', 1)
