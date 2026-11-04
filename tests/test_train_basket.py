"""验证训练入口参数，不启动模型训练或访问 GPU。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import train_basket as training


class BasketTrainingTests(unittest.TestCase):
    def test_current_dataset_and_training_arguments(self):
        yolo = Mock()
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=yolo)}), \
                patch.object(training, 'configure_local_cache'):
            training.main()
        yolo.assert_called_once_with(str(training.WEIGHTS))
        args = yolo.return_value.train.call_args.kwargs
        self.assertEqual(args['data'], str(training.DATA))
        self.assertEqual(args['name'], 'basket_4pt')
        self.assertEqual(args['mosaic'], 0)
        self.assertEqual(args['device'], 0)
        self.assertFalse(args['exist_ok'])
        self.assertNotIn('resume', args)

    def test_rejects_joint_dataset_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)/'data.yaml'
            data.write_text('names: {0: white_box, 1: basket}\nkpt_shape: [4, 3]\nflip_idx: [1,0,3,2]\n')
            yolo = Mock()
            with patch.object(training, 'DATA', data), \
                    patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=yolo)}):
                with self.assertRaises(ValueError):
                    training.main()
            yolo.assert_not_called()
