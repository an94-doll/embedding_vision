"""Closing the native Qt window must end the viewer, not report an app failure."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from view_d405_yolo import window_visible


class ViewerWindowTests(unittest.TestCase):
    def test_visible_window(self):
        with patch.object(cv2, "getWindowProperty", return_value=1):
            self.assertTrue(window_visible(cv2))

    def test_closed_window(self):
        with patch.object(cv2, "getWindowProperty", return_value=0):
            self.assertFalse(window_visible(cv2))

    def test_qt_window_already_destroyed(self):
        with patch.object(cv2, "getWindowProperty", side_effect=cv2.error("NULL guiReceiver")):
            self.assertFalse(window_visible(cv2))
