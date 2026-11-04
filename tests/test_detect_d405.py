"""Window rendering and lifecycle tests; no camera/GPU/window is opened."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import detect_d405 as live


class LiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(live, "FIRST_BOX_PATH", Path(directory.name)/"first.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.basket_model = Mock()
        self.basket_model.predict.return_value = [SimpleNamespace()]
        basket_patcher = patch.object(live, "load_basket_model", return_value=self.basket_model)
        basket_patcher.start()
        self.addCleanup(basket_patcher.stop)

    def test_headless_opencv_is_rejected_before_opening_window(self):
        with patch.dict(live.os.environ, {"DISPLAY":":fake"}), \
                patch.object(live.cv2,"getBuildInformation",return_value="  GUI: NONE\n"):
            with self.assertRaisesRegex(RuntimeError,"GUI"):
                live.check_display()

    def test_box_ids_are_hidden_without_hiding_missing_point_detections(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxes = np.array([[20., 20., 80., 50.], [100., 20., 170., 50.]])
        points = np.array([[[20., 40., .9], [80., 40., .9]], [[100., 40., .1], [170., 40., .1]]])
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as text:
            canvas, counts = live.render_frame(image, boxes, points, (0, 1), removed=2, show_boxes=False)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertFalse(any(label.startswith("#") for label in labels))
        self.assertEqual(counts, dict(boxes=2, stable=2, lost_tracks=0, edges=1, removed=2, partial=1))
        self.assertEqual(canvas.shape, (100 + live.HEADER_HEIGHT, 200, 3))
        self.assertFalse(image.any())

    def test_empty_frame_resets_count_and_labels(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as text:
            _, counts = live.render_frame(image, np.empty((0, 4)), np.empty((0, 2, 3)), ())
        self.assertEqual(counts["boxes"], 0)
        self.assertFalse(any(call.args[1].startswith("#") for call in text.call_args_list))

    def test_grid_labels_show_coordinates_counts_and_split_source(self):
        boxes = np.array([[100., 50., 180., 80.], [10., 10., 80., 40.]])
        points = np.array([[[110., 70., .9], [170., 70., .9]], [[20., 30., .9], [70., 30., .9]]])
        grid = SimpleNamespace(rows=2, cols=1, assignments=((0, 1, 0), (1, 0, 0)), unassigned_indices=(), cells=(
            SimpleNamespace(row=0,col=0,count=1,bounds=(10.,10.,80.,40.),source="gap"),
            SimpleNamespace(row=1,col=0,count=1,bounds=(100.,50.,180.,80.),source="count")))
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as text:
            _, counts = live.render_frame(np.zeros((100, 200, 3), dtype=np.uint8), boxes, points, (0, 1), grid=grid)
        labels = [call.args[1] for call in text.call_args_list]
        for expected in ("(0,0)", "(1,0)", "1/10 G", "1/10 C"):
            self.assertIn(expected, labels)
        self.assertFalse(any(label.startswith("#") for label in labels))
        self.assertEqual(counts["assigned"], 2)
        self.assertEqual(counts["unassigned"], 0)
        self.assertEqual((counts["rows"],counts["cols"]), (2,1))

    def test_temporal_defaults_are_enabled(self):
        self.assertTrue(live.TEMPORAL.tracking_enabled)
        self.assertTrue(live.TEMPORAL.smoothing_enabled)
        self.assertTrue(live.TEMPORAL.counting_enabled)
        self.assertTrue(live.GRID.enabled)
        self.assertEqual((live.GRID.rows, live.GRID.cols), (3, 5))
        self.assertEqual(live.GRID.hold_seconds, .5)
        self.assertGreater(live.GRID.confirmation_frames, 1)

    def test_empty_history_does_not_draw_ghost_grid_or_inflate_visible_dimensions(self):
        grid = SimpleNamespace(rows=26,cols=24,assignments=(),unassigned_indices=(),cells=(
            SimpleNamespace(row=25,col=23,count=0,bounds=(10.,10.,80.,80.),source="history"),))
        with patch.object(live.cv2,"rectangle",wraps=live.cv2.rectangle) as rectangle, \
                patch.object(live.cv2,"putText",wraps=live.cv2.putText) as text:
            _, counts = live.render_frame(np.zeros((100,200,3),np.uint8),np.empty((0,4)),
                                           np.empty((0,2,3)),(),grid=grid)
        rectangle.assert_not_called()
        self.assertEqual((counts["rows"],counts["cols"],counts["cells"]),(0,0,0))
        self.assertFalse(any(call.args[1].startswith("(") for call in text.call_args_list))
        self.assertTrue(any("waiting for confirmed boxes" in call.args[1] for call in text.call_args_list))

    def test_unconfirmed_detections_keep_current_count_without_cell_overlays(self):
        boxes=np.array([[20.,20.,80.,50.],[100.,20.,170.,50.]])
        points=np.array([[[20.,40.,.9],[80.,40.,.9]],[[100.,40.,.1],[170.,40.,.1]]])
        grid=SimpleNamespace(rows=0,cols=0,assignments=(),unassigned_indices=(0,1),cells=())
        with patch.object(live.cv2,"putText",wraps=live.cv2.putText) as text:
            _,counts=live.render_frame(np.zeros((100,200,3),np.uint8),boxes,points,(0,1),grid=grid)
        self.assertEqual(counts["boxes"],2)
        self.assertEqual(counts["assigned"]+counts["unassigned"],2)
        self.assertEqual(counts["edges"],1)
        self.assertFalse(any(call.args[1].startswith("(") for call in text.call_args_list))

    def test_stable_count_does_not_hide_current_detections_or_draw_lost_ones(self):
        boxes = np.array([[20., 20., 80., 50.], [100., 20., 170., 50.]])
        points = np.array([[[20., 40., .9], [80., 40., .9]], [[100., 40., .1], [170., 40., .1]]])
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as text:
            _, counts = live.render_frame(np.zeros((100, 200, 3), np.uint8), boxes, points, (0, 1),
                                           stable_count=3, lost_count=4)
        labels = [call.args[1] for call in text.call_args_list if call.args[1].startswith("#")]
        self.assertEqual(labels, [])
        self.assertEqual(counts["boxes"], 2)
        self.assertEqual(counts["stable"], 3)
        self.assertEqual(counts["lost_tracks"], 4)
        self.assertEqual(counts["partial"], 1)

    def test_loop_pairs_predictions_with_their_frames_and_releases_resources(self):
        class Capture:
            def __init__(self, config):
                self.info = {"name":"fake D405"};self.rate = SimpleNamespace(fps=lambda:30.)
                self.stopped = False;self.index = 0
            def start(self):return self
            def latest(self, sequence):
                self.index += 1
                return dict(sequence=self.index, image=np.full((100, 200, 3), self.index * 30, np.uint8),
                            frame_number=self.index, device_timestamp_ms=self.index * 10., received_at=time.monotonic())
            def stop(self):self.stopped=True
        capture = Capture(None)
        empty = (np.empty((0, 4)), np.empty((0, 2, 3)), np.empty(0), np.empty(0, dtype=int))
        model = Mock();model.predict.return_value=[SimpleNamespace()]
        with tempfile.TemporaryDirectory() as d, patch.object(live,"check_display"), patch.object(live,"configure_local_cache"), \
                patch.object(live,"load_model",return_value=(model,{})), patch.object(live,"D405Capture",return_value=capture), \
                patch.object(live,"prediction_arrays",return_value=empty), patch.object(live,"REPORT_PATH",Path(d)/"report.json"), \
                patch.object(live.cv2,"namedWindow"), patch.object(live.cv2,"resizeWindow"), \
                patch.object(live.cv2,"imshow") as show, patch.object(live.cv2,"waitKey",side_effect=[ord('d'),ord('q')]), \
                patch.object(live.cv2,"getWindowProperty",return_value=1), patch.object(live.cv2,"destroyAllWindows") as close:
            self.assertEqual(live.main(), 0)
            self.assertTrue(capture.stopped);close.assert_called_once()
            for index, call in enumerate(show.call_args_list, 1):
                np.testing.assert_array_equal(call.args[1][-1,199], [index*30]*3)
            report=json.loads((Path(d)/"report.json").read_text())
            self.assertEqual(report["last_counts"]["boxes"],0)
            self.assertEqual(report["processed_frames"],2)
            for box_call, basket_call in zip(model.predict.call_args_list, self.basket_model.predict.call_args_list[1:]):
                self.assertIs(box_call.args[0], basket_call.args[0])
            self.assertEqual(self.basket_model.predict.call_count, 3)  # warmup + two frames

            self.assertFalse(report["dedup_enabled_at_exit"])

    def test_toggle_without_new_frame_does_not_advance_temporal_processing(self):
        image = np.zeros((100, 200, 3), np.uint8)
        capture = Mock()
        capture.info = {"name": "fake D405"}
        capture.rate.fps.return_value = 30.
        now = time.monotonic()
        capture.latest.side_effect = [
            dict(sequence=1, image=image, frame_number=1, device_timestamp_ms=10., received_at=now), None, None,
            dict(sequence=2, image=image, frame_number=2, device_timestamp_ms=20., received_at=now + .1),
        ]
        arrays = (np.array([[20., 20., 80., 50.]]), np.array([[[20., 40., .9], [80., 40., .9]]]),
                  np.array([.9]), np.array([0]))
        model = Mock()
        model.predict.return_value = [SimpleNamespace()]
        processor = Mock(wraps=live.VideoPostprocessor(dedup_config=live.DEDUP,
                                                      temporal_config=live.TEMPORAL))
        grid_counter = Mock(wraps=live.GridCounter(live.GRID))
        with tempfile.TemporaryDirectory() as d, patch.object(live, "check_display"), \
                patch.object(live, "configure_local_cache"), patch.object(live, "load_model", return_value=(model, {})), \
                patch.object(live, "D405Capture", return_value=capture), \
                patch.object(live, "VideoPostprocessor", return_value=processor), \
                patch.object(live, "GridCounter", return_value=grid_counter), \
                patch.object(live, "prediction_arrays", return_value=arrays), \
                patch.object(live, "REPORT_PATH", Path(d)/"report.json"), \
                patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"), \
                patch.object(live.cv2, "imshow") as show, \
                patch.object(live.cv2, "waitKey", side_effect=[ord('d'), ord('d'), ord('d'), ord('q')]), \
                patch.object(live.cv2, "getWindowProperty", return_value=1), \
                patch.object(live.cv2, "destroyAllWindows"):
            self.assertEqual(live.main(), 0)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(show.call_count, 2)
        self.assertEqual([call.kwargs["sequence"] for call in processor.update.call_args_list], [1, 2])
        self.assertEqual([call.kwargs["dedup_enabled"] for call in processor.update.call_args_list], [True, False])
        self.assertEqual([call.kwargs["sequence"] for call in grid_counter.update.call_args_list], [1, 2])

    def test_stalled_camera_clears_overlays_without_updating_tracks(self):
        image = np.zeros((100, 200, 3), np.uint8)
        capture = Mock()
        capture.info = {}
        capture.rate.fps.return_value = 30.
        capture.latest.side_effect = [dict(sequence=1, image=image, frame_number=1, device_timestamp_ms=10., received_at=1.), None]
        model = Mock()
        model.predict.return_value = [SimpleNamespace()]
        arrays = (np.array([[20., 20., 80., 50.]]), np.array([[[20., 40., .9], [80., 40., .9]]]),
                  np.array([.9]), np.array([0]))
        processor = Mock(wraps=live.VideoPostprocessor())
        with tempfile.TemporaryDirectory() as d, patch.object(live, "check_display"), \
                patch.object(live, "configure_local_cache"), patch.object(live, "load_model", return_value=(model, {})), \
                patch.object(live, "D405Capture", return_value=capture), \
                patch.object(live, "VideoPostprocessor", return_value=processor), \
                patch.object(live, "prediction_arrays", return_value=arrays), \
                patch.object(live, "REPORT_PATH", Path(d)/"report.json"), \
                patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"), \
                patch.object(live.cv2, "imshow") as show, \
                patch.object(live.cv2, "waitKey", side_effect=[-1, ord('q')]), \
                patch.object(live.cv2, "getWindowProperty", return_value=1), \
                patch.object(live.cv2, "destroyAllWindows"), \
                patch.object(live, "render_waiting_frame", wraps=live.render_waiting_frame) as waiting:
            self.assertEqual(live.main(), 0)
        self.assertEqual(processor.update.call_count, 1)
        self.assertEqual(show.call_count, 2)
        waiting.assert_called_once()
        self.assertFalse(show.call_args_list[-1].args[1][live.HEADER_HEIGHT:, :image.shape[1]].any())

    def test_inference_failure_still_stops_camera(self):
        capture=Mock();capture.info={}
        capture.latest.return_value=dict(sequence=1,image=np.zeros((20,30,3),np.uint8),received_at=time.monotonic())
        model=Mock();model.predict.side_effect=RuntimeError("simulated inference failure")
        with tempfile.TemporaryDirectory() as d, patch.object(live,"check_display"), patch.object(live,"configure_local_cache"), \
                patch.object(live,"load_model",return_value=(model,{})), patch.object(live,"D405Capture",return_value=capture), \
                patch.object(live,"REPORT_PATH",Path(d)/"report.json"), patch.object(live.cv2,"namedWindow"), \
                patch.object(live.cv2,"resizeWindow"), patch.object(live.cv2,"destroyAllWindows"):
            self.assertEqual(live.main(),1)
            capture.stop.assert_called_once()
            self.assertEqual(json.loads((Path(d)/"report.json").read_text())["status"],"failed")


if __name__ == "__main__":
    unittest.main()
