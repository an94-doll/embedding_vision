"""Capture contract checks with a fake SDK; no test opens a physical camera."""

import importlib
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import d405_camera


class FakeDevice:
    def __init__(self, serial='test-serial', name='Intel RealSense D405'):
        self.info = {'name': name, 'serial_number': serial, 'firmware_version': 'test-fw',
                     'usb_type_descriptor': '3.2'}

    def get_info(self, key):
        return self.info[key]


class FakeConfig:
    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *profile):
        self.profile = profile


class FakeColor:
    def __init__(self, pixels, number):
        self.pixels = pixels
        self.number = number

    def get_data(self):
        return self.pixels

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.number * 10.0


class FakePipeline:
    def __init__(self, items=()):
        self.items = queue.Queue()
        for item in items:
            self.items.put(item)
        self.start_calls = 0
        self.stop_calls = 0
        self.start_error = None

    def start(self, config):
        self.start_calls += 1
        if self.start_error:
            raise self.start_error
        self.config = config
        _, width, height, color_format, fps = config.profile
        profile = SimpleNamespace(width=lambda: width, height=lambda: height,
                                  fps=lambda: fps, format=lambda: color_format,
                                  get_intrinsics=lambda: SimpleNamespace(width=width, height=height,
                                    fx=100., fy=100., ppx=width/2, ppy=height/2,
                                    model='distortion.none', coeffs=[0.]*5))
        stream = SimpleNamespace(as_video_stream_profile=lambda: profile)
        return SimpleNamespace(get_stream=lambda _: stream)

    def wait_for_frames(self, timeout_ms):
        try:
            item = self.items.get(timeout=.02)
        except queue.Empty:
            raise RuntimeError('fake SDK timeout')
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(get_color_frame=lambda: item)

    def stop(self):
        self.stop_calls += 1


def fake_sdk(pipeline, devices=None):
    devices = [FakeDevice()] if devices is None else devices
    return SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: devices),
        pipeline=lambda _: pipeline,
        config=FakeConfig,
        camera_info=SimpleNamespace(name='name', serial_number='serial_number',
                                    firmware_version='firmware_version', usb_type_descriptor='usb_type_descriptor'),
        stream=SimpleNamespace(color='color'), format=SimpleNamespace(bgr8='bgr8'))


class CaptureTests(unittest.TestCase):
    def make_capture(self, pipeline=None, devices=None, **config):
        pipeline = FakePipeline() if pipeline is None else pipeline
        with patch.dict(sys.modules, {'pyrealsense2': fake_sdk(pipeline, devices)}):
            capture = d405_camera.D405Capture(d405_camera.CaptureConfig(**config))
        self.addCleanup(capture.stop)
        return capture

    def test_import_and_configuration_need_no_sdk(self):
        with patch.dict(sys.modules, {'pyrealsense2': None}):
            module = importlib.reload(d405_camera)
            config = module.CaptureConfig()
        self.assertEqual((config.width, config.height, config.fps, config.serial, config.timeout),
                         (1280, 720, 30, None, 5.0))

    def test_constructor_does_not_start_and_stop_is_idempotent(self):
        capture = self.make_capture()
        self.assertEqual(capture.pipeline.start_calls, 0)
        self.assertIsNone(capture.latest(0, wait=0))
        capture.stop()
        capture.stop()
        self.assertEqual(capture.pipeline.stop_calls, 0)

    def test_newest_packet_is_owned_and_recorder_sees_every_frame(self):
        pixels = [np.full((4, 6, 3), n, dtype=np.uint8) for n in range(1, 4)]
        pipeline = FakePipeline([FakeColor(image, 40 + n) for n, image in enumerate(pixels)])
        capture = self.make_capture(pipeline, width=6, height=4, fps=15)
        recorded = []
        complete = threading.Event()

        def submit(packet):
            recorded.append(packet)
            if len(recorded) == 3:
                complete.set()

        capture.recorder = SimpleNamespace(submit=submit)
        self.assertIs(capture.start(), capture)
        self.assertTrue(complete.wait(1), capture.error)
        packet = capture.latest(0, wait=0)
        self.assertEqual(packet['sequence'], 3)
        self.assertEqual(packet['frame_number'], 42)
        self.assertEqual(packet['device_timestamp_ms'], 420.0)
        self.assertEqual(capture.received, 3)
        self.assertEqual([p['sequence'] for p in recorded], [1, 2, 3])
        self.assertIs(packet, recorded[-1])
        self.assertIsNone(capture.latest(3, wait=0))
        self.assertGreater(packet['received_at'], 0)
        self.assertGreater(capture.rate.fps(), 0)
        for source, saved in zip(pixels, recorded):
            source.fill(0)
            self.assertFalse(np.shares_memory(source, saved['image']))
            self.assertTrue(np.all(saved['image'] == saved['sequence']))
        self.assertEqual((capture.info['width'], capture.info['height'], capture.info['fps']), (6, 4, 15))
        capture.stop()
        self.assertFalse(capture.active)
        self.assertFalse(capture.thread.is_alive())
        self.assertIsNone(capture.error)
        capture.stop()
        self.assertEqual(pipeline.stop_calls, 1)

    def test_capture_error_wakes_latest_and_remains_visible(self):
        capture = self.make_capture(FakePipeline([ValueError('bad frame data')]))
        capture.start()
        with self.assertRaisesRegex(RuntimeError, 'bad frame data'):
            capture.latest(0, wait=1)
        self.assertEqual(capture.error, 'bad frame data')
        with self.assertRaisesRegex(RuntimeError, 'bad frame data'):
            capture.latest(0, wait=0)

    def test_repeated_sdk_timeout_reaches_caller(self):
        capture = self.make_capture(timeout=.03)
        capture.start()
        with self.assertRaisesRegex(RuntimeError, 'no color frame for 0.03s'):
            capture.latest(0, wait=1)

    def test_transient_sdk_timeout_can_recover(self):
        pipeline = FakePipeline([RuntimeError('temporary timeout'), FakeColor(np.zeros((4, 6, 3)), 7)])
        capture = self.make_capture(pipeline)
        capture.start()
        packet = capture.latest(0, wait=1)
        self.assertIsNotNone(packet)
        self.assertEqual(packet['frame_number'], 7)
        self.assertIsNone(capture.error)

    def test_recorder_error_is_propagated(self):
        capture = self.make_capture(FakePipeline([FakeColor(np.zeros((4, 6, 3)), 7)]))

        def submit(_):
            raise ValueError('recorder stopped')

        capture.recorder = SimpleNamespace(submit=submit)
        capture.start()
        capture.thread.join(timeout=1)
        self.assertFalse(capture.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'recorder stopped'):
            capture.latest(0, wait=0)

    def test_device_selection_requires_one_match_and_accepts_serial(self):
        devices = [FakeDevice('one'), FakeDevice('two'), FakeDevice('other', 'D435')]
        capture = self.make_capture(devices=devices)
        with self.assertRaisesRegex(RuntimeError, 'Found 2 matching D405'):
            capture.start()
        self.assertEqual(capture.pipeline.start_calls, 0)
        selected = self.make_capture(devices=devices, serial='two')
        selected.start()
        self.assertEqual(selected.info['serial'], 'two')
        self.assertEqual(selected.pipeline.config.serial, 'two')

    def test_start_failure_is_actionable_and_not_marked_active(self):
        pipeline = FakePipeline()
        pipeline.start_error = RuntimeError('unsupported profile')
        capture = self.make_capture(pipeline)
        with self.assertRaisesRegex(RuntimeError, 'D405 stream start failed: unsupported profile'):
            capture.start()
        self.assertFalse(capture.active)
        self.assertIsNone(capture.thread)
        capture.stop()
        self.assertEqual(pipeline.stop_calls, 0)


class RateTests(unittest.TestCase):
    def test_rate_is_bounded_and_zeros_when_stale(self):
        rate = d405_camera.Rate()
        with patch.object(d405_camera.time, 'monotonic', return_value=10):
            self.assertEqual(rate.fps(), 0)
            for n in range(200):
                rate.add(10 - (199 - n) / 30)
            self.assertEqual(len(rate.times), 120)
            self.assertAlmostEqual(rate.fps(), 30)
        with patch.object(d405_camera.time, 'monotonic', return_value=13):
            self.assertEqual(rate.fps(), 0)


if __name__ == '__main__':
    unittest.main()
