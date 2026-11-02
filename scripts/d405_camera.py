"""Reusable latest-frame D405 color capture; importing this module opens no device."""

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class CaptureConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    serial: str | None = None
    timeout: float = 5.0


class Rate:
    def __init__(self):
        self.times = deque(maxlen=120)

    def add(self, when=None):
        self.times.append(time.monotonic() if when is None else when)

    def fps(self):
        times = list(self.times)
        if len(times) < 2 or time.monotonic() - times[-1] > 2:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-9)


class D405Capture:
    """Keep one latest color frame, never a growing queue or a retained SDK buffer.

    ``args`` accepts a CaptureConfig or the original viewer's argparse Namespace.
    The optional ``recorder`` receives every captured packet through ``submit``.
    Packet images own their memory; callers must treat published packets as read-only.
    """

    def __init__(self, args):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("RealSense SDK missing: install pyrealsense2 in this Python environment.") from exc
        self.rs = rs
        self.args = args
        self.context = rs.context()
        self.pipeline = rs.pipeline(self.context)
        self.active = False
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread = None
        self.packet = None
        self.error = None
        self.received = 0
        self.rate = Rate()
        self.info = {}
        self.recorder = None

    def start(self):
        rs = self.rs
        devices = list(self.context.query_devices())
        candidates = [d for d in devices if 'D405' in d.get_info(rs.camera_info.name)]
        if self.args.serial:
            candidates = [d for d in candidates if d.get_info(rs.camera_info.serial_number) == self.args.serial]
        if len(candidates) != 1:
            raise RuntimeError(f'Found {len(candidates)} matching D405 cameras. Check USB 3 connection; '
                               'if multiple cameras are connected, select one serial number.')
        device = candidates[0]
        serial = device.get_info(rs.camera_info.serial_number)
        self.info = {'name': device.get_info(rs.camera_info.name), 'serial': serial,
                     'firmware': device.get_info(rs.camera_info.firmware_version),
                     'usb': device.get_info(rs.camera_info.usb_type_descriptor)}
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)
        try:
            profile = self.pipeline.start(config)
            self.active = True
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color.get_intrinsics()
            self.info['intrinsics'] = dict(width=intr.width, height=intr.height,
                fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                model=str(intr.model), coeffs=list(intr.coeffs))
            self.info.update(width=color.width(), height=color.height(), fps=color.fps(), format=str(color.format()))
        except RuntimeError as exc:
            raise RuntimeError(f'D405 stream start failed: {exc}. Close other camera apps and check USB/profile.') from exc
        self.thread = threading.Thread(target=self._capture, name='D405-latest-color', daemon=True)
        self.thread.start()
        return self

    def _capture(self):
        import numpy as np
        last_good = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(1000)
                except RuntimeError as exc:
                    if self.stop_event.is_set():
                        break
                    if time.monotonic() - last_good < self.args.timeout:
                        continue
                    raise RuntimeError(f'D405 received no color frame for {self.args.timeout:g}s: {exc}. '
                                       'Check the USB cable/port and reconnect the camera.') from exc
                color = frames.get_color_frame()
                if not color:
                    if time.monotonic() - last_good >= self.args.timeout:
                        raise RuntimeError('D405 stream contains no color frames.')
                    continue
                received_at = time.monotonic()
                pixels = np.asanyarray(color.get_data()).copy()
                packet = {'image': pixels, 'frame_number': color.get_frame_number(),
                          'device_timestamp_ms': color.get_timestamp(), 'received_at': received_at}
                with self.condition:
                    self.received += 1
                    packet['sequence'] = self.received
                    self.rate.add(received_at)
                    self.packet = packet
                    self.condition.notify_all()
                # Recording receives every captured raw frame, independently of
                # inference speed, preview pause, and latest-frame display drops.
                if self.recorder is not None:
                    self.recorder.submit(packet)
                last_good = received_at
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def latest(self, after_sequence, wait=0.01):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if self.packet is None or self.packet['sequence'] <= after_sequence:
                self.condition.wait(wait)
            if self.error:
                raise RuntimeError(self.error)
            return self.packet if self.packet is not None and self.packet['sequence'] > after_sequence else None

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.active:
            try:
                self.pipeline.stop()
            finally:
                self.active = False
        if self.thread:
            self.thread.join(timeout=1)
