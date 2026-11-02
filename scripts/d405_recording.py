"""Bounded, asynchronous recording of raw or annotated D405 BGR color frames.

The capture thread calls ``submit`` with its default raw mode for every frame,
independently of display, inference and pause. The viewer may instead record a
fresh, annotated camera-size image with ``submit(packet, mode='annotated')``.
Only submissions matching the active session's mode are accepted. Submitted
images must remain immutable until encoded; no header/display canvas is stored.
"""
from copy import deepcopy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import threading
import time
from zoneinfo import ZoneInfo


def _now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')


def _save_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class RawVideoRecorder:
    """One recording at a time; defaults remain compatible with raw capture."""

    MODES = ('raw', 'annotated')
    QUEUE_FRAMES = 16
    CSV_FIELDS = ('output_index', 'sequence', 'frame_number', 'device_timestamp_ms',
                  'received_at_monotonic', 'elapsed_seconds')

    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()
        self.recordings = []
        self._lock = threading.Lock()
        self._operations = threading.RLock()
        self._session = None

    def start(self, camera_info: dict, mode='raw', context=None):
        """Open the encoder before accepting frames; repeated start is harmless."""
        import cv2
        if mode not in self.MODES:
            raise ValueError(f'Recording mode must be one of {self.MODES}, got {mode!r}.')
        with self._operations:
            with self._lock:
                current = self._session
                active = current is not None and current['accepting']
            if active:
                return self.status()
            if current is not None:
                self.stop('restart_after_error')

            width, height = int(camera_info['width']), int(camera_info['height'])
            fps = float(camera_info['fps'])
            container_fps = float(camera_info.get('container_fps', fps))
            if (width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0
                    or not math.isfinite(container_fps) or container_fps <= 0):
                raise ValueError('Recording requires positive camera dimensions, camera FPS and container FPS.')
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
            folder = self.directory / f'recording_{stamp}'
            folder.mkdir(exist_ok=False)
            video_path, frames_path = folder / f'{mode}.mp4', folder / 'frames.csv'
            started = time.monotonic()
            metadata = {
                'status': 'recording', 'source': deepcopy(camera_info),
                'video_path': str(video_path), 'frames_path': str(frames_path),
                'metadata_path': str(folder / 'metadata.json'),
                'width': width, 'height': height, 'pixel_format': 'BGR8',
                'codec_requested': 'mp4v', 'container': 'mp4',
                'requested_fps': float(camera_info.get('requested_fps', fps)),
                'camera_profile_fps': fps, 'container_fps': container_fps,
                'started_at': _now(), 'ended_at': None, 'stop_reason': None,
                'received_frames': 0, 'written_frames': 0, 'dropped_frames': 0,
                'camera_missing_frames': 0, 'camera_frame_discontinuities': 0,
                'processed_frame_gaps': 0, 'processed_frame_discontinuities': 0,
                'camera_gap_tracking': mode == 'raw',
                'queue_capacity_frames': self.QUEUE_FRAMES,
                'recording_mode': mode, 'raw_pixels': mode == 'raw',
                'annotations_burned_in': mode == 'annotated', 'context': deepcopy(context),
                'timing_note': 'MP4 is constant frame rate. Missing frames are not duplicated. '
                               'Use frames.csv for original frame receive timing; fewer submitted or '
                               'queue-dropped frames shorten playback at the selected container FPS.',
                'gap_note': ('camera_missing_frames counts gaps between frame numbers submitted by '
                             'the capture thread; queue overflow is counted separately in dropped_frames.'
                             if mode == 'raw' else
                             'processed_frame_gaps counts skipped camera frame numbers between submitted '
                             'annotated frames. Processing, display rate and pauses can cause these gaps; '
                             'they do not measure camera frame loss. Camera gap counters are inactive '
                             '(camera_gap_tracking=false); queue overflow is counted separately.'),
                'error': None,
            }
            session = {
                'metadata': metadata, 'started': started, 'stopped': None,
                'accepting': False, 'queue': queue.Queue(maxsize=self.QUEUE_FRAMES),
                'stop_event': threading.Event(), 'thread': None,
                'first_received': None, 'last_received': None, 'previous_frame': None,
                'first_written': None, 'last_written': None,
            }
            writer = csv_file = None
            try:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                         container_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('OpenCV could not open the mp4v MP4 video encoder.')
                csv_file = frames_path.open('w', newline='', encoding='utf-8')
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
                csv_writer.writeheader()
                csv_file.flush()
                metadata['encoder_backend'] = writer.getBackendName()
                _save_json(folder / 'metadata.json', metadata)
            except Exception as exc:
                if writer is not None:
                    writer.release()
                if csv_file is not None:
                    csv_file.close()
                metadata.update(status='failed', error=str(exc), ended_at=_now(),
                                stop_reason='start_error', elapsed_seconds=time.monotonic() - started)
                session['stopped'] = time.monotonic()
                with self._lock:
                    self._session = session
                    self.recordings.append(deepcopy(metadata))
                _save_json(folder / 'metadata.json', metadata)
                raise
            session.update(writer=writer, csv_file=csv_file, csv_writer=csv_writer, accepting=True)
            thread = threading.Thread(target=self._write, args=(session,), name=f'D405-{mode}-recorder', daemon=True)
            session['thread'] = thread
            with self._lock:
                self._session = session
            thread.start()
            return self.status()

    def submit(self, packet, mode='raw'):
        """Non-blocking enqueue for the matching mode; supplied images must stay immutable."""
        if mode not in self.MODES:
            raise ValueError(f'Recording mode must be one of {self.MODES}, got {mode!r}.')
        with self._lock:
            session = self._session
            if (session is None or not session['accepting']
                    or session['metadata']['recording_mode'] != mode):
                return
            metadata = session['metadata']
            received = float(packet['received_at'])
            if received < session['started']:
                return  # A frame acquired just before R can arrive here after start().
            image = packet['image']
            if image.shape != (metadata['height'], metadata['width'], 3) or str(image.dtype) != 'uint8':
                self._error(session, 'Recording frame shape or dtype differs from the original camera image.')
                return
            frame_number = int(packet['frame_number'])
            previous = session['previous_frame']
            if previous is not None:
                gaps = 'camera_missing_frames' if mode == 'raw' else 'processed_frame_gaps'
                discontinuities = ('camera_frame_discontinuities' if mode == 'raw'
                                   else 'processed_frame_discontinuities')
                if frame_number > previous:
                    metadata[gaps] += frame_number - previous - 1
                else:
                    metadata[discontinuities] += 1
            session['previous_frame'] = frame_number
            if session['first_received'] is None:
                session['first_received'] = received
                metadata['first_device_timestamp_ms'] = float(packet['device_timestamp_ms'])
                metadata['first_frame_number'] = frame_number
            session['last_received'] = received
            metadata['last_device_timestamp_ms'] = float(packet['device_timestamp_ms'])
            metadata['last_frame_number'] = frame_number
            metadata['received_frames'] += 1
            item = {'image': image, 'sequence': int(packet['sequence']), 'frame_number': frame_number,
                    'device_timestamp_ms': float(packet['device_timestamp_ms']),
                    'received_at_monotonic': received, 'elapsed_seconds': received - session['started']}
            try:
                session['queue'].put_nowait(item)
            except queue.Full:
                metadata['dropped_frames'] += 1

    def _error(self, session, message):
        """Called with the state lock held."""
        metadata = session['metadata']
        metadata['error'] = (metadata['error'] + '; ' if metadata['error'] else '') + str(message)
        session['accepting'] = False
        if session['stopped'] is None:
            session['stopped'] = time.monotonic()
            metadata.update(ended_at=_now(), stop_reason='recording_error')
        session['stop_event'].set()

    def _write(self, session):
        metadata = session['metadata']
        try:
            while not session['stop_event'].is_set() or not session['queue'].empty():
                try:
                    item = session['queue'].get(timeout=.1)
                except queue.Empty:
                    continue
                session['writer'].write(item['image'])
                with self._lock:
                    index = metadata['written_frames']
                row = {key: item[key] for key in self.CSV_FIELDS if key != 'output_index'}
                session['csv_writer'].writerow({'output_index': index, **row})
                with self._lock:
                    metadata['written_frames'] += 1
                    if session['first_written'] is None:
                        session['first_written'] = item['received_at_monotonic']
                    session['last_written'] = item['received_at_monotonic']
                if (index + 1) % 30 == 0:
                    session['csv_file'].flush()
        except Exception as exc:
            with self._lock:
                self._error(session, exc)
        finally:
            for close in (session['writer'].release, session['csv_file'].close):
                try:
                    close()
                except Exception as exc:
                    with self._lock:
                        self._error(session, exc)
            self._finalize(session)

    def _finalize(self, session):
        import cv2
        metadata = session['metadata']
        capture = None
        try:
            path = Path(metadata['video_path'])
            if not metadata['written_frames'] or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('No valid video frames were recorded.')
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()  # Verify only one frame, not an unbounded full-video decode.
            if not capture.isOpened() or not ok or frame.shape[:2] != (metadata['height'], metadata['width']):
                raise RuntimeError('Final MP4 failed first-frame decoding or dimensions validation.')
            frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            if frames != metadata['written_frames']:
                raise RuntimeError(f'MP4 frame count {frames} differs from CSV count {metadata["written_frames"]}.')
            metadata['validation'] = {
                'first_frame_decoded': True, 'container_frame_count': frames,
                'container_fps': capture.get(cv2.CAP_PROP_FPS), 'size_bytes': path.stat().st_size,
                'decoded_shape': list(frame.shape),
            }
        except Exception as exc:
            with self._lock:
                self._error(session, exc)
        finally:
            if capture is not None:
                capture.release()

        with self._lock:
            if session['stopped'] is None:
                session['stopped'] = time.monotonic()
                metadata['ended_at'] = _now()
            session['accepting'] = False
            metadata.update(status='failed' if metadata['error'] else 'completed',
                            elapsed_seconds=session['stopped'] - session['started'],
                            finalized_at=_now(), video_duration_seconds=metadata['written_frames'] / metadata['container_fps'],
                            unwritten_frames=metadata['received_frames'] - metadata['written_frames'] - metadata['dropped_frames'])
            for label in ('received', 'written'):
                first, last = session[f'first_{label}'], session[f'last_{label}']
                count = metadata[f'{label}_frames']
                metadata[f'{label}_capture_span_seconds'] = last - first if count else 0.
                metadata[f'{label}_fps'] = (count - 1) / (last - first) if count > 1 and last > first else None
        try:
            _save_json(Path(metadata['metadata_path']), metadata)
        except Exception as exc:
            with self._lock:
                self._error(session, f'Could not save final metadata: {exc}')
                metadata['status'] = 'failed'
        with self._lock:
            self.recordings.append(deepcopy(metadata))

    def stop(self, reason='user'):
        """Stop accepting new frames, flush the bounded queue, and finalize MP4."""
        with self._operations:
            with self._lock:
                session = self._session
                if session is None:
                    return None
                if session['thread'] is None or not session['thread'].is_alive():
                    return None
                session['accepting'] = False
                if session['stopped'] is None:
                    session['stopped'] = time.monotonic()
                    session['metadata'].update(ended_at=_now(), stop_reason=reason)
                session['stop_event'].set()
            session['thread'].join(timeout=15)
            if session['thread'].is_alive():
                with self._lock:
                    self._error(session, 'Video writer did not finish within 15 seconds; finalization is incomplete.')
                raise RuntimeError(session['metadata']['error'])
            with self._lock:
                return deepcopy(session['metadata'])

    def status(self):
        with self._lock:
            session = self._session
            if session is None:
                return {'active': False, 'elapsed_seconds': 0., 'written_frames': 0,
                        'dropped_frames': 0, 'camera_missing_frames': 0,
                        'processed_frame_gaps': 0, 'processed_frame_discontinuities': 0,
                        'mode': None, 'error': None, 'video_path': None}
            metadata = session['metadata']
            end = session['stopped'] if session['stopped'] is not None else time.monotonic()
            return {'active': session['accepting'], 'elapsed_seconds': end - session['started'],
                    'written_frames': metadata['written_frames'], 'dropped_frames': metadata['dropped_frames'],
                    'mode': metadata['recording_mode'],
                    'camera_missing_frames': metadata['camera_missing_frames'],
                    'processed_frame_gaps': metadata['processed_frame_gaps'],
                    'processed_frame_discontinuities': metadata['processed_frame_discontinuities'],
                    'error': metadata['error'],
                    'video_path': metadata['video_path']}
