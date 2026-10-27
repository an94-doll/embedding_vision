#!/usr/bin/env python3
"""Live RealSense D405 color viewer with the user-selected two-endpoint YOLO base."""
import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo

from base_weights import resolve_base
from check_env import ROOT, configure_local_cache, project_path, write_report
from d405_recording import RawVideoRecorder

HEADER_HEIGHT = 136
WINDOW = 'D405 + YOLO | R record | A recording mode | Q/Esc quit | B boxes | D diagnostics'


def resolve_live_weights(explicit=None):
    """An explicit local checkpoint overrides only this viewer, never the shared base."""
    if explicit is None:
        return resolve_base()
    import torch
    weights = project_path(explicit)
    if not weights.is_file():
        raise FileNotFoundError(f'Explicit local weights do not exist: {weights}')
    # Hash and load the same open file, even if another process atomically replaces its path.
    with weights.open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=False)
    stored_epoch = int(checkpoint.get('epoch', -1))
    del checkpoint
    epoch = stored_epoch + 1 if stored_epoch >= 0 else 'unknown'
    actual_sha = checksum.hexdigest()
    sidecar = weights.with_suffix('.json')
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if saved.get('sha256') != actual_sha:
            raise ValueError(f'Explicit weights checksum differs from sidecar: {sidecar}')
        if saved.get('epoch') != epoch:
            raise ValueError(f'Explicit checkpoint epoch {epoch} differs from sidecar: {sidecar}')
    defaults = json.loads((ROOT / 'weights/base.json').read_text())
    return weights, {'sha256': actual_sha, 'epoch': epoch,
                     'stored_epoch_zero_based': stored_epoch,
                     'epoch_source': 'checkpoint.epoch + 1' if stored_epoch >= 0 else 'stripped checkpoint; unknown',
                     'selection': 'explicit local weights; shared base unchanged',
                     'sidecar_verified': str(sidecar) if sidecar.exists() else None,
                     'inference': dict(defaults['inference'])}


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
    """Keep one latest color frame, never a growing queue or a retained SDK buffer."""
    def __init__(self, args):
        import pyrealsense2 as rs
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
                               'if multiple cameras are connected, specify --serial.')
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


@dataclass
class Controls:
    conf: float
    endpoint_conf: float
    iou: float
    boxes: bool = False
    diagnostics: bool = False
    enabled: bool = True
    paused: bool = False
    quit: bool = False
    record_mode: str = 'raw'
    notice: str = ''
    notice_until: float = 0.0

    def notify(self, message):
        self.notice = message
        self.notice_until = time.monotonic() + 3

    def key(self, key, recording=None):
        """Return whether this key requests saving the currently displayed frame."""
        recording = recording or {'active': False}
        if key == ord('a'):
            if recording['active']:
                self.notify('Mode locked during recording. Press R to stop first.')
            else:
                self.record_mode = 'annotated' if self.record_mode == 'raw' else 'raw'
                self.notify(f'Record mode: {self.record_mode.upper()}. Press R to start.')
            return False
        if recording['active'] and recording.get('mode') == 'annotated' and key in (ord('y'), ord(' ')):
            self.notify('ANNOTATED recording needs live YOLO. Press R to stop first.')
            return False
        if key in (27, ord('q')):
            self.quit = True
        elif key == ord('b'):
            self.boxes = not self.boxes
        elif key == ord('d'):
            self.diagnostics = not self.diagnostics
        elif key == ord('y'):
            self.enabled = not self.enabled
        elif key == ord(' '):
            self.paused = not self.paused
        elif key in map(ord, '123'):
            self.conf = {ord('1'): .10, ord('2'): .25, ord('3'): .50}[key]
        else:
            adjustments = {ord('['): ('conf', -.05), ord(']'): ('conf', .05),
                           ord(','): ('endpoint_conf', -.05), ord('.'): ('endpoint_conf', .05),
                           ord('-'): ('iou', -.05), ord('='): ('iou', .05)}
            if key in adjustments:
                name, delta = adjustments[key]
                setattr(self, name, round(min(.95, max(.01, getattr(self, name) + delta)), 2))
        return key == ord('s')


def render(packet, detections, controls, metrics, epoch, overlap_iou, recording=None):
    import cv2
    import numpy as np
    image = packet['image'].copy()
    height, width = image.shape[:2]
    hidden = pairs = visible_edges = 0
    duplicate_ids = set()
    if controls.diagnostics:
        for i, a in enumerate(detections):
            for j in range(i + 1, len(detections)):
                b = detections[j]
                x1, y1 = max(a['bbox'][0], b['bbox'][0]), max(a['bbox'][1], b['bbox'][1])
                x2, y2 = min(a['bbox'][2], b['bbox'][2]), min(a['bbox'][3], b['bbox'][3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area = lambda box: max(0, box[2]-box[0]) * max(0, box[3]-box[1])
                union = area(a['bbox']) + area(b['bbox']) - inter
                if union > 0 and inter / union >= overlap_iou:
                    pairs += 1
                    duplicate_ids.update((i+1, j+1))
    for number, obj in enumerate(detections, 1):
        points = obj['keypoints']
        valid = [p[2] >= controls.endpoint_conf and 0 <= p[0] < width and 0 <= p[1] < height for p in points]
        coords = [tuple(round(v) for v in p[:2]) for p in points]
        hidden += sum(p[2] < controls.endpoint_conf for p in points)
        if all(valid):
            cv2.line(image, coords[0], coords[1], (121,111,255), 2, cv2.LINE_AA)
            visible_edges += 1
        for side, (coord, good) in enumerate(zip(coords, valid)):
            if good:
                cv2.circle(image, coord, 4, ((161,233,113),(255,198,119))[side], -1, cv2.LINE_AA)
                cv2.circle(image, coord, 4, (32,32,16), 1, cv2.LINE_AA)
            elif controls.diagnostics and 0 <= coord[0] < width and 0 <= coord[1] < height:
                cv2.drawMarker(image, coord, (0,165,255), cv2.MARKER_TILTED_CROSS, 9, 2)
        if controls.boxes or controls.diagnostics:
            x1, y1, x2, y2 = [round(v) for v in obj['bbox']]
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,220,70), 1)
            lines = [f"#{number} box={obj['confidence']:.3f}"]
            if controls.diagnostics:
                lines += [f"{side}={p[2]:.3f} {'VISIBLE' if good else 'HIDDEN'}" for side,p,good in zip(('L','R'),points,valid)]
                if number in duplicate_ids:
                    lines.append('OVERLAP? review manually')
            tx, ty = max(0,min(width-210,x1)), max(14,min(height-16*len(lines),y1))
            for line in lines:
                length = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .40, 1)[0][0]
                cv2.rectangle(image,(tx,ty-13),(tx+length+3,ty+3),(24,32,32),-1)
                cv2.putText(image,line,(tx+1,ty),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,240,80),1,cv2.LINE_AA)
                ty += 16
    canvas = np.zeros((height+HEADER_HEIGHT,width,3),dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = (30,37,42)
    canvas[HEADER_HEIGHT:] = image
    state = 'PAUSED' if controls.paused else 'YOLO ON' if controls.enabled else 'RAW / YOLO OFF'
    recording = recording or {'active': False}
    record_mode = recording.get('mode', controls.record_mode) if recording['active'] else controls.record_mode
    if recording.get('error'):
        recording_line = f'REC {record_mode.upper()} ERROR | ' + str(recording['error'])[:100]
    elif recording['active']:
        seconds = int(recording['elapsed_seconds'])
        recording_line = (f"REC {record_mode.upper()} {seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02} | "
                          f"written={recording['written_frames']} dropped={recording['dropped_frames']} | " +
                          ('R stop | Live YOLO locked' if record_mode == 'annotated'
                           else 'R stop | Raw capture continues while preview is paused'))
    else:
        recording_line = f'REC OFF | Mode: {record_mode.upper()} | R start | A switch RAW / ANNOTATED'
        if recording.get('video_path'):
            recording_line += ' | Last: ' + Path(recording['video_path']).parent.name
    if controls.notice_until > time.monotonic():
        recording_line = f'{record_mode.upper()} | ' + controls.notice
    lines = [
        f"D405 | epoch {epoch} | {state} | frame {packet['frame_number']} | boxes={len(detections)} edges={visible_edges}",
        recording_line,
        f"Camera RX {metrics['camera_fps']:.1f} FPS | infer {metrics['inference_fps']:.1f} FPS | display {metrics['display_fps']:.1f} FPS | infer {metrics['infer_ms']:.1f} ms",
        f"conf={controls.conf:.2f} kp={controls.endpoint_conf:.2f} NMS IoU={controls.iou:.2f} | hidden-kp={hidden} | post-NMS pairs IoU>={overlap_iou:.2f}: {pairs if controls.diagnostics else 'D to check'}",
        'Q/Esc quit | R record | A mode | Y YOLO | B boxes | D debug | Space pause | S photo | 1/2/3 conf .10/.25/.50',
        '[ / ] box conf | , / . endpoint conf | - / = NMS IoU | IDs are per frame; scores are not accuracy',
    ]
    for i,line in enumerate(lines):
        color = (80,100,255) if i == 1 and (recording['active'] or recording.get('error') or controls.notice_until > time.monotonic()) else (235,235,235)
        cv2.putText(canvas,line,(10,19+i*21),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
    return canvas, {'visible_edges':visible_edges,'hidden_keypoints_below_threshold':hidden,
                    'post_nms_overlap_pairs':pairs if controls.diagnostics else None,
                    'overlap_iou':overlap_iou}


def save_snapshot(directory, packet, canvas, detections, displayed, context):
    import cv2
    directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    prefix=directory/f"{stamp}_frame{packet['frame_number']}"
    raw=prefix.with_name(prefix.name+'_raw.png')
    overlay=prefix.with_name(prefix.name+'_overlay.png')
    for path,pixels in ((raw,packet['image']),(overlay,canvas)):
        if not cv2.imwrite(str(path),pixels):
            raise OSError(f'Could not save {path}')
    data={**context,'saved_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'frame_number':packet['frame_number'],'device_timestamp_ms':packet['device_timestamp_ms'],
          'raw_image':str(raw),'overlay_image':str(overlay),'overlay_header_height':HEADER_HEIGHT,
          'coordinate_system':'Original unresized color image pixels; no header offset.',
          'detections':detections,'display':displayed,'ground_truth_available':False}
    write_report(prefix.with_suffix('.json'),data)
    print(json.dumps({'snapshot':str(prefix.with_suffix('.json'))}),flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',help='Explicit local pose checkpoint for this viewer; default keeps weights/base.json selection')
    parser.add_argument('--serial',help='D405 serial; required only when multiple D405 cameras are connected')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--fps',type=int,default=30,help='Requested camera stream FPS')
    parser.add_argument('--display-fps',type=float,default=30,help='Display/processing update cap; 0 = uncapped')
    parser.add_argument('--device',type=int,default=0,help='CUDA GPU index')
    parser.add_argument('--imgsz',type=int,help='Override the base model input size')
    parser.add_argument('--conf',type=float,help='Detection threshold; default from base.json')
    parser.add_argument('--endpoint-conf',type=float,help='Endpoint display threshold; default from base.json')
    parser.add_argument('--iou',type=float,help='Box NMS IoU; default from base.json')
    parser.add_argument('--overlap-iou',type=float,default=.5,help='Diagnostic overlap threshold on post-NMS boxes; does not filter predictions')
    parser.add_argument('--show-boxes',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--headless',action='store_true',help='Measure capture/inference without opening a window')
    parser.add_argument('--max-frames',type=int,default=0,help='Stop after this many processed frames; 0 = unlimited')
    parser.add_argument('--timeout',type=float,default=5,help='Abort after this many seconds without a color frame')
    parser.add_argument('--snapshot-dir',help='S-key destination; must be inside project test/')
    parser.add_argument('--record-dir',default='test/d405_recordings',help='R-key video collection directory; must be inside project test/')
    parser.add_argument('--record-mode',choices=('raw','annotated'),default='raw',help='Initial R-key recording mode; A switches mode while not recording')
    parser.add_argument('--report',default='reports/d405_live_latest.json',help='Concise startup/final JSON, no per-frame logging')
    args=parser.parse_args(argv)
    for name in ('width','height','fps'):
        if getattr(args,name)<=0:parser.error(f'{name} must be positive')
    if args.imgsz is not None and args.imgsz<=0:parser.error('imgsz must be positive')
    if args.device<0 or args.max_frames<0:parser.error('device and max-frames must be nonnegative')
    if not math.isfinite(args.display_fps) or args.display_fps<0:parser.error('display-fps must be finite and nonnegative')
    if not math.isfinite(args.timeout) or args.timeout<1:parser.error('timeout must be at least one second')
    for name in ('conf','endpoint_conf','iou','overlap_iou'):
        value=getattr(args,name)
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):parser.error(f'{name} must be between 0 and 1')
    return args


def main(argv=None):
    args=parse_args(argv)
    configure_local_cache()
    import cv2
    # OpenCV's wheel sets a bundled font directory on import even when absent.
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    if not Path(os.environ.get('QT_QPA_FONTDIR','/nonexistent')).is_dir() and font_dir.is_dir():
        os.environ['QT_QPA_FONTDIR']=str(font_dir)
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    report_path=project_path(args.report)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    snapshots=project_path(args.snapshot_dir or f'test/d405_live_{stamp}')
    if not snapshots.is_relative_to((ROOT/'test').resolve()):
        raise ValueError('Camera snapshots must stay within this project test/ directory.')
    recordings=project_path(args.record_dir)
    if not recordings.is_relative_to((ROOT/'test').resolve()):
        raise ValueError('Camera recordings must stay within this project test/ directory.')
    recorder=RawVideoRecorder(recordings)
    recording_errors=[]
    recording_error=None
    weights,metadata=resolve_live_weights(args.weights)
    settings=dict(metadata['inference'])
    endpoint=settings.pop('endpoint_conf')
    settings['nms']=True  # Explicit one-to-many + box NMS; makes the IoU control effective.
    for name in ('conf','iou','imgsz'):
        if getattr(args,name) is not None:settings[name]=getattr(args,name)
    if args.endpoint_conf is not None:endpoint=args.endpoint_conf
    controls=Controls(settings['conf'],endpoint,settings['iou'],args.show_boxes,args.diagnostics,
                      record_mode=args.record_mode)
    report={'status':'starting','started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'weights':str(weights),'weights_sha256':metadata['sha256'],'epoch':metadata['epoch'],
            'weight_selection':'explicit' if args.weights is not None else 'default_base',
            'weights_sidecar_verified':metadata.get('sidecar_verified'),
            'stored_epoch_zero_based':metadata.get('stored_epoch_zero_based'),
            'epoch_source':metadata.get('epoch_source','base manifest'),
            'initial_inference_settings':settings.copy(),'initial_endpoint_conf':endpoint,
            'camera_requested':{'width':args.width,'height':args.height,'fps':args.fps,'serial':args.serial},
            'display_fps_cap':args.display_fps,'headless':args.headless,'saved_video':False,
            'recording_directory':str(recordings),'recording_starts_automatically':False,
            'initial_recording_mode':args.record_mode,'recording_mode':args.record_mode,
            'snapshot_directory':str(snapshots),'versions':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__}}
    write_report(report_path,report)
    capture=None;window=False;processed=inferences=skipped=saved=0
    infer_ms=deque(maxlen=300);infer_rate=Rate();display_rate=Rate()
    packet=None;canvas=None;detections=[];displayed={};used_settings=None;last_sequence=0
    first_input=[];prediction_signature=None;predicted_sequence=None;last_report=time.monotonic()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no CPU fallback.')
        if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('No desktop display. Run from the host desktop terminal, or use --headless for verification.')
        model=YOLO(str(weights)).to(f'cuda:{args.device}')
        if model.task!='pose' or list(model.model.kpt_shape)!=[2,3] or model.names!={0:'cassette_front'}:
            raise RuntimeError('Base must be the cassette_front two-endpoint pose model.')
        report['gpu']=torch.cuda.get_device_name(args.device)
        # Warm up before opening the camera, using no camera images or disk output.
        model.predict(np.zeros((args.height,args.width,3),dtype=np.uint8),**settings,
                      device=args.device,save=False,save_txt=False,show=False,verbose=False)
        torch.cuda.synchronize(args.device)
        report['predictor_end2end']=bool(model.predictor.model.end2end)
        if report['predictor_end2end']:raise RuntimeError('Expected box NMS, but backend is end-to-end; IoU control would be ineffective.')
        capture=D405Capture(args)
        capture.recorder=recorder
        capture.start()
        report['camera']=capture.info
        report['status']='running'
        write_report(report_path,report)
        print(json.dumps({'status':'running','camera':capture.info,'weights':str(weights),'epoch':metadata['epoch'],'settings':settings}),flush=True)
        if not args.headless:
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW,args.width,args.height+HEADER_HEIGHT)
            window=True
        def inspect_input(module,inputs):
            if not first_input:
                tensor=inputs[0]
                first_input.append({'device':str(tensor.device),'dtype':str(tensor.dtype),'shape':list(tensor.shape)})
                if tensor.device.type!='cuda':raise RuntimeError('Camera frame did not run on CUDA.')
        hook=model.predictor.model.model.register_forward_pre_hook(inspect_input)
        while not controls.quit:
            tick=time.monotonic()
            if capture.error:raise RuntimeError(capture.error)
            fresh=None
            if not controls.paused:
                fresh=capture.latest(last_sequence)
                if fresh is not None:
                    skipped+=max(0,fresh['sequence']-last_sequence-1)
                    last_sequence=fresh['sequence'];packet=fresh
                    processed+=1
            signature=(controls.enabled,controls.conf,controls.iou)
            # Threshold changes while paused re-run the frozen frame. Never label
            # predictions with a threshold that was not used to obtain them.
            if packet is not None and (fresh is not None or signature!=prediction_signature):
                detections=[]
                used_settings={**settings,'conf':controls.conf,'iou':controls.iou}
                if controls.enabled:
                    started=time.perf_counter()
                    result=model.predict(packet['image'],**used_settings,device=args.device,
                                         save=False,save_txt=False,save_crop=False,show=False,verbose=False)[0]
                    torch.cuda.synchronize(args.device)
                    duration=(time.perf_counter()-started)*1000
                    if result.boxes.data.device.type!='cuda' or result.keypoints.data.device.type!='cuda':
                        raise RuntimeError('Expected CUDA prediction outputs.')
                    boxes=result.boxes.xyxy.cpu().numpy();points=result.keypoints.data.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
                    if points.shape!=(len(boxes),2,3) or not all(np.isfinite(a).all() for a in (boxes,points,scores)):
                        raise RuntimeError('Invalid two-endpoint model output.')
                    order=sorted(range(len(boxes)),key=lambda i:(boxes[i,1],boxes[i,0]))
                    detections=[{'bbox':boxes[i].tolist(),'keypoints':points[i].tolist(),'confidence':float(scores[i]),'class_id':0} for i in order]
                    predicted_sequence=packet['sequence']
                    inferences+=1;infer_rate.add();infer_ms.append(duration)
                prediction_signature=signature
            if fresh is not None:
                display_rate.add()
            if packet is not None:
                metrics={'camera_fps':capture.rate.fps(),'inference_fps':infer_rate.fps() if controls.enabled and not controls.paused else 0,
                         'display_fps':display_rate.fps() if not controls.paused else 0,
                         'infer_ms':infer_ms[-1] if controls.enabled and infer_ms else 0}
                # The displayed image is always the exact frame used for its predictions.
                recording_status=recorder.status()
                if recording_error:
                    recording_status['error']=recording_error
                canvas,displayed=render(packet,detections if controls.enabled else [],controls,metrics,metadata['epoch'],args.overlap_iou,recording_status)
                if (fresh is not None and controls.enabled and predicted_sequence == packet['sequence']
                        and recording_status['active'] and recording_status.get('mode') == 'annotated'):
                    # Same fresh frame and predictions as the preview; omit the 136px HUD, never resize.
                    recorder.submit({**packet,'image':canvas[HEADER_HEIGHT:].copy()},mode='annotated')
                if not args.headless:cv2.imshow(WINDOW,canvas)
            key=cv2.waitKey(1)&0xff if not args.headless else 255
            if key == ord('r') and packet is not None:
                try:
                    if recorder.status()['active']:
                        finished=recorder.stop('user')
                        print(json.dumps({'recording_stopped':finished}),flush=True)
                    else:
                        if controls.record_mode == 'annotated':
                            controls.enabled=True
                            controls.paused=False
                        camera_fps=float(capture.info['fps'])
                        container_fps=min(camera_fps,args.display_fps) if controls.record_mode == 'annotated' and args.display_fps>0 else camera_fps
                        context={'weights':str(weights),'weights_sha256':metadata['sha256'],
                                 'epoch':metadata['epoch'],'container_fps':container_fps,
                                 'initial_inference_settings':{**settings,'conf':controls.conf,'iou':controls.iou},
                                 'initial_endpoint_conf':controls.endpoint_conf,
                                 'initial_show_boxes':controls.boxes,'initial_diagnostics':controls.diagnostics,
                                 'follows_display_annotations':controls.record_mode == 'annotated',
                                 'overlay_header_included':False,'display_fps_cap':args.display_fps}
                        recorder.start({**capture.info,'requested_fps':args.fps,'container_fps':container_fps},
                                       mode=controls.record_mode,context=context)
                        controls.notify(f'REC {controls.record_mode.upper()} started. Press R to stop.')
                        print(json.dumps({'recording_started':recorder.status()}),flush=True)
                    recording_error=None
                except Exception as exc:
                    recording_error=f'{type(exc).__name__}: {exc}'
                    recording_errors.append(recording_error)
                    print('Recording error: '+recording_errors[-1],file=sys.stderr,flush=True)
            if controls.key(key,recorder.status()) and packet is not None and canvas is not None:
                save_snapshot(snapshots,packet,canvas,detections if controls.enabled else [],displayed,
                              {'weights':str(weights),'weights_sha256':metadata['sha256'],'epoch':metadata['epoch'],
                               'camera':capture.info,'inference_settings_for_this_frame':used_settings,
                               'endpoint_display_threshold':controls.endpoint_conf,'yolo_enabled':controls.enabled})
                saved+=1
            if time.monotonic()-last_report>=5:
                report.update(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                              camera_received_frames=capture.received,
                              camera_received_fps_last_window=capture.rate.fps(),
                              processed_frames=processed,inference_frames=inferences,
                              skipped_capture_frames=skipped,saved_snapshots=saved,
                              first_camera_forward_input=first_input,
                              inference_fps_last_window=infer_rate.fps(),
                              display_update_fps_last_window=display_rate.fps(),
                              inference_ms_median=float(np.median(infer_ms)) if infer_ms else None,
                              last_frame_number=packet['frame_number'] if packet else None,
                              last_displayed_detection_count=len(detections) if controls.enabled else 0,
                              current_conf=controls.conf,current_endpoint_conf=controls.endpoint_conf,
                              current_iou=controls.iou,recording_active=recorder.status()['active'],
                              recording_mode=controls.record_mode,
                              preview_paused=controls.paused,yolo_enabled=controls.enabled)
                write_report(report_path,report)
                last_report=time.monotonic()
            if not args.headless and cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
            if args.max_frames and processed>=args.max_frames:break
            elapsed=time.monotonic()-tick
            time.sleep(max(0,(1/args.display_fps if args.display_fps else .001)-elapsed))
        hook.remove()
        report['status']='completed'
    except KeyboardInterrupt:
        report['status']='interrupted'
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        print(report['error'],file=sys.stderr,flush=True)
    finally:
        final_inference_fps=infer_rate.fps()
        final_display_fps=display_rate.fps()
        if capture:
            report['camera_received_frames']=capture.received
            report['camera_received_fps_last_window']=capture.rate.fps()
        # End the recording at the exit request, before USB pipeline shutdown
        # delays; the camera can continue publishing, but submit now ignores it.
        try:
            recorder.stop('camera_or_application_error' if report['status']=='failed' else 'viewer_exit')
        except Exception as exc:
            recording_errors.append(f'{type(exc).__name__}: {exc}')
            report['status']='failed'
        if capture:
            try:capture.stop()
            except Exception as exc:
                report['camera_stop_error']=str(exc)
                report['status']='failed'
        report['recordings']=recorder.recordings
        for item in recorder.recordings:
            if item.get('status')=='failed':
                recording_errors.append(f"{item['video_path']}: {item.get('error')}")
        if recording_errors:
            report['status']='failed'
        report['recording_errors']=recording_errors
        report['saved_video']=any(item.get('written_frames',0)>0 for item in recorder.recordings)
        if window:
            try:cv2.destroyAllWindows()
            except Exception as exc:
                report['window_close_error']=str(exc)
                report['status']='failed'
        report.update(processed_frames=processed,inference_frames=inferences,skipped_capture_frames=skipped,
                      saved_snapshots=saved,first_camera_forward_input=first_input,
                      inference_fps_last_window=final_inference_fps,display_update_fps_last_window=final_display_fps,
                      inference_ms_median=float(np.median(infer_ms)) if infer_ms else None,
                      final_conf=controls.conf,final_endpoint_conf=controls.endpoint_conf,final_iou=controls.iou,
                      recording_mode=controls.record_mode,recording_active=False,
                      last_frame_number=packet['frame_number'] if packet else None,
                      last_displayed_detection_count=len(detections) if controls.enabled else 0,
                      finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        report['fps_note']='Camera RX counts received frames. Infer counts completed predictions. Display/update counts fresh processed frames (headless: would-be updates); paused redraws excluded. No guaranteed 30 FPS.'
        write_report(report_path,report)
        print(json.dumps({k:report.get(k) for k in ('status','processed_frames','inference_frames','camera_received_fps_last_window','inference_fps_last_window','display_update_fps_last_window','inference_ms_median','error')}),flush=True)
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())
