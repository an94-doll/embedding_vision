#!/usr/bin/env python3
"""Execute an audited package's original train.py with persistent progress callbacks."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from check_env import configure_local_cache, project_path, write_report


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    job = config_path.parent
    status_path = job/'status.json'
    if status_path.exists():
        raise RuntimeError('This job has already been started; create a new job instead of overwriting its state.')
    state = {'status': 'starting', 'started_at': now(), 'pid': os.getpid(),
             'config': str(config_path), 'epochs_completed': 0,
             'epochs_requested': config['training_defaults']['epochs'],
             'source_dataset': config['source_dataset'], 'snapshot_dataset': config['snapshot_dataset'],
             'initial_weights': config['weights'], 'initial_weights_sha256': config['weights_sha256'],
             'initialization': config['initialization'], 'resume': False}
    started = time.perf_counter()

    def persist():
        state['updated_at'] = now()
        state['elapsed_seconds'] = time.perf_counter()-started
        # Ultralytics per-class summaries contain NumPy scalar image/instance counts.
        serializable = json.loads(json.dumps(state, default=lambda value: value.item()))
        write_report(status_path, serializable)

    persist()
    try:
        configure_local_cache()
        for name in ('dataset_check.json', 'supplemental_audit.json', 'gpu_smoke.json'):
            report = json.loads((job/name).read_text())
            if report['status'] != 'passed':
                raise RuntimeError(f'Preflight has not passed: {name}')
            if name == 'gpu_smoke.json' and (report['device'] != 'cuda:0' or not report['actual_forward_and_backward']):
                raise RuntimeError('A real GPU forward/backward check is required.')
        if config['resume'] is not False:
            raise RuntimeError('This runner only supports independent new training.')
        if hashlib.sha256((job/'dataset_assets.json').read_bytes()).hexdigest() != config['dataset_assets_sha256']:
            raise RuntimeError('Audited asset manifest changed.')
        dataset = Path(config['snapshot_dataset'])
        manifest = json.loads((job/'dataset_assets.json').read_text())
        for item in manifest['files']:
            if hashlib.sha256((dataset/item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError(f"Audited dataset asset changed: {item['path']}")
        weights = Path(config['weights'])
        if hashlib.sha256(weights.read_bytes()).hexdigest() != config['weights_sha256']:
            raise RuntimeError('Initial pretrained weights changed.')
        if config['weights_sha256'] != 'a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646':
            raise RuntimeError('Expected the official pretrained YOLO26s-pose checkpoint.')
        if Path(config['expected_save_dir']).exists():
            raise RuntimeError('Output directory already exists; refusing to reuse a previous run.')
        import torch
        import ultralytics
        from ultralytics.utils.callbacks.base import default_callbacks
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; CPU fallback is disabled.')
        state['versions'] = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__}
        state['gpu'] = torch.cuda.get_device_name(0)
        trainer_holder = {}

        def setup(trainer):
            trainer_holder['trainer'] = trainer
            if trainer.args.resume or getattr(trainer, 'resume', False):
                raise RuntimeError('Unexpected resume was requested.')
            if Path(trainer.save_dir).resolve() != Path(config['expected_save_dir']).resolve():
                raise RuntimeError('Unexpected training output directory.')
            if Path(trainer.args.model).resolve() != weights.resolve():
                raise RuntimeError('Training did not initialize from the specified official checkpoint.')
            actual_args = json.loads(json.dumps(vars(trainer.args), default=str))
            write_report(job/'actual_train_args.json', actual_args)
            state['save_dir'] = str(trainer.save_dir)
            state['actual_trainer'] = type(trainer).__name__
            persist()

        def training_started(trainer):
            head = trainer.model.model[-1]
            if trainer.device.type != 'cuda' or list(head.kpt_shape) != [4, 3] or head.nc != 2:
                raise RuntimeError('Expected a CUDA two-class, four-point model.')
            if type(trainer).__name__ != 'IgnorePoseTrainer' or type(trainer.validator).__name__ != 'IgnorePoseValidator':
                raise RuntimeError('Required partial-label trainer/validator not active.')
            criterion = trainer.model.criterion
            branches = [criterion.one2many, criterion.one2one] if hasattr(criterion, 'one2many') else [criterion]
            if not all(type(branch.bce).__name__ == 'IgnoreNegativeBCE' for branch in branches):
                raise RuntimeError('Class-aware ignore loss is not active in all branches.')
            state.update({'status': 'running', 'device': str(trainer.device), 'actual_kpt_shape': list(head.kpt_shape),
                          'classes': dict(trainer.model.names), 'actual_validator': type(trainer.validator).__name__,
                          'ignore_loss_branches': len(branches), 'train_images': len(trainer.train_loader.dataset),
                          'val_images': len(trainer.test_loader.dataset), 'amp': bool(trainer.amp),
                          'best_checkpoint': str(trainer.best), 'last_checkpoint': str(trainer.last)})
            def first_forward(module, inputs):
                batch = inputs[0]
                image = batch['img'] if isinstance(batch, dict) else batch
                if image.device.type != 'cuda':
                    raise RuntimeError('Actual training input is not on CUDA.')
                state['first_training_input'] = {'shape': list(image.shape), 'device': str(image.device), 'dtype': str(image.dtype)}
                handle.remove()
                persist()
            handle = trainer.model.register_forward_pre_hook(first_forward)
            persist()

        def epoch_started(trainer):
            state['current_epoch'] = trainer.epoch+1
            persist()

        def epoch_finished(trainer):
            # final_eval invokes this callback again with epoch+1; the CSV is authoritative.
            csv_path = trainer.save_dir/'results.csv'
            if csv_path.exists():
                with csv_path.open() as f:
                    rows = list(csv.DictReader(f))
                state['epochs_completed'] = int(float(rows[-1]['epoch'])) if rows else 0
            state['metrics'] = {str(k): float(v) for k, v in trainer.metrics.items()}
            losses = {str(k): float(v) for k, v in trainer.label_loss_items(trainer.tloss).items()}
            state['train_losses'] = losses
            state['losses_finite'] = all(math.isfinite(v) for v in losses.values())
            if not state['losses_finite']:
                raise RuntimeError('Non-finite training loss detected.')
            state['last_checkpoint_exists'] = trainer.last.exists()
            state['best_checkpoint_exists'] = trainer.best.exists()
            state['epoch_seconds'] = trainer.epoch_time
            state['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(trainer.device)
            state['validation_ignored_predictions'] = getattr(trainer.validator, 'ignored_predictions', None)
            persist()

        for event, callback in [('on_pretrain_routine_start', setup), ('on_train_start', training_started),
                                ('on_train_epoch_start', epoch_started), ('on_fit_epoch_end', epoch_finished)]:
            default_callbacks[event].append(callback)
        sys.path.insert(0, str(dataset))
        sys.argv = [config['entry'], *config['entry_arguments']]
        runpy.run_path(config['entry'], run_name='__main__')
        trainer = trainer_holder['trainer']
        state.update({'status': 'completed', 'finished_at': now(),
                      'stopped_early': state['epochs_completed'] < state['epochs_requested'],
                      'per_class_validation': trainer.validator.metrics.summary()})
        persist()
        return 0
    except BaseException as exc:
        state.update({'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc),
                                                   'traceback': traceback.format_exc(limit=14)}})
        persist()
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
