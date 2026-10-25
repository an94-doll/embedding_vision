#!/usr/bin/env python3
"""Freeze user-reviewed two-point data; train 100 epochs and save by validation loss."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from train_front_two_points import check_label

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'data/20261025T081334_943078Z/picture/work'
TEST_FRAMES = (31, 83, 134, 190, 259)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def setup_env(job):
    os.environ['YOLO_CONFIG_DIR'] = str(job/'cache/ultralytics')
    os.environ['MPLCONFIGDIR'] = str(job/'cache/matplotlib')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'


def prepare():
    import yaml
    from PIL import Image
    name = 'cassette_reviewed277_100_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    job = ROOT/'training'/name
    job.mkdir(parents=True, exist_ok=False)
    setup_env(job)
    from ultralytics.data.utils import verify_image_label
    dataset = job/'dataset'
    for split in ('train','val','test','gap'):
        for kind in ('images','labels'):
            (dataset/kind/split).mkdir(parents=True)
    manifest = json.loads((SOURCE/'batch/manifest.json').read_text())
    originals = {r['id']: r for r in manifest['images']}
    assert len(originals) == 277
    guards = ['batch_train.txt', 'data.yaml', 'annotation_standard.json', 'batch/frame_review_status.csv']
    preservation = {rel: digest(SOURCE/rel) for rel in guards}
    rows, sums, annotation_records = [], [], []
    sys.path.insert(0, str(SOURCE/'batch'))
    from review_server import clean_objects, labels
    for n in range(1,278):
        stem = f'color_{n:06d}'
        record_path = SOURCE/'batch/annotations'/f'{stem}.json'
        record_bytes = record_path.read_bytes()
        rec = json.loads(record_bytes)
        assert rec['user_reviewed'] and rec['training_eligible'] and rec['review_status']=='accepted', stem
        clean_objects(rec['objects'],1280,720)
        image_path, label_path = SOURCE/'images/train'/f'{stem}.png', SOURCE/'labels/train'/f'{stem}.txt'
        im_bytes, lb_bytes = image_path.read_bytes(), label_path.read_bytes()
        assert labels(rec['objects'],1280,720)==lb_bytes, f'JSON/YOLO mismatch: {stem}'
        assert hashlib.sha256(im_bytes).hexdigest()==originals[stem]['sha256']==digest(SOURCE.parent/'raw'/image_path.name)
        with Image.open(io.BytesIO(im_bytes)) as im:
            im.load()
            assert im.size==(1280,720)
        counts=check_label(lb_bytes.decode())
        assert counts['instances'] or rec.get('confirmed_empty'), stem
        near_test = any(abs(n-t)<=2 for t in TEST_FRAMES)
        split = 'test' if n in TEST_FRAMES else 'gap' if near_test or 213<=n<=222 else 'train' if n<=212 else 'val'
        for kind,suffix,content in [('images','.png',im_bytes),('labels','.txt',lb_bytes)]:
            dest=dataset/kind/split/(stem+suffix)
            dest.write_bytes(content)
            assert dest.read_bytes()==content
            sums.append(f'{digest(dest)}  {dest.relative_to(dataset)}')
        verified=verify_image_label((str(dataset/'images'/split/f'{stem}.png'),str(dataset/'labels'/split/f'{stem}.txt'),'',True,1,2,3,False))
        assert verified[0] is not None and not verified[-1] and verified[-2]==0, (stem,verified[-1])
        assert len(verified[1])==counts['instances']
        preservation[f'batch/annotations/{stem}.json']=hashlib.sha256(record_bytes).hexdigest()
        preservation[f'labels/train/{stem}.txt']=hashlib.sha256(lb_bytes).hexdigest()
        annotation_records.append(rec)
        rows.append({'frame':n,'id':stem,'split':split,'source_review_status':'accepted','image_sha256':digest(image_path),
                     'label_sha256':digest(label_path),'annotation_sha256':preservation[f'batch/annotations/{stem}.json'],**counts})
    for rel,sha in preservation.items():
        assert digest(SOURCE/rel)==sha, f'User saved during snapshot; retry: {rel}'
    groups={s:[r for r in rows if r['split']==s] for s in ('train','val','test','gap')}
    assert [len(groups[s]) for s in groups]==[192,50,5,30]
    for test in TEST_FRAMES:
        assert all(abs(r['frame']-test)>2 for r in groups['train']+groups['val'])
    assert set(r['frame'] for r in groups['train']).isdisjoint(r['frame'] for r in groups['val'])
    cfg={'path':str(dataset),'train':'images/train','val':'images/val','test':'images/test','names':{0:'cassette_front'},
         'kpt_shape':[2,3],'flip_idx':[1,0],'kpt_names':{0:['long_edge_left','long_edge_right']}}
    (dataset/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    sums.append(f'{digest(dataset/"data.yaml")}  data.yaml')
    (dataset/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    with (dataset/'manifest.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(job/'source_preservation_hashes.json',preservation)
    write_json(job/'user_annotation_snapshot.json',annotation_records)
    audit={'created_at':now(),'source':str(SOURCE),'source_images':277,'format_and_image_checks':'passed',
           'user_approved_images':277,'test_frames':TEST_FRAMES,'test_use':'Only inference after all 100 epochs; no fitting or checkpoint selection.',
           'split':{s:{'images':len(rs),**{k:sum(r[k] for r in rs) for k in ('instances','visible_endpoints','missing_endpoints')}} for s,rs in groups.items()},
           'temporal_gap':'Frames 213–222 separate train/validation. Two neighboring frames on each side of each test image are also excluded.',
           'limitation':'All splits are from one video; held-out frames do not establish generalization to other captures.'}
    write_json(job/'dataset_audit.json',audit)
    weights=ROOT/'weights/yolo26s-pose.pt'
    config={'created_at':now(),'run_name':name,'source_dataset':str(SOURCE),'snapshot_dataset':str(dataset),
            'weights':str(weights),'weights_sha256':digest(weights),'dataset_sha256_manifest':digest(dataset/'SHA256SUMS.txt'),
            'initialization':'Official pretrained YOLO26s-Pose; new one-class, two-keypoint head. No old candidate-label checkpoint.',
            'selection':'Minimum sum of all val/*_loss columns per epoch; fixed 50-frame validation set. Test frames excluded.',
            'test_frames':TEST_FRAMES,'inference':{'imgsz':1280,'rect':True,'conf':0.25,'iou':0.7,'max_det':300,'endpoint_conf':0.5,'half':False,'quantize':32},
            'train':{'data':str(dataset/'data.yaml'),'epochs':100,'patience':0,'imgsz':1280,'batch':4,
                     'device':'0','workers':4,'project':str(ROOT/'runs/pose'),'name':name,'exist_ok':False,
                     'optimizer':'AdamW','lr0':0.001,'lrf':0.01,'amp':False,'seed':42,'deterministic':True,
                     'rect':True,'cache':False,'resume':False,'mosaic':0.0,'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,
                     'close_mosaic':0,'degrees':0.0,'translate':0.0,'scale':0.0,'shear':0.0,'perspective':0.0,
                     'flipud':0.0,'fliplr':0.0,'hsv_h':0.01,'hsv_s':0.2,'hsv_v':0.2,
                     'save':True,'save_period':-1,'plots':False,'val':True}}
    write_json(job/'train_config.json',config)
    shutil.copy2(__file__,job/'entrypoint_snapshot.py')
    (ROOT/'reports/reviewed277_latest.txt').write_text(str(job)+'\n')
    print(json.dumps({'job':str(job),'audit':audit},ensure_ascii=False,indent=2))


def train(config_path):
    config_path=Path(config_path).resolve();job=config_path.parent
    cfg=json.loads(config_path.read_text());setup_env(job)
    started=time.perf_counter()
    state={'status':'starting','started_at':now(),'pid':os.getpid(),'epochs_completed':0,'epochs_requested':100,
           'best_loss_epoch':None,'best_val_total_loss':None,'selection':cfg['selection'],'test_in_training_or_selection':False}
    def persist():
        state.update(updated_at=now(),elapsed_seconds=round(time.perf_counter()-started,3));write_json(job/'status.json',state)
    persist()
    try:
        import torch
        from ultralytics import YOLO, settings
        settings.update({'sync':False})
        torch.set_num_threads(4)
        assert torch.cuda.is_available(), 'CUDA required'
        assert digest(cfg['weights'])==cfg['weights_sha256']
        ds=Path(cfg['snapshot_dataset']);sums=ds/'SHA256SUMS.txt'
        assert digest(sums)==cfg['dataset_sha256_manifest']
        for line in sums.read_text().splitlines():
            sha,rel=line.split('  ',1);assert digest(ds/rel)==sha,rel
        history=[]
        def start(t):
            assert t.epochs==100 and t.args.patience==0 and list(t.model.model[-1].kpt_shape)==[2,3]
            assert t.device.type=='cuda' and t.data['names']=={0:'cassette_front'}
            assert len(t.train_loader.dataset)==192 and len(t.test_loader.dataset)==50
            train_files=set(t.train_loader.dataset.im_files);val_files=set(t.test_loader.dataset.im_files)
            assert not train_files & val_files
            assert not any(Path(p).stem in {f'color_{n:06d}' for n in TEST_FRAMES} for p in train_files|val_files)
            state.update(status='running',save_dir=str(t.save_dir),gpu=torch.cuda.get_device_name(t.device),train_images=192,val_images=50,test_images=5)
            write_json(job/'actual_train_args.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(t.args).items()});persist()
        def epoch_start(t):
            state['current_epoch']=t.epoch+1;persist()
        def saved(t):
            epoch=t.epoch+1
            assert epoch==len(history)+1 and 1<=epoch<=100
            train_loss={k:float(v) for k,v in t.label_loss_items(t.tloss).items()}
            val_loss={k:float(v) for k,v in t.metrics.items() if k.startswith('val/') and k.endswith('_loss')}
            assert val_loss and all(math.isfinite(v) for v in list(train_loss.values())+list(val_loss.values()))
            total=sum(val_loss.values())
            row={'epoch':epoch,'train_total_loss':sum(train_loss.values()),'val_total_loss':total,**train_loss,**val_loss}
            history.append(row)
            def copy_checkpoint(name):
                dst=t.wdir/name;tmp=dst.with_suffix('.pt.tmp');shutil.copyfile(t.last,tmp);tmp.replace(dst);return str(dst)
            if state['best_val_total_loss'] is None or total<state['best_val_total_loss']:
                path=copy_checkpoint('best_loss.pt')
                state.update(best_loss_epoch=epoch,best_val_total_loss=total,best_loss_checkpoint=path)
                write_json(t.wdir/'best_loss.json',{'epoch':epoch,'val_total_loss':total,'loss_components':val_loss,'selection':cfg['selection']})
            if epoch==100:
                state['epoch100_checkpoint']=copy_checkpoint('epoch100.pt')
                write_json(t.wdir/'epoch100.json',{'epoch':100,'val_total_loss':total,'loss_components':val_loss})
            with (job/'loss_history.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            state.update(epochs_completed=epoch,latest_losses=row,losses_finite=True)
            persist()
        model=YOLO(cfg['weights'])
        model.add_callback('on_train_start',start)
        model.add_callback('on_train_epoch_start',epoch_start)
        model.add_callback('on_model_save',saved)
        model.train(**cfg['train'])
        assert state['epochs_completed']==100 and len(history)==100
        assert min(history,key=lambda r:r['val_total_loss'])['epoch']==state['best_loss_epoch']
        for key,epoch in [('best_loss_checkpoint',state['best_loss_epoch']),('epoch100_checkpoint',100)]:
            ckpt=torch.load(state[key],map_location='cpu',weights_only=False)
            assert ckpt['epoch']+1==epoch and ckpt['optimizer'] is not None
            assert list(ckpt['ema'].model[-1].kpt_shape)==[2,3]
            recorded=sum(float(v) for k,v in ckpt['train_metrics'].items() if k.startswith('val/') and k.endswith('_loss'))
            assert abs(recorded-history[epoch-1]['val_total_loss'])<1e-9
            state[key+'_sha256']=digest(state[key]);del ckpt
        results=list(csv.DictReader((Path(state['save_dir'])/'results.csv').open()))
        assert [int(r['epoch']) for r in results]==list(range(1,101))
        for r,h in zip(results,history):
            total=sum(float(v) for k,v in r.items() if k.startswith('val/') and k.endswith('_loss'))
            assert abs(total-h['val_total_loss'])<1e-4
        state.update(status='completed',finished_at=now(),checkpoint_integrity='passed',csv_100_epochs_check='passed')
        persist()
        print('TRAINING COMPLETE',json.dumps(state,ensure_ascii=False),flush=True)
        return 0
    except BaseException as exc:
        state.update(status='failed',error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});persist()
        traceback.print_exc();return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(train(args.config))
