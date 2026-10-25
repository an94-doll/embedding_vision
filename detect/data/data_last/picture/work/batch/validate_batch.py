#!/usr/bin/env python3
"""Validate files and YOLO Pose loading. This is NOT an annotation-quality test."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'batch'
RAW=ROOT.parent/'raw'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def strict_check(item):
    stem=item['id'];name=item['filename'];errors=[]
    ip=ROOT/'images/train'/name;lp=ROOT/'labels/train'/f'{stem}.txt'
    preview=ROOT/'previews'/f'{stem}_annotated.png'
    try:
        assert sha(RAW/name)==sha(ip)==item['sha256'],'source/copy/manifest SHA256 mismatch'
        with Image.open(ip) as im:
            im.load();assert im.size==(1280,720),'image dimensions'
        with Image.open(preview) as im:
            im.load();assert im.size==(1280,750),'preview dimensions, including status footer'
        rec=read(BATCH/'annotations'/f'{stem}.json')
        content=lp.read_text().strip()
        a=np.loadtxt(lp,ndmin=2) if content else np.empty((0,11))
        assert a.ndim==2 and a.shape[1]==11,'11 columns expected'
        if not len(a):
            assert not rec.get('training_eligible') or rec.get('confirmed_empty'), 'Empty accepted image must be explicitly confirmed as negative'
        assert np.isfinite(a).all(),'non-finite value'
        assert (a[:,0]==0).all(),'unexpected class'
        assert ((a[:,1:5]>=0)&(a[:,1:5]<=1)).all(),'box normalization'
        assert (a[:,3:5]>0).all(),'box dimensions must be positive'
        lo=a[:,1:3]-a[:,3:5]/2;hi=a[:,1:3]+a[:,3:5]/2
        assert (lo>=-2e-8).all() and (hi<=1+2e-8).all(),'box extends outside image'
        kp=a[:,5:].reshape(-1,2,3)
        assert np.isin(kp[:,:,2],[0,2]).all(),'visibility must be 0 or 2'
        visible=kp[:,:,2]==2
        assert (kp[~visible]==0).all(),'missing endpoint must be 0 0 0'
        assert ((kp[:,:,:2]>=0)&(kp[:,:,:2]<=1)).all(),'keypoint normalization'
        for side in range(2):
            v=visible[:,side];xy=kp[v,side,:2]
            assert (xy>=lo[v]-2e-8).all() and (xy<=hi[v]+2e-8).all(),'visible endpoint outside box'
        v=visible.all(1)
        assert (kp[v,0,0]<kp[v,1,0]).all(),'left/right ordering'
        assert len(np.unique(a,axis=0))==len(a),'identical duplicate label rows'
        # Identical boxes with different keypoints are suspicious, not silently removed.
        duplicate_boxes=len(a)-len(np.unique(a[:,:5],axis=0))
        assert duplicate_boxes==0,'identical duplicate bounding boxes'
        assert len(rec['objects'])==len(a),'JSON/label count differs'
        assert rec['screening_complete'] is True,'missing screening record'
        assert rec['training_eligible']==(rec['review_status']=='accepted'),'review status inconsistency'
        if not rec['training_eligible']:assert rec['unresolved_issues'],'missing unresolved reason'
        return {'id':stem,'pass':True,'instances':len(a),'visible_endpoints':int(visible.sum()),
                'missing_endpoints':int((~visible).sum()),'training_eligible':rec['training_eligible']}
    except Exception as exc:
        errors.append(str(exc))
        return {'id':stem,'pass':False,'errors':errors}


def main():
    items=read(BATCH/'manifest.json')['images']
    assert len(items)==277
    expected={x['id'] for x in items}
    assert expected=={p.stem for p in RAW.glob('color_*.png')}
    assert expected=={p.stem for p in (ROOT/'images/train').glob('color_*.png')}
    assert expected=={p.stem for p in (ROOT/'labels/train').glob('color_*.txt')}
    assert expected=={p.stem.removesuffix('_annotated') for p in (ROOT/'previews').glob('color_*_annotated.png')}
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(strict_check,items))
    failures=[r for r in rows if not r['pass']]
    summary={'input_images':277,'strict_file_and_geometry_passed':len(rows)-len(failures),
             'strict_failures':failures,'per_image':rows,
             'quality_scope':'Only file integrity, numeric geometry, and loader compatibility. Not semantic completeness, duplicate-object identity, or endpoint accuracy.'}
    save(BATCH/'validation_report.json',summary)
    print('strict passed',len(rows)-len(failures),'failed',len(failures),flush=True)
    if failures:
        print(json.dumps(failures[:20],ensure_ascii=False));sys.exit(1)
    cfg=yaml.safe_load((ROOT/'data.yaml').read_text())
    assert cfg['kpt_shape']==[2,3] and cfg['flip_idx']==[1,0]
    assert cfg['names']=={0:'cassette_front'}
    assert not list((ROOT/'images/val').glob('*')),'validation split should remain empty'
    accepted=[str(ROOT/'images/train'/f'{r["id"]}.png') for r in rows if r['training_eligible']]
    allowlist=(ROOT/'batch_train.txt').read_text().splitlines()
    assert allowlist==accepted and len(set(allowlist))==len(allowlist)
    assert not any('9bd4f8' in p for p in allowlist)
    import ultralytics
    import torch
    from ultralytics.data.utils import verify_image_label
    from ultralytics.data.dataset import YOLODataset
    torch.set_num_threads(2)
    def ultra(item):
        stem=item['id']
        result=verify_image_label((str(ROOT/'images/train'/item['filename']),str(ROOT/'labels/train'/f'{stem}.txt'),'',True,1,2,3,False))
        im,lb,shape,segments,kpts,nm,nf,ne,nc,msg=result
        count=len(read(BATCH/'annotations'/f'{stem}.json')['objects'])
        ok=im is not None and (nm,nf,ne,nc)==(0,1,int(count==0),0) and not msg and lb.shape==(count,5) and kpts.shape==(count,2,3)
        return {'id':stem,'pass':bool(ok),'message':msg,'instances':count}
    with ThreadPoolExecutor(max_workers=6) as pool:ur=list(pool.map(ultra,items))
    assert all(r['pass'] for r in ur),[r for r in ur if not r['pass']]
    summary['ultralytics_version']=ultralytics.__version__
    summary['ultralytics_verify_passed']=len(ur)
    summary['ultralytics_verify_failures']=[]
    print('Ultralytics verify passed',len(ur),flush=True)
    def load_dataset(list_path,expected_files):
        dataset=YOLODataset(img_path=str(list_path),imgsz=640,batch_size=8,augment=False,cache=False,
                            rect=False,stride=32,data=cfg,task='pose',prefix='Pose loading check: ')
        assert len(dataset)==len(expected_files)
        assert set(dataset.im_files)==set(expected_files)
        total=0;nk=0
        loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
        for batch in loader:
            assert batch['keypoints'].ndim==3 and tuple(batch['keypoints'].shape[1:])==(2,3)
            assert torch.isfinite(batch['keypoints']).all()
            assert torch.isfinite(batch['bboxes']).all()
            total+=len(batch['im_file']);nk+=len(batch['keypoints'])
        assert total==len(expected_files)
        return {'images_loaded':total,'instances_loaded':nk,'keypoint_shape':[2,3],'augmentation':False,'training':False}
    all_files=(BATCH/'all_images_for_format_check.txt').read_text().splitlines()
    summary['all_candidate_pose_loading']=load_dataset(BATCH/'all_images_for_format_check.txt',all_files)
    assert summary['all_candidate_pose_loading']['instances_loaded']==sum(r['instances'] for r in rows)
    summary['approved_pose_loading']=load_dataset(ROOT/'batch_train.txt',accepted) if accepted else {'images_loaded':0,'instances_loaded':0,'skipped':'No images approved by the user yet'}
    assert summary['approved_pose_loading']['instances_loaded']==sum(r['instances'] for r in rows if r['training_eligible'])
    summary.update(allowlist_pass=True,source_images_preserved=True,validation_images=0,training_started=False,
                   overall_format_status='passed',overall_annotation_quality='incomplete')
    save(BATCH/'validation_report.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='per_image'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
