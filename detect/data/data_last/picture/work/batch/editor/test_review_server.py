"""Data-integrity tests use isolated temporary datasets, never the real annotations."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import threading
import urllib.request
import urllib.error

import cv2
import numpy as np
import pytest

MODULE=Path(__file__).resolve().parents[1]/'review_server.py'
spec=importlib.util.spec_from_file_location('review_server',MODULE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


@pytest.fixture
def store(tmp_path):
    root=tmp_path/'work';items=[]
    for i in (1,2):
        stem=f'color_{i:06d}'
        items.append({'id':stem,'filename':stem+'.png','width':160,'height':100})
        for folder in ['batch/annotations','images/train','labels/train','previews','images/val']:(root/folder).mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(root/'images/train'/f'{stem}.png'),np.full((100,160,3),140,np.uint8))
        obj={'id':1,'bbox_xyxy':[20,20,100,60],'keypoints_xyv':[[25,50,2],[95,49,2]]}
        rec={'frame':i,'objects':[obj],'review_status':'accepted' if i==2 else 'requires_correction','training_eligible':i==2,
             'unresolved_issues':['old issue'] if i==1 else [],'screening_complete':True}
        (root/'batch/annotations'/f'{stem}.json').write_bytes(mod.json_bytes(rec))
        (root/'labels/train'/f'{stem}.txt').write_bytes(mod.labels([obj],160,100))
        (root/'previews'/f'{stem}_annotated.png').write_bytes(b'old-preview')
    (root/'batch/manifest.json').write_bytes(mod.json_bytes({'images':items,'count':2}))
    (root/'labels/train/sample.txt').write_text('SAMPLE PRESERVED')
    (root/'batch_train.txt').write_text(str(root/'images/train/color_000002.png')+'\n')
    return mod.Store(root)


def payload(store,decision='accepted'):
    rec=store.get('color_000001')
    return {'objects':rec['objects'],'version':rec['version'],'decision':decision,'note':'已检查'}


def test_accept_save_and_reject_sync(store):
    p=payload(store);p['objects'].append({'bbox_xyxy':[110,20,150,60],'keypoints_xyv':[[145,50,2],[115,49,2]]})
    result=store.save('color_000001',p)
    assert result['overview']['stats']['accepted']==2
    assert result['frame']['objects'][1]['keypoints_xyv'][0][0]==115
    assert len((store.root/'batch_train.txt').read_text().splitlines())==2
    rows=np.loadtxt(store.root/'labels/train/color_000001.txt',ndmin=2)
    assert rows.shape==(2,11) and np.all((rows[:,1:7]>=0)&(rows[:,1:7]<=1))
    assert cv2.imread(str(store.root/'previews/color_000001_annotated.png')).shape==(130,160,3)
    assert (store.root/result['history']/'batch/annotations/color_000001.json').is_file()
    assert (store.editor/'overrides/color_000001.json').is_file()
    assert (store.root/'labels/train/sample.txt').read_text()=='SAMPLE PRESERVED'
    p=payload(store,'rejected');store.save('color_000001',p)
    assert 'color_000001' not in (store.root/'batch_train.txt').read_text()
    assert json.loads((store.batch/'frame_review_status.json').read_text())[0]['review_status']=='rejected'
    assert json.loads((store.batch/'manifest.json').read_text())['images'][0]['training_eligible'] is False
    assert json.loads((store.batch/'export_summary.json').read_text())['unresolved_images']==0
    assert json.loads((store.batch/'run_state.json').read_text())['status']=='user_review_complete'


def test_missing_endpoint_and_draft_not_approved(store):
    p=payload(store,'pending');p['objects'][0]['keypoints_xyv'][0]=[88,90,0]
    r=store.save('color_000001',p)
    assert r['frame']['objects'][0]['keypoints_xyv'][0]==[0,0,0]
    assert r['frame']['decision']=='pending'
    row=np.loadtxt(store.root/'labels/train/color_000001.txt')
    assert list(row[5:8])==[0,0,0]


@pytest.mark.parametrize('case',['out_of_box','invalid_box','nan','duplicate','empty'])
def test_invalid_save_does_not_mutate(store,case):
    p=payload(store);original=store.record_path('color_000001').read_bytes()
    if case=='out_of_box':p['objects'][0]['keypoints_xyv'][0]=[1,1,2]
    if case=='invalid_box':p['objects'][0]['bbox_xyxy']=[-1,20,100,60]
    if case=='nan':p['objects'][0]['bbox_xyxy'][0]=float('nan')
    if case=='duplicate':p['objects']*=2
    if case=='empty':p['objects']=[]
    with pytest.raises(mod.Invalid):store.save('color_000001',p)
    assert store.record_path('color_000001').read_bytes()==original


def test_confirmed_negative_requires_explicit_choice(store):
    p=payload(store);p.update(objects=[],confirmed_empty=True)
    result=store.save('color_000001',p)
    assert result['frame']['confirmed_empty']
    assert (store.root/'labels/train/color_000001.txt').read_text()==''
    assert result['overview']['stats']['accepted']==2


def test_optimistic_concurrency(store):
    p=payload(store);store.save('color_000001',p)
    with pytest.raises(mod.Conflict):store.save('color_000001',p)


def test_recovery_after_interrupted_save(store):
    old=store.record_path('color_000001').read_bytes()
    result=store.save('color_000001',payload(store))
    journal_path=store.root/result['history']/'change.json'
    journal=json.loads(journal_path.read_text());journal['state']='prepared'
    journal_path.write_bytes(mod.json_bytes(journal))
    assert store.recover_transactions()==['color_000001']
    assert store.record_path('color_000001').read_bytes()==old
    assert not (store.editor/'overrides/color_000001.json').exists()
    assert 'color_000001' not in (store.root/'batch_train.txt').read_text()
    assert store.recover_transactions()==[]


def test_derived_failure_rolls_back(store,monkeypatch):
    old=store.record_path('color_000001').read_bytes();real=mod.atomic;failures=[]
    def fail_once(path,content):
        if path.name=='frame_review_status.csv' and not failures:
            failures.append(True);raise OSError('simulated disk error')
        return real(path,content)
    monkeypatch.setattr(mod,'atomic',fail_once)
    with pytest.raises(OSError):store.save('color_000001',payload(store))
    assert store.record_path('color_000001').read_bytes()==old
    assert 'color_000001' not in (store.root/'batch_train.txt').read_text()


def test_http_token_and_path_restriction(store):
    asset=store.root/'page.html';asset.write_text('<h1>test</h1>')
    server=mod.ThreadingHTTPServer(('127.0.0.1',0),mod.handler_for(store,'known-token',asset))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    local_open=urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with local_open(base+'/api/session') as r:assert json.load(r)['stats']['total']==2
        request=urllib.request.Request(base+'/api/frame/color_000001',data=mod.json_bytes(payload(store)),headers={'Content-Type':'application/json'},method='POST')
        with pytest.raises(urllib.error.HTTPError) as e:local_open(request)
        assert e.value.code==403
        request.add_header('X-Editor-Token','known-token')
        with local_open(request) as r:assert json.load(r)['frame']['decision']=='accepted'
        with pytest.raises(urllib.error.HTTPError) as e:local_open(base+'/images/../../annotations.json')
        assert e.value.code==404
    finally:server.shutdown();server.server_close();thread.join()


def test_supplement_review_metadata_and_original_point_edit(store):
    path=store.record_path('color_000001');rec=json.loads(path.read_text())
    rec['supplementation']={'run':'20261024_233859','added_objects':1,'filled_endpoints':0,'needs_user_review':True}
    rec['objects'][0]['supplement']={'run':'20261024_233859','kind':'added_object'}
    path.write_bytes(mod.json_bytes(rec))
    assert store.get('color_000001')['supplementation']['added_objects']==1
    row=store.overview()['frames'][0]
    assert row['supplemented'] and row['added_objects']==1 and row['missing_endpoints']==0
    p=payload(store,'pending');p['objects'][0]['keypoints_xyv'][0][0]=27
    r=store.save('color_000001',p)
    assert r['frame']['decision']=='pending' and r['frame']['objects'][0]['keypoints_xyv'][0][0]==27
    assert r['frame']['objects'][0]['supplement']['kind']=='added_object'
    assert r['frame']['supplementation']['needs_user_review']
    r=store.save('color_000001',payload(store,'accepted'))
    assert r['frame']['supplementation']['needs_user_review'] is False
