#!/usr/bin/env python3
"""Local cassette Pose annotation editor. No third-party web framework required."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
from urllib.parse import urlsplit
import webbrowser

import cv2
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
FRAME = re.compile(r'color_\d{6}')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+secrets.token_hex(5)+'.tmp')
    temp.write_bytes(content)
    temp.replace(path)


class Invalid(ValueError):
    pass


class Conflict(Exception):
    pass


def clean_objects(objects, width, height):
    if not isinstance(objects, list) or len(objects)>1000:
        raise Invalid('标注对象必须是列表，且不超过 1000 个。')
    cleaned, seen = [], set()
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise Invalid('坐标必须是有限数字。')
        return round(float(value), 4)
    for i, obj in enumerate(objects, 1):
        try:
            box = [number(v) for v in obj['bbox_xyxy']]
            if len(box)!=4:
                raise Invalid(f'目标 {i} 的边界框需要四个坐标。')
            x1,y1,x2,y2 = box
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise Invalid(f'目标 {i} 的边界框无效或超出图像。')
            raw = obj['keypoints_xyv']
            if len(raw)!=2 or any(len(p)!=3 for p in raw):
                raise Invalid(f'目标 {i} 必须有两个端点。')
            points=[]
            for x,y,v in raw:
                if type(v) not in (int,float) or v not in (0,2):
                    raise Invalid(f'目标 {i} 的可见性只能是 0 或 2。')
                if v==0:
                    points.append([0,0,0]);continue
                x,y=number(x),number(y)
                if not (x1<=x<=x2 and y1<=y<=y2):
                    raise Invalid(f'目标 {i} 的端点不在边界框内，请扩大或调整边界框。')
                points.append([x,y,2])
            if all(p[2]==2 for p in points):
                points.sort(key=lambda p:p[0])
                if points[0][0]==points[1][0]:
                    raise Invalid(f'目标 {i} 的左右端点不能重合或位于同一竖线。')
            key=tuple(box)
            if key in seen:
                raise Invalid(f'目标 {i} 与其他目标的边界框完全重复，请检查重复标注。')
            seen.add(key)
            cleaned.append({'id':i,'class_id':0,'group':'browser_editor','bbox_xyxy':box,
                            'keypoints_xyv':points,'source':'user_browser_annotation'})
            meta=obj.get('supplement')
            if isinstance(meta,dict) and isinstance(meta.get('run'),str) and re.fullmatch(r'\d{8}_\d{6}',meta['run']):
                cleaned[-1]['supplement']={'run':meta['run'],'kind':'added_object' if meta.get('kind')=='added_object' else 'filled_endpoint',
                    'filled_endpoints':[side for side in (0,1) if side in meta.get('filled_endpoints',[])],
                    'user_edited':True}

        except (TypeError, KeyError) as exc:
            raise Invalid(f'目标 {i} 的数据结构无效。') from exc
    return cleaned


def labels(objects, width, height):
    lines=[]
    for obj in objects:
        x1,y1,x2,y2=obj['bbox_xyxy']
        a=[0,(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height]
        for x,y,v in obj['keypoints_xyv']:a.extend([x/width,y/height,v])
        lines.append(' '.join(str(int(v)) if j in (0,7,10) else f'{v:.8f}' for j,v in enumerate(a)))
    return ('\n'.join(lines)+('\n' if lines else '')).encode()


def preview(image_path, objects, caption, show_boxes=False, show_numbers=False):
    im=cv2.imread(str(image_path))
    if im is None:raise Invalid('原始图片读取失败，未保存。')
    for o in objects:
        a,b=o['keypoints_xyv'];x1,y1,x2,y2=np.rint(o['bbox_xyxy']).astype(int)
        is_new=o.get('supplement',{}).get('kind')=='added_object'
        if a[2] and b[2]:cv2.line(im,tuple(np.rint(a[:2]).astype(int)),tuple(np.rint(b[:2]).astype(int)),(235,100,225) if is_new else (65,65,255),1,cv2.LINE_AA)
        elif show_boxes:cv2.rectangle(im,(x1,y1),(x2,y2),(40,180,255),1)
        for side,(p,c) in enumerate(zip((a,b),((70,255,80),(255,180,60)))):
            if p[2]:
                cv2.circle(im,tuple(np.rint(p[:2]).astype(int)),2,c,-1,cv2.LINE_AA)
                if side in o.get('supplement',{}).get('filled_endpoints',[]):cv2.circle(im,tuple(np.rint(p[:2]).astype(int)),4,(235,100,225),1,cv2.LINE_AA)
        if show_numbers:cv2.putText(im,('+' if is_new else '')+str(o['id']),(max(0,x1-15),max(12,y1)),0,.35,(0,0,0),2,cv2.LINE_AA)
        if show_numbers:cv2.putText(im,('+' if is_new else '')+str(o['id']),(max(0,x1-15),max(12,y1)),0,.35,(235,150,255) if is_new else (255,255,255),1,cv2.LINE_AA)
    h=im.shape[0];im=cv2.copyMakeBorder(im,0,30,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(im,caption,(8,h+20),0,.45,(255,255,255),1,cv2.LINE_AA)
    ok,data=cv2.imencode('.png',im,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not ok:raise Invalid('预览生成失败，未保存。')
    return data.tobytes()


class Store:
    def __init__(self, root):
        self.root=Path(root).resolve();self.batch=self.root/'batch'
        self.lock=threading.RLock()
        self.manifest=json.loads((self.batch/'manifest.json').read_text())
        self.items={r['id']:r for r in self.manifest['images']}
        if not all(FRAME.fullmatch(s) for s in self.items):raise Invalid('清单文件名无效。')
        self.editor=self.batch/'editor'

    def record_path(self, stem):
        if stem not in self.items:raise Invalid('图片不在本批次清单内。')
        return self.batch/'annotations'/f'{stem}.json'

    def get(self, stem):
        with self.lock:
            raw=self.record_path(stem).read_bytes();rec=json.loads(raw);item=self.items[stem]
            return {'id':stem,'filename':item['filename'],'width':item['width'],'height':item['height'],
                    'objects':rec['objects'],'version':hashlib.sha256(raw).hexdigest(),
                    'decision':'accepted' if rec.get('training_eligible') else 'rejected' if rec.get('review_status')=='rejected' else 'pending',
                    'note':rec.get('user_review_note',''),'confirmed_empty':rec.get('confirmed_empty',False),
                    'issues':rec.get('initial_review_issues',rec.get('resolved_screening_findings',rec.get('unresolved_issues',[]))),
                    'supplementation':rec.get('supplementation',{}),'image_url':f'/images/{stem}.png'}

    def overview(self):
        with self.lock:
            rows=[]
            for stem,item in self.items.items():
                rec=json.loads(self.record_path(stem).read_text())
                status='accepted' if rec.get('training_eligible') else 'rejected' if rec.get('review_status')=='rejected' else 'pending'
                supp=rec.get('supplementation',{})
                rows.append({'id':stem,'filename':item['filename'],'status':status,'instances':len(rec['objects']),
                             'supplemented':bool(supp),'added_objects':supp.get('added_objects',0),
                             'filled_endpoints':supp.get('filled_endpoints',0),
                             'missing_endpoints':sum(not p[2] for o in rec['objects'] for p in o['keypoints_xyv'])})
            return {'frames':rows,'stats':{'total':len(rows),**{s:sum(r['status']==s for r in rows) for s in ('accepted','rejected','pending')}}}

    def save(self, stem, payload):
        with self.lock:
            current=self.get(stem)
            if payload.get('version')!=current['version']:
                raise Conflict('这张图片已被其他页面或程序修改。请先下载当前草稿，再重新加载图片，避免覆盖。')
            decision=payload.get('decision')
            if decision not in ('pending','accepted','rejected'):raise Invalid('请选择待复核、参与训练或不参与训练。')
            objects=clean_objects(payload.get('objects'),current['width'],current['height'])
            note=payload.get('note','')
            if not isinstance(note,str) or len(note)>4000:raise Invalid('备注最多 4000 字。')
            empty=payload.get('confirmed_empty') is True and not objects
            if not objects and decision=='accepted' and not empty:
                raise Invalid('没有标注目标。若这是无目标图片，请明确勾选“已确认画面无包埋盒”；否则请补充标注。')
            rec=json.loads(self.record_path(stem).read_text())
            rec.setdefault('initial_review_issues',rec.get('unresolved_issues',[]))
            old_decision=rec.get('review_status')
            when=datetime.now(timezone.utc).isoformat()
            rec.update(objects=objects,review_status='accepted' if decision=='accepted' else 'rejected' if decision=='rejected' else 'requires_correction',
                       training_eligible=decision=='accepted',user_review_note=note,confirmed_empty=empty,
                       user_reviewed=decision!='pending',screening_complete=True,
                       native_and_enlarged_review_complete=decision=='accepted',
                       unresolved_issues=[] if decision=='accepted' else [note or ('用户选择不参与训练' if decision=='rejected' else '用户保存草稿，尚未确认参与训练')],
                       review_evidence={'reviewer':'user_browser_editor','scope':'user decision','reviewed_at':when},
                       edited_at=when,flags=[] if decision=='accepted' else rec.get('flags',[]))
            if rec.get('supplementation'):
                rec['supplementation'].update(needs_user_review=decision=='pending',user_saved=True)
            png=preview(self.root/'images/train'/current['filename'],objects,f'{stem} | {len(objects)} objects | USER '+decision.upper())
            changes={self.record_path(stem):json_bytes(rec),self.root/'labels/train'/f'{stem}.txt':labels(objects,current['width'],current['height']),
                     self.root/'previews'/f'{stem}_annotated.png':png,self.editor/'overrides'/f'{stem}.json':json_bytes(rec)}
            history=self.editor/'history'/stem/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+secrets.token_hex(3))
            history.mkdir(parents=True)
            old={}
            for path in changes:
                old[path]=path.read_bytes() if path.exists() else None
                if old[path] is not None:
                    backup=history/path.relative_to(self.root);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(old[path])
            journal={'frame':stem,'time':when,'old_status':old_decision,'new_status':decision,'instances':len(objects),'note':note,
                     'state':'prepared','missing_before':[str(p.relative_to(self.root)) for p,v in old.items() if v is None]}
            atomic(history/'change.json',json_bytes(journal))
            try:
                for path,content in changes.items():atomic(path,content)
                self.rebuild()
                atomic(history/'change.json',json_bytes({**journal,'state':'complete'}))
            except Exception:
                for path,content in old.items():
                    if content is None:path.unlink(missing_ok=True)
                    else:atomic(path,content)
                self.rebuild()
                atomic(history/'change.json',json_bytes({**journal,'state':'rolled_back'}))
                raise
            # These are regenerable Ultralytics caches; label edits invalidate them.
            (self.root/'labels/train.cache').unlink(missing_ok=True)
            atomic(self.editor/'last_change.json',json_bytes({'time':when,'frame':stem,'initial_validation_is_historical':True,
                    'geometry_validated_on_save':True,'training_started':False}))
            return {'frame':self.get(stem),'overview':self.overview(),'history':str(history.relative_to(self.root))}

    def recover_transactions(self):
        """Roll back interrupted saves before accepting new edits after a restart."""
        recovered=[]
        for path in sorted((self.editor/'history').glob('color_*/*/change.json')):
            journal=json.loads(path.read_text())
            if journal.get('state')!='prepared':continue
            stem=journal['frame'];self.record_path(stem)
            for rel in (f'batch/annotations/{stem}.json',f'labels/train/{stem}.txt',
                        f'previews/{stem}_annotated.png',f'batch/editor/overrides/{stem}.json'):
                backup=path.parent/rel;target=self.root/rel
                if rel in journal.get('missing_before',[]):target.unlink(missing_ok=True)
                elif backup.exists():atomic(target,backup.read_bytes())
                else:raise Invalid(f'恢复备份缺失：{backup}')
            self.rebuild()
            atomic(path,json_bytes({**journal,'state':'recovered_after_interruption'}))
            recovered.append(stem)
        return recovered

    def rebuild(self):
        """All aggregate files derive from frame JSON; safe to retry after interruption."""
        rows=[]
        for stem,item in self.items.items():
            rec=json.loads(self.record_path(stem).read_text());objects=rec['objects']
            visible=sum(p[2]==2 for o in objects for p in o['keypoints_xyv'])
            rows.append({'frame':int(stem.split('_')[1]),'filename':item['filename'],'instances':len(objects),
                         'visible_endpoints':visible,'missing_endpoints':2*len(objects)-visible,
                         'screening_complete':rec.get('screening_complete',False),
                         'native_and_enlarged_review_complete':rec.get('native_and_enlarged_review_complete',False),
                         'review_status':rec.get('review_status','requires_correction'),'training_eligible':rec.get('training_eligible',False),
                         'unresolved_issues':' | '.join(rec.get('unresolved_issues',[])),
                         'screening_evidence':rec.get('review_evidence',{}).get('screening_sheet','user_browser_editor'),
                         'label':f'labels/train/{stem}.txt','preview':f'previews/{stem}_annotated.png',
                         'supplement_run':rec.get('supplementation',{}).get('run',''),
                         'supplement_added_objects':rec.get('supplementation',{}).get('added_objects',0),
                         'supplement_filled_endpoints':rec.get('supplementation',{}).get('filled_endpoints',0)})
        approved=[r for r in rows if r['training_eligible']];excluded=[r for r in rows if not r['training_eligible']]
        pending=[r for r in excluded if r['review_status']!='rejected']
        csv_out=io.StringIO(newline='');writer=csv.DictWriter(csv_out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest=copy.deepcopy(self.manifest)
        for item,row in zip(manifest['images'],rows):
            item.update(review_status=row['review_status'],training_eligible=row['training_eligible'])
        manifest['batch_quality_status']='incomplete' if pending else 'complete_with_exclusions' if excluded else 'complete'
        summary={'input_images':len(rows),'exported_labels':len(rows),'previews':len(rows),
                 'instances_all_exports':sum(r['instances'] for r in rows),'visible_endpoints_all_exports':sum(r['visible_endpoints'] for r in rows),
                 'missing_endpoints_all_exports':sum(r['missing_endpoints'] for r in rows),'screened_images':sum(r['screening_complete'] for r in rows),
                 'quality_review_completed_and_accepted':len(approved),'unresolved_images':len(pending),'excluded_images':len(excluded),
                 'rejected_images':sum(r['review_status']=='rejected' for r in rows),
                 'accepted_instances':sum(r['instances'] for r in approved),'accepted_visible_endpoints':sum(r['visible_endpoints'] for r in approved),
                 'accepted_missing_endpoints':sum(r['missing_endpoints'] for r in approved),
                 'quality_acceptance':'INCOMPLETE' if pending else 'REVIEW_COMPLETE_WITH_EXCLUSIONS' if excluded else 'COMPLETE',
                 'training_started':False,'note':'User editor decisions are authoritative. Initial validation report is historical after edits.'}
        status=f'''# 浏览器标注当前状态

更新时间：{datetime.now(timezone.utc).isoformat()}

- 总图片：{len(rows)}；参与训练：{len(approved)}；不参与训练：{summary['rejected_images']}；待复核：{len(excluded)-summary['rejected_images']}。
- 已纳入训练的实例：{summary['accepted_instances']}；可见端点：{summary['accepted_visible_endpoints']}；缺失端点：{summary['accepted_missing_endpoints']}。
- 训练清单：`../../batch_train.txt`。只有明确通过的图片才会进入。
- 本页、逐帧 CSV/JSON 和训练清单随用户保存更新。原 CHECK_REPORT.md 和 validation_report.json 的验证结论是编辑前历史记录；修改后的标签在保存时进行几何与格式检查，完整 Ultralytics 检查需重新执行。
- 没有启动训练；验证集没有改变。用户修改历史位于 `history/`，权威覆盖记录位于 `overrides/`。
'''
        changes={self.batch/'frame_review_status.json':json_bytes(rows),self.batch/'frame_review_status.csv':csv_out.getvalue().encode('utf-8-sig'),
                 self.root/'batch_train.txt':(''.join(str(self.root/'images/train'/r['filename'])+'\n' for r in approved)).encode(),
                 self.batch/'excluded_images.txt':(''.join(r['filename']+'\n' for r in excluded)).encode(),
                 self.batch/'manifest.json':json_bytes(manifest),self.batch/'export_summary.json':json_bytes(summary),
                 self.batch/'run_state.json':json_bytes({'status':'user_review_in_progress' if pending else 'user_review_complete','source_images':len(rows),
                     'precision_review_accepted':len(approved),'unresolved':len(pending),'excluded':len(excluded),'training_started':False,'initial_format_check_is_historical':True}),
                 self.editor/'CURRENT_STATUS.md':status.encode()}
        for path,content in changes.items():atomic(path,content)


def handler_for(store, token, asset):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if '/images/' not in self.path:super().log_message(fmt,*args)

        def send(self, status, content, content_type='application/json; charset=utf-8'):
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(content)

        def safe_host(self):
            # Local service only; do not serve data to unrelated origins via DNS rebinding.
            return self.headers.get('Host','').split(':')[0] in ('localhost','127.0.0.1')

        def do_GET(self):
            if not self.safe_host():return self.send(403,json_bytes({'error':'仅支持本机访问或 SSH 本地端口转发。'}))
            path=urlsplit(self.path).path
            try:
                if path in ('/','/annotation_editor.html'):
                    return self.send(200,asset.read_bytes(),'text/html; charset=utf-8')
                if path=='/api/session':return self.send(200,json_bytes({'token':token,**store.overview()}))
                if path=='/api/frames':return self.send(200,json_bytes(store.overview()))
                m=re.fullmatch(r'/api/frame/(color_\d{6})',path)
                if m:return self.send(200,json_bytes(store.get(m[1])))
                m=re.fullmatch(r'/images/(color_\d{6})\.png',path)
                if m:
                    store.record_path(m[1]);return self.send(200,(store.root/'images/train'/f'{m[1]}.png').read_bytes(),'image/png')
                if path=='/api/train-list':return self.send(200,(store.root/'batch_train.txt').read_bytes(),'text/plain; charset=utf-8')
                return self.send(404,json_bytes({'error':'未找到页面。'}))
            except (Invalid,FileNotFoundError) as exc:return self.send(404,json_bytes({'error':str(exc)}))

        def do_POST(self):
            if not self.safe_host() or self.headers.get('X-Editor-Token')!=token:
                return self.send(403,json_bytes({'error':'页面凭据已失效，请刷新。'}))
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://{self.headers.get("Host")}':return self.send(403,json_bytes({'error':'不允许跨站保存。'}))
            m=re.fullmatch(r'/api/frame/(color_\d{6})',urlsplit(self.path).path)
            if not m:return self.send(404,json_bytes({'error':'未知操作。'}))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise Invalid('请求大小无效。')
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise Invalid('需要 JSON 数据。')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise Invalid('请求必须是对象。')
                return self.send(200,json_bytes(store.save(m[1],payload)))
            except Conflict as exc:return self.send(409,json_bytes({'error':str(exc)}))
            except (Invalid,ValueError,TypeError) as exc:return self.send(400,json_bytes({'error':str(exc)}))
            except Exception as exc:
                self.log_error('Save failed: %r',exc)
                return self.send(500,json_bytes({'error':'保存失败，已尝试恢复原记录。请保留页面草稿并检查服务日志。'}))
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--open',action='store_true')
    parser.add_argument('--rebuild-index',action='store_true',help='Recover derived lists from authoritative frame JSON, then exit.')
    args=parser.parse_args();store=Store(args.root)
    import fcntl
    store.editor.mkdir(parents=True,exist_ok=True)
    lock=(store.editor/'server.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('本目录的标注服务已在运行，请使用已有页面。')
    recovered=store.recover_transactions()
    if recovered:print('已恢复中断的保存：'+', '.join(recovered),flush=True)
    if args.rebuild_index:store.rebuild();print('已从逐帧 JSON 重建清单。');return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(store,secrets.token_urlsafe(32),DEFAULT_ROOT/'annotation_editor.html'))
    server.daemon_threads=True
    url=f'http://127.0.0.1:{args.port}/'
    print(f'标注工具已启动：{url}\n数据目录：{store.root}\n按 Ctrl+C 停止。',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();lock.close()


if __name__=='__main__':main()
