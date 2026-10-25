#!/usr/bin/env python3
"""Render untouched five-frame test predictions after the fixed 100-epoch run."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil

from train_reviewed_two_points import digest, setup_env, write_json, now
from evaluate_front_two_points import match_reviewed_boxes


def main(job):
    job=Path(job).resolve();cfg=json.loads((job/'train_config.json').read_text());state=json.loads((job/'status.json').read_text())
    requested=cfg['train']['epochs']
    assert state['status']=='completed' and state['epochs_completed']==requested
    setup_env(job)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    ds=Path(cfg['snapshot_dataset']);out=job/'test_predictions';out.mkdir(exist_ok=False)
    history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((job/'loss_history.csv').open())]
    assert len(history)==requested and [int(r['epoch']) for r in history]==list(range(1,requested+1))
    best=min(history,key=lambda r:r['val_total_loss']);assert int(best['epoch'])==state['best_loss_epoch']
    manifest=list(csv.DictReader((ds/'manifest.csv').open()));test=[r for r in manifest if r['split']=='test']
    assert len(test)==5 and [int(r['frame']) for r in test]==cfg['test_frames']
    final_tag=f'epoch{requested}'
    specs=[('best_loss',state['best_loss_checkpoint'],state['best_loss_epoch'],'最低验证 loss'),(final_tag,state[f'{final_tag}_checkpoint'],requested,f'第{requested}轮')]
    records={}
    def draw(im, detections, threshold):
        im=im.copy();d=ImageDraw.Draw(im)
        for obj in detections:
            points=obj['keypoints'];visible=[p[2]>=threshold for p in points]
            if all(visible):d.line([tuple(points[0][:2]),tuple(points[1][:2])],fill='#ff6f79',width=2)
            for side,(x,y,_) in enumerate(points):
                if visible[side]:d.ellipse((x-3,y-3,x+3,y+3),fill='#71e9a1' if side==0 else '#77c6ff',outline='#102029',width=1)
        return im
    for r in test:
        stem=r['id'];image=ds/'images/test'/f'{stem}.png';label=ds/'labels/test'/f'{stem}.txt'
        assert digest(image)==r['image_sha256'] and digest(label)==r['label_sha256']
        lb=np.loadtxt(label,ndmin=2);boxes=np.concatenate([lb[:,1:3]-lb[:,3:5]/2,lb[:,1:3]+lb[:,3:5]/2],1)*[1280,720,1280,720]
        points=lb[:,5:].reshape(-1,2,3)*[1280,720,1]
        gt=[{'bbox':b.tolist(),'keypoints':p.tolist()} for b,p in zip(boxes,points)]
        shutil.copy2(image,out/f'{stem}_original.png')
        draw(Image.open(image).convert('RGB'),gt,2).save(out/f'{stem}_reference.png')
        records[stem]={'id':stem,'frame':int(r['frame']),'original':f'{stem}_original.png','reference':f'{stem}_reference.png','ground_truth':gt,'models':{}}
    summary=[]
    for tag,weight,epoch,title in specs:
        assert digest(weight)==state[('best_loss_checkpoint' if tag=='best_loss' else f'{final_tag}_checkpoint')+'_sha256']
        model=YOLO(weight)
        assert model.ckpt['epoch']+1==epoch and model.task=='pose' and list(model.model.model[-1].kpt_shape)==[2,3]
        assert model.names=={0:'cassette_front'}
        totals={'ground_truth_objects':0,'predicted_objects':0,'matched_objects_iou50':0,'ground_truth_visible_endpoints':0,
                'confident_endpoints_within_5px':0,'confident_endpoints_within_10px':0,'low_confidence_predicted_endpoints':0,'left_right_order_violations':0}
        distances=[]
        for r in test:
            stem=r['id'];rec=records[stem];gt=rec['ground_truth']
            settings={k:v for k,v in cfg['inference'].items() if k!='endpoint_conf'}
            result=model.predict(str(ds/'images/test'/f'{stem}.png'),device=0,save=False,verbose=False,**settings)[0]
            assert result.orig_shape==(720,1280) and tuple(result.keypoints.data.shape[1:])==(2,3)
            boxes,points,scores=[a.cpu().numpy() for a in (result.boxes.xyxy,result.keypoints.data,result.boxes.conf)]
            assert all(np.isfinite(a).all() for a in (boxes,points,scores))
            detections=[{'bbox':b.tolist(),'keypoints':p.tolist(),'confidence':float(s)} for b,p,s in zip(boxes,points,scores)]
            name=f'{stem}_{tag}.png';draw(Image.open(out/rec['original']).convert('RGB'),detections,.5).save(out/name)
            gbox=np.array([o['bbox'] for o in gt]);gp=np.array([o['keypoints'] for o in gt]);pairs=[]
            if len(boxes):
                lo=np.maximum(gbox[:,None,:2],boxes[None,:,:2]);hi=np.minimum(gbox[:,None,2:],boxes[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2)
                union=(gbox[:,2:]-gbox[:,:2]).prod(1)[:,None]+(boxes[:,2:]-boxes[:,:2]).prod(1)[None,:]-inter
                pairs=match_reviewed_boxes(inter/np.maximum(union,1e-9))
            vis=points[:,:,2]>=.5
            measure={'reference_objects':len(gt),'predicted_objects':len(boxes),'matched_objects_iou50':len(pairs),'low_confidence_predicted_endpoints':int((~vis).sum())}
            totals['ground_truth_objects']+=len(gt);totals['predicted_objects']+=len(boxes);totals['matched_objects_iou50']+=len(pairs)
            totals['ground_truth_visible_endpoints']+=int((gp[:,:,2]==2).sum());totals['low_confidence_predicted_endpoints']+=measure['low_confidence_predicted_endpoints']
            totals['left_right_order_violations']+=int((vis.all(1)&(points[:,0,0]>=points[:,1,0])).sum())
            for g,p in pairs:
                for side in range(2):
                    if gp[g,side,2]==2 and vis[p,side]:
                        delta=float(np.linalg.norm(gp[g,side,:2]-points[p,side,:2]));distances.append(delta)
                        totals['confident_endpoints_within_5px']+=int(delta<=5);totals['confident_endpoints_within_10px']+=int(delta<=10)
            rec['models'][tag]={'epoch':epoch,'title':title,'overlay':name,'detections':detections,**measure}
        totals['missed_objects_at_iou50']=totals['ground_truth_objects']-totals['matched_objects_iou50']
        totals['unmatched_predictions']=totals['predicted_objects']-totals['matched_objects_iou50']
        totals['matched_confident_endpoint_median_error_px']=float(np.median(distances)) if distances else None
        summary.append({'tag':tag,'title':title,'epoch':epoch,'weight':weight,'weight_sha256':digest(weight),
                        'val_total_loss':history[epoch-1]['val_total_loss'],**totals})
        del model,result;torch.cuda.empty_cache()
    report={'created_at':now(),'epochs':requested,'test_frames':cfg['test_frames'],'test_images':5,'test_used_for_training':False,'test_used_for_checkpoint_selection':False,
            'predictions_manually_corrected':False,'settings':cfg['inference'],'models':summary,'frames':list(records.values()),
            'pixel_error_scope':'Median covers IoU>=0.5 matched objects and endpoints with confidence>=0.5. Within-5/10px rates use all visible reference endpoints as denominator.',
            'limitation':'Same video with temporal gaps; these five frames are not a new recording.'}
    write_json(out/'predictions.json',report)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=[r['epoch'] for r in history]
    for ax,key,title in [(axes[0],'train_total_loss','Training total loss'),(axes[1],'val_total_loss','Validation total loss (selection)')]:
        ax.plot(x,[r[key] for r in history],linewidth=1.5);ax.set(title=title,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.25)
    axes[1].scatter([best['epoch'],requested],[best['val_total_loss'],history[-1]['val_total_loss']],c=['#15a878','#e37025'],zorder=3)
    if cfg.get('resume_from_epoch'):
        for ax in axes:ax.axvline(cfg['resume_from_epoch'],color='#7f8790',linestyle='--',alpha=.6)
    axes[1].annotate(f"min: epoch {int(best['epoch'])}",(best['epoch'],best['val_total_loss']),xytext=(8,18),textcoords='offset points')
    fig.savefig(job/'loss_curves.png',dpi=160);plt.close(fig)
    create_html(out,report)
    source=Path(cfg['source_dataset']);hashes=json.loads((job/'source_preservation_hashes.json').read_text())
    changed=[rel for rel,sha in hashes.items() if digest(source/rel)!=sha]
    write_json(job/'source_preservation_check.json',{'checked_files':len(hashes),'unchanged':not changed,'changed_since_snapshot':changed,'training_snapshot_independent':True})
    lines=[f'# 用户修正数据：{requested}轮训练与五张留出测试','',
           f"累计完整完成{requested}轮，未提前停止。两阶段运行耗时合计约 {state['elapsed_seconds']/60:.2f} 分钟。YOLO26s-Pose，1280输入，batch=4，AdamW，FP32，seed=42。",'',
           '277张均来自用户已批准标注的独立快照：192张训练、50张验证、5张最终测试、30张时间间隔。测试编号31、83、134、190、259；未用于权重更新或最低loss选择。', '',
           '**最低loss定义：**每轮验证集全部 `val/*_loss` 项之和。训练损失和框架自带best.pt的mAP选择均不是此次选择标准。', '',
           '| 保存权重 | 轮数 | 验证总loss | 测试匹配目标 / 43 | 额外预测 | 匹配点中位误差px | ≤5px / 86 | ≤10px / 86 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in summary:
        label='best_loss.pt' if m['tag']=='best_loss' else f'{final_tag}.pt'
        median=m['matched_confident_endpoint_median_error_px'];median_text=f'{median:.2f}' if median is not None else '无匹配点'
        lines.append(f"| [{label}]({os.path.relpath(m['weight'],job)}) | {m['epoch']} | {m['val_total_loss']:.5f} | {m['matched_objects_iou50']} | {m['unmatched_predictions']} | {median_text} | {m['confident_endpoints_within_5px']} | {m['confident_endpoints_within_10px']} |")
    lines+=['','[五张图片对比网页](test_predictions/index.html) · [loss曲线](loss_curves.png) · [逐轮loss](loss_history.csv) · [数据划分与审计](dataset_audit.json)', '',
            f'两份指定权重均保留模型、EMA及优化器状态；已加载核对内部轮数和对应loss。{requested}行训练记录连续无缺失。所有预测使用相同阈值：检测0.25、端点0.5，无人工修正；默认只显示两点和连线，网页可选择显示框或人工标注。', '',
            '目标匹配使用IoU≥0.5一对一匹配。像素中位误差仅统计匹配且达到置信度的端点；≤5/10px以全部86个可见参考端点为分母，漏检也计入分母。五张测试仍来自同一视频，不能代替新场景测试。', '',
            f"源数据保护检查：{len(hashes)}个标注及配置文件，{'全部未变' if not changed else '检测到用户后续保存，具体见source_preservation_check.json；训练使用此前快照'}。没有修改浏览器审核结果。",'']
    if cfg.get('resume_from_epoch'):
        lines+=['第101–200轮通过Ultralytics恢复第100轮的优化器、EMA和scaler状态，已核对优化器步数与EMA更新数。学习率从第100轮实际值继续线性衰减到其10%；这是用户追加轮数后的续训，不等同于从首轮预设200轮的学习率计划。', '',
                f"第100轮权重也保留：[epoch100.pt]({os.path.relpath(state['epoch100_checkpoint'],job)})。前100轮原运行目录和权重未覆盖。",'']
    (job/'TRAINING_REPORT.md').write_text('\n'.join(lines))
    write_json(job/'training_summary.json',{'epochs':requested,'status':'completed','models':summary,'test_report':str(out/'predictions.json'),'html':str(out/'index.html'),'source_unchanged':not changed})
    print(json.dumps({'output':str(out),'models':summary},ensure_ascii=False,indent=2))


def create_html(out, report):
    data=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5张留出图片 · YOLO100轮对比</title>
<style>*{box-sizing:border-box}body{margin:0;background:#11191f;color:#e4edf2;font:15px/1.6 system-ui,sans-serif}header,main{padding:18px 28px}header{border-bottom:1px solid #33434d}h1{font-size:24px;margin:0 0 8px}p{margin:8px 0;color:#b7c7d0}a{color:#78cef0}button,select{background:#25343e;color:#edf7ff;border:1px solid #4b6474;border-radius:5px;padding:7px 12px;font:inherit;cursor:pointer}label{margin-right:18px}nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.panels.reference{grid-template-columns:repeat(3,minmax(0,1fr))}figure{margin:0;background:#1c2830;border:1px solid #334853;border-radius:8px;overflow:hidden}figcaption{padding:10px 14px;font-size:17px}.small{font-size:13px;color:#a9bdc8;padding:8px 14px}canvas{display:block;width:100%;height:auto;background:#080c10}#referencePanel[hidden]{display:none}.note{color:#b8d9cb}#stats{padding:12px 0}details{margin:20px 0}img{max-width:100%}@media(max-width:900px){.panels,.panels.reference{grid-template-columns:1fr}header,main{padding:14px}}</style>
<header><h1>5 张留出图片 · 两版模型对比</h1><p>图片未参与训练或最低loss权重选择。绿点为左端、蓝点为右端，红线连接两点。以下为模型直接预测，未人工修改。</p><nav><button id="prev">上一张</button><select id="frame"></select><button id="next">下一张</button><label><input type="checkbox" id="crop" checked>放大盒体区域</label><label><input type="checkbox" id="boxes">显示框</label><label><input type="checkbox" id="reference">对照你的标注</label></nav><p class="note">放大视图只截取盒体区域；取消勾选可检查完整画面的误检。点击下方完整图链接可按原始分辨率查看。</p></header>
<main><div id="stats"></div><div class="panels" id="panels"><figure><figcaption id="bestTitle"></figcaption><canvas id="bestCanvas"></canvas><div class="small" id="bestStats"></div><div class="small"><a id="bestLink" target="_blank">打开完整预测图</a></div></figure><figure><figcaption id="lastTitle">第100轮</figcaption><canvas id="lastCanvas"></canvas><div class="small" id="lastStats"></div><div class="small"><a id="lastLink" target="_blank">打开完整预测图</a></div></figure><figure id="referencePanel" hidden><figcaption>你的人工标注</figcaption><canvas id="referenceCanvas"></canvas><div class="small"><a id="referenceLink" target="_blank">完整人工标注图</a> · <a id="originalLink" target="_blank">原图</a></div></figure></div><p>检测阈值0.25，端点阈值0.5，输入1280；两份模型设置一致。最低loss按50张验证图每轮总loss选择。测试仍来自同一视频。</p><details><summary>100轮训练曲线</summary><img src="../loss_curves.png" alt="训练和验证loss曲线"></details><p><a href="predictions.json">原始预测坐标与检查结果</a></p></main>
<script>const DATA=__DATA__, $=id=>document.getElementById(id);let currentImage=null,serial=0;const selector=$('frame');DATA.frames.forEach((f,i)=>selector.add(new Option(`${i+1}/5 · ${f.id}`,i)));
function region(f){if(!$('crop').checked)return[0,0,1280,720];const b=f.ground_truth.map(o=>o.bbox);const x=Math.max(0,Math.min(...b.map(v=>v[0]))-35),y=Math.max(0,Math.min(...b.map(v=>v[1]))-30),right=Math.min(1280,Math.max(...b.map(v=>v[2]))+35),bottom=Math.min(720,Math.max(...b.map(v=>v[3]))+30);return[x,y,right-x,bottom-y]}
function paint(id,objects,threshold,r){const c=$(id),ctx=c.getContext('2d');c.width=r[2];c.height=r[3];ctx.drawImage(currentImage,...r,0,0,r[2],r[3]);ctx.translate(-r[0],-r[1]);for(const o of objects){const p=o.keypoints,vis=p.map(v=>v[2]>=threshold);if($('boxes').checked){ctx.strokeStyle=vis.every(Boolean)?'#c4d8e799':'#ffc35d';ctx.lineWidth=1;ctx.strokeRect(o.bbox[0],o.bbox[1],o.bbox[2]-o.bbox[0],o.bbox[3]-o.bbox[1])}if(vis.every(Boolean)){ctx.strokeStyle='#ff6f79';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);ctx.lineTo(p[1][0],p[1][1]);ctx.stroke()}p.forEach((v,i)=>{if(vis[i]){ctx.beginPath();ctx.arc(v[0],v[1],2.7,0,Math.PI*2);ctx.fillStyle=i?'#77c6ff':'#71e9a1';ctx.fill();ctx.strokeStyle='#102029';ctx.lineWidth=.7;ctx.stroke()}})}}
function render(){if(!currentImage)return;const f=DATA.frames[Number(selector.value)],r=region(f);paint('bestCanvas',f.models.best_loss.detections,.5,r);paint('lastCanvas',f.models.epoch100.detections,.5,r);paint('referenceCanvas',f.ground_truth,2,r);$('referencePanel').hidden=!$('reference').checked;$('panels').classList.toggle('reference',$('reference').checked);$('stats').textContent=`${f.id} · 人工标注 ${f.ground_truth.length} 个盒体`;for(const [tag,prefix] of [['best_loss','best'],['epoch100','last']]){const m=f.models[tag];$(prefix+'Title').textContent=tag==='best_loss'?`${m.title} · 第 ${m.epoch} 轮`:`第 ${m.epoch} 轮`;$(prefix+'Stats').textContent=`预测 ${m.predicted_objects} 个 · 匹配 ${m.matched_objects_iou50} 个 · 低置信端点 ${m.low_confidence_predicted_endpoints} 个`;$(prefix+'Link').href=m.overlay}$('referenceLink').href=f.reference;$('originalLink').href=f.original;$('prev').disabled=selector.selectedIndex===0;$('next').disabled=selector.selectedIndex===4}
function load(){const seq=++serial,im=new Image();im.onload=()=>{if(seq!==serial)return;currentImage=im;render();document.body.dataset.loaded=DATA.frames[Number(selector.value)].id};im.src=DATA.frames[Number(selector.value)].original}
selector.onchange=load;$('prev').onclick=()=>{selector.selectedIndex--;load()};$('next').onclick=()=>{selector.selectedIndex++;load()};['crop','boxes','reference'].forEach(id=>$(id).onchange=render);load();</script></html>'''
    requested=report['epochs']
    template=template.replace('epoch100',f'epoch{requested}').replace('100轮',f'{requested}轮')
    (out/'index.html').write_text(template.replace('__DATA__',data))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);args=p.parse_args();main(args.job)
