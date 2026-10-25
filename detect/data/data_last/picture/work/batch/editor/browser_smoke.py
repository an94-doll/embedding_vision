#!/usr/bin/env python3
"""Headless Chrome UI integration checks against copied frames in /tmp only."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import tempfile
import threading

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('review_server',ROOT/'batch/review_server.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

TEST=r'''
<script>
const assertions=[];
const check=(value,message)=>{if(!value)throw Error(message);assertions.push(message)};
const until=async(fn,label)=>{for(let n=0;n<600;n++){if(fn())return;await new Promise(r=>setTimeout(r,20))}throw Error('超时: '+label)};
const pointer=(type,x,y,buttons=1)=>{const r=canvas.getBoundingClientRect();canvas.dispatchEvent(new PointerEvent(type,{clientX:r.left+S.ox+x*S.scale,clientY:r.top+S.oy+y*S.scale,pointerId:1,button:0,buttons,bubbles:true}))};
const drag=(x1,y1,x2,y2)=>{pointer('pointerdown',x1,y1);pointer('pointermove',x2,y2);pointer('pointerup',x2,y2,0)};
const clickAt=(x,y)=>{pointer('pointerdown',x,y);pointer('pointerup',x,y,0)};
// Synthetic events do not create a native active pointer. Capture itself is a browser primitive.
canvas.setPointerCapture=()=>{};canvas.releasePointerCapture=()=>{};canvas.hasPointerCapture=()=>false;
(async()=>{try{
 await until(()=>S.frame?.id==='color_000001'&&!S.busy,'load first frame');
 check(!$('showBoxes').checked&&!$('showNumbers').checked,'默认隐藏边界框和编号');
 $('referenceHelp').click();check($('referenceDialog').open,'打开第83张固定标注示例');
 await until(()=>$('referenceImage').complete&&$('referenceImage').naturalWidth===870,'reference image');
 check($('referenceImage').naturalHeight===570,'示例图片正常加载');
 $('closeReference').click();check(!$('referenceDialog').open&&!S.dirty,'关闭示例后继续编辑且无意外修改');
 const initial=S.objects.length;
 $('addTool').click();drag(1100,580,1210,665);check(S.objects.length===initial+1&&S.tool==='left','拖框新增目标');
 clickAt(1110,640);check(S.tool==='right','左端点后进入右端点模式');clickAt(1200,637);
 check(S.objects.at(-1).keypoints_xyv.every(p=>p[2]===2)&&S.tool==='select','完成两端点标注');
 drag(1110,640,1115,643);check(Math.abs(S.objects.at(-1).keypoints_xyv[0][0]-1115)<.1,'拖动端点');
 const sc=S.scale;$('zoomIn').click();check(S.scale>sc,'放大画面');
 $('showBoxes').click();
 drag(1210,665,1220,680);check(S.objects.at(-1).bbox_xyxy[2]>=1219,'放大后拖动框角');
 $('leftMissing').click();check(S.objects.at(-1).keypoints_xyv[0].every(v=>v===0),'设置缺失端点');
 $('leftMissing').click();clickAt(1115,643);check(S.objects.at(-1).keypoints_xyv[0][2]===2,'重新放置缺失端点');
 $('delete').click();check(S.objects.length===initial,'删除目标');$('undo').click();check(S.objects.length===initial+1,'撤销删除');$('redo').click();check(S.objects.length===initial,'重做删除');$('undo').click();
 $('lx').value='1118';$('lx').dispatchEvent(new Event('input',{bubbles:true}));
 $('next').click();await until(()=>$('dirtyDialog').open,'dirty dialog');check($('dirtyDialog').open,'未保存翻页提示');$('cancelNav').click();
 $('saveDraft').click();await until(()=>!S.busy&&!S.dirty,'save draft');check(S.frame.decision==='pending','保存草稿不批准训练');
 check(S.objects.at(-1).keypoints_xyv[0][0]===1118,'保存时自动应用数值坐标');
 $('approve').click();await until(()=>!S.busy&&S.frame.decision==='accepted','approve');check($('acceptedStat').textContent==='2','通过后同步训练统计');
 $('next').click();await until(()=>!S.busy&&S.frame.id==='color_000140','next');check(S.frame.objects.length>0,'加载另一张图片');
 $('prev').click();await until(()=>!S.busy&&S.frame.id==='color_000001','previous');check(S.objects.length===initial+1&&S.frame.decision==='accepted','重新加载保留编辑及批准状态');
 $('reject').click();await until(()=>!S.busy&&S.frame.decision==='rejected','reject');check($('acceptedStat').textContent==='1','拒绝后移出训练清单');
 $('filter').value='rejected';$('filter').dispatchEvent(new Event('change'));check($('frames').children.length===1,'按复核结果筛选');
 $('fit').click();$('objects').firstChild.click();
 const pre=document.createElement('pre');pre.id='smoke-result';pre.textContent='PASS\n'+assertions.join('\n');pre.style.display='none';document.body.append(pre);
}catch(e){const pre=document.createElement('pre');pre.id='smoke-result';pre.textContent='FAIL '+e.stack+'\n'+assertions.join('\n');document.body.append(pre)}})();
</script>
'''


def main():
    tmp=Path(tempfile.mkdtemp(prefix='cassette_ui_check_'));fixture=tmp/'work'
    manifest=json.loads((ROOT/'batch/manifest.json').read_text());manifest['images']=[x for x in manifest['images'] if x['id'] in ('color_000001','color_000140')];manifest['count']=2
    for folder in ['batch/annotations','labels/train','images/train','previews']:(fixture/folder).mkdir(parents=True,exist_ok=True)
    for item in manifest['images']:
        stem=item['id']
        for folder,suffix in [('batch/annotations','.json'),('labels/train','.txt'),('images/train','.png'),('previews','_annotated.png')]:
            shutil.copy2(ROOT/folder/(stem+suffix),fixture/folder/(stem+suffix))
    fp=fixture/'batch/annotations/color_000140.json';record=json.loads(fp.read_text());record.update(review_status='accepted',training_eligible=True);fp.write_bytes(mod.json_bytes(record))
    (fixture/'batch/manifest.json').write_bytes(mod.json_bytes(manifest))
    (fixture/'batch_train.txt').write_text(str(fixture/'images/train/color_000140.png')+'\n')
    asset=tmp/'test.html';asset.write_text((ROOT/'annotation_editor.html').read_text().replace('</body>',TEST+'</body>'))
    store=mod.Store(fixture)
    server=mod.ThreadingHTTPServer(('127.0.0.1',0),mod.handler_for(store,'ui-test-only',asset))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        args=['google-chrome','--headless','--no-sandbox','--disable-gpu','--no-proxy-server','--disable-dev-shm-usage',
              '--disable-background-networking','--no-first-run','--no-default-browser-check',f'--user-data-dir={tmp}/chrome',
              '--window-size=1600,1100','--virtual-time-budget=18000','--dump-dom',f'--screenshot={tmp}/editor.png',
              f'http://127.0.0.1:{server.server_port}/']
        r=subprocess.run(args,capture_output=True,text=True,timeout=55)
        (tmp/'dom.html').write_text(r.stdout);(tmp/'chrome.log').write_text(r.stderr)
        import re,html
        result=re.search(r'<pre id="smoke-result"[^>]*>(.*?)</pre>',r.stdout,re.S)
        if not result:raise AssertionError(f'浏览器测试未返回结果: {tmp}')
        text=html.unescape(result[1]);print(text);print('Artifacts:',tmp)
        assert text.startswith('PASS'),text
        assert len((fixture/'batch_train.txt').read_text().splitlines())==1
        assert store.get('color_000001')['decision']=='rejected'
        out=ROOT/'batch/editor/browser_test_result.json'
        out.write_bytes(mod.json_bytes({'passed':True,'checks':text.splitlines()[1:],'fixture':str(fixture),'screenshot':str(tmp/'editor.png'),'real_dataset_modified':False}))
    finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
