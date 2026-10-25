#!/usr/bin/env python3
"""Exercise the local five-frame comparison page without changing its data."""
from pathlib import Path
import argparse
import html
import json
import re
import subprocess
import tempfile

SCRIPT=r'''<script>
(async()=>{const checks=[];const check=(ok,msg)=>{if(!ok)throw Error(msg);checks.push(msg)};
const until=async(fn)=>{for(let i=0;i<500;i++){if(fn())return;await new Promise(r=>setTimeout(r,10))}throw Error('image loading timed out')};
try{await until(()=>document.body.dataset.loaded===DATA.frames[0].id);
check(DATA.frames.length===5,'五张测试图片');check(!$('boxes').checked,'默认隐藏检测框');
for(let i=0;i<5;i++){selector.selectedIndex=i;selector.dispatchEvent(new Event('change'));await until(()=>document.body.dataset.loaded===DATA.frames[i].id);
check($('bestCanvas').width>0&&$('lastCanvas').width>0&&currentImage.naturalWidth===1280,'加载并绘制 '+DATA.frames[i].id)}
$('reference').click();check(!$('referencePanel').hidden&&$('panels').classList.contains('reference'),'人工标注对照');
$('crop').click();check($('bestCanvas').width===1280&&$('bestCanvas').height===720,'完整画面查看');
$('boxes').click();check($('boxes').checked,'框显示开关');$('boxes').click();$('crop').click();$('reference').click();
$('prev').click();await until(()=>document.body.dataset.loaded===DATA.frames[3].id);check(selector.selectedIndex===3,'上一张导航');
$('next').click();await until(()=>document.body.dataset.loaded===DATA.frames[4].id);check($('next').disabled,'最后一张导航边界');
const p=document.createElement('pre');p.id='test-result';p.hidden=true;p.textContent=JSON.stringify({passed:true,checks});document.body.append(p);
}catch(e){const p=document.createElement('pre');p.id='test-result';p.textContent=JSON.stringify({passed:false,error:e.stack,checks});document.body.append(p)}})();
</script>'''


def main(job):
    job=Path(job).resolve();out=job/'test_predictions';tmp=Path(tempfile.mkdtemp(prefix='heldout_page_'))
    page=(out/'index.html').read_text()
    page=page.replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="'+out.as_uri()+'/">')
    (tmp/'check.html').write_text(page.replace('</html>',SCRIPT+'</html>'))
    shot=out/'browser_preview.png'
    run=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-proxy-server',
                        '--disable-background-networking','--no-first-run','--no-default-browser-check',f'--user-data-dir={tmp}/chrome',
                        '--window-size=1600,1100','--virtual-time-budget=12000','--dump-dom',f'--screenshot={shot}',(tmp/'check.html').as_uri()],capture_output=True,text=True,timeout=45)
    (tmp/'dom.html').write_text(run.stdout);(tmp/'chrome.log').write_text(run.stderr)
    match=re.search(r'<pre id="test-result"[^>]*>(.*?)</pre>',run.stdout,re.S)
    assert match, f'No result: {tmp}'
    result=json.loads(html.unescape(match[1]));assert result['passed'],result
    result.update(screenshot=str(shot),artifacts=str(tmp),source_data_changed=False)
    (out/'browser_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args();main(args.job)
