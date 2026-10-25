from pathlib import Path
import json
import cv2
import numpy as np
WORK=Path(__file__).resolve().parents[1]
JOB=WORK/"batch"
def read(path):return json.loads(path.read_text())
def iou(a,b):
    a=np.array(a);b=np.array(b)
    area=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(area/max(1,(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-area))

def match(a,b):
    if iou(a['bbox'],b['bbox'])<.45:return False
    pa=np.array(a['points']);pb=np.array(b['points'])
    width=min(a['bbox'][2]-a['bbox'][0],b['bbox'][2]-b['bbox'][0])
    height=min(a['bbox'][3]-a['bbox'][1],b['bbox'][3]-b['bbox'][1])
    # Models often disagree along the same long edge; box overlap and normal
    # displacement distinguish adjacent boxes without discarding real targets.
    dx=np.abs(pa[:,0]-pb[:,0]).max()
    mid=np.mean(pb[:,:2],axis=0)
    line_y=np.interp(mid[0],pa[:,0],pa[:,1])
    return dx<max(14,width*.24) and abs(mid[1]-line_y)<max(5,height*.22)

def consensus(stem):
    pool=[]
    for tag in ['epoch007','epoch034','epoch100','tile007','tile100']:
        for obj in read(JOB/'predictions'/tag/f'{stem}.json'):
            if obj['confidence']>=(.60 if tag.startswith('tile') else .40):
                pool.append({**obj,'model':tag})
    pool.sort(key=lambda x:x['confidence'],reverse=True)
    used=set();groups=[]
    for i,a in enumerate(pool):
        if i in used:continue
        chosen=[i]
        for tag in ['epoch007','epoch034','epoch100','tile007','tile100']:
            if tag==a['model']:continue
            options=[j for j,b in enumerate(pool) if j not in used and b['model']==tag and match(a,b)]
            if options:chosen.append(max(options,key=lambda j:iou(a['bbox'],pool[j]['bbox'])))
        if len(chosen)<2:continue
        used.update(chosen)
        obs=[pool[j] for j in chosen]
        bbox=np.median([o['bbox'] for o in obs],axis=0)
        points=np.median([o['points'] for o in obs],axis=0)
        spread=np.max(np.linalg.norm(np.array([o['points'] for o in obs])[:,:,:2]-points[None,:,:2],axis=2),axis=0)
        groups.append({'bbox':bbox.tolist(),'points':points.tolist(),'models':[o['model'] for o in obs],
                       'confidence':float(np.mean([o['confidence'] for o in obs])),
                       'point_spread':spread.tolist(),'votes':len(obs)})
    return sorted(groups,key=lambda x:(x['votes'],x['confidence']),reverse=True)

def evidence(gray,points,refine=True):
    p=np.array(points,float)[:,:2];direction=p[1]-p[0];length=np.linalg.norm(direction)
    if length<22 or direction[0]<=10:return None
    normal=np.array([-direction[1],direction[0]])/length
    t=np.linspace(.12,.88,43);base=p[0]+t[:,None]*direction
    def sample(coords):
        return cv2.remap(gray,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(float)
    shifts=np.arange(-5,5.01,.5) if refine else np.array([0.])
    coords=base[None,:,:]+shifts[:,None,None]*normal
    upper=sample(coords-normal*2);lower=sample(coords+normal*2)
    contrast=upper-lower
    valid=(coords[:,:,0]>3)&(coords[:,:,0]<1276)&(coords[:,:,1]>3)&(coords[:,:,1]<716)
    score=np.where(valid,contrast,-100)-np.abs(shifts[:,None])*1.4
    idx=score.argmax(axis=0);diff=contrast[idx,np.arange(len(t))];offset=shifts[idx]
    keep=(diff>7)&valid[idx,np.arange(len(t))]
    if keep.sum()<23:return None
    coef=np.polyfit(t[keep],offset[keep],1)
    inlier=keep&(np.abs(np.polyval(coef,t)-offset)<1.6)
    if inlier.sum()<21:return None
    coef=np.polyfit(t[inlier],offset[inlier],1)
    correction=np.clip(np.polyval(coef,[0,1]),-5,5)
    refined=p+normal*correction[:,None]
    if np.median(upper[idx,np.arange(len(t))][inlier])<55:return None
    return {'points':refined.tolist(),'edge_contrast':round(float(np.median(diff[inlier])),2),
            'edge_support':round(float(inlier.mean()),3),'normal_correction':correction.tolist()}

def front_surface(gray,bbox,points):
    # A bright shelf lip can resemble a long edge. Require a visible front
    # surface above it, rather than only a narrow bright metal/plastic strip.
    p=np.array(points,float)[:,:2];direction=p[1]-p[0];length=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])/max(1,length)
    base=p[0]+np.linspace(.16,.84,37)[:,None]*direction
    h=bbox[3]-bbox[1]
    offsets=np.array([2,.25*h,.4*h,.55*h,.7*h])
    coords=base[None]-normal*offsets[:,None,None]
    values=cv2.remap(gray,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(float)
    edge_level=float(np.median(values[0]));face_level=float(np.median(values[1:]))
    return {'front_brightness':round(face_level,2),'edge_brightness':round(edge_level,2),
            'front_to_edge_ratio':round(face_level/max(1,edge_level),3),
            'supported':(face_level>=105 and face_level>=edge_level*.45) or (face_level>=80 and face_level>=edge_level*.65)}

def represented(candidate,objects):
    box=candidate['bbox'];p=np.array(candidate['points'])[:,:2]
    for o in objects:
        b=o['bbox_xyxy'];lo=max(b[0],box[0]);hi=min(b[2],box[2])
        xover=max(0,hi-lo)/max(1,min(b[2]-b[0],box[2]-box[0]))
        q=np.array(o['keypoints_xyv']);vis=q[:,2]>0
        if iou(box,b)>.40:return True
        if xover<.55:continue
        tolerance=max(6,min(b[3]-b[1],box[3]-box[1])*.25)
        if vis.sum()==2:
            x=(lo+hi)/2
            # Bounding rectangles can overlap across adjacent front faces.
            if abs(np.interp(x,p[:,0],p[:,1])-np.interp(x,q[:,0],q[:,1]))<tolerance:return True
        elif vis.sum()==1:
            if iou(box,b)>.22 and abs(np.interp(q[vis,0][0],p[:,0],p[:,1])-q[vis,1][0])<tolerance:return True
        elif iou(box,b)>.45:return True
    return False

