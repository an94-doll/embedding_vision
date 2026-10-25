from pathlib import Path
import cv2,json,numpy as np
W=Path(__file__).resolve().parents[1];B=W/'batch';frames=[]
for n in range(1,278):
 stem=f'color_{n:06d}';r=json.loads((B/'annotations'/f'{stem}.json').read_text());boxes=np.array([o['bbox_xyxy'] for o in r['objects']]);im=cv2.imread(str(W/'previews'/f'{stem}_annotated.png'))[:720]
 if len(boxes):
  x1,y1=np.floor(boxes[:,:2].min(0)-25).astype(int);x2,y2=np.ceil(boxes[:,2:].max(0)+25).astype(int)
  x1,y1=max(0,x1),max(0,y1);x2,y2=min(1280,x2),min(720,y2)
 else:x1,y1,x2,y2=0,0,1280,720
 crop=im[y1:y2,x1:x2];scale=min(310/crop.shape[1],405/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
 tile=np.full((440,320,3),18,np.uint8);h,w=crop.shape[:2];tile[28:28+h,(320-w)//2:(320-w)//2+w]=crop
 missing=sum(not p[2] for o in r['objects'] for p in o['keypoints_xyv'])
 cv2.putText(tile,f'{n:03d} | {len(boxes)} boxes | {missing} missing',(5,18),0,.44,(230,230,230),1)
 frames.append(tile)
for start in range(0,len(frames),20):
 sheet=np.full((4*440,5*320,3),12,np.uint8)
 for k,tile in enumerate(frames[start:start+20]):sheet[(k//5)*440:(k//5+1)*440,(k%5)*320:(k%5+1)*320]=tile
 cv2.imwrite(str(B/'review'/f'sheet_{start//20+1:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
print('Created',int(np.ceil(len(frames)/20)),'review sheets')
