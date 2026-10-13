from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np
from card_utils import CARD_W,CARD_H,PAD,edge_corner_sheet,corner_closeup_sheet,card_cut_off

def prepare(source,annotation,out):
 source=Path(source);out=Path(out);data=source.read_bytes()
 if hashlib.sha256(data).hexdigest()!=annotation['source_sha256']:raise ValueError('Correction belongs to another source image')
 img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);q=np.asarray(annotation['quad'],np.float32)
 if img is None or q.shape!=(4,2) or not np.isfinite(q).all() or not cv2.isContourConvex(q) or cv2.contourArea(q,oriented=True)<=0:raise ValueError('Invalid TL/TR/BR/BL coordinates')
 if card_cut_off(img.shape,q):raise ValueError('Boundary outside source image')
 if abs(cv2.contourArea(q))<.05*img.shape[0]*img.shape[1]:raise ValueError('Implausibly small boundary')
 out.mkdir(parents=True,exist_ok=True)
 # Same inspection-sheet format; change only the source boundary, not shared centering logic.
 dst=np.float32([[PAD,PAD],[PAD+CARD_W-1,PAD],[PAD+CARD_W-1,PAD+CARD_H-1],[PAD,PAD+CARD_H-1]])
 warped=cv2.warpPerspective(img,cv2.getPerspectiveTransform(q,dst),(CARD_W+2*PAD,CARD_H+2*PAD))
 for name,value in [('corrected_sheet.png',edge_corner_sheet(warped,'reviewed')),('corrected_tips.png',corner_closeup_sheet(warped,'reviewed'))]:
  if not cv2.imwrite(str(out/name),value):raise ValueError('Save failed')
 overlay=img.copy();cv2.polylines(overlay,[q.astype(int)],True,(0,220,0),3)
 radius=max(30,int(min(np.linalg.norm(q[(i+1)%4]-q[i]) for i in range(4))*.14));boxes={}
 for loc,(x,y) in zip(['TL','TR','BR','BL'],q):
  x1,y1=max(0,int(x)-radius),max(0,int(y)-radius);x2,y2=min(img.shape[1],int(x)+radius),min(img.shape[0],int(y)+radius);boxes[loc]=[x1,y1,x2,y2];cv2.imwrite(str(out/(loc+'.png')),img[y1:y2,x1:x2]);cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,180,255),2)
 cv2.imwrite(str(out/'alignment.jpg'),overlay)
 rec={**annotation,'source':str(source.resolve()),'native_corner_boxes':boxes,'note':'Explicit visually reviewed boundary; not proof of card condition. Sheet/tips are perspective-resampled, native corner PNGs retain source pixels.','api_calls':0};(out/'crop_record.json').write_text(json.dumps(rec,indent=2));return rec
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--annotation',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();prepare(a.source,json.loads(Path(a.annotation).read_text()),a.out)
