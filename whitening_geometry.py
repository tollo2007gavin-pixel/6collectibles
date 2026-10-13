"""Automatic whitening-only outer-boundary refinement; no model/network calls."""
import cv2
import numpy as np
from card_utils import find_card_quad as seed_quad, card_cut_off, CARD_W, CARD_H, PAD
VERSION='supported-outer-lines-2'
def detect(img):
 seed=seed_quad(img)
 if seed is None:
  # Legacy seeding assumes an almost front-on 1.4 aspect ratio. Allow perspective here.
  from card_utils import _candidates, _order, _refine
  scale0=min(1.,800/max(img.shape[:2]));small=cv2.resize(img,None,fx=scale0,fy=scale0) if scale0<1 else img;choices=[]
  for contour,method in _candidates(small):
   area=cv2.contourArea(contour);rect=cv2.minAreaRect(contour);w,h=rect[1]
   if min(w,h)<40 or not .1<area/(small.shape[0]*small.shape[1])<.9 or not 1.15<max(w,h)/min(w,h)<1.9 or area/(w*h)<.75:continue
   polygon=cv2.approxPolyDP(cv2.convexHull(contour),.025*cv2.arcLength(cv2.convexHull(contour),True),True)
   if len(polygon)!=4:continue
   candidate=_refine(contour,_order(polygon.reshape(4,2).astype(np.float32)))
   if card_cut_off(small.shape,candidate):continue
   choices.append((area*(2 if method=='blue' else 1),candidate/scale0))
  if not choices:return None,{'version':VERSION,'status':'needs_review','reason':'no_seed'}
  seed=max(choices,key=lambda x:x[0])[1]
 scale=min(1.,800/max(img.shape[:2]));sm=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img
 q=seed*scale;center=q.mean(0);width=min(np.linalg.norm(q[(i+1)%4]-q[i]) for i in range(4));lab=cv2.cvtColor(sm,cv2.COLOR_BGR2LAB).astype(float);hsv=cv2.cvtColor(sm,cv2.COLOR_BGR2HSV)
 edges=cv2.Canny(cv2.GaussianBlur(sm,(3,3),0),10,30)
 segments=cv2.HoughLinesP(edges,1,np.pi/720,threshold=20,minLineLength=max(20,int(width*.15)),maxLineGap=20)
 if segments is None:return None,{'version':VERSION,'status':'needs_review','reason':'no_supported_lines'}
 chosen=[];metrics=[];side_options=[]
 for i in range(4):
  a,b=q[i],q[(i+1)%4];v=b-a;length0=np.linalg.norm(v);v=v/length0;n=np.array([-v[1],v[0]]);n*=1 if np.dot(n,(a+b)/2-center)>0 else -1;options=[]
  for x1,y1,x2,y2 in segments.reshape(-1,4):
   p=np.array([x1,y1],float);z=np.array([x2,y2],float);d=z-p;length=np.linalg.norm(d)
   if length<1:continue
   d/=length;offset=np.dot((p+z)/2-(a+b)/2,n);t=np.dot((p+z)/2-a,v)/length0
   if abs(np.dot(d,v))<.965 or not -.2*width<offset<.2*width or not .15<t<.85:continue
   normal=np.array([-d[1],d[0]]);midpoints=np.array([a+t0*(b-a) for t0 in np.linspace(.15,.85,50)]);projected=p+((midpoints-p)@d)[:,None]*d
   left=np.rint(projected+normal*3).astype(int);right=np.rint(projected-normal*3).astype(int)
   if min(left[:,0].min(),right[:,0].min(),left[:,1].min(),right[:,1].min())<0 or max(left[:,0].max(),right[:,0].max())>=sm.shape[1] or max(left[:,1].max(),right[:,1].max())>=sm.shape[0]:continue
   delta=np.linalg.norm(lab[left[:,1],left[:,0]]-lab[right[:,1],right[:,0]],axis=1);support=float(np.mean(delta>22));contrast=float(np.median(delta))
   if support<.6 or contrast<28:continue
   # Sleeve edges also have contrast. Require printed border color just INSIDE the line.
   outward=normal if np.dot(normal,projected.mean(0)-center)>0 else -normal
   border_support=0.;blue_support=0.;gold_support=0.
   for inset in [.006,.012]:
    xy=np.rint(projected-outward*width*inset).astype(int)
    if xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=sm.shape[1] or xy[:,1].max()>=sm.shape[0]:continue
    samples=hsv[xy[:,1],xy[:,0]];h,sat,val=samples.T
    colored=((h>=85)&(h<=140)&(sat>75)&(val>20)&(val<210))|((h>=5)&(h<=45)&(sat>70)&(val>25))
    blue_support=max(blue_support,float(((h>=85)&(h<=140)&(sat>75)&(val>20)&(val<210)).mean()));gold_support=max(gold_support,float(((h>=5)&(h<=45)&(sat>70)&(val>25)).mean()));border_support=max(blue_support,gold_support)
   # An internal artwork edge has printed border on BOTH sides; a sleeve
   # usually has no saturated printed color immediately inside it.
   xy=np.rint(projected+outward*max(3,width*.012)).astype(int)
   if xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=sm.shape[1] or xy[:,1].max()>=sm.shape[0]:continue
   h,sat,val=hsv[xy[:,1],xy[:,0]].T
   outer_blue=float(((h>=85)&(h<=140)&(sat>75)&(val>20)&(val<210)).mean())
   outer_gold=float(((h>=5)&(h<=45)&(sat>70)&(val>25)).mean())
   if outer_blue>.35:blue_support=0.
   if outer_gold>.35:gold_support=0.
   border_support=max(blue_support,gold_support)
   if border_support<.6:continue
   options.append((offset+contrast*.03,p,d,{'support':support,'printed_border_support':border_support,'blue_support':blue_support,'gold_support':gold_support,'median_contrast':contrast,'offset_source_pixels':float(offset/scale)}))
  if not options:return None,{'version':VERSION,'status':'needs_review','reason':'unsupported_side','side':i}
  side_options.append(options)
 families=[]
 for family in ['blue_support','gold_support']:
  selected=[]
  for options in side_options:
   compatible=[x for x in options if x[3][family]>=.6]
   if not compatible:break
   selected.append(max(compatible,key=lambda x:x[0]))
  if len(selected)==4:families.append((sum(x[0] for x in selected),selected))
 if not families:return None,{'version':VERSION,'status':'needs_review','reason':'inconsistent_border_color'}
 for _,p,d,stats in max(families,key=lambda x:x[0])[1]:chosen.append((p,d));metrics.append(stats)
 points=[]
 try:
  for i in range(4):
   p,d=chosen[i-1];z,v=chosen[i];t=np.linalg.solve(np.column_stack((d,-v)),z-p)[0];points.append(p+t*d)
 except np.linalg.LinAlgError:return None,{'version':VERSION,'status':'needs_review','reason':'parallel_sides'}
 result=np.float32(points)/scale
 if not np.isfinite(result).all() or not cv2.isContourConvex(result) or cv2.contourArea(result,oriented=True)<=0 or card_cut_off(img.shape,result):return None,{'version':VERSION,'status':'needs_review','reason':'invalid_or_cut_off_boundary'}
 area=abs(cv2.contourArea(result));oldarea=abs(cv2.contourArea(seed));sides=[np.linalg.norm(result[(i+1)%4]-result[i]) for i in range(4)]
 if not .65<area/oldarea<1.65 or max(sides)/min(sides)>2.2:return None,{'version':VERSION,'status':'needs_review','reason':'implausible_change'}
 return result,{'version':VERSION,'status':'ready','seed_quad':seed.tolist(),'quad':result.tolist(),'sides':metrics,'max_corner_shift_pixels':float(np.linalg.norm(result-seed,axis=1).max()),'heuristic_not_guaranteed':True}
def warp(img,quad):
 q=quad.copy()
 if np.linalg.norm(q[1]-q[0])>np.linalg.norm(q[2]-q[1]):q=q[[3,0,1,2]]
 dst=np.float32([[PAD,PAD],[PAD+CARD_W-1,PAD],[PAD+CARD_W-1,PAD+CARD_H-1],[PAD,PAD+CARD_H-1]])
 return cv2.warpPerspective(img,cv2.getPerspectiveTransform(q,dst),(CARD_W+2*PAD,CARD_H+2*PAD))

def refine_manifest(card_dir,manifest):
    from pathlib import Path
    import json
    from card_utils import sharpness
    card_dir=Path(card_dir)
    for photo in manifest.get('photos',[]):
        if photo.get('side')!='back' or not photo.get('original_file'):continue
        image=cv2.imread(str(card_dir/photo['original_file']))
        if image is None:
            photo['card_found']=False
            photo['boundary_review']={'status':'needs_review','reason':'unreadable_source'}
            continue
        quad,diagnostics=detect(image);photo['boundary_review']=diagnostics
        photo['card_found']=quad is not None
        if quad is None:
            photo.pop('boundary_overlay',None)
            continue
        normalized=warp(image,quad)
        if not cv2.imwrite(str(card_dir/photo['file']),normalized,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('Could not save normalized card')
        photo['sharpness']=round(sharpness(normalized),1);photo['cut_off']=False
        overlay=image.copy();cv2.polylines(overlay,[quad.astype(int)],True,(0,255,0),4)
        photo['boundary_overlay']='boundary_'+photo['file'];cv2.imwrite(str(card_dir/photo['boundary_overlay']),overlay)
    manifest['has_back']=any(p.get('side')=='back' and p.get('card_found') for p in manifest.get('photos',[]))
    (card_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
