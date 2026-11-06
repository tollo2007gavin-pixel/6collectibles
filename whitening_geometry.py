"""Automatic whitening-only outer-boundary refinement; no model/network calls."""
import cv2
import numpy as np
from card_utils import find_card_quad as seed_quad, card_cut_off, CARD_W, CARD_H, PAD
VERSION='supported-outer-lines-6'
def _detect(img, blue_sat=75, blue_value=210, perimeter_only=False, band_width=.22, fabric_recovery=False):
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
 # Limit line voting to the neighborhood of the seeded perimeter. Background
 # texture and the Pokemon artwork otherwise overwhelm the actual border lines.
 if perimeter_only:
  mask=np.zeros(edges.shape,np.uint8)
  cv2.polylines(mask,[np.rint(q).astype(np.int32)],True,255,max(12,int(width*band_width)))
  edges=cv2.bitwise_and(edges,mask)
 segments=cv2.HoughLinesP(edges,1,np.pi/720,threshold=20,minLineLength=max(20,int(width*.15)),maxLineGap=20)
 if fabric_recovery:
  # LSD supplies short/low-contrast straight segments that global Hough voting
  # can miss. Candidate edges still require contrast, printed border and valid geometry.
  gray=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
  lines=cv2.createLineSegmentDetector().detect(gray)[0]
  if lines is not None:
   segments=lines if segments is None else np.concatenate((segments.astype(np.float32),lines),axis=0)
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
   if support<(.85 if fabric_recovery else .6) or contrast<28:continue
   # Sleeve edges also have contrast. Require printed border color just INSIDE the line.
   outward=normal if np.dot(normal,projected.mean(0)-center)>0 else -normal
   border_support=0.;blue_support=0.;gold_support=0.
   for inset in [.006,.012]:
    xy=np.rint(projected-outward*width*inset).astype(int)
    if xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=sm.shape[1] or xy[:,1].max()>=sm.shape[0]:continue
    samples=hsv[xy[:,1],xy[:,0]];h,sat,val=samples.T
    colored=((h>=85)&(h<=140)&(sat>blue_sat)&(val>20)&(val<blue_value))|((h>=5)&(h<=45)&(sat>70)&(val>25))
    blue_support=max(blue_support,float(((h>=85)&(h<=140)&(sat>blue_sat)&(val>20)&(val<blue_value)).mean()));gold_support=max(gold_support,float(((h>=5)&(h<=45)&(sat>70)&(val>25)).mean()));border_support=max(blue_support,gold_support)
   # An internal artwork edge has printed border on BOTH sides; a sleeve
   # usually has no saturated printed color immediately inside it.
   xy=np.rint(projected+outward*max(3,width*.012)).astype(int)
   if xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=sm.shape[1] or xy[:,1].max()>=sm.shape[0]:continue
   h,sat,val=hsv[xy[:,1],xy[:,0]].T
   outer_blue=float(((h>=85)&(h<=140)&(sat>blue_sat)&(val>20)&(val<blue_value)).mean())
   outer_gold=float(((h>=5)&(h<=45)&(sat>70)&(val>25)).mean())
   # Blue-tinted fabric is not the printed border: require a strong
   # saturation drop across a supported edge before allowing that exterior.
   inside_xy=np.rint(projected-outward*max(3,width*.012)).astype(int)
   inside_sat=hsv[inside_xy[:,1],inside_xy[:,0],1].astype(float)
   fabric_separation=fabric_recovery and float(np.median(inside_sat))>=100 and float(np.mean(inside_sat-sat.astype(float)>60))>=.8
   if outer_blue>.35 and not fabric_separation:blue_support=0.
   if outer_gold>.35:gold_support=0.
   border_support=max(blue_support,gold_support)
   if border_support<.6:continue
   options.append(((support*100+border_support*100+contrast*.1-abs(offset)*.1) if fabric_recovery else offset+contrast*.03,p,d,{'support':support,'printed_border_support':border_support,'blue_support':blue_support,'gold_support':gold_support,'median_contrast':contrast,'offset_source_pixels':float(offset/scale)}))
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
def detect(img):
 # Retain successful legacy geometry exactly. Retry failed seeds near the
 # perimeter; wider bands also cover seeds placed on the inner artwork panel.
 quad, diagnostics = _detect(img)
 diagnostics['recovery'] = 'legacy'
 if quad is not None:return quad, diagnostics
 for band in [.22,.42]:
  for sat,value,profile in [(75,210,'strict'),(35,245,'pale_blue')]:
   quad, diagnostics = _detect(img, blue_sat=sat, blue_value=value, perimeter_only=True, band_width=band)
   diagnostics.update(recovery='perimeter_lines_'+profile,band_width=band)
   if quad is not None:return quad, diagnostics
 # Preserve every successful prior result before trying tinted-background recovery.
 for sat,value in [(75,210),(35,245)]:
  quad,candidate=_detect(img,blue_sat=sat,blue_value=value,perimeter_only=True,band_width=.42,fabric_recovery=True)
  if quad is not None:
   candidate.update(recovery='tinted_background_lines',band_width=.42)
   return quad,candidate
 quad,extra=back_contour_recovery(img)
 if quad is not None:return quad,extra
 return None, diagnostics


def back_contour_recovery(img):
 """Recover perspective-distorted backs using independently supported physical edges."""
 from card_utils import _candidates,_order,_refine,classify_side
 scale=min(1.,800/max(img.shape[:2]));sm=cv2.resize(img,None,fx=scale,fy=scale)
 lab=cv2.cvtColor(sm,cv2.COLOR_BGR2LAB).astype(float)
 for contour,method in _candidates(sm):
  area=cv2.contourArea(contour);rect=cv2.minAreaRect(contour);w,h=rect[1]
  if min(w,h)<40 or not .1<area/sm.shape[0]/sm.shape[1]<.95 or area/(w*h)<.88 or not 1.2<max(w,h)/min(w,h)<1.65:continue
  hull=cv2.convexHull(contour);poly=cv2.approxPolyDP(hull,.018*cv2.arcLength(hull,True),True)
  if len(poly)!=4:continue
  q=_refine(contour,_order(poly.reshape(4,2).astype(np.float32)))
  if card_cut_off(sm.shape,q) or not cv2.isContourConvex(q):continue
  sides=[];center=q.mean(0)
  for i in range(4):
   a,b=q[i],q[(i+1)%4];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v);n*=1 if np.dot(n,center-(a+b)/2)>0 else -1
   mid=a+np.linspace(.12,.88,48)[:,None]*v
   delta=max(4,min(w,h)*.018);inside=np.rint(mid+n*delta).astype(int);outside=np.rint(mid-n*delta).astype(int)
   if any(x[:,0].min()<0 or x[:,1].min()<0 or x[:,0].max()>=sm.shape[1] or x[:,1].max()>=sm.shape[0] for x in (inside,outside)):break
   inn=tuple(inside.T[::-1]);ext=tuple(outside.T[::-1]);contrast=np.linalg.norm(lab[inn]-lab[ext],axis=1)
   B,G,R=sm[inn].astype(float).T
   # Relative blue dominance tolerates neutral lighting changes. A short glare
   # patch may interrupt the ink, but cannot substitute for a missing side.
   ink=(B>G*1.04)&(B>R*1.08)&(B>18)
   shadow=ink.mean()<.6 and np.median(contrast)>65 and np.median(sm[ext].astype(float).mean(axis=1)-sm[inn].astype(float).mean(axis=1))>55
   if (ink.mean()<.6 and not shadow) or min((part>20).mean() for part in np.array_split(contrast,3))<.65:break
   sides.append(dict(shadow=bool(shadow),blue_support=float(ink.mean()),median_contrast=float(np.median(contrast))))
  if len(sides)==4 and sum(x['shadow'] for x in sides)<=1 and back_print(warp(img,q/scale)):
   return q/scale,dict(version=VERSION,status='ready',quad=(q/scale).tolist(),sides=sides,recovery='supported_back_contour',candidate=method)
 return None,{}


def back_print(card):
 # Two yellow logo bands and the central red ball on the upright Pokemon back.
 body=card[PAD:PAD+CARD_H,PAD:PAD+CARD_W]
 hsv=cv2.cvtColor(cv2.resize(body,(100,140)),cv2.COLOR_BGR2HSV)
 h,s,v=hsv.transpose(2,0,1)
 yellow=(h>=12)&(h<=45)&(s>85)&(v>60)
 red=((h<13)|(h>170))&(s>70)&(v>50)
 return min(yellow[9:36,8:92].mean(),yellow[104:131,8:92].mean())>.08 and red[47:93,23:77].mean()>.05

def warp(img,quad):
 q=quad.copy()
 if np.linalg.norm(q[1]-q[0])>np.linalg.norm(q[2]-q[1]):q=q[[3,0,1,2]]
 dst=np.float32([[PAD,PAD],[PAD+CARD_W-1,PAD],[PAD+CARD_W-1,PAD+CARD_H-1],[PAD,PAD+CARD_H-1]])
 return cv2.warpPerspective(img,cv2.getPerspectiveTransform(q,dst),(CARD_W+2*PAD,CARD_H+2*PAD))

def detect_whitening_back(image, initial_side):
    """Cross-check the physical outline against the location of recognizable back art."""
    from card_utils import classify_side
    from back_landmarks import detect_printed_back, outlines_agree

    artwork_quad, artwork = detect_printed_back(image)
    diagnostics = {"reason": "initial_front"}
    if initial_side != "front":
        quad, diagnostics = detect(image)
        if quad is not None:
            side, _ = classify_side(warp(image, quad))
            blue_edges = all(s.get("blue_support", 0) >= .6 for s in diagnostics["sides"])
            valid = diagnostics.get("recovery") == "supported_back_contour" or (side == "back" and blue_edges)
            if valid:
                if "design" not in artwork:
                    return quad, diagnostics
                reference = artwork_quad if artwork_quad is not None else np.float32(artwork["seed_quad"])
                if outlines_agree(quad, reference):
                    return quad, diagnostics
                diagnostics = {"reason": "outline_disagrees_with_print_location"}
            else:
                diagnostics = dict(diagnostics, status="needs_review", reason="not_supported_blue_back")

    if "design" in artwork:
        artwork.update(version=VERSION, previous_reason=diagnostics.get("reason"))
        if artwork_quad is None:
            artwork["status"] = "needs_review"
        return artwork_quad, artwork
    return None, diagnostics


def refine_manifest(card_dir,manifest):
    from pathlib import Path
    import json
    from card_utils import sharpness, classify_side
    card_dir=Path(card_dir)
    for photo in manifest.get('photos',[]):
        if not photo.get('original_file'):continue
        image=cv2.imread(str(card_dir/photo['original_file']))
        if image is None:
            photo['card_found']=False
            photo['boundary_review']={'status':'needs_review','reason':'unreadable_source'}
            continue
        quad,diagnostics=detect_whitening_back(image,photo.get('side'))
        if diagnostics.get('reason') == 'initial_front':
            continue
        if diagnostics.get('design'):
            photo['side']='back'
        photo['boundary_review']=diagnostics
        photo['card_found']=quad is not None
        if quad is None:
            photo.pop('boundary_overlay',None)
            continue
        normalized=warp(image,quad)
        photo['side']='back'
        _,photo['blue_ratio']=classify_side(normalized)
        if not cv2.imwrite(str(card_dir/photo['file']),normalized,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('Could not save normalized card')
        photo['sharpness']=round(sharpness(normalized),1);photo['cut_off']=False
        overlay=image.copy();cv2.polylines(overlay,[quad.astype(int)],True,(0,255,0),4)
        photo['boundary_overlay']='boundary_'+photo['file'];cv2.imwrite(str(card_dir/photo['boundary_overlay']),overlay)
    manifest['has_front']=any(p.get('side')=='front' for p in manifest.get('photos',[]))
    manifest['has_back']=any(p.get('side')=='back' and p.get('card_found') for p in manifest.get('photos',[]))
    (card_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
