import cv2,numpy as np
from .gx_before import _order_corners,locate_card_on_photo

def proposals(bgr):
 h,w=bgr.shape[:2];scale=min(1.,900/max(h,w));im=cv2.resize(bgr,None,fx=scale,fy=scale)
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);sat=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[:,:,1]
 masks=[]
 # Coarse foreground proposal; its pixels never determine measured edges.
 gc_mask=np.full(gray.shape,cv2.GC_PR_BGD,np.uint8)
 gh,gw=gray.shape
 gc_mask[3:-3,3:-3]=cv2.GC_PR_FGD
 gc_mask[:3]=cv2.GC_BGD;gc_mask[-3:]=cv2.GC_BGD;gc_mask[:,:3]=cv2.GC_BGD;gc_mask[:,-3:]=cv2.GC_BGD
 gc_mask[int(.45*gh):int(.55*gh),int(.45*gw):int(.55*gw)]=cv2.GC_FGD
 try:
  cv2.setRNGSeed(0)
  cv2.grabCut(im,gc_mask,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
  masks.append(np.where((gc_mask==1)|(gc_mask==3),255,0).astype(np.uint8))
 except cv2.error:pass

 for ch in [gray,sat]:
  for level in [40,65,90,120,150,180,210]:
   mask=(ch>level).astype(np.uint8)*255;masks += [mask,255-mask] if ch is gray else [mask]
 for ch in [cv2.medianBlur(gray,15),cv2.medianBlur(sat,15)]:
  for level in [40,70,100,130,160,190,220]:
   mask=(ch>level).astype(np.uint8)*255;masks +=[mask,255-mask]
 masks+=[cv2.Canny(gray,30,90)]
 found=[]
 for mask in masks:
  clean=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
  contours,_=cv2.findContours(clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
   area=cv2.contourArea(c);frac=area/gray.size
   if not .08<frac<.94:continue
   x,y,ww,hh=cv2.boundingRect(c)
   if x<=1 or y<=1 or x+ww>=gray.shape[1]-1 or y+hh>=gray.shape[0]-1:continue
   hull=cv2.convexHull(c);rect=cv2.boxPoints(cv2.minAreaRect(hull))
   candidates=[rect]
   approx=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
   if len(approx)==4:candidates.insert(0,approx.reshape(4,2))
   for q in candidates:
    q=_order_corners(q);ss=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1);ratio=(ss[0]+ss[2])/(ss[1]+ss[3])
    if not .52<ratio<.92:continue
    fill=area/max(1,cv2.contourArea(q))
    score=frac*2+fill-3*abs(ratio-63/88)
    found.append((score,q/scale))
 found.sort(key=lambda x:-x[0]);chosen=[]
 for _,q in found:
  if all(np.mean(np.linalg.norm(q-prev,axis=1))>.035*min(w,h) for prev in chosen):chosen.append(q)
  if len(chosen)==7:break
 try:
  q=locate_card_on_photo(bgr)
  chosen.append(q)
 except ValueError:pass
 return chosen
