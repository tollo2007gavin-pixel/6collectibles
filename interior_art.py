"""Locate large pale interior-art rectangles in normalized corner-tip crops."""
import cv2
import numpy as np


def estimate(crop, corner, interior, uncertainty, edge_status):
    empty=np.zeros(crop.shape[:2],bool)
    if edge_status!='agreement':
        return dict(status='unconfirmed_edge'),empty
    size=crop.shape[0]
    # Mirror each tile so its inward (artwork) corner is top-left.
    def orient(a):
        if corner in ('TL','BL'):a=np.fliplr(a)
        if corner in ('TL','TR'):a=np.flipud(a)
        return a.copy()
    image=orient(crop); core=orient(interior & ~uncertainty)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    blue=core & (hsv[:,:,0]>=85)&(hsv[:,:,0]<=140)&(hsv[:,:,1]>100)
    if blue.sum()<size*size*.05:return dict(status='insufficient_blue'),empty
    reference_light=float(np.median(lab[:,:,0][blue]))
    reference_sat=float(np.percentile(hsv[:,:,1][blue],75))
    pale=(core & (lab[:,:,0].astype(float)>reference_light+12) &
          (hsv[:,:,1]<reference_sat*.85)).astype(np.uint8)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,round(size*.014)|1),)*2)
    pale=cv2.morphologyEx(pale,cv2.MORPH_CLOSE,kernel)
    count,labels,stats,_=cv2.connectedComponentsWithStats(pale,8)
    mask=np.zeros_like(pale);boxes=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        # Artwork extends through BOTH inward tile sides. A pale edge chip or
        # detached interior spot does not provide this rectangular layout evidence.
        if x>size*.02 or y>size*.02:continue
        fill=area/(w*h)
        regular=min(w,h)>=size*.06 and fill>=.6
        # A crop can expose only a thin slice of the printed rectangle. Require
        # a long, dense strip at the actual inward corner rather than lowering
        # the minimum size for every pale mark.
        clipped=(min(w,h)>=size*.03 and max(w,h)>=size*.20 and
                 max(x,y)<=size*.006 and fill>=.85)
        if not (regular or clipped) or area<size*size*.008 or area>size*size*.4:continue
        pad=max(2,round(size*.02))
        mask[:min(size,y+h+pad),:min(size,x+w+pad)]=1
        boxes.append(dict(width=int(w),height=int(h),fill=float(fill),
                          layout='rectangle' if regular else 'clipped_strip'))
    mask=mask.astype(bool)&core
    return dict(status='detected' if boxes else 'not_detected',rectangles=boxes,
                masked_fraction=float(mask.mean())),orient(mask)
