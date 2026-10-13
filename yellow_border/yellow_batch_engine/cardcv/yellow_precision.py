"""Local, source-pixel fits after yellow-border edge identities are selected."""
import cv2
import numpy as np
from .consensus import ransac_td
from .refine import fit_line_huber, line_residuals, intersect


def fit_boundary(img, a, b, ppm, kind, capture="photo"):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not np.isfinite([a,b]).all() or not np.isfinite(ppm) or ppm<=0 or np.linalg.norm(b-a)<2:
        return None
    direction=(b-a)/np.linalg.norm(b-a)
    inward=np.array([-direction[1],direction[0]])
    ts=np.linspace(.12,.88,180)
    base=a+ts[:,None]*(b-a)
    radius=max(2.0,.8*ppm)
    ds=np.arange(-radius,radius+.125,.25)
    pts=base[:,None,:]+ds[None,:,None]*inward
    # Float RGB avoids quantization before interpolation.
    x0=max(0,int(np.floor(pts[:,:,0].min()))-2)
    y0=max(0,int(np.floor(pts[:,:,1].min()))-2)
    x1=min(img.shape[1],int(np.ceil(pts[:,:,0].max()))+3)
    y1=min(img.shape[0],int(np.ceil(pts[:,:,1].max()))+3)
    if x1<=x0 or y1<=y0:return None
    patch=img[y0:y1,x0:x1].astype(np.float32)
    if patch.size == 0:return None
    rgb=cv2.remap(patch,(pts[:,:,0]-x0).astype(np.float32),
                  (pts[:,:,1]-y0).astype(np.float32),cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    rgb=cv2.GaussianBlur(rgb,(0,0),sigmaX=2.6,sigmaY=.7)
    grad=np.gradient(rgb,.25,axis=1)
    # Project onto the observed local colour transition. This excludes
    # reversals from text/texture and keeps the position tied to source pixels.
    flank=max(1.0,.35*ppm)
    lo=np.median(rgb[:,ds < -flank],axis=1)
    hi=np.median(rgb[:,ds > flank],axis=1)
    contrast=hi-lo
    norm=np.linalg.norm(contrast,axis=1)
    unit=contrast/np.maximum(norm[:,None],1)
    strength=np.maximum(0,(grad*unit[:,None,:]).sum(axis=2))
    good_rows=norm>12
    peaks=(strength[:,1:-1]>strength[:,:-2]) & (strength[:,1:-1]>=strength[:,2:])
    rows,cols=np.nonzero(peaks);cols+=1
    keep=good_rows[rows] & (strength[rows,cols]>.15*np.max(strength,axis=1)[rows])
    rows,cols=rows[keep],cols[keep]
    if len(rows)<30:return None
    g0,g1,g2=strength[rows,cols-1],strength[rows,cols],strength[rows,cols+1]
    den=g0-2*g1+g2
    delta=np.divide(.5*(g0-g2),den,out=np.zeros_like(den),where=np.abs(den)>1e-8)
    offsets=ds[cols]+.25*np.clip(delta,-.5,.5)
    tolerance_mm = .025 if capture == "scan" else .075
    model,inl=ransac_td(rows.astype(float),offsets,max(.65,tolerance_mm*ppm),unique_rows=True)
    if model is None:return None
    rr,dd=rows[inl],offsets[inl]
    points=base[rr]+dd[:,None]*inward
    line=fit_line_huber(points)
    res=line_residuals(line,points)
    good=np.abs(res-np.median(res))<max(.65,3*1.4826*np.median(np.abs(res-np.median(res))))
    points=points[good];rr=rr[good]
    if len(points)<30:return None
    line=fit_line_huber(points)
    rms=float(np.sqrt(np.mean(line_residuals(line,points)**2)))
    info={'support':len(points)/180,'span':float(np.ptp(rr)/179),
          'rms_px':rms,'offset_px':float(np.median(dd)),
          'contrast':float(np.median(norm[rr]))}
    return line,info


def refine_boundaries(img,outer,inner,ppm,capture="photo"):
    outputs=[];diagnostics={}
    for kind,quad in [('outer',outer),('inner',inner)]:
        lines=[];infos=[]
        for i in range(4):
            fit=fit_boundary(img,quad[i],quad[(i+1)%4],ppm,kind,capture=capture)
            if fit is None:return None
            line,info=fit;lines.append(line);infos.append(info)
        corners=[intersect(lines[i-1],lines[i]) for i in range(4)]
        if any(c is None for c in corners):return None
        quad=np.asarray(corners)
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
            return None
        outputs.append(quad)
        diagnostics[kind]=infos
    return *outputs,diagnostics


def frame_evidence(img, outer, inner):
    """Sample the final geometry, including real background pixels at the cut.

    Rounded corners are excluded. Clearance is measured where the straight
    sides are observed, not at extrapolated corner intersections.
    """
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ts=np.linspace(.12,.88,160)
    fractions=np.linspace(.25,.75,12)
    yellow=[];uniform=[];chroma=[];outside=[];visible=[];frame_yellow=[]
    h,w=img.shape[:2]
    lengths=[np.linalg.norm(outer[(i+1)%4]-outer[i]) for i in range(4)]
    ppm=min((lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2)/63.
    for i in range(4):
        a,b=outer[i],outer[(i+1)%4]
        c,d=inner[i],inner[(i+1)%4]
        o=a+ts[:,None]*(b-a);inn=c+ts[:,None]*(d-c)
        points=o[:,None,:]*(1-fractions[None,:,None])+inn[:,None,:]*fractions[None,:,None]
        values=cv2.remap(hsv,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        H,S,V=values[:,:,0],values[:,:,1],values[:,:,2]
        yellow.append(float(((H>=12)&(H<=42)&(S>=55)&(V>=60)).mean()))
        ang=H.astype(float)*np.pi/90
        cs=np.stack([S*np.cos(ang),S*np.sin(ang)],axis=-1)
        med=np.median(cs.reshape(-1,2),axis=0);chroma.append(med)
        uniform.append(float((np.linalg.norm(cs-med,axis=2)<.6*max(np.linalg.norm(med),20)).mean()))
        v=(b-a)/np.linalg.norm(b-a);n=np.array([v[1],-v[0]])
        # One full source pixel outside plus the interpolation footprint.
        probe=o+1.25*n
        valid=(probe[:,0]>=0)&(probe[:,0]<=w-1)&(probe[:,1]>=0)&(probe[:,1]<=h-1)
        visible.append(float(valid.mean()))
        sample=cv2.remap(hsv,probe[:,0,None].astype(np.float32),probe[:,1,None].astype(np.float32),cv2.INTER_NEAREST)
        H,S,V=sample[:,:,0],sample[:,:,1],sample[:,:,2]
        outside.append(float(((H>=12)&(H<=42)&(S>=55)&(V>=60)).mean()))
        # A fitted line inside a cropped yellow band may still have pixels
        # outside it. Check whether that band continues to the image boundary.
        tx=np.full(len(o),np.inf);ty=tx.copy()
        if n[0]>.0001:tx=(w-1-o[:,0])/n[0]
        elif n[0]<-.0001:tx=-o[:,0]/n[0]
        if n[1]>.0001:ty=(h-1-o[:,1])/n[1]
        elif n[1]<-.0001:ty=-o[:,1]/n[1]
        distance=np.minimum(tx,ty)
        frame=o+np.maximum(distance,0)[:,None]*n
        frame_sample=cv2.remap(hsv,frame[:,0,None].astype(np.float32),frame[:,1,None].astype(np.float32),cv2.INTER_NEAREST)
        H,S,V=frame_sample[:,:,0],frame_sample[:,:,1],frame_sample[:,:,2]
        near=distance < max(4.,3.0*ppm)
        yel=((H>=12)&(H<=42)&(S>=55)&(V>=60))[:,0]
        frame_yellow.append(float((near&yel).mean()))
    ch=np.asarray(chroma);ref=np.median(ch,axis=0)
    dev=np.linalg.norm(ch-ref,axis=1)/max(np.linalg.norm(ref),20)
    return {'yellow_in':yellow,'yellow_out':outside,'uniform':uniform,
            'chroma_dev':dev.tolist(),'outside_visible':visible,
            'border_reaches_frame':frame_yellow}
