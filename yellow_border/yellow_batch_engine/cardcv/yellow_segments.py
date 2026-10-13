"""Experimental boundary fitting from the yellow plateau and separated spans."""
import cv2
import numpy as np
from .refine import intersect


def profiles(img, outer, inner, index, kind, ppm, *, trim=.08, n_samples=240,
             color_space='bgr', smoothing_mm=0., transition_mm=.20,along_sigma=1.2):
    quad=outer if kind=='outer' else inner
    a,b=quad[index],quad[(index+1)%4]
    direction=(b-a)/np.linalg.norm(b-a)
    inward=np.array([-direction[1],direction[0]])
    ts=np.linspace(trim,1-trim,n_samples)
    base=a+ts[:,None]*(b-a)
    radius=max(3.,.85*ppm)
    ds=np.arange(-radius,radius+.125,.25)
    points=base[:,None,:]+ds[None,:,None]*inward
    oi=outer[index]+ts[:,None]*(outer[(index+1)%4]-outer[index])
    ii=inner[index]+ts[:,None]*(inner[(index+1)%4]-inner[index])
    fs=np.linspace(.22,.78,17)
    yellow=oi[:,None,:]*(1-fs[None,:,None])+ii[:,None,:]*fs[None,:,None]
    x0=max(0,int(np.floor(min(points[:,:,0].min(),yellow[:,:,0].min())))-3)
    y0=max(0,int(np.floor(min(points[:,:,1].min(),yellow[:,:,1].min())))-3)
    x1=min(img.shape[1],int(np.ceil(max(points[:,:,0].max(),yellow[:,:,0].max())))+4)
    y1=min(img.shape[0],int(np.ceil(max(points[:,:,1].max(),yellow[:,:,1].max())))+4)
    patch=img[y0:y1,x0:x1].astype(np.float32)
    if color_space=='lab_chroma':
        patch=cv2.cvtColor(patch.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        patch[:,:,0]*=.25
    elif color_space=='luma':
        channel=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        patch=np.repeat(channel[:,:,None],3,axis=2)
    def sample(p):
        return cv2.remap(patch,(p[:,:,0]-x0).astype(np.float32),(p[:,:,1]-y0).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    colour=cv2.GaussianBlur(sample(points),(0,0),sigmaX=max(2.2,smoothing_mm*ppm/.25),sigmaY=along_sigma)
    ref=np.median(sample(yellow),axis=1)
    ref=cv2.GaussianBlur(ref[:,None,:],(1,9),0)[:,0,:]
    distance=np.linalg.norm(colour-ref[:,None,:],axis=2)
    gradient=np.gradient(distance,.25,axis=1)*(1 if kind=='inner' else -1)
    rows,cols=np.nonzero((gradient[:,1:-1]>gradient[:,:-2])&(gradient[:,1:-1]>=gradient[:,2:]))
    cols+=1
    gap=max(2,int(transition_mm*ppm/.25))
    left=distance[rows,np.maximum(0,cols-gap)]
    right=distance[rows,np.minimum(len(ds)-1,cols+gap)]
    plateau=left if kind=='inner' else right
    other=right if kind=='inner' else left
    good=(gradient[rows,cols]>3)&(other-plateau>6)&(plateau<.75*other)
    rows,cols=rows[good],cols[good]
    if len(rows)<30:return None
    g0,g1,g2=gradient[rows,cols-1],gradient[rows,cols],gradient[rows,cols+1]
    den=g0-2*g1+g2
    delta=np.divide(.5*(g0-g2),den,out=np.zeros_like(den),where=np.abs(den)>1e-8)
    offsets=ds[cols]+.25*np.clip(delta,-.5,.5)
    weight=np.sqrt(np.maximum(g1,0))
    rowmax=np.zeros(len(ts));np.maximum.at(rowmax,rows,weight)
    weight/=np.maximum(rowmax[rows],1e-8)
    t=ts[rows]
    return dict(a=a,b=b,inward=inward,ts=ts,base=base,radius=radius,
                rows=rows,offsets=offsets,weight=weight,t=t)


def boundary(img, outer, inner, index, kind, ppm):
    data=profiles(img,outer,inner,index,kind,ppm)
    if data is None:return None
    a,b,inward,ts,base,radius,rows,offsets,weight,t=(data[k] for k in
        ('a','b','inward','ts','base','radius','rows','offsets','weight','t'))
    rng=np.random.default_rng(6)
    tolerance=max(.65,.035*ppm)
    # Fit separated end regions independently, as in the GX corner work.
    # A whole-side consensus can fit the centre perfectly and miss both ends.
    anchors=[];local_details=[];chosen=[]
    for low,high in [(0,60),(180,240)]:
        use=np.flatnonzero((rows>=low)&(rows<high))
        if len(use)<20:return None
        local_best=None;local_score=-1
        for _ in range(350):
            u,v=rng.choice(use,2,replace=False)
            if abs(t[u]-t[v])<.055:continue
            model=np.polyfit(t[[u,v]],offsets[[u,v]],1)
            if max(abs(np.polyval(model,ts[[low,high-1]])))>radius:continue
            good=use[np.abs(offsets[use]-np.polyval(model,t[use]))<tolerance]
            votes=np.zeros(high-low);np.maximum.at(votes,rows[good]-low,weight[good])
            score=votes.sum()
            if score>local_score:local_best,local_score=model,score
        if local_best is None:return None
        for _ in range(3):
            good=use[np.abs(offsets[use]-np.polyval(local_best,t[use]))<tolerance]
            selected=[]
            for row in np.unique(rows[good]):
                ix=good[rows[good]==row];selected.append(ix[np.argmax(weight[ix])])
            selected=np.asarray(selected)
            if len(selected)<33:return None
            local_best=np.polyfit(t[selected],offsets[selected],1,w=np.sqrt(weight[selected]))
        at=float(np.median(t[selected]));anchors.append([at,float(np.polyval(local_best,at))])
        local_details.append({'support':len(selected)/60,'anchor':anchors[-1]})
        chosen.extend(selected)
    chosen=np.asarray(chosen)
    best=np.polyfit(*np.array(anchors).T,1)
    dd=np.polyval(best,ts)
    pts=base+dd[:,None]*inward
    vector=pts[-1]-pts[0];vector/=np.linalg.norm(vector)
    residual=offsets[chosen]-np.polyval(best,t[chosen])
    # Check the intervening visible edge as well as the two anchors. A chord
    # can match both ends of a bowed card yet sit in the background midway.
    strongest=[]
    for row in np.unique(rows):
        ix=np.flatnonzero(rows==row)
        strongest.append(ix[np.argmax(weight[ix])])
    strongest=np.asarray(strongest)
    along_residuals=[];trace=[]
    for low in range(0,240,30):
        ix=strongest[(rows[strongest]>=low)&(rows[strongest]<low+30)]
        if len(ix)<15:continue
        at=float(np.median(t[ix]));offset=float(np.median(offsets[ix]))
        delta=float(np.median(offsets[ix]-np.polyval(best,t[ix])))
        along_residuals.append(delta)
        point=a+at*(b-a)+offset*inward
        trace.append(point.tolist())
    detail={'support':len(chosen)/120,'span':float(np.ptp(t[chosen])/.84),
            'rms_px':float(np.sqrt(np.mean(residual**2))),
            'offset_px':float(np.median(dd)),
            'endpoint_offsets_px':np.polyval(best,[0,1]).tolist(),'end_segments':local_details,
            'intervening_residuals_px':along_residuals,'observed_edge_samples':trace,
            'max_segment_residual_px':max(map(abs,along_residuals),default=float('inf'))}
    return (pts.mean(axis=0),vector),detail


def refine(img,outer,inner,ppm,*,capture='photo'):
    if capture!='photo':
        return outer,inner,{'skipped':'scan path is preserved'}
    from .orientation import card_orientation
    top_side,_=card_orientation(img,outer,inner)
    bottom_side=(top_side+2)%4
    result=[];details={}
    for kind,quad in [('outer',outer),('inner',inner)]:
        lines=[];infos=[];changed=False
        for i in range(4):
            # The yellow inner-bottom transition is interrupted by copyright
            # and rules text. Leave other inner sides on the original fitter.
            fit=boundary(img,outer,inner,i,kind,ppm) if kind=='outer' or i==bottom_side else None
            if fit is not None:
                line,info=fit
                move=np.abs(info['endpoint_offsets_px'])
                # Correct measurable end drift while retaining an anchor near
                # the existing boundary. These are source-resolution and mm
                # limits, never card names or desired centering percentages.
                repair=(max(move)>max(1.,.20*ppm) and
                        min(move)<max(1.,.16*ppm) and max(move)<max(4.,.85*ppm))
                if repair and info['max_segment_residual_px']>max(2.,.10*ppm):
                    repair=False
                    info['blocked']='straight proposal misses the intervening visible boundary'
                info['applied']=bool(repair)
                if repair:
                    changed=True
                else:
                    a,b=quad[i],quad[(i+1)%4]
                    fit=((a,(b-a)/np.linalg.norm(b-a)),info)
            if fit is None:
                a,b=quad[i],quad[(i+1)%4]
                fit=((a,(b-a)/np.linalg.norm(b-a)),{'unchanged':'no supported segmented transition'})
            line,info=fit;lines.append(line);infos.append(info)
        corners=[intersect(lines[i-1],lines[i]) for i in range(4)]
        if any(c is None for c in corners):return None
        result.append(np.array(corners) if changed else quad);details[kind]=infos
    return *result,details
