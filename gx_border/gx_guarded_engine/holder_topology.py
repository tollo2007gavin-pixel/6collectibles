"""Experimental outer-edge hypotheses scored at rounded card corners.

Search extends beyond the initial middle anchor so a sleeve seam is not locked
in permanently. Corner probes are evidence tests, never measurement extensions.
"""
import cv2,numpy as np
from .anchored_polynomial import fit_outer
from .tracks import project,SIDES

FRAME=np.float32([[160,160],[1420,160],[1420,1920],[160,1920]])

def sample(lab,points,Hinv):
    shape=points.shape[:-1];source=project(points.reshape(-1,2),Hinv).reshape(*shape,2)
    valid=(source[...,0]>=1)&(source[...,0]<lab.shape[1]-2)&(source[...,1]>=1)&(source[...,1]<lab.shape[0]-2)
    colors=cv2.remap(lab,source[...,0].astype('float32'),source[...,1].astype('float32'),cv2.INTER_LINEAR)
    return colors,valid

def corner_evidence(lab,Hinv,side,p):
    a,b=FRAME[[side,(side+1)%4]];v=b-a;u=v/np.linalg.norm(v);n=np.array([-u[1],u[0]])
    result=[]
    for end in [0,1]:
        along=u if end==0 else -u
        d=float(np.polynomial.polynomial.polyval(end-.5,p))
        corner=(a if end==0 else b)+n*d
        best=None
        for radius in np.arange(18.,61.,3.):
            angles=np.linspace(.16,np.pi/2-.16,25)
            radial=-np.cos(angles)[:,None]*along-np.sin(angles)[:,None]*n
            center=corner+radius*(along+n)
            points=center+radius*radial
            probes=points[:,None]+np.array([-2.,0,2.])[None,:,None]*radial[:,None]
            colors,valid=sample(lab,probes,Hinv)
            gradient=np.linalg.norm(colors[:,2]-colors[:,0],axis=1)/4
            gradient[~valid.all(1)]=0
            bins=[float(np.median(g)) for g in np.array_split(gradient,3)]
            score=min(bins)
            if best is None or score>best['score']:best=dict(score=score,radius=float(radius),bins=bins)
        result.append(best)
        # A holder seam often continues straight past the rounded card corner.
        fractions=np.linspace(-.012,.012,17) if end==0 else np.linspace(.988,1.012,17)
        ds=np.polynomial.polynomial.polyval(fractions-.5,p)
        probes=a+fractions[:,None,None]*(b-a)+(ds[:,None,None]+np.array([-2.,0,2.])[None,:,None])*n
        colors,valid=sample(lab,probes,Hinv)
        g=np.linalg.norm(colors[:,2]-colors[:,0],axis=1)/4
        best['straight_continuation']=float(np.median(g[valid.all(1)])) if valid.all(1).any() else None
        best['arc_over_continuation']=best['score']/(1+best['straight_continuation']) if best['straight_continuation'] is not None else 0.
    return result

def side_hypotheses(lab,Hinv,k):
    a,b=FRAME[[k,(k+1)%4]];u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]])
    ts=np.linspace(.05,.95,241);offsets=np.arange(-140.,201.,1.)
    points=a+ts[:,None,None]*(b-a)+offsets[None,:,None]*n
    colors,valid=sample(lab,points,Hinv);colors=cv2.GaussianBlur(colors,(3,1),0)
    strength=np.linalg.norm(np.gradient(colors,axis=1),axis=2)
    peaks=(strength[:,1:-1]>=strength[:,:-2])&(strength[:,1:-1]>=strength[:,2:])&(strength[:,1:-1]>1.5)
    peaks &= valid[:,:-2]&valid[:,1:-1]&valid[:,2:]
    rows,cols=np.nonzero(peaks);cols+=1
    cs=np.column_stack([rows,ts[rows],offsets[cols],strength[rows,cols]**.25])
    middle=(ts[rows]>.38)&(ts[rows]<.62)
    hist=np.bincount(cols[middle],weights=np.minimum(strength[rows[middle],cols[middle]],8),minlength=len(offsets))
    hist=cv2.GaussianBlur(hist[None].astype('float32'),(9,1),0)[0]
    modes=[0.]
    for index in np.argsort(hist)[::-1]:
        if hist[index]<5:break
        d=offsets[index]
        if all(abs(d-x)>9 for x in modes):modes.append(float(d))
        if len(modes)==14:break
    results=[]
    for mode in modes:
        # Widen curvature only around the original middle edge. A relocated
        # branch must not bend from one parallel seam to another at the ends.
        bend=50 if abs(mode)<8 else 14
        fits=[]
        for band in ([50,55,60] if abs(mode)<8 else [20]):
            c=cs[abs(cs[:,2]-mode)<band].copy();c[:,2]-=mode
            fitted=fit_outer(c,a,b,n,1,2,max_offset=bend)
            if fitted.get('supported'):fits.append(fitted)
        if not fits:continue
        track=max(fits,key=lambda t:t['coverage'])
        if not track.get('supported') or track['coverage']<.45:continue
        p=np.array(track['coefficients']);p[0]+=mode
        corners=corner_evidence(lab,Hinv,k,p)
        # Shift the evidence-only track back to the searched side, then source.
        track['segments']=[project(np.asarray(s)+mode*n,Hinv).tolist() for s in track['segments']]
        track['coefficients']=p.tolist();track['method']='rounded_corner_outer_hypothesis'
        results.append(dict(track=track,offset=float(p[0]),corners=corners,
                            corner_score=min(c['score'] for c in corners),
                            topology_score=min(c['arc_over_continuation'] for c in corners)))
    return sorted(results,key=lambda r:-r['corner_score'])

def hypotheses(image,geometry):
    H=cv2.getPerspectiveTransform(np.float32(geometry['outer']),FRAME)
    inv=np.linalg.inv(H);lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    return {side:side_hypotheses(lab,inv,k) for k,side in enumerate(SIDES)}

def baseline_evidence(image,geometry):
    inv=np.linalg.inv(cv2.getPerspectiveTransform(np.float32(geometry['outer']),FRAME))
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    return {s:corner_evidence(lab,inv,k,[0,0]) for k,s in enumerate(SIDES)}
