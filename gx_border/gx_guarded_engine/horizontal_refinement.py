"""Experimental horizontal proposals; no change to production eligibility.

Local darkening candidates are anchored to the existing GX keyline seed. They
lack the full GX shoulder signature, so remain explicitly review-only.
"""
import cv2
import numpy as np
from .tracks import extract
from .anchored_polynomial import fit_outer


def local_darkening(lab, quad, side, scale):
    a,b=np.asarray(quad,float)[[side,(side+1)%4]]
    tangent=(b-a)/np.linalg.norm(b-a);normal=np.array([-tangent[1],tangent[0]])
    t=np.linspace(.05,.95,241);offsets=np.arange(-16.,16.5,.5)
    points=a+t[:,None,None]*(b-a)+offsets[None,:,None]*scale*normal
    colours=cv2.remap(lab,points[:,:,0].astype('float32'),points[:,:,1].astype('float32'),cv2.INTER_LINEAR)
    signed=np.gradient(cv2.GaussianBlur(colours,(3,1),0)[:,:,0],.5,axis=1)
    candidates=[]
    for row in range(len(t)):
        if np.any(points[row,:,0]<0) or np.any(points[row,:,0]>=lab.shape[1]-1) or np.any(points[row,:,1]<0) or np.any(points[row,:,1]>=lab.shape[0]-1):continue
        score=-signed[row]
        peaks=np.flatnonzero((score[1:-1]>=score[:-2])&(score[1:-1]>=score[2:])&(score[1:-1]>1.5))+1
        for p in peaks[abs(offsets[peaks])<=12]:
            u,v,w=score[p-1:p+2];delta=np.clip(.5*(u-w)/(u-2*v+w),-.5,.5) if abs(u-2*v+w)>1e-6 else 0
            candidates.append((row,t[row],offsets[p]+.5*delta,float(score[p])**.25))
    return np.asarray(candidates).reshape(-1,4),a,b,normal


def select_outer(current, straight):
    """Prefer a simpler model only with ten additional supported profiles.

    A curve stays in place on ties. This provisional rule is not an accuracy
    gate and does not change observed support into inferred corner evidence.
    """
    if (straight.get('supported') and current.get('supported')
        and straight.get('support',0)-current.get('support',0)>=10/241
        and straight.get('coverage',0)>=current.get('coverage',0)):
        return straight,'straight has at least ten more supported profiles and no span loss'
    return current,'retain connected curve'


def candidates(image,geometry):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.array(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    result={}
    for side,k in [('top',0),('bottom',2)]:
        result[side]={'outer_straight':fit_outer(*extract(lab,q,k,'outer',scale),scale,1)}
        for degree in [1,2]:
            track=fit_outer(*local_darkening(lab,geometry['inner'],k,scale),scale,degree)
            track.update(evidence='local GX darkening without full shoulder signature',review_only=True)
            result[side]['inner_dark_'+str(degree)]=track
    return result
