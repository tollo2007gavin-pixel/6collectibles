"""Shadow outer fit constrained to a supported middle transition.

Preserves the initial physical-edge identity in the middle, not a centering
ratio. Does not claim that every initial outer edge is correct.
"""
import numpy as np
from .tracks import extract
import cv2


def connected_candidates(candidates):
    """Discard parallel branches unreachable from the middle physical edge.

    A fit cannot gain votes by bridging a blank region to a disconnected printed
    line. Missing source observations stay missing; no new samples are inserted.
    """
    groups={int(row):candidates[candidates[:,0]==row] for row in np.unique(candidates[:,0])}
    central=[row for row in groups if abs(groups[row][0,1]-.5)<.12 and np.any(abs(groups[row][:,2])<=2)]
    if not central:return np.empty((0,4))
    anchor=min(central,key=lambda row:abs(groups[row][0,1]-.5))
    initial=groups[anchor][abs(groups[anchor][:,2])<=2];chosen=[initial]
    for direction in [-1,1]:
        previous=initial;last=anchor
        indices=sorted([row for row in groups if (row-anchor)*direction>0],reverse=direction<0)
        for row in indices:
            gap=abs(row-last)
            if gap>8:break
            current=groups[row]
            reachable=np.any(abs(current[:,2,None]-previous[:,2][None,:])<=3.+.35*gap,axis=1)
            if not reachable.any():continue
            previous=current[reachable];chosen.append(previous);last=row
    return np.concatenate(chosen)


def fit_outer(candidates,a,b,normal,scale,degree=2,*,max_offset=14.0):
    if len(candidates)<20:return dict(supported=False,reason='too few profiles')
    rows,t,d,weight=candidates.T;x=t-.5
    central=(abs(x)<.12)&(abs(d)<=2)
    if len(np.unique(rows[central]))<20:return dict(supported=False,reason='no supported middle anchor')
    candidates=connected_candidates(candidates)
    if len(candidates)<20:return dict(supported=False,reason='insufficient connected outer evidence')
    rows,t,d,weight=candidates.T;x=t-.5
    rng=np.random.default_rng(712);best=None
    for _ in range(550):
        ix=rng.choice(len(t),degree+1,replace=False)
        if np.ptp(t[ix])<.3 or len(np.unique(t[ix]))<=degree:continue
        p=np.polynomial.polynomial.polyfit(x[ix],d[ix],degree)
        predicted=np.polynomial.polynomial.polyval(x,p)
        if abs(p[0])>2 or max(abs(np.polynomial.polynomial.polyval(np.linspace(-.45,.45,31),p)))>max_offset:continue
        mask=abs(d-predicted)<1.5
        votes=np.zeros(241);np.maximum.at(votes,rows[mask].astype(int),weight[mask])
        score=votes.sum()
        if best is None or score>best[0]:best=score,p
    if best is None:return dict(supported=False,reason='no middle-anchored smooth fit')
    p=best[1]
    for _ in range(3):
        mask=abs(d-np.polynomial.polynomial.polyval(x,p))<1.5
        ids=[]
        for row in np.unique(rows[mask]):
            choices=np.flatnonzero(mask&(rows==row));ids.append(choices[np.argmax(weight[choices])])
        ids=np.array(ids)
        if len(ids)<20:return dict(supported=False,reason='too few unique profiles')
        proposed=np.polynomial.polynomial.polyfit(x[ids],d[ids],degree,w=np.sqrt(weight[ids]))
        if abs(proposed[0])>2:
            proposed[0]=np.clip(proposed[0],-2,2)
            design=np.column_stack([x[ids]**k for k in range(1,degree+1)])
            proposed[1:]=np.linalg.lstsq(design*np.sqrt(weight[ids,None]),(d[ids]-proposed[0])*np.sqrt(weight[ids]),rcond=None)[0]
        if max(abs(np.polynomial.polynomial.polyval(np.linspace(-.45,.45,31),proposed)))>max_offset:break
        p=proposed
    mask=abs(d-np.polynomial.polynomial.polyval(x,p))<1.5
    ids=[]
    for row in np.unique(rows[mask]):
        choices=np.flatnonzero(mask&(rows==row));ids.append(choices[np.argmax(weight[choices])])
    ids=np.array(ids);ids=ids[np.argsort(rows[ids])]
    groups=np.split(ids,np.flatnonzero(np.diff(rows[ids])>3)+1)
    groups=[g for g in groups if len(g)>=4];segments=[]
    for g in groups:
        ts=np.linspace(t[g[0]],t[g[-1]],len(g)*2)
        ds=np.polynomial.polynomial.polyval(ts-.5,p)*scale
        segments.append((a+ts[:,None]*(b-a)+ds[:,None]*normal).tolist())
    coverage=sum(t[g[-1]]-t[g[0]] for g in groups)
    return dict(supported=bool(coverage>=.2),coverage=float(coverage),support=len(ids)/241,segments=segments,
                coefficients=p.tolist(),degree=degree,anchor_limit_normalized_px=2,
                method='middle_anchored_polynomial',calibrated=False)


def outer_tracks(image,geometry,degree=2):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.asarray(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    return {s:fit_outer(*extract(lab,q,k,'outer',scale),scale,degree) for k,s in enumerate(['top','right','bottom','left'])}
