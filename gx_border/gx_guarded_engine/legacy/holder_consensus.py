"""Experimental independent-window consensus for weak holder boundaries."""
import itertools
import cv2,numpy as np
from .holder_edge_evidence import profiles
from .frame_recovery import combine_sides

def fit_windows(records):
    points=np.array([(i,r['t'],r['expected']+d,min(s,8.)**.5) for i,r in enumerate(records) for d,s in r['peaks']['lab'] if s>=1.])
    if len(points)<12:raise ValueError('Too few local median transitions')
    ids,t,x,w=points.T;models=[]
    for i,j in itertools.combinations(range(len(points)),2):
        if abs(t[i]-t[j])<.45*(records[-1]['t']-records[0]['t']):continue
        m=np.polyfit(t[[i,j]],x[[i,j]],1)
        if abs(m[0])>.015:continue
        for _ in range(3):
            residual=np.abs(x-np.polyval(m,t));chosen=[]
            for row in range(len(records)):
                near=np.flatnonzero((ids==row)&(residual<=1.5))
                if len(near):chosen.append(near[np.argmax(w[near])])
            if len(chosen)<8:break
            m=np.polyfit(t[chosen],x[chosen],1)
        residual=np.abs(x-np.polyval(m,t));chosen=[]
        for row in range(len(records)):
            near=np.flatnonzero((ids==row)&(residual<=1.5))
            if len(near):chosen.append(near[np.argmax(w[near])])
        if len(chosen)<8:continue
        chosen=np.array(chosen);bins=ids[chosen].astype(int)
        score=len(chosen)+.1*np.sum(w[chosen])
        if any(np.max(np.abs(np.polyval(m-old[1],[records[0]['t'],records[-1]['t']])))<.2 for old in models):continue
        models.append((score,m,bins,float(np.sqrt(np.mean(residual[chosen]**2)))))
    if not models:raise ValueError('No coherent median-window line')
    models.sort(key=lambda v:-v[0])
    return models

def propose_consensus(image,candidate,side,edge):
    records,H,q,indices,coord,axis,dimension,prior=profiles(image,candidate,side,edge)
    models=fit_windows(records);score,model,bins,rms=models[0]
    ends=q[indices,axis];movement=np.polyval(model-prior,ends)
    detail=dict(model=model.tolist(),prior=prior.tolist(),movement_px=movement.tolist(),bins=bins.tolist(),
                support=len(bins)/len(records),rms_px=rms,alternatives=[dict(score=s,model=m.tolist(),bins=b.tolist()) for s,m,b,r in models[:5]])
    updated=q.copy()
    for index in indices:updated[index,coord]=np.polyval(model,updated[index,axis])
    if side in ('right','bottom'):updated[:,coord]=dimension-1-updated[:,coord]
    source=cv2.perspectiveTransform(updated[None],np.linalg.inv(H))[0]
    return combine_sides(candidate[edge],[source],{side:0}),detail

def confirm_outer(image,candidate):
    """Confirm an existing outer line; never move it or repair an inner keyline."""
    weak=[key for key,value in candidate['support'].items() if value<.65]
    if len(weak)!=1 or not weak[0].endswith('_outer') or candidate['support'][weak[0]]<.5:
        raise ValueError('Confirmation requires one moderately supported outer side')
    side=weak[0].split('_')[0]
    records,H,q,indices,coord,axis,dimension,prior=profiles(image,candidate,side,'outer')
    models=fit_windows(records);score,model,bins,rms=models[0]
    ends=q[indices,axis]
    if len(bins)<12 or bins[0]>2 or bins[-1]<14 or any(np.sum((bins>=lo)&(bins<hi))<3 for lo,hi in [(0,6),(6,12),(12,17)]):
        raise ValueError('Median evidence does not span the physical side')
    if np.max(np.abs(np.polyval(model-prior,ends)))>2:
        raise ValueError('Median evidence disagrees with the existing physical line')
    alternatives=[m for s,m,b,r in models if s>=score-1.]
    disagreement=max(float(np.max(np.abs(np.polyval(m-model,ends)))) for m in alternatives)
    if disagreement>2:
        raise ValueError('Median edge hypotheses disagree')
    # Check the retained line itself, independently of the proposed median fit.
    hits=[i for i,r in enumerate(records) if any(abs(d)<=2 and s>=1 for d,s in r['peaks']['lab'])]
    if len(hits)<12:
        raise ValueError('Too few independent windows confirm the retained line')
    return dict(side=weak[0],confirmed=True,window_support=len(hits)/17,confirmed_windows=hits,
                fitted_windows=bins.tolist(),movement_px=np.polyval(model-prior,ends).tolist(),
                max_alternative_disagreement_px=disagreement,rms_px=rms,
                method='independent_local_median_confirmation',geometry_changed=False)
