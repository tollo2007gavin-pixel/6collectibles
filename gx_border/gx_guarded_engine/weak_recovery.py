"""Experimental multi-window recovery of already localized weak GX sides.

No global threshold reduction: every weak side must have distributed evidence,
agree across three median scales, and have no competitive displaced hypothesis.
Only existing weak sides are changed. Strong fitted lines are kept exactly.
"""
import copy
import cv2
import numpy as np
from .legacy.holder_edge_evidence import profiles
from .legacy.holder_consensus import fit_windows
from .legacy.frame_recovery import combine_sides
from .legacy.frame_holders import validate_card_geometry
from .production_gx import centering_from_frame_quads


def agree_windows(fits,edge):
    if len(fits)!=3:raise ValueError('Three independent median scales required')
    for f in fits:
        bins=np.array(f['bins'])
        if len(bins)<11 or bins[0]>2 or bins[-1]<14 or any(np.sum((bins>=lo)&(bins<hi))<2 for lo,hi in [(0,6),(6,12),(12,17)]):
            raise ValueError('Local evidence does not span the side')
        if f['alternative_disagreement']>2:raise ValueError('Competing local transitions')
    movements=np.array([f['movement'] for f in fits])
    if np.max(np.ptp(movements,axis=0))>2:raise ValueError('Median scales disagree on boundary')
    if np.max(np.abs(movements))>(2 if edge=='inner' else 8):raise ValueError('Proposed move exceeds local recovery bound')
    return True


def recover_candidate(image,candidate):
    weak=[k for k,v in candidate['support'].items() if v<.65]
    if not 1<=len(weak)<=2 or any(candidate['support'][k]<.5 for k in weak):
        raise ValueError('Recovery requires at most two moderately supported sides')
    validate_card_geometry(image,candidate['outer'],candidate['inner'],candidate['stripe'])
    if min(candidate['stripe_support'].values())<.55:raise ValueError('Weak printed stripe')
    updated=copy.deepcopy(candidate);details={};alternatives={'inner':[],'outer':[]};choices={'inner':{},'outer':{}}
    for key in weak:
        side,edge=key.split('_');fits=[];quads=[]
        for half_window in [25,35,45]:
            records,H,q,ids,coord,axis,dim,prior=profiles(image,candidate,side,edge,half_window=half_window)
            models=fit_windows(records);score,model,bins,rms=models[0];ends=q[ids,axis]
            alternatives_delta=max(float(np.max(np.abs(np.polyval(m-model,ends)))) for s,m,b,r in models if s>=score-1.)
            fits.append(dict(window=half_window,bins=bins.tolist(),movement=np.polyval(model-prior,ends).tolist(),alternative_disagreement=alternatives_delta,rms=rms))
            adjusted=q.copy()
            for index in ids:adjusted[index,coord]=np.polyval(model,adjusted[index,axis])
            if side in ('right','bottom'):adjusted[:,coord]=dim-1-adjusted[:,coord]
            quads.append(cv2.perspectiveTransform(adjusted[None],np.linalg.inv(H))[0])
        agree_windows(fits,edge)
        # Middle scale is selected in advance, never by the resulting ratio.
        choices[edge][side]=len(alternatives[edge]);alternatives[edge].append(quads[1]);details[key]=fits
    for edge in alternatives:
        if alternatives[edge]:updated[edge]=combine_sides(candidate[edge],alternatives[edge],choices[edge]).tolist()
    validate_card_geometry(image,updated['outer'],updated['inner'],updated['stripe'])
    updated['centering']=centering_from_frame_quads(np.float32(updated['outer']),np.float32(updated['inner']))
    updated['independent_recovery']=dict(weak_sides=weak,scales=details,calibrated=False)
    return updated
