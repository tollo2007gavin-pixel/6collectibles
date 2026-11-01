"""Experimental outer-edge continuity from a supported middle anchor.

This is a shadow comparison, not a replacement for physical-edge validation.
No card names, scan ratios, user labels, or preferred centering are inputs.
"""
import cv2
import numpy as np
from .tracks import extract


def trace(candidates,a,b,normal,scale):
    if len(candidates)<20:return dict(supported=False,reason='too few outer profiles')
    groups={int(row):candidates[candidates[:,0]==row] for row in np.unique(candidates[:,0])}
    central=[row for row in groups if abs(groups[row][0,1]-.5)<.12 and np.any(abs(groups[row][:,2])<=2)]
    if len(central)<20:return dict(supported=False,reason='no supported middle anchor')
    # Only anchor at an observed profile; the prior identifies which of several
    # parallel transitions is the physical edge at the middle of the side.
    anchor=min(central,key=lambda row:abs(groups[row][0,1]-.5))
    initial=groups[anchor];cost=(initial[:,2]/2)**2
    cost[abs(initial[:,2])>2]=np.inf
    def walk(indices):
        previous=initial;loss=cost.copy();history=[];last=anchor
        for row in indices:
            current=groups[row];gap=abs(row-last)
            if gap>8:break
            delta=current[:,2,None]-previous[:,2][None,:]
            transition=(delta**2)/gap
            transition[abs(delta)>3.+.35*gap]=np.inf
            emission=-.35*np.log(np.maximum(current[:,3],.01))
            combined=transition+loss[None,:]
            choices=np.argmin(combined,axis=1);next_loss=combined[np.arange(len(current)),choices]+emission
            if not np.isfinite(next_loss).any():continue
            history.append((row,current,choices));loss=next_loss;previous=current;last=row
        if not history:return []
        state=int(np.argmin(loss));selected=[]
        for row,current,choices in history[::-1]:
            selected.append(current[state]);state=int(choices[state])
        selected.append(initial[state]);return selected[::-1]
    before=walk(sorted((r for r in groups if r<anchor),reverse=True))
    after=walk(sorted(r for r in groups if r>anchor))
    selected=np.array(before+after)
    if len(selected)<40:return dict(supported=False,reason='continuity does not span enough observed profiles')
    selected=selected[np.argsort(selected[:,0])];_,ids=np.unique(selected[:,0],return_index=True);selected=selected[ids]
    rows,t,d,w=selected.T
    # Local fits avoid a global polynomial moving a well-supported middle to
    # accommodate one end. Do not extrapolate beyond observed samples.
    smooth=[]
    for ti in t:
        ix=abs(t-ti)<=.045
        degree=min(2,int(ix.sum())-1)
        if degree<1:smooth.append(float(d[np.argmin(abs(t-ti))]));continue
        p=np.polynomial.polynomial.polyfit(t[ix]-ti,d[ix],degree,w=np.sqrt(w[ix]))
        smooth.append(float(p[0]))
    smooth=np.array(smooth)
    # Suppress sections where smoothing no longer follows the traced evidence.
    keep=abs(smooth-d)<=1.5
    selected_ix=np.flatnonzero(keep)
    pieces=np.split(selected_ix,np.flatnonzero(np.diff(rows[selected_ix])>3)+1)
    pieces=[p for p in pieces if len(p)>=4]
    segments=[(a+t[p,None]*(b-a)+smooth[p,None]*scale*normal).tolist() for p in pieces]
    coverage=sum(float(t[p[-1]]-t[p[0]]) for p in pieces)
    bins=np.histogram(t[keep],bins=np.linspace(.05,.95,13))[0]
    return dict(supported=bool(coverage>=.2),segments=segments,coverage=coverage,support=float(keep.sum()/241),
                method='middle_anchored_continuity_local_curve',anchor_fraction=float(groups[anchor][0,1]),
                observations=(a+t[:,None]*(b-a)+d[:,None]*scale*normal).tolist(),
                observed_fractions=t.tolist(),observed_offsets=d.tolist(),smoothed_offsets=smooth.tolist(),
                distributed_counts=bins.tolist(),rms_normalized_px=float(np.sqrt(np.mean((smooth[keep]-d[keep])**2))))


def outer_tracks(image,geometry):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.asarray(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    result={}
    for k,side in enumerate(['top','right','bottom','left']):
        c,a,b,n=extract(lab,q,k,'outer',scale)
        result[side]=trace(c,a,b,n,scale)
    return result
