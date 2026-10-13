"""Narrow colour-gate recovery for one desaturated but supported yellow side.

This validates existing source-pixel paths; it never moves or creates a fit.
"""
import cv2
import numpy as np


def validate_pale_border(image,seed,tracks):
    reasons=seed['geometry_reasons']
    evidence=dict(applied=False,original_reasons=list(reasons))
    def reject(reason):
        evidence['reason']=reason
        return evidence
    if not reasons or any(not r.startswith(('yellow_min ','chroma_dev ')) for r in reasons):
        return reject('Recovery requires colour-only rejection')
    for kind in ('outer','inner'):
        for t in tracks[kind]:
            if (not t or not t.get('retained') or t['support']<.90 or
                    not t.get('resampling_stable') or t['covered_segments']<10):
                return reject('All eight boundaries need strong distributed, resize-stable support')
    h,w=image.shape[:2]
    def sample(points):
        if not np.isfinite(points).all() or not ((points[...,0]>=0)&(points[...,0]<=w-1)&
                (points[...,1]>=0)&(points[...,1]<=h-1)).all():return None
        return cv2.remap(image,points[...,0].astype(np.float32),points[...,1].astype(np.float32),cv2.INTER_LINEAR)
    bands=[];boundaries=[]
    for side in range(4):
        pair=[tracks[k][side] for k in ('outer','inner')]
        lo=max(.12,*(t['t_range'][0] for t in pair));hi=min(.88,*(t['t_range'][1] for t in pair))
        if hi-lo<.70:return reject('Insufficient shared observed span')
        ts=np.linspace(lo,hi,160);paths=[]
        for t in pair:
            p=np.asarray(t['points']);tp=np.linspace(*t['t_range'],len(p))
            paths.append(np.column_stack([np.interp(ts,tp,p[:,j]) for j in (0,1)]))
        outer,inner=paths;f=np.linspace(.25,.75,12)
        band=sample(outer[:,None,:]*(1-f[None,:,None])+inner[:,None,:]*f[None,:,None])
        if band is None:return reject('Border sampling leaves the image')
        hsv=cv2.cvtColor(band,cv2.COLOR_BGR2HSV);H,S,V=[hsv[:,:,j] for j in (0,1,2)]
        bands.append(dict(yellow_fraction=float(((H>=12)&(H<=42)&(S>=55)&(V>=60)).mean()),
            yellow_hue_fraction=float(((H>=12)&(H<=42)).mean()),
            median_hue=float(np.median(H)),median_saturation=float(np.median(S)),median_value=float(np.median(V))))
        boundaries.append((outer,inner,band))
    evidence['bands']=bands
    healthy=[i for i,b in enumerate(bands) if b['yellow_fraction']>=.80]
    if len(healthy)!=3:return reject('Exactly three sides must independently support yellow ink')
    side=next(i for i in range(4) if i not in healthy);pale=bands[side]
    hue=float(np.median([bands[i]['median_hue'] for i in healthy]))
    saturation=float(np.median([bands[i]['median_saturation'] for i in healthy]))
    if any(abs(bands[i]['median_hue']-hue)>8 for i in healthy):
        return reject('The three yellow sides disagree in hue')
    if not (pale['yellow_hue_fraction']>=.90 and abs(pale['median_hue']-hue)<=8 and
            8<=pale['median_saturation']<min(55,.60*saturation) and pale['median_value']>=60):
        return reject('Weak side is not a supported desaturated version of the yellow ink')
    outer,inner,band=boundaries[side];delta=inner-outer;length=np.linalg.norm(delta,axis=1)
    if np.min(length)<2:return reject('Pale strip is too narrow to validate')
    inward=delta/length[:,None];ref=np.median(band.astype(float),axis=1)
    contrasts=[]
    for points in (outer-.18*seed['ppm']*inward,inner+.18*seed['ppm']*inward):
        colour=sample(points[:,None,:])
        if colour is None:return reject('Contrast probe leaves the image')
        distance=np.linalg.norm(colour[:,0,:].astype(float)-ref,axis=1)
        contrasts.append(dict(median_distance=float(np.median(distance)),supported_fraction=float((distance>=15).mean())))
    evidence.update(side=side,contrasts=contrasts)
    if any(c['supported_fraction']<.85 for c in contrasts):
        return reject('Pale strip lacks distinct artwork and background transitions')
    evidence.update(applied=True,reason='One desaturated yellow side supported by both physical transitions and three yellow sides')
    return evidence
