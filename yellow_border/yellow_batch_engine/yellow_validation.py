"""Recover chroma-only rejections using already supported yellow strips.

The original chroma statistic includes saturation, which can change under
reflections. This fallback validates hue, yellow-pixel coverage and both
physical transitions separately. It neither creates paths nor changes pixels,
and it cannot override missing-yellow, geometry or perspective rejections.
"""
import cv2
import numpy as np


def validate_yellow_illumination(image,seed,tracks):
    reasons=seed['geometry_reasons']
    evidence=dict(applied=False,original_reasons=list(reasons))
    def reject(reason):
        evidence['reason']=reason
        return evidence
    if not reasons or any(not r.startswith('chroma_dev ') for r in reasons):
        return reject('Recovery requires chroma-only rejection')
    for kind in ('outer','inner'):
        if len(tracks.get(kind,[]))!=4:return reject('Exactly eight boundary tracks are required')
        for side,t in enumerate(tracks[kind]):
            if not t or not t.get('retained') or t['support']<.85 or not t.get('resampling_stable') or t['covered_segments']<11:
                evidence['unsupported_boundary']=dict(kind=kind,side=side)
                return reject('All eight boundaries need strong distributed stable support')
    h,w=image.shape[:2]
    def sample(points):
        if not np.isfinite(points).all() or not ((points[...,0]>=0)&(points[...,0]<=w-1)&(points[...,1]>=0)&(points[...,1]<=h-1)).all():return None
        return cv2.remap(image,points[...,0].astype(np.float32),points[...,1].astype(np.float32),cv2.INTER_LINEAR)
    bands=[]
    for side in range(4):
        pair=[tracks[k][side] for k in ('outer','inner')]
        lo=max(.12,*(t['t_range'][0] for t in pair));hi=min(.88,*(t['t_range'][1] for t in pair))
        if hi-lo<.70:return reject('Insufficient shared observed span')
        ts=np.linspace(lo,hi,160);paths=[]
        for t in pair:
            p=np.asarray(t['points']);tp=np.linspace(*t['t_range'],len(p))
            paths.append(np.column_stack([np.interp(ts,tp,p[:,j]) for j in (0,1)]))
        outer,inner=paths;delta=inner-outer;length=np.linalg.norm(delta,axis=1)
        if np.min(length)<2:return reject('Yellow strip is too narrow')
        f=np.linspace(.25,.75,12)
        band=sample(outer[:,None,:]*(1-f[None,:,None])+inner[:,None,:]*f[None,:,None])
        if band is None:return reject('Border sampling leaves image')
        hsv=cv2.cvtColor(band,cv2.COLOR_BGR2HSV);H,S,V=[hsv[:,:,j] for j in (0,1,2)]
        yellow=(H>=12)&(H<=42)&(S>=55)&(V>=60)
        ref=np.median(band.astype(float),axis=1);n=delta/length[:,None];contrasts=[]
        for points in (outer-.18*seed['ppm']*n,inner+.18*seed['ppm']*n):
            colour=sample(points[:,None,:])
            if colour is None:return reject('Contrast probe leaves image')
            distance=np.linalg.norm(colour[:,0,:].astype(float)-ref,axis=1)
            contrasts.append(float((distance>=15).mean()))
        bands.append(dict(yellow_fraction=float(yellow.mean()),median_hue=float(np.median(H)),
            median_saturation=float(np.median(S)),median_value=float(np.median(V)),contrast_fractions=contrasts))
    evidence['bands']=bands
    if any(b['yellow_fraction']<.85 for b in bands):return reject('Every side must independently support yellow ink')
    if np.ptp([b['median_hue'] for b in bands])>8:return reject('Sides disagree in hue')
    if any(min(b['contrast_fractions'])<.80 for b in bands):return reject('Yellow strip lacks distinct transitions')
    evidence.update(applied=True,reason='Four yellow sides retain matching hue and distinct boundaries despite illumination variation')
    return evidence
