"""Fresh replay of stripe selection -> colour fusion -> recovery -> segments.

Ported from saved scripts. No cached image geometry or card-specific targets.
"""
import cv2
import numpy as np
from .legacy import frame_trial,frame_adaptive
from .legacy.localizer import proposals
from .legacy.frame_recovery import combine_sides,recover_frame
from .legacy.frame_keyline_segments import refine_vertical_keylines
from .legacy.gx_before import centering_from_frame_quads

SIDES=['left','right','top','bottom']
INDICES={'left':[0,3],'right':[1,2],'top':[0,1],'bottom':[3,2]}


def inset(q,side):
    k=0 if side in ['left','right'] else 1
    v=q[INDICES[side],k]
    return (1259 if k==0 else 1759)-v if side in ['right','bottom'] else v


def measure_bare(image,production=None,quality=None):
    candidates=[]
    for seed in proposals(image):
        try:
            stripe,ss=frame_trial.stripe_frame(image,seed)
            if min(ss.values())<.5:continue
            outer,inner,support,_=frame_trial.fit_frame(image,seed,stripe)
            score=min(support.values())+.5*np.mean(list(support.values()))+.3*min(ss.values())
            candidates.append(dict(seed=seed,stripe=stripe,outer=outer,inner=inner,support=support,score=score))
        except (ValueError,cv2.error,np.linalg.LinAlgError):continue
    if not candidates:
        return refine_vertical_keylines(image,recover_frame(image,None,known_bare=True),known_bare=True)
    best=max(candidates,key=lambda c:c['score'])
    colour_outer,colour_inner,_,_=frame_adaptive.fit_frame(image,best['seed'],best['stripe'])
    H=cv2.getPerspectiveTransform(np.float32(best['stripe']),np.float32([[0,0],[1259,0],[1259,1759],[0,1759]]))
    def transform(q):return cv2.perspectiveTransform(np.float32(q)[None],H)[0]
    co,ci=transform(colour_outer),transform(colour_inner)
    ro=[];ri=[];reasons=[]
    if production is not None:
        bo,bi=transform(production.outer_quad),transform(production.inner_quad)
        diag=production.fit_diagnostics
        trusted=quality['accepted'] and min(v['support'] for v in diag['outer'].values())>=.9
        for side in SIDES:
            o,inn=inset(bo,side),inset(bi,side)
            so,si=diag['outer'][side]['support'],diag['inner'][side]['support']
            if (not (-65<o.mean()<-13) or np.ptp(o)>12) and not trusted:ro.append(side)
            if ((inn.mean()>31 or ((inn.mean()<16 or np.ptp(inn)>10 or si<.55) and not trusted)) and ((-65<o.mean()<-13) or not trusted)):ri.append(side)
            if side not in ro and so<.9 and abs(o.mean()-inset(co,side).mean())>1:
                if best['support'][side+'_outer']>so+.08:ro.append(side)
                else:reasons.append(side+' outer ambiguous')
            if (best['support'][side+'_outer'] if side in ro else so)<.60:reasons.append(side+' outer weak')
        fo=combine_sides(bo,[co],{s:0 for s in ro});fi=combine_sides(bi,[ci],{s:0 for s in ri})
    else:fo,fi=co,ci
    outer=cv2.perspectiveTransform(fo[None],np.linalg.inv(H))[0]
    inner=cv2.perspectiveTransform(fi[None],np.linalg.inv(H))[0]
    geometry=dict(outer=outer.tolist(),inner=inner.tolist(),centering=centering_from_frame_quads(outer,inner),reasons=reasons,
                  replace_outer=ro,replace_inner=ri,fresh_candidate_count=len(candidates))
    geometry=recover_frame(image,geometry,best['stripe'],candidates,known_bare=True)
    return refine_vertical_keylines(image,geometry,known_bare=True)
