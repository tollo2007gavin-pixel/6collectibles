"""Fallback localization that detaches thin, similarly coloured backgrounds.

The cleaned masks propose geometry only. Measurements and validation always
sample the original image; accepted fits and hard failures never enter here.
"""
import copy
import cv2
import numpy as np
from .cardcv.localize import MASKERS,_mask_yellow,_mask_wb_yellow,_mask_vivid_yellow
from .cardcv.pipeline import analyze,rejection_reasons,DEFAULT_GATES
from .cardcv.orientation import card_orientation
from .exterior_gate import inspect_exterior_background,apply_exterior_gate


def _opened(masker,fraction):
    def mask(image):
        k=max(3,int(max(image.shape[:2])*fraction)|1)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        return cv2.morphologyEx(masker(image),cv2.MORPH_OPEN,kernel)
    return mask


RECOVERY_POOL=[]
for _name,_mask in [('yellow',_mask_yellow),('wb_yellow',_mask_wb_yellow),('vivid_yellow',_mask_vivid_yellow)]:
    for _fraction in (.006,.01):
        _key=_name+'_background_open_'+str(_fraction)
        MASKERS[_key]=_opened(_mask,_fraction)
        RECOVERY_POOL.append(_key)


def should_attempt(analysis):
    if analysis.get('image_gate',{}).get('fail'):return False
    if any(a['resolved'] for m in analysis['methods'].values() for a in m['axes'].values()):return False
    seeds=analysis.get('seeds',[])
    # A border-support failure on an otherwise accepted frame is a different
    # category. A perspective-only rejection is not background contamination.
    return not seeds or all(any(not r.startswith('fitted perspective ') for r in s['geometry_reasons']) for s in seeds)


def recover_background(image,analysis,*,measurement_frame='initial'):
    if not should_attempt(analysis):return analysis
    from .compare import fit_seed,combine_trials,combine_methods
    result=copy.deepcopy(analysis)
    # Geometry repair must not be suppressed by the old perspective gate.
    # The actual configured gate is reapplied below before reporting margins.
    geometry_gates=dict(DEFAULT_GATES,keystone=1.0)
    best,candidates=analyze(image,meta=dict(capture='photo',holder='unknown'),
        pool=RECOVERY_POOL,gates=geometry_gates,fit_tracks=False)
    info=dict(applied=False,method='opened_background_masks',candidate_diagnostics=[
        {k:c[k] for k in ('method','ok','reason','reject_reasons','score') if k in c} for c in candidates])
    result['background_recovery']=info
    if best is None:
        info['reason']='No alternative frame passed physical validation'
        return result
    supporters={c['method'].split('+')[0] for c in candidates if c.get('ok') and
        not rejection_reasons(c,geometry_gates,meta=dict(holder='unknown')) and
        max(abs(c['centering'][key]-best['centering'][key]) for key in ('left_pct','top_pct'))<1.5}
    if len(supporters)<2:
        info['reason']='Alternative localization lacks agreement across mask variants'
        return result
    r=best['result'];top,_=card_orientation(image,r['outer'],r['inner'])
    seed=dict(method=best['method']+'+background_recovery',outer=np.asarray(r['outer']).tolist(),
        inner=np.asarray(r['inner']).tolist(),ppm=r['px_per_mm'],top_side=int(top),
        keystone=best['diag']['keystone'],geometry_reasons=rejection_reasons(best,DEFAULT_GATES,meta=dict(holder='unknown')))
    # Fit while keeping perspective rejection out of the *search* only. Actual
    # axes are measured again with all configured rejection reasons restored.
    search_seed=dict(seed,geometry_reasons=[reason for reason in seed['geometry_reasons'] if not reason.startswith('fitted perspective ')])
    modes={}
    from .compare import measure_seed_tracks
    for mode in ('adaptive','forced'):
        trial=fit_seed(image,search_seed,mode,measurement_frame=measurement_frame)
        if seed['geometry_reasons']!=search_seed['geometry_reasons']:
            trial=measure_seed_tracks(image,seed,trial['candidates'],measurement_frame=measurement_frame)
        modes[mode]=combine_trials([trial])
    proposal=dict(seeds=[seed],methods=modes)
    gate=inspect_exterior_background(image,proposal)
    modes={m:apply_exterior_gate(v,gate) for m,v in modes.items()}
    # Each axis must have a valid measurement or only the configured perspective
    # rejection left. Geometry, colour, missing-edge and crop failures remain.
    usable=all(any(a['resolved'] or (a.get('frame_diagnostics',{}).get('initial_percentage') is not None and
        a['reasons'] and all(reason.startswith('fitted perspective ') for reason in a['reasons']))
        for a in (m['axes'][axis] for m in modes.values())) for axis in ('lr','tb'))
    combined=combine_methods(modes)
    if gate['fail'] or not usable:
        info['reason']='Alternative frame lacks fully validated source-pixel boundary fits'
        info['fit_status']=combined['status']
        return result
    info.update(applied=True,reason='Mask variants agree and all measurement boundaries pass validation',
        original_seeds=analysis.get('seeds',[]),agreeing_candidates=len(supporters),agreeing_methods=sorted(supporters))
    result.update(seeds=[seed],methods=modes,image_gate=gate)
    # Callers may provide cached analyses with a combined result.
    if 'combined' in result:result['combined']=combined
    return result
