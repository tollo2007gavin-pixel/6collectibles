"""Reconsider wide physical strips only after the existing pipeline rejects a frame."""
import copy
import numpy as np
from .background_recovery import should_attempt
from .cardcv.pipeline import analyze,rejection_reasons,DEFAULT_GATES
from .cardcv.orientation import card_orientation
from .exterior_gate import inspect_exterior_background,apply_exterior_gate


def recover_boundary_selection(image,analysis,*,measurement_frame='initial'):
    # Preserve any already usable axis, hard crop failure, and otherwise valid
    # frame rejected only by perspective. New candidates cannot displace them.
    if not should_attempt(analysis):return analysis
    from .compare import fit_seed,measure_seed_tracks,combine_trials,combine_methods
    result=copy.deepcopy(analysis)
    gates=dict(DEFAULT_GATES,keystone=1.)
    best,candidates=analyze(image,meta=dict(capture='photo',holder='unknown',retry_wide_pairs=True),
                            gates=gates,fit_tracks=False)
    evidence=dict(applied=False,candidate_diagnostics=[
        {k:c[k] for k in ('method','ok','reason','reject_reasons','score') if k in c} for c in candidates])
    result['boundary_selection_recovery']=evidence
    if best is None:
        evidence['reason']='Wider pair retry did not produce a physically validated frame'
        return result
    if not best['result'].get('pair_width_recovery'):
        evidence['reason']='Selected frame does not use a recovered wider boundary pair'
        return result
    supporters={c['method'].split('+')[0] for c in candidates if c.get('ok') and
        not rejection_reasons(c,gates,meta=dict(holder='unknown')) and
        max(abs(c['centering'][key]-best['centering'][key]) for key in ('left_pct','top_pct'))<1.5}
    if len(supporters)<2:
        evidence['reason']='Recovered frame lacks agreement across localization variants'
        return result
    r=best['result'];top,_=card_orientation(image,r['outer'],r['inner'])
    seed=dict(method=best['method']+'+boundary_recovery',outer=np.asarray(r['outer']).tolist(),
        inner=np.asarray(r['inner']).tolist(),ppm=r['px_per_mm'],top_side=int(top),
        keystone=best['diag']['keystone'],geometry_reasons=rejection_reasons(best,DEFAULT_GATES,meta=dict(holder='unknown')))
    search_seed=dict(seed,geometry_reasons=[s for s in seed['geometry_reasons'] if not s.startswith('fitted perspective ')])
    modes={}
    for mode in ('adaptive','forced'):
        trial=fit_seed(image,search_seed,mode,measurement_frame=measurement_frame)
        if search_seed['geometry_reasons']!=seed['geometry_reasons']:
            trial=measure_seed_tracks(image,seed,trial['candidates'],measurement_frame=measurement_frame)
        modes[mode]=combine_trials([trial])
    proposal=dict(seeds=[seed],methods=modes)
    gate=inspect_exterior_background(image,proposal)
    modes={m:apply_exterior_gate(v,gate) for m,v in modes.items()}
    usable=all(any(a['resolved'] or (a.get('frame_diagnostics',{}).get('initial_percentage') is not None and
        a['reasons'] and all(s.startswith('fitted perspective ') for s in a['reasons']))
        for a in (m['axes'][axis] for m in modes.values())) for axis in ('lr','tb'))
    if gate['fail'] or not usable:
        evidence['reason']='Recovered frame lacks fully validated measurement boundaries'
        return result
    evidence.update(applied=True,reason='Wider yellow strip supported by agreeing frames and validated boundary tracks',
        original_seeds=analysis.get('seeds',[]),agreeing_methods=sorted(supporters),
        recovered_pairs=best['result']['pair_width_recovery'])
    result.update(seeds=[seed],methods=modes,image_gate=gate)
    if 'combined' in result:result['combined']=combine_methods(modes)
    return result
