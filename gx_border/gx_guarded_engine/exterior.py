"""GX source-canvas availability; missing fits are not missing borders.

Use an explicit two-unavailable-sides rejection using geometry only. GX has no
yellow colour band assumption. Unknown evidence never counts as unavailable.
"""
import copy
import numpy as np

SIDES=('top','right','bottom','left')


def inspect_exterior(image,candidates):
    h,w=image.shape[:2];evidence={s:[] for s in SIDES}
    for c in candidates:
        q=np.asarray(c.get('outer',[]),float)
        if q.shape!=(4,2) or not np.isfinite(q).all():continue
        if c.get('support') and min(c['support'].values())<.45:continue
        if c.get('stripe_support') and min(c['stripe_support'].values())<.55:continue
        for k,side in enumerate(SIDES):
            a,b=q[[k,(k+1)%4]];direction=b-a;length=np.linalg.norm(direction)
            if length<10:continue
            outward=np.array([direction[1],-direction[0]])/length
            # Probe actual source pixels, never the black padding of a warp.
            p=a+np.linspace(.15,.85,160)[:,None]*direction+3*outward
            unavailable=(p[:,0]<0)|(p[:,0]>w-1)|(p[:,1]<0)|(p[:,1]>h-1)
            evidence[side].append(float(unavailable.mean()))
    sides={s:dict(state='unavailable' if values and min(values)>=.8 else 'unknown',
                  unavailable_fraction=min(values) if values else None,
                  candidate_count=len(values)) for s,values in evidence.items()}
    missing=[s for s,v in sides.items() if v['state']=='unavailable']
    return dict(fail=len(missing)>=2,unavailable_sides=missing,unavailable_count=len(missing),
                minimum_unavailable_sides=2,
                sides=sides,method='source_canvas_outward_probe',stock_classifier=False)


def apply_exterior(result,gate):
    updated=dict(result,exterior_gate=gate)
    if gate['fail']:
        updated=copy.deepcopy(updated)
        updated['unvalidated_axes']=updated.get('axes',{})
        updated['axes']={};updated['status']='hard_failure'
        updated['failure_code']='insufficient_exterior_background'
        updated['rejection_category']='missing_exterior_edges'
        updated['rejection_label']='Rejected: 2+ exterior edges unavailable'
        updated['margin_report_allowed']=False
        updated['reasons']=list(updated.get('reasons',[]))+[
            'Physical exterior unavailable at source image boundary on '+
            str(gate['unavailable_count'])+'/4 sides: '+', '.join(gate['unavailable_sides'])]
    return updated
