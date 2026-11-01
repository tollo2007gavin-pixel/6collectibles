"""Recovery from source pixels after a legacy localization gate stops early.

Weak stripe fits and canvas rectangles are search seeds, never approved edges.
All supported competing frames must converge before any recovery is released.
"""
import copy,cv2,numpy as np
from .tracks import SIDES,extract,fit,measure,rebuild_frame
from .quality_evidence import disagreement
from .observed_recovery import recover_missing_outer,recover_missing_inner
from .legacy.frame_holders import structured_stripe,fit_frame,validate_card_geometry
from .frame_completion import extent_seeds


def extend_outer(image,g,tracks):
    updated=copy.deepcopy(tracks);changes=[]
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.asarray(g['outer'],float)
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    for k,s in enumerate(SIDES):
        old=tracks['outer'][s]
        if old.get('supported') and old.get('coverage',0)>=.55:continue
        args=extract(lab,q,k,'outer',scale,allow_partial_profiles=True)
        choices=[fit(*args,scale,d) for d in [1,2]]
        if any(not t.get('supported') or t.get('coverage',0)<.55 for t in choices):continue
        _,a,b,n=args
        cmp=disagreement(*choices,a,b,n,scale)
        if cmp['common_profiles']<54 or cmp.get('p95_normalized_px',float('inf'))>3:continue
        candidate=max(choices,key=lambda t:t['coverage'])
        if old.get('supported'):
            c=disagreement(old,candidate,a,b,n,scale)
            if c['common_profiles']<54 or c.get('p95_normalized_px',float('inf'))>3:continue
        updated['outer'][s]=candidate;changes.append(dict(side=s,method='agreeing straight/curve source observations',comparison=cmp))
    return updated,changes


def recover_frame(image,candidates=()):
    from .modern import preferred_tracks
    from .final_frame import certify_frame
    from .holder_topology import hypotheses,baseline_evidence
    from .quality_evidence import inspect_quality
    from .sparse_keyline import recover as recover_sparse_keyline
    seeds=list(candidates);attempts=[];records=[]
    initial=extent_seeds(image)
    # Intermediate search windows help avoid a single stripe seed landing on
    # glare or a sleeve. They do not change the required source evidence.
    if initial:
        h,w=image.shape[:2]
        initial += [np.float32([[f*w,f*h],[(1-f)*w,f*h],[(1-f)*w,(1-f)*h],[f*w,(1-f)*h]]) for f in [.025,.045,.065]]
    for index,seed in enumerate(initial):
        try:
            stripe,ss=structured_stripe(image,seed)
            outer,inner,support,_=fit_frame(image,seed,stripe)
            # Stripe support may be weak; source evidence must subsequently
            # establish every physical boundary and the reported inner tracks.
            seeds.append(dict(seed_index=index,method='source_extent_recovery',outer=outer.tolist(),
                              inner=inner.tolist(),stripe=stripe.tolist(),support=support,stripe_support=ss))
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:attempts.append(dict(stage='seed',index=index,reason=str(error)))
    for index,g in enumerate(seeds):
        try:
            tracks=preferred_tracks(image,g)
            tracks,_=recover_missing_outer(image,g,tracks)
            tracks,changes=extend_outer(image,g,tracks)
            tracks,_=recover_missing_inner(image,g,tracks)
            frame=rebuild_frame(tracks,np.asarray(g['outer']))
            if frame is None:raise ValueError('insufficient source-observed physical frame')
            g=copy.deepcopy(g);g['outer']=frame.tolist()
            validate_card_geometry(image,g['outer'],g['inner'],g['stripe'])
            e=dict(hypotheses=hypotheses(image,g),baseline=baseline_evidence(image,g))
            quality=inspect_quality(image,g,tracks,e)
            proof=certify_frame(image,g,tracks,quality,e)
            if not proof['verified']:raise ValueError(proof['reason'])
            tracks,inner_changes=recover_sparse_keyline(image,g,tracks)
            axes=measure(tracks,g['outer'])
            if not any(a.get('measured') for a in axes.values()):raise ValueError('no axis has common observed support')
            records.append(dict(geometry=g,tracks=tracks,certificate=proof,source_index=index,outer_recovery=changes,sparse_keyline_recovery=inner_changes))
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:attempts.append(dict(stage='source',index=index,reason=str(error)))
    proof=dict(method='source-track recovery from retained and extent candidates',attempts=attempts,qualified_candidates=len(records),
               qualified_records=records)
    if len(records)<2:return None,dict(proof,reason='fewer than two independently initialized source-supported frames')
    for i,row in enumerate(records):
        g=row['geometry'];q=np.asarray(g['outer']);scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
        for other in records[:i]:
            for k,s in enumerate(SIDES):
                a,b=q[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
                for kind in ['outer','inner']:
                    c=disagreement(row['tracks'][kind][s],other['tracks'][kind][s],a,b,n,scale)
                    if c['common_profiles']<54 or c.get('p95_normalized_px',float('inf'))>3:
                        return None,dict(proof,reason='source-supported candidate frames still disagree',comparison=c,side=s,kind=kind)
    return records[0],dict(proof,reason='independent source-supported frames converge on all eight boundaries')
