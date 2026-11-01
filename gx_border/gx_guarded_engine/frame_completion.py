"""Fallback localization for cards that nearly fill the source image.

Canvas-relative rectangles are search regions, never measured card edges.
Multiple independently initialized GX fits must agree on all eight observed
tracks. The perspective frame is then rebuilt from supported physical tracks.
"""
import copy,cv2,numpy as np
from .legacy.frame_holders import structured_stripe,fit_frame,candidate_quality,validate_card_geometry
from .production_gx import centering_from_frame_quads
from .tracks import SIDES,measure,rebuild_frame
from .observed_recovery import recover_missing_outer
from .quality_evidence import disagreement


def extent_seeds(image):
    h,w=image.shape[:2]
    if not .5<w/h<.95:return []
    return [np.float32([[f*w,f*h],[(1-f)*w,f*h],[(1-f)*w,(1-f)*h],[f*w,(1-f)*h]])
            for f in [.015,.035,.055,.075]]


def complete_frame(image):
    from .modern import preferred_tracks
    records=[];attempts=[]
    for index,seed in enumerate(extent_seeds(image)):
        try:
            stripe,ss=structured_stripe(image,seed)
            if min(ss.values())<.55:raise ValueError('insufficient GX stripe evidence')
            outer,inner,support,_=fit_frame(image,seed,stripe)
            g=dict(seed_index=index,method='image_extent_source_consensus',outer=outer.tolist(),inner=inner.tolist(),
                   stripe=stripe.tolist(),support=support,stripe_support=ss,centering=centering_from_frame_quads(outer,inner))
            score,_=candidate_quality(image,g)
            tracks,_=recover_missing_outer(image,g,preferred_tracks(image,g))
            if not all(v.get('measured') for v in measure(tracks,g['outer']).values()):raise ValueError('incomplete observed tracks')
            rebuilt=rebuild_frame(tracks,np.asarray(g['outer']))
            if rebuilt is None:raise ValueError('physical tracks cannot rebuild a supported perspective frame')
            validate_card_geometry(image,rebuilt,g['inner'],g['stripe'])
            g['outer']=rebuilt.tolist();g['centering']=centering_from_frame_quads(rebuilt,np.asarray(g['inner']))
            records.append(dict(geometry=g,tracks=tracks,score=score))
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:attempts.append(dict(seed=index,reason=str(error)))
    diagnostic=dict(method='independent extent seeds and eight observed-track consensus',qualified_seeds=len(records),attempts=attempts)
    if len(records)<3:return None,dict(diagnostic,reason='fewer than three supported independent frames')
    worst=0.;comparisons=[]
    for i,row in enumerate(records):
        for other in records[:i]:
            g=row['geometry'];q=np.asarray(g['outer']);scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
            for k,side in enumerate(SIDES):
                for kind in ['outer','inner']:
                    a,b=np.asarray(g[kind])[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
                    c=disagreement(row['tracks'][kind][side],other['tracks'][kind][side],a,b,n,scale)
                    comparisons.append(dict(side=side,kind=kind,**c))
                    worst=max(worst,c.get('p95_normalized_px',float('inf')))
                    if c['common_profiles']<54 or c.get('p95_normalized_px',float('inf'))>3:
                        return None,dict(diagnostic,reason='supported source tracks disagree across extent seeds',comparisons=comparisons)
    selected=max(records,key=lambda r:r['score'])
    return selected,dict(diagnostic,reason='all eight observed boundaries agree across independent seeds',
                         max_p95_normalized_px=worst,comparisons=comparisons)
