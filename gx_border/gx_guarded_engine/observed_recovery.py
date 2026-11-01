"""Opt-in recovery of missing outer tracks from valid source pixels only."""
import copy
import cv2
import numpy as np
from .tracks import extract
from .anchored_polynomial import fit_outer
from .horizontal_refinement import select_outer

def recover_missing_outer(image,geometry,prior_tracks):
    tracks=copy.deepcopy(prior_tracks);lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.array(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260;rescued=[]
    for k,side in enumerate(['top','right','bottom','left']):
        if tracks['outer'][side].get('supported'):continue
        args=extract(lab,q,k,'outer',scale,allow_partial_profiles=True)
        curve=fit_outer(*args,scale,2);line=fit_outer(*args,scale,1);candidate,_=select_outer(curve,line)
        if not candidate.get('supported') and line.get('supported'):candidate=line
        if candidate.get('supported'):tracks['outer'][side]=candidate;rescued.append(side)
    return tracks,rescued


def recover_missing_inner(image,geometry,prior_tracks):
    """Recover absent keylines from valid partial profiles, never padded pixels.

    An off-image tail of a search strip is not a reason to discard an in-image
    GX band/shoulder/keyline sequence. Supported existing keylines are untouched.
    """
    from .keyline_selection import extend_observed_support,sampled_track
    tracks=copy.deepcopy(prior_tracks);rescued=[]
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');q=np.asarray(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    for k,side in enumerate(['top','right','bottom','left']):
        if tracks['inner'][side].get('supported'):continue
        c,a,b,n=extract(lab,geometry['inner'],k,'inner_colour',scale,allow_partial_profiles=True)
        line=extend_observed_support(fit_outer(c,a,b,n,scale,1),c,a,b,n,scale)
        curve=extend_observed_support(fit_outer(c,a,b,n,scale,2),c,a,b,n,scale)
        grid,l=sampled_track(line,a,b,n,scale);_,v=sampled_track(curve,a,b,n,scale)
        common=np.isfinite(l)&np.isfinite(v)
        line_seen=np.isfinite(l)
        enough=(line.get('supported') and line.get('coverage',0)>=.3 and line_seen.sum()>=54
                and min(np.sum(line_seen&(grid<.5)),np.sum(line_seen&(grid>=.5)))>=12)
        agreement=bool(common.sum()>=54 and np.quantile(abs(l[common]-v[common]),.95)<=3)
        # A weaker curve can jump to a parallel branch after resampling. Do not
        # give it a veto over a simpler model with more actual signature votes.
        # Both candidates count the same full set of source observations.
        dominates=(curve.get('supported') and line.get('support',0)>=curve.get('support',0)+10/241
                   and line.get('coverage',0)>=curve.get('coverage',0)+.1)
        if not enough or not (agreement or dominates):continue
        line['recovery_method']='valid partial GX signature profiles; '+('straight/curve agreement' if agreement else 'straight model has greater observed signature support')
        tracks['inner'][side]=line;rescued.append(side)
    return tracks,rescued


def audit_inner_recovery(image,geometry,prior_tracks,result_tracks,sides):
    from .quality_evidence import disagreement
    checks=[]
    for label,im in [('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                     ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]:
        factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        g={k:(np.asarray(geometry[k])*factor).tolist() for k in ['outer','inner']}
        prior=copy.deepcopy(prior_tracks)
        for kind in prior.values():
            for t in kind.values():t['segments']=[(np.asarray(s)*factor).tolist() for s in t.get('segments',[])]
        recovered,_=recover_missing_inner(im,g,prior)
        for side in sides:
            t=copy.deepcopy(recovered['inner'][side]);t['segments']=[(np.asarray(s)/factor).tolist() for s in t.get('segments',[])]
            k=['top','right','bottom','left'].index(side);a,b=np.asarray(geometry['inner'])[[k,(k+1)%4]]
            v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v);q=np.asarray(geometry['outer'])
            scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
            cmp=disagreement(result_tracks['inner'][side],t,a,b,n,scale)
            checks.append(dict(variant=label,side=side,comparison=cmp,
                stable=bool(t.get('supported') and cmp['common_profiles']>=54 and cmp.get('p95_normalized_px',float('inf'))<=3)))
    return dict(stable=all(c['stable'] for c in checks),checks=checks,scope='recovered keyline only; fixed/scaled seed')
