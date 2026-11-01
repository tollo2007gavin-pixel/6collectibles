"""Certify final source boundaries independently of early localization status.

Search frames and projected intersections never create observed margin samples.
These checks are provisional GX evidence rules, not calibrated accuracy bounds.
"""
import copy
import cv2
import numpy as np
from .tracks import SIDES,project,rebuild_frame,extract,fit,measure
from .holder_topology import FRAME,baseline_evidence
from .soft_seam import certify


def reobserve_curve(image,geometry,track,k,*,resampling_uncertainty_px=0.):
    """Reobserve only existing fragments in two displaced source windows.

    Search follows the candidate curve; it does not assume that a curved card
    edge stays inside a narrow strip around a straight projected side.
    """
    from .keyline_selection import sampled_track
    q=np.asarray(geometry['outer'],float);a,b=q[[k,(k+1)%4]];v=b-a
    n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    t,d=sampled_track(track,a,b,n,scale);seen=np.isfinite(d)
    t=t[seen];d=d[seen]
    if len(t)<54:return dict(verified=False,reason='too few observed profiles')
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');checks=[]
    for shift in [-3.,3.]:
        offsets=np.arange(-6.,6.01,.5)+shift
        pts=a+t[:,None,None]*v+(d[:,None,None]+offsets[None,:,None])*scale*n
        valid=(pts[:,:,0]>=1)&(pts[:,:,0]<image.shape[1]-2)&(pts[:,:,1]>=1)&(pts[:,:,1]<image.shape[0]-2)
        colors=cv2.remap(lab,pts[:,:,0].astype('float32'),pts[:,:,1].astype('float32'),cv2.INTER_LINEAR)
        colors=cv2.GaussianBlur(colors,(3,1),0);strength=np.linalg.norm(np.gradient(colors,.5,axis=1),axis=2)
        peaks=(strength[:,1:-1]>=strength[:,:-2])&(strength[:,1:-1]>=strength[:,2:])&(strength[:,1:-1]>2)
        peaks &= valid[:,:-2]&valid[:,1:-1]&valid[:,2:]
        # Texture outside the card can be stronger than its physical edge.
        # Check that this already corner-identified edge is observed; the
        # separate holder-candidate gate handles competing physical identities.
        # Resizing quantizes the transition in output pixels. Original/JPEG
        # checks retain their strict tolerance; this never adds observed span.
        tolerance=min(3.,1.5+resampling_uncertainty_px/scale)
        agrees=np.any(peaks&(abs(offsets[1:-1])[None,:]<=tolerance),axis=1)
        halves=[int(np.sum(agrees&(t<.5))),int(np.sum(agrees&(t>=.5)))]
        good=agrees.sum()>=54 and agrees.mean()>=.8 and min(halves)>=12
        checks.append(dict(shift=shift,profiles=len(t),agreeing_profiles=int(agrees.sum()),
                           fraction=float(agrees.mean()),half_counts=halves,verified=bool(good),
                           tolerance_normalized_px=float(tolerance)))
    return dict(verified=all(c['verified'] for c in checks),checks=checks,
                scope='observed local source transition in two curve-following windows; physical identity checked separately; no new span')


def frame_candidate(track,geometry,k):
    """Convert source segments to the canonical coordinates expected by certify.

    Track coefficients belong to their fitting frame; never reuse them after a
    frame rebuild or for a fit produced by a different source sampler.
    """
    if not track.get('supported') or not track.get('segments'):return None
    H=cv2.getPerspectiveTransform(np.float32(geometry['outer']),FRAME)
    a,b=FRAME[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    p=project(np.concatenate(track['segments']),H);t=(p-a)@v/(v@v);d=(p-a)@n
    coeff=np.polynomial.polynomial.polyfit(t-.5,d,2)
    copied=copy.deepcopy(track);copied['coefficients']=coeff.tolist()
    return dict(track=copied,offset=float(coeff[0]))


def audit_curve(image,geometry,track,k):
    checks=[]
    for label,im in [('original',image),('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                     ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]:
        factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        translation=(factor-1)/2 if label=='resize75' else np.zeros(2)
        g=dict(outer=(np.asarray(geometry['outer'])*factor+translation).tolist());t=copy.deepcopy(track)
        t['segments']=[(np.asarray(seg)*factor+translation).tolist() for seg in t.get('segments',[])]
        c=reobserve_curve(im,g,t,k,resampling_uncertainty_px=.5 if label=='resize75' else 0.)
        checks.append(dict(variant=label,**c))
    return dict(verified=all(c['verified'] for c in checks),checks=checks,
                scope='fixed candidate curve reobserved in original, resized and JPEG source pixels')


def repair_certificate(image,holder):
    checks=[];g=holder['geometry'];tracks=holder['tracks']
    if not holder.get('repairs') or holder.get('stability',{}).get('stable') is not True:
        return dict(verified=False,checks=checks,reason='no stable repair')
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    inv=np.linalg.inv(cv2.getPerspectiveTransform(np.float32(g['outer']),FRAME))
    corners=baseline_evidence(image,g)
    for repair in holder['repairs']:
        side=repair['side'];k=SIDES.index(side)
        candidate=frame_candidate(tracks['outer'][side],g,k)
        sequence=certify(lab,inv,k,candidate,tracks['inner'][side]) if candidate else None
        rounded=all(c['score']>=1.25 and c['arc_over_continuation']>=.75 for c in corners[side])
        checks.append(dict(side=side,sequence=sequence,corners=corners[side],verified=bool(sequence and rounded)))
    return dict(verified=bool(checks and all(c['verified'] for c in checks)),checks=checks,
                scope='final physical rim, GX band, shoulder and keyline; rounded endpoints; existing repair perturbation audit')


def audit_sequence(image,geometry,tracks,k):
    """Check physical-edge identity when background texture wins a free fit."""
    checks=[];side=SIDES[k]
    variants=[('original',image),('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
              ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]
    for label,im in variants:
        factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        g=dict(outer=(np.asarray(geometry['outer'])*factor).tolist());observed={}
        for kind in ['outer','inner']:
            t=copy.deepcopy(tracks[kind][side]);t['segments']=[(np.asarray(seg)*factor).tolist() for seg in t.get('segments',[])]
            observed[kind]=t
        candidate=frame_candidate(observed['outer'],g,k)
        inv=np.linalg.inv(cv2.getPerspectiveTransform(np.float32(g['outer']),FRAME))
        lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype('float32')
        sequence=certify(lab,inv,k,candidate,observed['inner']) if candidate else None
        checks.append(dict(variant=label,sequence=sequence,verified=sequence is not None))
    return dict(verified=all(c['verified'] for c in checks),checks=checks,
                scope='four-transition GX sequence reobserved at fixed candidate tracks; original, resized and JPEG pixels')


def certify_frame(image,geometry,tracks,quality,holder_evidence=None):
    """Require all four physical edges before releasing any measured axis.

    Independent normal-offset searches must return the same observed outer
    tracks. This is a local frame audit, not a full localization repeatability
    claim. Earlier unresolved state is superseded only on this explicit proof.
    """
    from .quality_evidence import disagreement
    from .legacy.gx_before import _validate_frame_quad
    from .legacy.frame_holders import frame_shape
    proof=dict(verified=False,checks=[],scope='four final physical tracks and independent offset searches')
    if quality.get('issues'):return dict(proof,reason='final whole-frame quality issue remains')
    q=np.asarray(geometry['outer'],float)
    try:_validate_frame_quad(q,image.shape,'final physical frame')
    except ValueError as error:return dict(proof,reason=str(error))
    if not .55<frame_shape(q)<.90:return dict(proof,reason='implausible frame proportions')
    rebuilt=rebuild_frame(tracks,q)
    if rebuilt is None:return dict(proof,reason='four sufficiently observed outer tracks cannot rebuild the frame')
    # The GX bright shoulder can mimic a physical rim. A rounded dark-band
    # candidate just before it (the existing GX 9..26 px transition interval)
    # leaves that identity unresolved. Farther background/sleeve tracks are
    # assessed by the holder checks, not mistaken for this printed sequence.
    if holder_evidence is None:
        from .holder_topology import hypotheses
        holder_evidence=dict(hypotheses=hypotheses(image,geometry))
    outward=[]
    for side,candidates in holder_evidence['hypotheses'].items():
        for c in candidates:
            if -26<c['offset']<-8 and c['corner_score']>=1.25 and c['topology_score']>=.75 and c['track'].get('coverage',0)>=.55:
                outward.append(dict(side=side,offset=c['offset'],coverage=c['track']['coverage'],corner_score=c['corner_score'],topology_score=c['topology_score']))
    if outward:return dict(proof,reason='competing rounded edge outside selected physical frame',outward_candidates=outward)
    corners=baseline_evidence(image,geometry)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    for k,s in enumerate(SIDES):
        ends=corners[s]
        if min(c['score'] for c in ends)<1.25 or max(c['arc_over_continuation'] for c in ends)<.75:
            return dict(proof,reason='physical corner evidence incomplete: '+s)
        a,b=q[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        for offset in [-3.,3.]:
            seed=q+offset*scale*n
            # Reobserve this physical edge under displaced search windows.
            # A broad strongest-gradient search can jump into the printed band.
            args=extract(lab,seed,k,'outer',scale,allow_partial_profiles=True,search_radius=6)
            alternatives=[fit(*args,scale,tracks['outer'][s].get('degree',2))]
            supported=[t for t in alternatives if t.get('supported')]
            if not supported:return dict(proof,reason='independent outer search unsupported: '+s)
            fresh=max(supported,key=lambda t:t.get('support',0))
            cmp=disagreement(tracks['outer'][s],fresh,a,b,n,scale)
            good=cmp['common_profiles']>=54 and cmp.get('p95_normalized_px',float('inf'))<=3
            proof['checks'].append(dict(side=s,offset=offset,comparison=cmp,verified=bool(good)))
            if not good:
                curved=audit_curve(image,geometry,tracks['outer'][s],k)
                proof['checks'][-1]['curve_reobservation']=curved
                if not curved['verified']:
                    sequence=audit_sequence(image,geometry,tracks,k)
                    proof['checks'][-1]['sequence_reobservation']=sequence
                    if not sequence['verified']:return dict(proof,reason='independent outer search disagrees: '+s)
    return dict(proof,verified=True,reason='all four final physical edges pass independent source searches',
                rebuilt_outer=rebuilt.tolist())
