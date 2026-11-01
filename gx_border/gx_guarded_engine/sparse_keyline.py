"""Recover a sparse keyline only from a certified physical-edge GX sequence.

Used within unresolved localization recovery. Existing well-observed keylines
are retained, and failed audits never replace a prior track.
"""
import copy,cv2,numpy as np
from .tracks import SIDES
from .holder_repair import inner_from_physical_edge
from .quality_evidence import disagreement
from .final_frame import audit_sequence


def compare(first,second,outer,k):
    q=np.asarray(outer);a,b=q[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    scale=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2520
    c=disagreement(first,second,a,b,n,scale)
    return dict(c,verified=bool(c['common_profiles']>=54 and c.get('p95_normalized_px',float('inf'))<=3))


def select(image,outer,k):
    candidates=inner_from_physical_edge(image,outer,k)
    if not candidates:return None,dict(reason='no inward GX signature candidates')
    selected=max(candidates,key=lambda t:t['coverage'])
    if selected['coverage']<.55:return None,dict(reason='inward GX signature remains sparse')
    comparisons=[compare(selected,t,outer,k) for t in candidates if t is not selected]
    if any(not c['verified'] for c in comparisons):return None,dict(reason='competing inward GX signatures',comparisons=comparisons)
    return selected,dict(comparisons=comparisons)


def recover(image,geometry,tracks):
    updated=copy.deepcopy(tracks);changes=[]
    for k,side in enumerate(SIDES):
        old=tracks['inner'][side]
        if old.get('coverage',0)>=.55:continue
        candidate,proof=select(image,geometry['outer'],k)
        if candidate is None or candidate['coverage']<old.get('coverage',0)+.15:continue
        proposed=copy.deepcopy(updated);proposed['inner'][side]=candidate
        sequence=audit_sequence(image,geometry,proposed,k)
        if not sequence['verified']:continue
        checks=[]
        for label,im in [('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                         ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]:
            factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
            translation=(factor-1)/2 if label=='resize75' else np.zeros(2)
            outer=np.asarray(geometry['outer'])*factor+translation
            fresh,detail=select(im,outer,k)
            if fresh is None:checks.append(dict(variant=label,verified=False,detail=detail));continue
            fresh=copy.deepcopy(fresh);fresh['segments']=[((np.asarray(s)-translation)/factor).tolist() for s in fresh['segments']]
            checks.append(dict(variant=label,**compare(candidate,fresh,geometry['outer'],k)))
        if not all(c['verified'] for c in checks):continue
        updated['inner'][side]=candidate
        changes.append(dict(side=side,previous_coverage=old.get('coverage',0),coverage=candidate['coverage'],
                            sequence=sequence,stability=checks,identity=proof,
                            method='sparse keyline recovered inward from final physical edge'))
    return updated,changes
