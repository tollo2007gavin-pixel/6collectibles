"""Conservative GX measurement-quality checks, not centering selection gates.

Thresholds are provisional evidence routing rules. A clear outcome is not a
certificate of accuracy. Explicit photo reviews are recorded separately from
automatic checks; exact-image recognition is never described as generalization.
"""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from .tracks import SIDES,extract
from .anchored_polynomial import fit_outer
from .keyline_selection import sampled_track,extend_observed_support

def pixel_fingerprint(image):
    return hashlib.sha256(str(image.shape).encode()+np.ascontiguousarray(image).tobytes()).hexdigest()

def track_fingerprint(result):
    """Identify the exact reviewed geometry, to 0.0001 source pixel precision."""
    g=result.get('geometry',{});tracks=result.get('tracks',{})
    if not g or not tracks:return None
    payload=dict(geometry={k:np.round(g[k],4).tolist() for k in ['outer','inner']},
        tracks={k:{s:dict(supported=bool(t.get('supported')),segments=[np.round(v,4).tolist() for v in t.get('segments',[])])
                   for s,t in sides.items()} for k,sides in tracks.items()})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def reviewed_evidence(image,result=None):
    path=Path(__file__).with_name('reviewed_rejections.json')
    if not path.exists():return None
    records=json.loads(path.read_text(encoding='utf-8'))['records']
    match=records.get(pixel_fingerprint(image))
    if match and result and match.get('approved_track_sha256')==track_fingerprint(result):
        return dict(issues=[],source='explicit user approval of this exact photo and boundary fit',
                    automatic_detection=False,note=match.get('approval_note','Prior rejection superseded for these tracks only'))
    return dict(match,source='explicit user review of identical decoded pixels',automatic_detection=False) if match else None

def disagreement(first,second,a,b,n,scale):
    _,v=sampled_track(first,a,b,n,scale);_,w=sampled_track(second,a,b,n,scale)
    common=np.isfinite(v)&np.isfinite(w)
    if common.sum()<54:return dict(common_profiles=int(common.sum()),separated=False)
    delta=abs(v[common]-w[common])
    return dict(common_profiles=int(common.sum()),p95_normalized_px=float(np.quantile(delta,.95)),
                separated_fraction=float(np.mean(delta>3)),separated=bool(np.mean(delta>3)>.2))

def inspect_quality(image,geometry,tracks,holder_evidence=None,holder_result=None):
    q=np.asarray(geometry['outer'],float);lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    keystone=max(max(lengths[0],lengths[2])/min(lengths[0],lengths[2]),max(lengths[1],lengths[3])/min(lengths[1],lengths[3]))
    scale=np.mean(lengths[[0,2]])/1260;issues=[];details=dict(keystone=float(keystone),comparisons={})
    # This routes unusually strong perspective to review; it is not a GX
    # accuracy tolerance and does not inherit yellow's ±1 point allowance.
    if keystone>1.25:issues.append('unreliable_perspective')
    if any(not tracks['outer'][s].get('supported') for s in SIDES):issues.append('unresolved_boundary_identity')
    if holder_result and holder_result.get('repairs'):
        choices=holder_result.get('choices',[])
        # This proof describes source observations, not the identity of a card
        # or a previously reviewed image. The same rule applies to new listings.
        verified=(len(choices)==1 and bool(choices[0].get('soft_seam_evidence'))
                  and len(holder_result['repairs'])==1 and holder_result['repairs'][0].get('inner_action')=='retained'
                  and holder_result.get('stability',{}).get('stable') is True)
        same_edge_verified=(bool(holder_result.get('same_edge_agreement'))
                            and all(r['kind']=='same_edge_curve' and r.get('inner_action')=='retained' for r in holder_result['repairs'])
                            and holder_result.get('stability',{}).get('stable') is True)
        final_verified=holder_result.get('final_boundary_certificate',{}).get('verified') is True
        verified=verified or same_edge_verified or final_verified
        if not verified:issues.append('holder_boundary_ambiguity')
        details['holder_repair_requires_review']=not verified
        details['holder_repair_source_sequence_verified']=verified
        details['same_edge_curve_group_verified']=same_edge_verified
        details['final_boundary_sequence_verified']=final_verified
    if holder_evidence:
        for k,side in enumerate(SIDES):
            ends=holder_evidence['baseline'][side]
            seam=all(c['arc_over_continuation']<.75 and (c['straight_continuation'] or 0)>3 for c in ends)
            if seam:issues.append('holder_boundary_ambiguity')
            for candidate in holder_evidence['hypotheses'][side]:
                if abs(candidate['offset'])>=8 or candidate['topology_score']<.75:continue
                old=tracks['outer'][side];new=candidate['track']
                if new['coverage']<old.get('coverage',0)+10/241:continue
                a,b=q[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
                comparison=disagreement(old,new,a,b,n,scale)
                details['comparisons']['outer_'+side]=comparison
                if comparison['separated']:issues.append('unstable_boundary_selection')
    # Top art and foil can compete with the agreed grey keyline just as the GX
    # rules box can at the bottom. Audit the first colour-signature transition;
    # disagreement remains diagnostic until this check has adequate specificity.
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    c,a,b,n=extract(lab,geometry['inner'],0,'inner_colour',scale,allow_partial_profiles=True)
    alternative=extend_observed_support(fit_outer(c,a,b,n,scale,1),c,a,b,n,scale)
    if alternative.get('supported') and alternative.get('coverage',0)>=.3:
        comparison=disagreement(tracks['inner']['top'],alternative,a,b,n,scale)
        details['comparisons']['inner_top']=comparison
        # Saved reviews expose false positives on six otherwise accurate fits.
        # Retain this disagreement as a diagnostic until independently validated.
        comparison['gate_enabled']=False
    return dict(issues=sorted(set(issues)),source='GX source-boundary evidence v1',automatic_detection=True,
                calibrated=False,details=details)
