"""Experimental card-frame fitting through transparent sleeves and holders.

The caller supplies holder eligibility. Pixel patterns, never holder dimensions
or card-specific centering targets, determine measured card boundaries.
"""
import cv2
import numpy as np
from .frame_adaptive import _frame_ransac,quad_from_models,fit_frame,stripe_frame
from .gx_before import _validate_frame_quad,centering_from_frame_quads
from .localizer import proposals
from .frame_recovery import combine_sides
from .frame_keyline_segments import propose_side
from .holder_consensus import confirm_outer

SIDES=('left','right','top','bottom')


class AmbiguousHolderFrame(ValueError):
    """Conflicting supported frames must not be bypassed by another search."""

def structured_stripe(image,seed):
    w,h,pad=1260,1760,110
    H=cv2.getPerspectiveTransform(np.float32(seed),np.float32([[pad,pad],[pad+w-1,pad],[pad+w-1,pad+h-1],[pad,pad+h-1]]))
    rect=cv2.warpPerspective(image,H,(w+2*pad,h+2*pad))
    gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY).astype(np.float32)
    models={};diagnostics={}
    for side in SIDES:
        vertical=side in ('left','right');length=gray.shape[0] if vertical else gray.shape[1]
        along=np.linspace(pad+.10*(length-2*pad),pad+.90*(length-2*pad),220).astype(int)
        profiles=np.array([gray[t] if vertical else gray[:,t] for t in along])
        if side in ('right','bottom'):profiles=profiles[:,::-1].copy()
        profiles=cv2.GaussianBlur(profiles,(5,1),0);grad=np.gradient(profiles,axis=1)
        candidates=[]
        for row,t in enumerate(along):
            g=grad[row];v=profiles[row];lo,hi=pad-85,pad+275
            neg=np.flatnonzero((g[lo+1:hi-1]<=g[lo:hi-2])&(g[lo+1:hi-1]<=g[lo+2:hi])&(g[lo+1:hi-1]<-1.0))+lo+1
            pos=np.flatnonzero((g[lo+1:hi-1]>=g[lo:hi-2])&(g[lo+1:hi-1]>=g[lo+2:hi])&(g[lo+1:hi-1]>1.0))+lo+1
            for a in neg:
                for b in pos[(pos-a>=8)&(pos-a<=26)]:
                    outer=pos[(a-pos>=8)&(a-pos<=40)]
                    inner=neg[(neg-b>=7)&(neg-b<=30)]
                    if len(outer)==0 or len(inner)==0:continue
                    if min(v[a-4],v[b+4])-np.mean(v[a+2:b-1])<8:continue
                    score=min(-g[a],g[b],np.max(g[outer]),np.max(-g[inner]))**.5
                    candidates.append((t,(a+b)/2,score))
        model,support=_frame_ransac(np.asarray(candidates),pad+27,140,tol_px=2.5,iters=2200,max_slope=.055)
        if abs(model[0])>.055:raise ValueError('Frame stripe orientation unstable')
        models[side]=model;diagnostics[side]=support
    stripe=cv2.perspectiveTransform(quad_from_models(models,*gray.shape[::-1])[None],np.linalg.inv(H))[0]
    _validate_frame_quad(stripe,image.shape,'printed stripe')
    return stripe,diagnostics

def frame_shape(outer):
    q=np.array(outer);lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    ratio=(lengths[0]+lengths[2])/(lengths[1]+lengths[3])
    return float(ratio)


def validate_card_geometry(image,outer,inner,stripe):
    """Reject inverted, clipped, nested-in-the-wrong-order, or implausible frames."""
    for name,quad in [('outer',outer),('inner',inner),('stripe',stripe)]:
        _validate_frame_quad(quad,image.shape,name)
    H=cv2.getPerspectiveTransform(np.float32(outer),np.float32([[0,0],[1259,0],[1259,1759],[0,1759]]))
    qi=cv2.perspectiveTransform(np.float32(inner)[None],H)[0]
    qs=cv2.perspectiveTransform(np.float32(stripe)[None],H)[0]
    def insets(q):
        return np.array([q[[0,3],0],1259-q[[1,2],0],q[[0,1],1],1759-q[[3,2],1]])
    inside,rim=insets(qi),insets(qs)
    if np.any(inside<=3) or np.any(inside>=150):
        raise ValueError('Inner frame is not contained at a plausible card inset')
    if np.any(rim<=2) or np.any(rim>=inside-3):
        raise ValueError('Printed stripe must lie between card edge and inner keyline')
    if np.max(np.ptp(inside,axis=1))>30:
        raise ValueError('Inner and physical edges diverge excessively')
    if not .55<frame_shape(outer)<.90:
        raise ValueError('Card-frame proportions are implausible')
    return dict(inner_insets_px=inside.tolist(),stripe_insets_px=rim.tolist(),
                apparent_width_height_ratio=frame_shape(outer))


def candidate_quality(image,candidate):
    support=candidate['support'];stripe_support=candidate['stripe_support']
    if set(support)!={s+'_'+edge for s in SIDES for edge in ('outer','inner')} or set(stripe_support)!=set(SIDES):
        raise ValueError('Missing side evidence')
    weak=[key for key,value in support.items() if value<.65]
    confirmation=candidate.get('outer_confirmation',{})
    confirmed=(len(weak)==1 and weak[0].endswith('_outer') and support[weak[0]]>=.5
               and confirmation.get('side')==weak[0] and confirmation.get('confirmed') is True
               and confirmation.get('window_support',0)>=12/17)
    if (weak and not confirmed) or min(stripe_support.values())<.55:
        raise ValueError('Insufficient card-frame support through holder')
    detail=validate_card_geometry(image,candidate['outer'],candidate['inner'],candidate['stripe'])
    value=min(support.values())+.5*np.mean(list(support.values()))+.3*min(stripe_support.values())
    # A weak plausibility term, not a perspective correction or measurement offset.
    value-=2*abs(frame_shape(candidate['outer'])-63/88)
    return float(value),detail


def select_candidate(image,candidates):
    qualified=[];rejected=[]
    for candidate in candidates:
        try:
            score,detail=candidate_quality(image,candidate)
            qualified.append((score,candidate,detail))
        except ValueError as error:
            rejected.append(dict(seed_index=candidate.get('seed_index'),reason=str(error)))
    if not qualified:
        raise ValueError('No fully supported card frame inside holder')
    qualified.sort(key=lambda item:-item[0]);score,best,detail=qualified[0]
    short=np.mean(np.linalg.norm(np.diff(np.array(best['outer'])[[0,1,2,3,0]],axis=0),axis=1)[[0,2]])
    plausible=[entry for entry in qualified if entry[0]>=score-.12]
    disagreement=max(max(np.linalg.norm(np.array(entry[1][key])-np.array(best[key]),axis=1))*1260/short
                     for entry in plausible for key in ['outer','inner'])
    if disagreement>8:
        raise AmbiguousHolderFrame('Supported holder candidates disagree on the card boundary')
    return dict(best),dict(score=score,geometry=detail,qualified_candidates=len(qualified),
        max_close_candidate_disagreement_px=float(disagreement),rejected_candidates=rejected)


def refine_holder_inner(image,geometry):
    """Reuse visible-segment proposals without claiming the holder is a bare card."""
    candidates=[];details={}
    for side in ['left','right']:
        try:
            candidate,detail=propose_side(image,geometry['outer'],geometry['inner'],side)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
            candidate=np.float32(geometry['inner']);detail={'error':str(error)}
        candidates.append(candidate);details[side]=detail
    repair=all('error' not in detail for detail in details.values()) and any(
        abs(d['movement_px'][0])<=2 and 4<=abs(d['movement_px'][1])<14 for d in details.values())
    result=dict(geometry)
    if repair:
        inner=combine_sides(geometry['inner'],candidates,{'left':0,'right':1})
        try:
            validate_card_geometry(image,geometry['outer'],inner,geometry['stripe'])
            result.update(inner=inner.tolist(),centering=centering_from_frame_quads(geometry['outer'],inner))
        except ValueError as error:
            repair=False;details['validation_error']=str(error)
    result.update(segment_repair=bool(repair),segment_diagnostics=details)
    return result


def measure_holder(image,*,holder,capture='photo',include_candidates=False):
    """Known transparent-holder entry point; all measurements are diagnostic-only.

    This is not a holder classifier. Bare images and scans stay on their existing
    paths. A returned supported geometry is not calibrated grading acceptance.
    """
    if holder not in ('penny_sleeve','rigid_toploader','unknown') or capture!='photo':
        raise ValueError('Known sleeve/toploader photo eligibility is required')
    if image is None or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<300:
        raise ValueError('A readable colour photograph at least 300 pixels wide/high is required')
    seeds=proposals(image);attempts=[];chosen=None;selection=None;selected_method=None;all_candidates=[]
    for method,fitter in [('structured',structured_stripe),('standard_fallback',stripe_frame),('inset_fallback',stripe_frame)]:
        candidates=[]
        for index,seed in enumerate(seeds):
            try:
                if method=='inset_fallback':
                    # A holder proposal can put the narrow stripe search outside
                    # the card. This changes only the search ROI; all eight final
                    # lines are subsequently fitted from image pixels as before.
                    centre=np.mean(seed,axis=0)
                    seed=centre+.90*(seed-centre)
                stripe,ss=fitter(image,seed)
                if min(ss.values())<.55:raise ValueError('Weak printed-stripe support')
                outer,inner,support,_=fit_frame(image,seed,stripe)
                candidate=dict(seed_index=index,method=method,outer=outer.tolist(),inner=inner.tolist(),
                    stripe=stripe.tolist(),support=support,stripe_support=ss,
                    centering=centering_from_frame_quads(outer,inner))
                candidates.append(candidate)
                weak={key:float(value) for key,value in support.items() if value<.65}
                if weak:
                    attempts.append(dict(method=method,seed_index=index,reason='Weak final side evidence',weak_side_support=weak))
            except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
                attempts.append(dict(method=method,seed_index=index,reason=str(error)))
        all_candidates.extend(candidates)
        try:
            chosen,selection=select_candidate(image,candidates);selected_method=method
            break
        except AmbiguousHolderFrame as error:
            return dict(holder=holder,capture=capture,accepted=False,geometry_supported=False,status='rejected',
                        reasons=[str(error)],attempts=attempts,ambiguous=True,
                        diagnostic_candidates=all_candidates if include_candidates else [])
        except ValueError as error:
            attempts.append(dict(method=method,reason=str(error)))
    if chosen is None:
        confirmed=[]
        for candidate in all_candidates:
            try:
                validate_card_geometry(image,candidate['outer'],candidate['inner'],candidate['stripe'])
                detail=confirm_outer(image,candidate)
                confirmed.append(dict(candidate,outer_confirmation=detail))
            except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
                attempts.append(dict(method='outer_confirmation',seed_index=candidate['seed_index'],reason=str(error)))
        try:
            chosen,selection=select_candidate(image,confirmed);selected_method='outer_confirmation'
        except ValueError as error:
            attempts.append(dict(method='outer_confirmation',reason=str(error)))
    if chosen is None:
        return dict(holder=holder,capture=capture,accepted=False,geometry_supported=False,status='rejected',
                    reasons=['No unambiguous, fully supported card frame inside holder'],attempts=attempts,
                    diagnostic_candidates=all_candidates if include_candidates else [])
    result=refine_holder_inner(image,chosen)
    result.update(holder=holder,capture=capture,accepted=False,geometry_supported=True,status='diagnostic_only',
        reasons=['Holder-photo accuracy and acceptance are not yet calibrated'],
        selection=selection,selected_method=selected_method,proposal_count=len(seeds),attempts=attempts)
    return result
