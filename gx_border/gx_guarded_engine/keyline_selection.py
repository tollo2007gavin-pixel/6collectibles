"""Experimental GX inner-bottom identity selection, independent of card names.

An earlier color transition must retain the GX band/shoulder signature and
connected middle anchor. Distinct parallel transitions are an identity conflict,
not confidence in a centering score. A replacement remains review-required.
Straight color fits are the conservative comparison; these rules do not establish
that curves are less accurate generally. No production eligibility is changed.
"""
import copy
import cv2
import numpy as np
from .tracks import extract, measure
from .anchored_polynomial import fit_outer
from .observed_recovery import recover_missing_outer
from .reportability import apply_reportability


def extend_observed_support(track, candidates, a, b, normal, scale):
    """Reconnect observed collinear signature fragments without moving the fit.

    The connected middle fit establishes identity and slope. Across an evidence
    gap, accept only actual GX-signature samples within its original residual
    tolerance. Never fill the gap or fit to the strongest disconnected branch.
    """
    if not track.get('supported') or len(candidates)<20:
        return track
    rows,t,d,weight=candidates.T
    p=track['coefficients']
    mask=abs(d-np.polynomial.polynomial.polyval(t-.5,p))<1.5
    ids=[]
    for row in np.unique(rows[mask]):
        choices=np.flatnonzero(mask & (rows==row))
        ids.append(choices[np.argmax(weight[choices])])
    ids=np.asarray(ids,dtype=int)
    groups=np.split(ids,np.flatnonzero(np.diff(rows[ids])>3)+1)
    groups=[g for g in groups if len(g)>=4]
    segments=[]
    for group in groups:
        ts=np.linspace(t[group[0]],t[group[-1]],2*len(group))
        ds=np.polynomial.polynomial.polyval(ts-.5,p)*scale
        segments.append((a+ts[:,None]*(b-a)+ds[:,None]*normal).tolist())
    result=copy.deepcopy(track)
    result.update(segments=segments,coverage=float(sum(t[g[-1]]-t[g[0]] for g in groups)),
                  support=len(ids)/241,original_connected_coverage=track['coverage'],
                  extension_method='observed signature fragments; fixed middle-anchored straight fit')
    return result


def sampled_track(track, a, b, normal, scale):
    """Sample only the union of observed segments, retaining all missing spans."""
    grid = np.linspace(.05, .95, 241)
    values = np.full(len(grid), np.nan)
    if not track.get('supported'):
        return grid, values
    vector = b-a
    for segment in track.get('segments', []):
        points = np.asarray(segment)
        ts = (points-a) @ vector / (vector @ vector)
        ds = (points-a) @ normal / scale
        order = np.argsort(ts); ts, ds = ts[order], ds[order]
        if len(ts)<2 or np.any(np.diff(ts)<=0):
            continue
        inside = (grid>=ts[0]) & (grid<=ts[-1])
        values[inside] = np.interp(grid[inside], ts, ds)
    return grid, values


def select_bottom(prior, colour_straight, a, b, normal, scale):
    """Propose a different boundary only beyond the fitter's ±1.5 px band.

    Floors are provisional evidence requirements, not calibrated accuracy gates.
    Returning the prior does not certify its correctness (e.g. glare or holders).
    """
    decision = dict(action='retain', reason='', review_required=False,
                    calibrated=False, selected_method='prior')
    if not prior.get('supported'):
        decision['reason']='missing prior: boundary identity cannot be compared'
        decision['review_required']=True
        return prior, decision
    if not colour_straight.get('supported') or colour_straight.get('coverage',0)<.30:
        decision['reason']='insufficient connected color-signature span'
        return prior, decision
    grid, old = sampled_track(prior,a,b,normal,scale)
    _, new = sampled_track(colour_straight,a,b,normal,scale)
    common = np.isfinite(old) & np.isfinite(new)
    decision['common_profile_count']=int(common.sum())
    if common.sum()<54:  # .20 of the side at the sampler's .00375 step.
        decision['reason']='insufficient common observed evidence'
        decision['review_required']=True
        return prior, decision
    shift = old[common]-new[common]  # Positive means proposed boundary is outward.
    separated = shift>3.0  # Two nonoverlapping ±1.5 normalized-pixel fit bands.
    decision.update(median_outward_shift_normalized_px=float(np.median(shift)),
                    outward_separated_fraction=float(separated.mean()))
    if np.median(shift)<=3 or separated.mean()<.8:
        decision['reason']='no consistently separate earlier boundary'
        return prior, decision
    middle=common & (abs(grid-.5)<.12)
    decision['middle_common_profiles']=int(middle.sum())
    if middle.sum()<12 or np.median((old-new)[middle])<=3:
        decision['reason']='earlier branch not distinctly supported at the middle anchor'
        decision['review_required']=True
        return prior, decision
    # Require distinct evidence on both halves, not an extrapolated corner vote.
    half_counts=[int(np.sum(common & (grid<.5))),int(np.sum(common & (grid>=.5)))]
    decision['common_half_counts']=half_counts
    if min(half_counts)<12:
        decision['reason']='boundary conflict lacks observed evidence on both halves'
        decision['review_required']=True
        return prior, decision
    decision.update(action='propose_colour_straight',selected_method='colour_straight',
                    reason='earlier connected GX color signature across both halves',
                    review_required=True)
    return colour_straight, decision


def refine_inner_bottom(image, geometry, tracks):
    """Return a copy, an explicit decision, and straight/curve review alternatives."""
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    q=np.asarray(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    candidates,a,b,n=extract(lab,geometry['inner'],2,'inner_colour',scale,
                              allow_partial_profiles=True)
    choices={label:fit_outer(candidates,a,b,n,scale,degree)
             for label,degree in [('straight',1),('curve',2)]}
    choices['straight']=extend_observed_support(choices['straight'],candidates,a,b,n,scale)
    selected,decision=select_bottom(tracks['inner']['bottom'],choices['straight'],a,b,n,scale)
    result=copy.deepcopy(tracks)
    result['inner']['bottom']=copy.deepcopy(selected)
    before=measure(tracks,geometry['outer']); after=measure(result,geometry['outer'])
    if decision['action']!='retain' and before['tb'].get('measured') and not after['tb'].get('measured'):
        result=copy.deepcopy(tracks)
        decision.update(action='retain',selected_method='prior',
                        reason='color alternative loses common measured span',review_required=True)
    return result,decision,choices


def audit_selection_stability(image, geometry, prior_tracks, result):
    """Isolated selector perturbations; does not rerun localization or other fits."""
    variants=[('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
              ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]
    checks=[]
    for label,im in variants:
        factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        g={k:(np.array(geometry[k])*factor).tolist() for k in ['outer','inner']}
        fixed=copy.deepcopy(result['tracks'])
        fixed['inner']['bottom']=copy.deepcopy(prior_tracks['inner']['bottom'])
        for sides in fixed.values():
            for track in sides.values():
                track['segments']=[(np.array(s)*factor).tolist() for s in track.get('segments',[])]
        tracks,decision,_=refine_inner_bottom(im,g,fixed)
        axes=measure(tracks,g['outer'])
        shifts={a:axes[a]['first_pct']-result['axes'][a]['first_pct']
                if axes[a].get('measured') and result['axes'][a].get('measured') else None for a in ['lr','tb']}
        checks.append(dict(variant=label,decision=decision,axes=axes,signed_shift_pp=shifts,
                           same_selection=decision['action']==result['bottom_decision']['action']))
    return checks


def refine_observed_tracks(image, geometry, prior_tracks, *, check_stability=True, quality_evidence=None):
    """Opt-in experimental composition; preserves all existing non-bottom fits."""
    tracks,rescued=recover_missing_outer(image,geometry,prior_tracks)
    result,decision,choices=refine_inner_bottom(image,geometry,tracks)
    output=dict(tracks=result,axes=measure(result,geometry['outer']),
                rescued_outer_sides=rescued,bottom_decision=decision,
                bottom_alternatives=choices,review_required=decision['review_required'],
                calibrated=False,centering_selection_gates_enabled=False,
                automatic_promotion_allowed=False)
    if check_stability:
        output['perturbations']=audit_selection_stability(image,geometry,prior_tracks,output)
        output['selection_stable']=all(v['same_selection'] for v in output['perturbations'])
        output['review_required'] |= not output['selection_stable']
    # Recovery of drawable geometry must not override a quality exclusion.
    return apply_reportability(output,quality_evidence)
