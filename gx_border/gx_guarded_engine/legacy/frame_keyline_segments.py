"""Segment-supported inner-line proposals for known bare GX photographs.

Fit a dark band / silver strip / darkening sequence in source-derived pixels.
Separate visible upper/lower segments have equal geometric influence even when
artwork interrupts most of a side. Each output side remains one straight line.
No image names, card identities, labels, or centering targets enter selection.
"""
from __future__ import annotations

import cv2
import numpy as np

from .frame_adaptive import _frame_ransac
from .frame_recovery import combine_sides
from .gx_before import centering_from_frame_quads, _validate_frame_quad

WIDTH, HEIGHT, PAD = 1260, 1760, 100


def _side_candidates(image, outer, inner, side):
    dst = np.float32([[100,100],[1359,100],[1359,1859],[100,1859]])
    transform = cv2.getPerspectiveTransform(np.float32(outer), dst)
    rectified = cv2.warpPerspective(image, transform, (1460,1960))
    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    quad = cv2.perspectiveTransform(np.float32(inner)[None], transform)[0]
    indices = [0,3] if side == 'left' else [1,2]
    if side == 'right':
        lab = lab[:,::-1].copy()
        quad[:,0] = lab.shape[1]-1-quad[:,0]
    prior = np.polyfit(quad[indices,1], quad[indices,0], 1)
    # Suppress texture along the edge, without broadening it across its normal.
    lab = cv2.GaussianBlur(lab, (1,15), 0)
    along = np.linspace(PAD+.035*HEIGHT, PAD+.965*HEIGHT, 500).astype(int)
    colour = cv2.GaussianBlur(lab[along], (3,1), 0)
    gradient = np.gradient(colour[:,:,0], axis=1)
    candidates = []
    for row, t in enumerate(along):
        expected = np.polyval(prior,t)
        lo, hi = int(expected-50), int(expected+15)
        if lo < 5 or hi >= colour.shape[1]-5:
            continue
        g = gradient[row]
        negative = np.flatnonzero((g[lo+1:hi-1] <= g[lo:hi-2]) &
                                  (g[lo+1:hi-1] <= g[lo+2:hi]) &
                                  (g[lo+1:hi-1] < -1.5))+lo+1
        positive = np.flatnonzero((g[lo+1:hi-1] >= g[lo:hi-2]) &
                                  (g[lo+1:hi-1] >= g[lo+2:hi]) &
                                  (g[lo+1:hi-1] > 1.5))+lo+1
        for a in negative:
            for b in positive[(positive-a >= 9) & (positive-a <= 24)]:
                # Both shoulders must actually be brighter than the dark band.
                if min(colour[row,a-4,0],colour[row,b+4,0])-np.mean(colour[row,a+2:b-1,0]) < 10:
                    continue
                for c in negative[(negative-b >= 7) & (negative-b <= 23) &
                                   (abs(negative-expected) < 13)]:
                    weight = min(-g[a],g[b],-g[c])**.25
                    ga,gb,gc = -g[c-1:c+2]
                    delta = .5*(ga-gc)/(ga-2*gb+gc) if abs(ga-2*gb+gc) > 1e-6 else 0.
                    candidates.append((t,c+np.clip(delta,-.5,.5),weight))
    return np.asarray(candidates), prior, transform, quad, indices


def _segment(candidates, prior, interval, project_to=None):
    lo, hi = interval
    subset = candidates[(candidates[:,0] > PAD+lo*HEIGHT) &
                        (candidates[:,0] < PAD+hi*HEIGHT)]
    model, support = _frame_ransac(subset, np.polyval(prior,PAD+(lo+hi)*HEIGHT/2),
                                   12, tol_px=1.5, iters=2000, max_slope=.01)
    inliers = np.abs(subset[:,1]-np.polyval(model,subset[:,0])) < 1.5
    positions = np.unique(subset[inliers,0])
    if len(positions) < 15 or np.ptp(positions) <= 40 or support <= .65:
        raise ValueError('Insufficient supported span in a visible segment')
    # Initial RANSAC is slope-bounded; enforce a bound after its refinement too.
    if abs(model[0]) > .04:
        raise ValueError('Local line extrapolation is unstable')
    coordinate = float(np.median(positions)) if project_to is None else float(project_to)
    point = [coordinate,float(np.polyval(model,coordinate))]
    return point, dict(interval=list(interval), support=float(support),
                       unique_profiles=len(positions), span_px=float(np.ptp(positions)),
                       model=model.tolist(), anchor=point)


def propose_side(image, outer, inner, side, *, local_extrapolation=False):
    """Propose a straight side from two independent, visible frame segments.

    local_extrapolation is an explicitly unvalidated diagnostic variant for
    corners hidden by nearby labels. It is never used by the conservative gate.
    """
    if side not in ('left','right'):
        raise ValueError('Only vertical-side refinement is implemented')
    candidates, prior, transform, quad, indices = _side_candidates(image,outer,inner,side)
    if candidates.ndim != 2 or len(candidates) < 20:
        raise ValueError('Too few frame-pattern candidates')
    windows = [(.075,.15),(.865,.945)] if local_extrapolation else [(.045,.22),(.865,.945)]
    points, diagnostics = [], []
    for index, window in enumerate(windows):
        end = quad[indices[index],1] if local_extrapolation else None
        point, detail = _segment(candidates,prior,window,end)
        points.append(point)
        diagnostics.append(detail)
    model = np.polyfit(*np.asarray(points).T,1)
    movement = (model-prior) @ np.array([[160,1800],[1,1]])
    if np.max(np.abs(movement)) >= 14:
        raise ValueError('Proposed line moves too far from its supported prior')
    updated = quad.copy()
    for index in indices:
        updated[index,0] = np.polyval(model,updated[index,1])
    if side == 'right':
        updated[:,0] = 1459-updated[:,0]
    source = cv2.perspectiveTransform(updated[None],np.linalg.inv(transform))[0]
    result = combine_sides(inner,[source],{side:0})
    return result, dict(segments=diagnostics,model=model.tolist(),
                        prior_model=prior.tolist(),movement_px=movement.tolist(),
                        candidate_count=len(candidates),
                        strategy='local_extrapolation' if local_extrapolation else 'separated_segments')


def refine_vertical_keylines(image, geometry, *, known_bare=False, capture='photo'):
    """Conservative repair; substantial end drift must be independently shown.

    When one side demonstrates substantial lower-end drift with a stable upper
    anchor, use the compatible segment fits for both vertical sides. Scans and
    uncertain/weak frames remain on the calling pipeline's unchanged path.
    """
    if capture == 'scan':
        return dict(geometry)
    if capture != 'photo' or not known_bare:
        raise ValueError('Confirmed bare-photo eligibility is required')
    outer, previous = np.float32(geometry['outer']), np.float32(geometry['inner'])
    _validate_frame_quad(outer,image.shape,'outer')
    _validate_frame_quad(previous,image.shape,'inner')
    details, alternatives = {}, []
    for side in ('left','right'):
        try:
            proposal, detail = propose_side(image,outer,previous,side)
            details[side] = detail
            alternatives.append(proposal)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
            details[side] = {'error':str(error)}
            alternatives.append(previous)
    supported = all('error' not in d for d in details.values())
    substantial_drift = supported and any(abs(d['movement_px'][0]) <= 2 and
                                          4 <= abs(d['movement_px'][1]) < 14
                                          for d in details.values())
    result = dict(geometry)
    if substantial_drift:
        inner = combine_sides(previous,alternatives,{'left':0,'right':1})
        _validate_frame_quad(inner,image.shape,'refined inner')
        result.update(inner=inner.tolist(),centering=centering_from_frame_quads(outer,inner))
    result.update(segment_repair=bool(substantial_drift),segment_diagnostics=details,
                  accepted=False,status='diagnostic_only',
                  acceptance_reason='Experimental correct-line acceptance is not calibrated')
    # Historical errors describe another geometry/target, so evaluation recomputes them.
    result.pop('evaluation_error_pp',None)
    result.pop('errors',None)
    return result
