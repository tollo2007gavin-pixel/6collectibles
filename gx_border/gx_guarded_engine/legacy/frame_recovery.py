"""Pixel-verified recovery stage for the existing experimental GX frame fitter.

This is a diagnostic experiment, not a production acceptance policy. Input
frames/candidates come from the existing pixel fitter. No card identity, target
centering, labels, API coordinates, or numerical centering offsets are read.
"""
from __future__ import annotations

import cv2
import numpy as np

from .frame_adaptive import fit_frame, quad_from_models, _frame_ransac
from .gx_before import centering_from_frame_quads, _validate_frame_quad
from .localizer import proposals

SIDES = ("left", "right", "top", "bottom")
INDICES = {"left": (0, 3), "right": (1, 2), "top": (0, 1), "bottom": (3, 2)}


def combine_sides(base, alternatives, choices):
    """Extrapolate each selected source-image line; intersect adjacent lines."""
    lines = {}
    for side in SIDES:
        quad = alternatives[choices[side]] if side in choices else base
        a, b = np.asarray(quad)[list(INDICES[side])]
        line = np.cross(np.r_[a, 1.], np.r_[b, 1.])
        if not np.all(np.isfinite(line)) or np.linalg.norm(line[:2]) < 1e-6:
            raise ValueError("Degenerate side line")
        lines[side] = line

    def meet(a, b):
        point = np.cross(lines[a], lines[b])
        if abs(point[2]) < 1e-8:
            raise ValueError("Parallel adjacent sides")
        return point[:2] / point[2]

    return np.float32([meet("left", "top"), meet("right", "top"),
                       meet("right", "bottom"), meet("left", "bottom")])


def physical_evidence(image, quad):
    """Compare near exterior pixels with farther exterior and interior pixels.

    Distances are normalized by observed card width. Across 120 side profiles,
    a physical edge should retain similar background outside the edge and
    introduce a larger colour change inside. This complements line support;
    it is not a calibrated confidence or a general foreground classifier.
    """
    q = np.float32(quad)
    _validate_frame_quad(q, image.shape, "outer candidate")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    scale = np.mean([np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])])/1260
    offsets = np.arange(-90, 32, dtype=np.float32)*scale
    result = {}
    for k, side in enumerate(("top", "right", "bottom", "left")):
        a, b = q[k], q[(k+1) % 4]
        tangent = (b-a)/np.linalg.norm(b-a)
        normal = np.array([-tangent[1], tangent[0]])
        fractions = np.linspace(.08, .92, 120)
        points = (a+fractions[:, None]*(b-a))[:, None, :] + offsets[None, :, None]*normal
        valid = ((points[:, :, 0] >= 0) & (points[:, :, 0] < image.shape[1]-1) &
                 (points[:, :, 1] >= 0) & (points[:, :, 1] < image.shape[0]-1)).all(axis=1)
        colours = cv2.remap(lab, points[:, :, 0].astype(np.float32),
                            points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)[valid]
        if len(colours) < 30:
            result[side] = dict(score=-1., valid_profiles=int(len(colours)))
            continue
        far = np.median(colours[:, :50], axis=1)
        near = np.median(colours[:, 65:82], axis=1)
        inside = np.median(colours[:, 98:112], axis=1)
        mismatch = np.linalg.norm(near-far, axis=1)
        contrast = np.linalg.norm(inside-far, axis=1)
        evidence = (contrast-mismatch)/(contrast+mismatch+10)
        result[side] = dict(score=float(np.median(evidence)),
                            valid_profiles=int(len(colours)),
                            good_profile_fraction=float(np.mean(evidence > .25)),
                            exterior_mismatch=float(np.median(mismatch)),
                            interior_contrast=float(np.median(contrast)))
    return result


def bounded_stripe_frame(image, seed):
    """Wide fallback ROI search with bounded line orientation.

    The old 85-pixel inward search can miss a side when a coarse contour
    includes part of a hand. An unconstrained fit can then extrapolate artwork
    into a false side. Expansion is for coarse stripe localization only;
    final die-cut and keyline positions are independently fitted afterward.
    """
    width, height, pad = 1260, 1760, 110
    destination = np.float32([[pad, pad], [pad+width-1, pad],
                              [pad+width-1, pad+height-1], [pad, pad+height-1]])
    transform = cv2.getPerspectiveTransform(np.float32(seed), destination)
    rectified = cv2.warpPerspective(image, transform, (width+2*pad, height+2*pad))
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    models, support = {}, {}
    for side in SIDES:
        vertical = side in ("left", "right")
        length = gray.shape[0] if vertical else gray.shape[1]
        along = np.linspace(pad+.12*(length-2*pad), pad+.88*(length-2*pad), 180).astype(int)
        profiles = np.array([gray[t] if vertical else gray[:, t] for t in along], np.float32)
        if side in ("right", "bottom"):
            profiles = profiles[:, ::-1].copy()
        gradient = np.gradient(cv2.GaussianBlur(profiles, (5, 1), 0), axis=1)
        candidates = []
        lo, hi = pad-65, pad+245
        for row, t in enumerate(along):
            g = gradient[row]
            negative = np.flatnonzero((g[lo+1:hi-1] <= g[lo:hi-2]) &
                                      (g[lo+1:hi-1] <= g[lo+2:hi]) & (g[lo+1:hi-1] < -1))+lo+1
            positive = np.flatnonzero((g[lo+1:hi-1] >= g[lo:hi-2]) &
                                      (g[lo+1:hi-1] >= g[lo+2:hi]) & (g[lo+1:hi-1] > 1))+lo+1
            for a in negative:
                for b in positive[(positive-a >= 10) & (positive-a <= 22)]:
                    candidates.append((t, .5*(a+b), min(-g[a], g[b])))
        model, value = _frame_ransac(np.asarray(candidates), pad+27, 140,
                                      tol_px=3, iters=3500, max_slope=.055)
        if abs(model[0]) > .055:
            raise ValueError("Stripe refinement exceeded orientation bound")
        models[side], support[side] = model, value
    stripe = cv2.perspectiveTransform(quad_from_models(models, *gray.shape[::-1])[None],
                                      np.linalg.inv(transform))[0]
    _validate_frame_quad(stripe, image.shape, "stripe")
    return stripe, support


def recover_frame(image, previous, stripe=None, candidates=(), *, known_bare=False):
    """Recover an existing experimental photo hypothesis, or generate fallback.

    previous has outer/inner source quads. candidates are the existing coarse
    stripe fitter's alternatives, each with outer and stripe quads. Unchanged
    hypotheses are returned numerically intact. Callers must route scans and
    holders through the existing production paths.
    """
    if not known_bare:
        raise ValueError("Recovery requires confirmed bare-card photo eligibility")
    if previous is None:
        best = None
        for seed in proposals(image):
            try:
                coarse, stripe_support = bounded_stripe_frame(image, seed)
                if min(stripe_support.values()) < .5:
                    continue
                outer, inner, support, _ = fit_frame(image, seed, coarse)
                if min(support.values()) < .55:
                    continue
                _validate_frame_quad(outer, image.shape, "outer")
                _validate_frame_quad(inner, image.shape, "inner")
                value = min(support.values())+.5*np.mean(list(support.values()))+.3*min(stripe_support.values())
                row = dict(outer=outer.tolist(), inner=inner.tolist(), stripe=coarse.tolist(),
                           centering=centering_from_frame_quads(outer, inner),
                           fallback_support=support, stripe_support=stripe_support,
                           recovery="bounded_stripe_fallback", replaced_sides=list(SIDES))
                if best is None or value > best[0]:
                    best = (value, row)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
        if best is None:
            raise ValueError("No supported fallback frame")
        result = best[1]
    else:
        before = physical_evidence(image, previous['outer'])
        usable, evidence = [], []
        for candidate in candidates:
            try:
                evidence.append(physical_evidence(image, candidate['outer']))
                usable.append(candidate)
            except (ValueError, cv2.error):
                continue
        choices = {}
        for side in SIDES:
            if not usable:
                break
            index = max(range(len(usable)), key=lambda j: evidence[j][side]['score'])
            if (before[side]['score'] < .4 and
                    evidence[index][side]['score'] >= .4 and
                    evidence[index][side]['score'] > before[side]['score']+.15):
                choices[side] = index
        result = dict(previous)
        if choices:
            outer = combine_sides(previous['outer'], [c['outer'] for c in usable], choices)
            coarse = combine_sides(stripe, [c['stripe'] for c in usable], choices)
            _validate_frame_quad(outer, image.shape, "recovered outer")
            _, candidate_inner, _, _ = fit_frame(image, outer, coarse)
            inner = combine_sides(previous['inner'], [candidate_inner], {s: 0 for s in choices})
            _validate_frame_quad(inner, image.shape, "recovered inner")
            result.update(outer=outer.tolist(), inner=inner.tolist(), stripe=coarse.tolist(),
                          centering=centering_from_frame_quads(outer, inner))
        result.update(recovery="exterior_verified_sides" if choices else "unchanged",
                      replaced_sides=list(choices), physical_before=before,
                      physical_after=physical_evidence(image, result['outer']))
    # Legacy fusion stored errors against an obsolete target. Detection must
    # not propagate those numbers as a current accuracy claim.
    result.pop('errors', None)
    result.pop('evaluation_error_pp', None)
    if 'reasons' in result:
        result['legacy_reasons'] = result.pop('reasons')
    result.update(accepted=False, status="diagnostic_only",
                  acceptance_reason="Experimental acceptance policy is not validated")
    return result
