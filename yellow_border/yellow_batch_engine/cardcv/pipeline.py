"""Yellow-border centering with source-pixel fits and validated seed consensus.

Each segmentation proposes a seed. One enumeration pass identifies boundaries;
local signed colour profiles fit their positions on the original pixels.
Candidates are validated independently before selecting a representative fit.
No reference-card labels or target centering values enter detection.
"""
import numpy as np
import cv2

from .localize import localize, MASKERS
from .enumerate_edges import refine_by_enumeration
from .centering import centering
from .geom import order_corners

DEFAULT_POOL = ("saturation", "yellow", "vivid_yellow", "wb_yellow", "lab_b", "wb_lab_b",
                "luma")


def refine_iterated(img, seed, meta=None, max_iter=1, tol_pct=0.25,
                    wide=(4.0, 7.0), narrow=(1.2, 3.4)):
    """Refine repeatedly, reseeding with the previous result, until stable.

    DEFAULTS TO A SINGLE PASS. Iterating was tried and must not be re-enabled
    without fixing the underlying feedback: the search band is defined relative
    to the CURRENT estimate and the pair selector prefers the outermost
    qualifying line, so an outward error shifts the band outward, which admits
    a still-more-outward line. Measured on the leather photo, the left margin
    walked 0.0991 -> 0.1828 in and the left border width 2.15 -> 3.56mm over
    six rounds, moving the result from 52.6 to 67.2. Reseeding AMPLIFIES error
    here rather than damping it, so this estimator is not a contraction and
    fixed-point iteration is invalid for it.

    A correct version would anchor the band to the initial seed (removing the
    feedback path) or fix the chosen line identities after the first pass and
    only re-fit them. Until then, one pass.
    """
    quad = order_corners(seed)
    last = None
    history = []
    prev_r = None
    for it in range(max_iter):
        band = wide if it == 0 else narrow
        r = refine_by_enumeration(img, quad, meta=meta,
                                  out_mm=band[0], in_mm=band[1])
        if r is None:
            return prev_r
        new = order_corners(r["outer"])
        # Converge on the MEASUREMENT, not on corner pixel positions. Corners
        # are extrapolated from line intersections, so a fraction of a degree
        # of line-angle change moves a corner tens of pixels while changing the
        # centering result by almost nothing. Corner motion is the wrong metric.
        c = centering(r["outer"], r["inner"], portrait=_portrait(r["outer"]))
        cur = (c["left_pct"], c["top_pct"])
        if last is not None:
            move = max(abs(cur[0] - last[0]), abs(cur[1] - last[1]))
            history.append(move)
            if move < tol_pct:
                r["n_iter"] = it + 1
                r["converged"] = True
                r["move_history"] = history
                return r
        last = cur
        quad = new
        prev_r = r
    r["n_iter"] = max_iter
    r["converged"] = False
    r["move_history"] = history
    return r


def _portrait(quad):
    q = order_corners(quad)
    return np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1])


def frame_clearance(quad, img_shape, ppm):
    """Distance in mm from each card side to the image boundary.

    A card running off the edge of the frame cannot be measured on that side:
    the fitted "edge" is the crop, not the die cut, so the margin is invented.
    Line extrapolation cannot recover it -- the pixels simply are not there.
    Returned per side so an axis can be rejected individually; a card clipped
    top and bottom still has usable left/right centering.
    """
    q = order_corners(quad)
    h, w = img_shape[:2]
    top = float(min(q[0][1], q[1][1]))
    right = float(w - 1 - max(q[1][0], q[2][0]))
    bottom = float(h - 1 - max(q[2][1], q[3][1]))
    left = float(min(q[0][0], q[3][0]))
    return {k: v / ppm for k, v in
            (('top', top), ('right', right), ('bottom', bottom), ('left', left))}


def shape_error(quad, img_shape):
    """Deviation of the quad from a true 63x88mm rectangle, plus reliability.

    Returns (error_fraction, reliable). Zhang recovery is perspective-correct
    but ill-conditioned on steeply angled views: on one photo it returned an
    aspect of 0.4999 with an implied focal length of 8248px for a 3024px-wide
    phone image -- physically impossible, and a 30% "shape error" that rejected
    a detection whose colour validators were all perfect. The recovered focal
    length is therefore sanity-checked against a plausible range, and naive
    aspect (which is perspective-biased and unusable under keystone) is no
    longer silently substituted. When neither estimator is trustworthy the
    shape is reported UNRELIABLE so the caller can decline to penalise it,
    rather than penalising a measurement that was never made.
    """
    from .geom import zhang_aspect, naive_aspect, CARD_ASPECT
    h, w = img_shape[:2]
    a, f = zhang_aspect(quad, img_shape)
    na = naive_aspect(quad)
    # Zhang is perspective-correct but ill-conditioned on steep views. Two
    # independent sanity checks: the recovered focal length must be physically
    # plausible for a camera, AND Zhang must broadly agree with the naive
    # estimate. They diverge only when the solve is unstable -- on one photo
    # Zhang gave 0.4999 against a naive 0.7698 (35% apart) with a plausible-
    # looking focal length, and that alone would have rejected a detection
    # whose colour validators were all perfect.
    reliable = (a is not None and f is not None and 0.5 * w <= f <= 3.5 * w
                and na > 0 and abs(a - na) / na < 0.15)
    if not reliable:
        # Naive aspect is only meaningful when the view is near-fronto-parallel,
        # which we detect by opposite sides being nearly equal in length.
        import numpy as _np
        q = _np.asarray(quad, float)
        s0 = [float(_np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        keystone = max(abs(s0[0] - s0[2]) / max(s0[0], s0[2]),
                       abs(s0[1] - s0[3]) / max(s0[1], s0[3]))
        if keystone > 0.06:
            return 0.0, False          # too skewed for naive, Zhang unusable
        a = na
        reliable = True
    err = float(min(abs(a - CARD_ASPECT) / CARD_ASPECT,
                    abs(a - 1.0 / CARD_ASPECT) * CARD_ASPECT))
    return err, reliable


def score(r, c, img_shape=None, orient_ok=True):
    """Physical plausibility of a candidate result. Higher is better.

    Deliberately uses only localizer-independent evidence, so it can compare
    results proposed by different segmentation rules without favouring any.
    """
    widths = np.array(r["border_mm"])
    spread = float(widths.max() - widths.min())
    # Opposite-pair sums. For ANY card, border[0]+border[2] and
    # border[1]+border[3] are each (card dimension - print dimension) on their
    # axis, and both come out near-equal because the printed frame is inset
    # similarly on both axes. Crucially those sums are INDEPENDENT of how the
    # print sits within the border -- an off-centre card moves width between a
    # pair without changing its total. Raw spread cannot tell those apart: on a
    # genuinely off-centre card (left 2.87 / right 2.13) it read 0.54mm and
    # rejected a detection accurate to 3px, because uneven widths ARE the
    # signal being measured, not evidence of error.
    sum_a = float(widths[0] + widths[2])
    sum_b = float(widths[1] + widths[3])
    sum_asym = abs(sum_a - sum_b)
    med = float(np.median(widths))
    ymin = float(min(r["yellow_in"]))
    umin = float(min(r.get("uniform", [1.0])))
    cdev = float(max(r.get("chroma_dev", [0.0])))
    rms = float(np.mean(r["rms_outer"] + r["rms_inner"]))
    qc = all(c["qc_inward_ok"]) and all(c["qc_bounds_ok"])

    # Shape gate. Verified against a 600dpi scan: a detection that placed two
    # edges a border width inside the card gave 63x84mm, only 4.3% off in
    # aspect, and passed the old 8% tolerance with a positive score. A real
    # card, perspective-corrected, holds to a couple of percent.
    if img_shape is not None:
        serr, s_reliable = shape_error(r["outer"], img_shape)
    else:
        serr, s_reliable = 0.0, False
    # A recovered camera aspect is a ranking diagnostic, not proof of invalid
    # geometry. Physical containment and supported sides are validated below.
    hard_fail = not orient_ok

    s = 0.0
    s += 1.2 * ymin                                    # band matches the expected hue
    s += 0.8 * umin                                    # band is uniform along its length
    s -= 1.5 * min(cdev, 1.0)                          # all four sides agree on colour
    s -= 0.3 * min(rms, 5.0)                           # tight line fits
    # Steep viewing angle degrades accuracy even when the homography is applied
    # correctly: extrapolated corners amplify small errors and the far edge is
    # sampled at lower effective resolution. Measured across six non-glare
    # photos of one card, the two steepest views (keystone 0.215, 0.242) gave
    # the two largest centering errors (5.7 and 4.7 points combined) while every
    # view under 0.08 stayed below 3.7. Keystone is measurable up front, so it
    # is priced into the score rather than discovered afterwards.
    import numpy as _np
    _q = _np.asarray(r["outer"], float)
    _s = [float(_np.linalg.norm(_q[(i + 1) % 4] - _q[i])) for i in range(4)]
    keystone = max(abs(_s[0] - _s[2]) / max(_s[0], _s[2]),
                   abs(_s[1] - _s[3]) / max(_s[1], _s[3]))
    s -= 5.0 * max(0.0, keystone - 0.10)

    s += 1.5 if qc else 0.0
    s -= 8.0 * min(serr, 0.25) if s_reliable else 0.0
    if hard_fail:
        s -= 5.0
    return s, {"spread": spread, "sum_asym": sum_asym, "median_mm": med,
               "yellow_min": ymin,
               "uniform_min": umin, "chroma_dev": cdev, "shape_err": serr,
               "shape_reliable": s_reliable, "keystone": keystone,
               "orient_ok": orient_ok, "rms": rms, "qc": qc,
               "hard_fail": hard_fail}


DEFAULT_GATES = {"yellow_min": 0.50, "chroma_dev": 0.50, "keystone": 0.10}


def analyze(img, meta=None, pool=DEFAULT_POOL, min_score=None, gates=None, *, fit_tracks=True):
    """Run the full pipeline. Returns the best validated result, or None."""
    gates = dict(DEFAULT_GATES, **(gates or {}))
    meta=dict(meta or {})
    capture=meta.get('capture','auto')
    if capture not in ('auto','scan','photo'):
        raise ValueError('capture must be auto, scan, or photo')
    if capture=='auto':
        pixels=img[::max(1,img.shape[0]//200),::max(1,img.shape[1]//200)]
        # Platen-like white surroundings select a sharp-edge tolerance. This
        # is a processing heuristic; callers can explicitly declare capture.
        capture='scan' if float((pixels.min(axis=2)>235).mean())>.70 else 'photo'
    meta['capture']=capture
    cands = []
    for meth in pool:
        if meth not in MASKERS:
            continue
        try:
            seed = localize(img, method=meth)
        except (ValueError, RuntimeError, cv2.error) as exc:
            cands.append({"method": meth, "stage": "localize", "ok": False,
                          "reason": str(exc)})
            continue
        if seed is None:
            cands.append({"method": meth, "stage": "localize", "ok": False})
            continue
        try:
            r = refine_iterated(img, seed, meta=meta)
        except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
            cands.append({"method": meth, "stage": "refine", "ok": False, "reason": str(exc)})
            continue
        if r is None:
            cands.append({"method": meth, "stage": "refine", "ok": False,
                          "reason": "insufficient supported edge pairs"})
            continue
        c = centering(r["outer"], r["inner"], portrait=_portrait(r["outer"]))
        # Orientation must be resolvable: if neither short side shows the
        # printed bottom-border text, the inner quad is not on the border.
        try:
            from .orientation import card_orientation
            _, oinfo = card_orientation(img, r["outer"], r["inner"])
            dk = sorted(oinfo["dark"].values())
            orient_ok = dk[-1] > 0.02 and dk[-1] > 3 * max(dk[0], 1e-4)
        except Exception:
            orient_ok = False
        sc, diag = score(r, c, img_shape=img.shape, orient_ok=orient_ok)
        diag['capture']=capture
        clr = frame_clearance(r["outer"], img.shape, r["px_per_mm"])
        diag["clearance_mm"] = clr
        # Check observed straight-side pixels, excluding rounded corners.
        visible=r.get('outside_visible',[0.,0.,0.,0.])
        diag['outside_visible']=dict(zip(('top','right','bottom','left'),visible))
        diag["clipped"] = [k for k, v in diag['outside_visible'].items() if v < .85]
        # Axes are named in QUAD order here; orientation mapping happens later.
        diag["axis_ok"] = {
            "vertical": visible[0]>=.85 and visible[2]>=.85,
            "horizontal": visible[1]>=.85 and visible[3]>=.85,
        }
        if not (diag["axis_ok"]["vertical"] or diag["axis_ok"]["horizontal"]):
            sc -= 5.0          # both axes unusable -> reject outright
            diag["hard_fail"] = True
        c["axis_ok"] = diag["axis_ok"]
        cands.append({"method": meth, "ok": True, "score": sc, "diag": diag,
                      "result": r, "centering": c})

    best=select_validated(cands, gates, meta=meta, min_score=min_score)
    if best is not None and capture=='photo':
        best=refine_selected_segments(img,best,cands,gates,meta,min_score)
        from .yellow_curves import attach_boundary_tracks
        if fit_tracks:
            attach_boundary_tracks(img,best)
    return best,cands


def refine_selected_segments(img,best,cands,gates,meta,min_score):
    """Repair supported end drift, then validate the resulting geometry anew."""
    from .yellow_segments import refine
    from .yellow_precision import frame_evidence
    original=best['result']
    try:
        fit=refine(img,original['outer'],original['inner'],original['px_per_mm'])
    except (ValueError,cv2.error,np.linalg.LinAlgError):
        return best
    if fit is None:return best
    outer,inner,details=fit
    best['diag']['segment_refinement']=details
    blocked=[kind+' '+('top','right','bottom','left')[i]
             for kind in ('outer','inner') for i,d in enumerate(details[kind]) if d.get('blocked')]
    if blocked:
        best['diag']['review_required']=True
        best['diag']['review_reasons']=['Straight-line fit does not follow the full visible edge: '+', '.join(blocked)]
    if not any(d.get('applied') for kind in ('outer','inner') for d in details[kind]):
        return best
    r=dict(original,outer=outer,inner=inner,segment_diagnostics=details)
    r.update(frame_evidence(img,outer,inner))
    r['fit_diagnostics']={kind:[details[kind][i] if details[kind][i].get('applied')
                               else original['fit_diagnostics'][kind][i] for i in range(4)]
                          for kind in ('outer','inner')}
    r['rms_outer']=[d['rms_px'] for d in r['fit_diagnostics']['outer']]
    r['rms_inner']=[d['rms_px'] for d in r['fit_diagnostics']['inner']]
    c=centering(outer,inner,portrait=_portrait(outer))
    r['border_mm']=[25.4*c['margins_in'][k] for k in ('top','right','bottom','left')]
    r['median_border_mm']=float(np.median(r['border_mm']))
    from .orientation import card_orientation
    _,orientation=card_orientation(img,outer,inner)
    dark=sorted(orientation['dark'].values())
    orient_ok=dark[-1]>.02 and dark[-1]>3*max(dark[0],1e-4)
    value,diag=score(r,c,img.shape,orient_ok)
    visible=r['outside_visible']
    diag.update(capture='photo',clearance_mm=frame_clearance(outer,img.shape,r['px_per_mm']),
                outside_visible=dict(zip(('top','right','bottom','left'),visible)),
                clipped=[k for k,v in zip(('top','right','bottom','left'),visible) if v<.85],
                axis_ok={'vertical':visible[0]>=.85 and visible[2]>=.85,
                         'horizontal':visible[1]>=.85 and visible[3]>=.85},
                segment_refinement=details)
    if blocked:
        diag['review_required']=True
        diag['review_reasons']=best['diag']['review_reasons']
    c['axis_ok']=diag['axis_ok']
    candidate=dict(method=best['method']+'+segments',ok=True,score=value,diag=diag,result=r,centering=c)
    reasons=rejection_reasons(candidate,gates,meta,min_score)
    candidate.update(accepted=not reasons,reject_reasons=reasons)
    cands.append(candidate)
    return best if reasons else candidate


def rejection_reasons(candidate, gates, meta=None, min_score=None):
    """Validate each candidate before it can participate in selection."""
    d = candidate['diag']
    r = candidate['result']
    c = candidate['centering']
    reasons = []
    holder=(meta or {}).get('holder','unknown')
    if holder == 'slab':
        reasons.append('unsupported holder: '+holder)
    if not all(c['qc_bounds_ok']):
        reasons.append('inner frame is outside the physical card')
    touching=r.get('border_reaches_frame',[0.,0.,0.,0.])
    if max(touching)>.65:
        sides=[k for k,v in zip(('top','right','bottom','left'),touching) if v>.65]
        reasons.append('yellow border continues into crop at sides: '+','.join(sides))
    widths=np.asarray(r['border_mm'])
    if not np.isfinite(widths).all() or np.any(widths<=0) or np.any(widths>7):
        reasons.append('invalid border geometry')
    if not d['orient_ok']:
        reasons.append('card orientation is not supported by bottom-border text')
    fd=r.get('fit_diagnostics')
    if not fd:
        reasons.append('source-pixel boundary refinement failed')
    else:
        for kind in ('outer','inner'):
            if any(x['support']<.40 or x['span']<.50 for x in fd[kind]):
                reasons.append(kind+' boundary has insufficient support or span')
    # Side slopes are allowed to differ under perspective and print rotation;
    # enforce finite, convex, nested frames without a near-50/50 assumption.
    for kind in ('outer','inner'):
        q=np.asarray(r[kind],np.float32)
        if not np.isfinite(q).all() or not cv2.isContourConvex(q):
            reasons.append(kind+' frame is not a finite convex quadrilateral')
    if d["yellow_min"] < gates["yellow_min"]:
        reasons.append("yellow_min %.2f < %.2f" % (d["yellow_min"], gates["yellow_min"]))
    if d["chroma_dev"] > gates["chroma_dev"]:
        reasons.append("chroma_dev %.2f > %.2f" % (d["chroma_dev"], gates["chroma_dev"]))
    if d["keystone"] > gates["keystone"]:
        reasons.append("fitted perspective %.3f > %.3f" % (d["keystone"], gates["keystone"]))
    if d["clipped"]:
        reasons.append("insufficient visible background at sides: %s" % ",".join(d["clipped"]))
    if min_score is not None and candidate['score']<min_score:
        reasons.append('score %.2f < %.2f' % (candidate['score'],min_score))
    return reasons


def select_validated(cands, gates=None, meta=None, min_score=None):
    gates=dict(DEFAULT_GATES,**(gates or {}))
    valid=[]
    for candidate in cands:
        if not candidate.get('ok'):
            continue
        reasons=rejection_reasons(candidate,gates,meta,min_score)
        candidate['reject_reasons']=reasons
        candidate['accepted']=not reasons
        if not reasons:valid.append(candidate)
    if not valid:return None
    # Choose an actual fitted candidate near the consensus of independent
    # seeds. Do not average coordinates or nudge measurements toward 50/50.
    values=np.array([[x['centering']['left_pct'],x['centering']['top_pct']] for x in valid])
    distances=np.max(np.abs(values[:,None,:]-values[None,:,:]),axis=2)
    counts=(distances<1.5).sum(axis=1)
    if len(valid)>1 and counts.max()==1:
        for candidate in valid:
            candidate['accepted']=False
            candidate['reject_reasons']=['supported candidates disagree on centering']
        return None
    index=max(range(len(valid)),key=lambda i:(int(counts[i]),-float(np.median(distances[i])),valid[i]['score']))
    best=valid[index]
    neighbors=values[distances[index]<1.5]
    best['diag']['agreeing_candidates']=int(counts[index])
    best['diag']['candidate_count']=len(valid)
    best['diag']['seed_spread_pp']=np.ptp(neighbors,axis=0).tolist()
    return best
