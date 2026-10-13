"""Enumerate every consistent edge on a side, then choose the card pair.

Why enumeration: consensus RANSAC filters noise, but a toploader rim, a penny
sleeve lip and a cast shadow are all geometrically real straight lines. No
purely geometric rule separates them from the die cut. So we stop trying to
pick the right edge and instead extract ALL of them, then disambiguate with a
physical test that geometry cannot do:

    the true card pair has a ~2.2mm YELLOW band between the two lines,
    and is NOT yellow immediately outside the outer line.

Colour is used here as a classifier on an already-located band, not as a
detector. That is the job it is actually good at -- the reverse arrangement is
what made the yellow localizer fail under warm light.
"""
import cv2
import numpy as np

from .consensus import ransac_td
from .geom import order_corners
from .refine import _outward_normals, fit_line_huber, line_residuals, intersect, MM_PER_IN

# Physical priors for a modern Pokemon card border.
BORDER_MM_MIN = 1.3
BORDER_MM_MAX = 3.4
BORDER_MM_NOMINAL = 2.2


def _sample_lab(img_lab, base, normal, ds, px_per_mm):
    """Sample L*, a*, b* along the normal for colour-aware edge detection."""
    inward = -normal
    coords = base[:, None, :] + (ds * px_per_mm)[None, :, None] * inward[None, None, :]
    mx = coords[:, :, 0].astype(np.float32)
    my = coords[:, :, 1].astype(np.float32)
    out = []
    for ch in range(3):
        p = cv2.remap(img_lab[:, :, ch], mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        out.append(_smooth_profile(p, ds, px_per_mm))
    return out


def _sample(img_gray, img_hsv, base, normal, ds, px_per_mm):
    inward = -normal
    coords = base[:, None, :] + (ds * px_per_mm)[None, :, None] * inward[None, None, :]
    mx = coords[:, :, 0].astype(np.float32)
    my = coords[:, :, 1].astype(np.float32)
    g = cv2.remap(img_gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    h = cv2.remap(img_hsv[:, :, 0], mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    s = cv2.remap(img_hsv[:, :, 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    v = cv2.remap(img_hsv[:, :, 2], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _smooth_profile(g, ds, px_per_mm), h, s, v


def _smooth_profile(profile, ds, ppm):
    # Smooth along the normal in SOURCE pixels. Oversampling a bilinear remap
    # without this creates multiple staircase peaks from one physical edge.
    sigma = max(0.5, 0.65 / ((ds[1] - ds[0]) * ppm))
    return cv2.GaussianBlur(profile, (0, 0), sigmaX=sigma, sigmaY=0.7)


def _candidates(work, ds, rel_peak=0.02, max_per_row=14, noise_k=2.5):
    """All plausible gradient peaks, with sub-pixel offsets.

    The threshold is ABSOLUTE (a multiple of the profile's own noise floor)
    rather than a fraction of the row maximum. Relative thresholding lets a
    strong line suppress a genuine weaker one nearby: on a 600dpi scan the true
    die cut registered ~797 while printed content just inside it registered
    ~2163, so at rel_peak=0.08 the real card edge was discarded on two of four
    sides and no selection rule downstream could recover it.

    The outermost surviving peak in each row is also force-kept regardless of
    strength -- nothing outside the card edge can itself be the card edge, so
    the outermost real step is always worth carrying as a candidate.
    """
    peak = np.zeros_like(work, dtype=bool)
    peak[:, 1:-1] = (work[:, 1:-1] >= work[:, :-2]) & (work[:, 1:-1] >= work[:, 2:])

    # Robust per-side noise floor from the median absolute gradient.
    noise = float(np.median(work)) + 1e-9
    rowmax = np.maximum(work.max(axis=1, keepdims=True), 1e-9)
    thresh = np.maximum(noise_k * noise, rel_peak * rowmax)
    strong = peak & (work >= thresh)

    step = ds[1] - ds[0]
    ts, dv, sv, sh = [], [], [], []
    for r in range(work.shape[0]):
        k = np.flatnonzero(strong[r])
        if not len(k):
            continue
        keep = set(k[np.argsort(work[r, k])[::-1][:max_per_row]].tolist())
        keep.add(int(k[0]))            # outermost peak on this row, always
        for kk in sorted(keep):
            kc = int(np.clip(kk, 1, work.shape[1] - 2))
            y0, y1, y2 = work[r, kc - 1], work[r, kc], work[r, kc + 1]
            den = y0 - 2 * y1 + y2
            dl = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            ts.append(r)
            dv.append(ds[kc] + float(np.clip(dl, -1, 1)) * step)
            sv.append(work[r, kc])
            # Peak sharpness: a die cut is an abrupt step, while a cast shadow
            # ramps gradually, so the two are separable by peak WIDTH even when
            # their heights are comparable. Normalised curvature is scale-free.
            sh.append(float((2 * y1 - y0 - y2) / (y1 + 1e-9)))
    return (np.array(ts, float), np.array(dv, float),
            np.array(sv, float), np.array(sh, float))


def extract_lines(t, d, s, tol, max_lines=9, min_frac=0.12, sharp=None):
    """Repeatedly RANSAC out consistent affine d(t) lines, strongest support first."""
    n_rows = len(np.unique(t))
    lines = []
    live = np.ones(len(t), bool)
    for _ in range(max_lines):
        if live.sum() < 12:
            break
        m, inl = ransac_td(t[live], d[live], tol, unique_rows=True)
        if m is None:
            break
        idx = np.flatnonzero(live)[inl]
        if len(np.unique(t[idx])) < min_frac * n_rows:
            break
        lines.append({"model": m, "idx": idx,
                      "sharp": float(np.median(sharp[idx])) if sharp is not None else 0.0,
                      "mean_d": float(np.mean(d[idx])),
                      "support": int(len(np.unique(t[idx]))),
                      "strength": float(np.median(s[idx]))})
        live[idx] = False
    lines.sort(key=lambda L: L["mean_d"])
    return lines


def plateau_end(A, B, ds, ppm, ref_lo=0.25, ref_hi=0.75, tol=9.0, n_seg=8):
    """Where the border colour LAST matches, per sample row.

    Gradient-peak detection asks "where is the first strong step inward", which
    is wrong on the bottom edge of a Pokemon card: the illustrator credit, set
    number and copyright line are printed ON the yellow border, so the first
    strong step is a letter, not the border's end. Asking instead where the
    border colour last appears is immune to that -- text interrupts the plateau
    but the border resumes after it.

    The reference colour is estimated PER SEGMENT along the side, not once for
    the whole edge. Glare, holo shimmer and shading change the border's own
    apparent colour over part of its length; a single global reference then
    stops matching early and the prior comes back short. Measured across eight
    phone photos of one card, the two with badly wrong bottom margins were
    exactly the two whose global-reference prior read low (1.73, 1.81 vs
    2.05-2.55 elsewhere), leaving the prior too far from the truth to constrain
    anything.

    Used for SELECTION only. The plateau ends where the transition ramp begins,
    so it sits ~half a ramp inside the true 50% crossing; the gradient peak
    still provides the sub-pixel position.
    """
    m = (ds >= ref_lo) & (ds <= ref_hi)
    if m.sum() < 2:
        return None
    n_rows = A.shape[0]
    outward = ds > ref_hi
    seg = max(1, min(n_seg, n_rows // 20))
    bounds = np.linspace(0, n_rows, seg + 1).astype(int)

    ts, dv = [], []
    for si in range(seg):
        r0, r1 = bounds[si], bounds[si + 1]
        if r1 - r0 < 5:
            continue
        ref = np.array([np.median(A[r0:r1, m]), np.median(B[r0:r1, m])])
        sub = np.hypot(A[r0:r1] - ref[0], B[r0:r1] - ref[1]) < tol
        for k in range(r1 - r0):
            idx = np.flatnonzero(sub[k] & outward)
            if len(idx):
                ts.append(r0 + k)
                dv.append(ds[idx[-1]])
    if len(ts) < 20:
        return None
    ts = np.array(ts, float)
    dv = np.array(dv, float)
    model, inl = ransac_td(ts, dv, max(0.05, 2.0 / ppm))
    if model is None or inl.sum() < 15:
        return None
    return float(np.mean(dv[inl]))


def _band_chroma(hue, sat, ds, d_lo, d_hi):
    """Median chroma vector of the band, plus how uniform it is along its length.

    Chroma (s*cos h, s*sin h) is effectively LAB a*/b*. Colour-agnostic: it
    characterises WHAT colour the band is without assuming which colour to
    expect, so the same statistic works for yellow, silver, gold or black.
    """
    m = (ds >= d_lo) & (ds <= d_hi)
    if m.sum() < 2:
        return np.array([0.0, 0.0]), 0.0
    H = hue[:, m].astype(np.float32) * np.pi / 90.0
    S = sat[:, m].astype(np.float32)
    cx, cy = (S * np.cos(H)).ravel(), (S * np.sin(H)).ravel()
    med = np.array([np.median(cx), np.median(cy)])
    scale = max(float(np.hypot(*med)), 20.0)
    uni = float((np.hypot(cx - med[0], cy - med[1]) < 0.6 * scale).mean())
    return med, uni


def _yellow_frac(hue, sat, val, ds, d_lo, d_hi):
    """Fraction of samples between two offsets that look like border yellow."""
    m = (ds >= d_lo) & (ds <= d_hi)
    if m.sum() < 2:
        return 0.0
    H, S, V = hue[:, m], sat[:, m], val[:, m]
    yel = (H >= 12) & (H <= 42) & (S >= 55) & (V >= 60)
    return float(yel.mean())


def score_pairs(lines, hue, sat, val, ds, n_rows, meta=None,
                wmin=BORDER_MM_MIN, wmax=BORDER_MM_MAX, inner_prior=None):
    """Score every (outer, inner) line pair on physical plausibility."""
    # Holder metadata does not specify a count of visible lines.
    # Shadows, seams and multiple sleeve rims make line ordering unreliable.
    # A POSITIONAL prior: the caller says roughly where the card edge sits,
    # expressed as a tolerance in mm around the seed side (d = 0). Unlike a
    # holder count, this says where the card IS rather than requiring us to
    # infer it from line ordering -- which fails because shadows and seams also
    # produce consistent lines outside the card.
    edge_tol = (meta or {}).get("edge_tol_mm")
    out = []
    for i in range(len(lines)):
        if edge_tol is not None and abs(lines[i]["mean_d"]) > edge_tol:
            continue
        for j in range(i + 1, len(lines)):
            lo, li = lines[i], lines[j]
            w = li["mean_d"] - lo["mean_d"]
            if not (wmin <= w <= wmax):
                continue
            inside = _yellow_frac(hue, sat, val, ds,
                                  lo["mean_d"] + 0.25 * w, li["mean_d"] - 0.25 * w)
            outside = _yellow_frac(hue, sat, val, ds,
                                   lo["mean_d"] - 0.9, lo["mean_d"] - 0.2)
            beyond = _yellow_frac(hue, sat, val, ds,
                                  li["mean_d"] + 0.2, li["mean_d"] + 0.9)
            support = (lo["support"] + li["support"]) / (2.0 * n_rows)
            width_pen = abs(w - BORDER_MM_NOMINAL) / BORDER_MM_NOMINAL
            prior_pen = 0.0
            if inner_prior is not None:
                # Plateau end sits slightly inside the 50% crossing, so allow a
                # small offset before penalising.
                prior_pen = max(0.0, abs(li["mean_d"] - inner_prior) - 0.25)
            sharp_bonus = 0.7 * float(np.clip(lo.get("sharp", 0.0), 0.0, 1.0))
            score = (sharp_bonus + 1.6 * inside - 1.0 * outside - 0.5 * beyond
                     + 0.6 * support - 0.5 * width_pen
                     - 1.4 * min(prior_pen, 1.5))
            chroma, uni = _band_chroma(hue, sat, ds,
                                       lo["mean_d"] + 0.25 * w,
                                       li["mean_d"] - 0.25 * w)
            out.append({"outer": lo, "inner": li, "width_mm": w, "score": score,
                        "chroma": chroma, "uniform": uni,
                        "yellow_in": inside, "yellow_out": outside,
                        "yellow_beyond": beyond, "support": support})
    out.sort(key=lambda p: -p["score"])
    return out


def analyze_side(gray, hsv, a, b, normal, px_per_mm, trim_px,
                 out_mm=4.0, in_mm=7.0, step_mm=0.02, n_samples=220, meta=None,
                 lab_img=None, l_weight=1.0, adaptive=True,
                 rel_peak=0.08, min_frac=0.20, wmin=BORDER_MM_MIN,
                 wmax=BORDER_MM_MAX, max_lines=6):
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a)
    if L <= 2 * trim_px + 4:
        return None
    t0, t1 = trim_px / L, 1.0 - trim_px / L
    ts = np.linspace(t0, t1, n_samples)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    step_mm = max(step_mm, 0.25 / px_per_mm)
    ds = np.arange(-out_mm, in_mm + 1e-9, step_mm)

    g, hue, s_, v_ = _sample(gray, hsv, base, normal, ds, px_per_mm)

    # Colour-aware edge strength. Grayscale alone nearly misses the die cut on
    # a yellow border against white paper -- similar luminance, very different
    # chroma -- while border-to-artwork is strong in both. Measured on a 600dpi
    # flatbed scan, the luminance-only profile put two of four edges a full
    # border width inside the card. Combining L* with a*/b* makes the real cut
    # edge visible on every side.
    prior = None
    if lab_img is not None:
        L, A, B = _sample_lab(lab_img, base, normal, ds, px_per_mm)
        prior = plateau_end(A, B, ds, px_per_mm)
        gL = np.gradient(L, step_mm, axis=1)
        gA = np.gradient(A, step_mm, axis=1)
        gB = np.gradient(B, step_mm, axis=1)
        if adaptive:
            # Weight each channel by how well it actually separates features in
            # THIS image, rather than by a fixed constant. A white scanner bed
            # separates from a yellow border in chroma but barely in luminance
            # (and luminance additionally carries the card's cast shadow); a
            # cream wall separates in luminance but barely in chroma. A single
            # global weighting therefore cannot serve both, which is exactly
            # what the l_weight sweep showed: 0.35 halved between-scan
            # disagreement while driving one photo to an impossible negative
            # margin. Weight = per-channel dynamic range over per-channel noise.
            ws = []
            for ch in (L, A, B):
                med = np.median(ch, axis=0)
                span = float(np.percentile(med, 95) - np.percentile(med, 5))
                noise = float(np.median(np.abs(np.diff(med)))) + 1e-6
                ws.append(span / noise)
            ws = np.array(ws, dtype=float)
            ws = ws / (ws.max() + 1e-9)
            wL, wA, wB = ws
        else:
            wL, wA, wB = l_weight, 1.0, 1.0

        # Luminance is downweighted because a card casts a faint shadow on a
        # light background: L* starts falling BEFORE any colour change, so a
        # luminance-weighted gradient peaks outside the true die cut. Measured
        # on a 600dpi scan, the L* 50% crossing sat up to 0.18mm outside the
        # chroma crossing, and the offsets rotated with the SCANNER frame
        # rather than the card -- i.e. directional illumination, not the card.
        work = np.sqrt((wL * gL) ** 2 + (wA * gA) ** 2 + (wB * gB) ** 2)
    else:
        work = np.abs(np.gradient(g, step_mm, axis=1))
    t, d, sv, shv = _candidates(work, ds, rel_peak=rel_peak)
    if len(t) < 20:
        return None

    tol = max(0.04, 0.8 / px_per_mm)
    lines = extract_lines(t, d, sv, tol, max_lines=max_lines,
                          min_frac=min_frac, sharp=shv)
    if len(lines) < 2:
        return None
    pairs = score_pairs(lines, hue, s_, v_, ds, n_samples, meta=meta,
                        wmin=wmin, wmax=wmax, inner_prior=prior)
    # A nonempty list can consist entirely of printed transitions. Such junk
    # pairs previously prevented the wider retry even when both real edges had
    # already been detected. Only extend the nominal width window when it has
    # no convincing yellow strip with non-yellow surroundings; require that
    # stronger evidence for every added pair. Keep identical peaks and search.
    def physical_strip(pair):
        return pair['yellow_in']>=.85 and pair['yellow_out']<=.25 and pair['support']>=.50
    if (meta or {}).get('retry_wide_pairs',False) and wmax<4.2 and not any(physical_strip(p) for p in pairs):
        expanded=score_pairs(lines,hue,s_,v_,ds,n_samples,meta=meta,
                             wmin=wmax+1e-6,wmax=4.2,inner_prior=prior)
        for p in expanded:
            if physical_strip(p):
                p['width_recovery']='supported yellow strip beyond nominal width window'
                pairs.append(p)
        pairs.sort(key=lambda p:-p['score'])
    if not pairs:
        return None
    out = {"base": base, "normal": normal, "lines": lines, "pairs": pairs,
           "t": t, "d": d, "inner_prior": prior}
    if lab_img is not None:
        out["A"], out["B"], out["ds"], out["Lp"] = A, B, ds, L
    return out


def _pts(side, line, prior=None, prior_tol=0.4):
    """Points belonging to a fitted line, optionally filtered by a prior.

    On the bottom border the illustrator credit and copyright line are printed
    ON the yellow band, so some inlier points sit on letter edges rather than
    the border's end. Which of those survive RANSAC varies with the seed, and
    that alone moved the fitted border width by up to 0.195mm between
    localizers even though every one of them selected the same line. Filtering
    against the plateau-derived position removes the contaminated points before
    the sub-pixel fit.
    """
    base, normal = side["base"], side["normal"]
    idx = line["idx"]
    tt = side["t"][idx].astype(int)
    dd = side["d"][idx]
    if prior is not None:
        keep = np.abs(dd - prior) < prior_tol
        if keep.sum() >= 25:
            tt, dd = tt[keep], dd[keep]
    return base[tt] + (dd * side.get("ppm", 1.0))[:, None] * (-normal)[None, :]


def chroma_outer_points(side, line, ppm, win=0.35, min_contrast=12.0,
                        shadow_mm=0.10):
    """Relocate the outer edge using chroma only.

    A cast shadow changes L* but not hue, so a luminance-weighted gradient
    peaks OUTSIDE the true die cut wherever the card shades its background.
    Measured on 600dpi scans the offset reached 0.18mm and rotated with the
    SCANNER frame, so it is illumination geometry, not a property of the card.
    Chroma is immune to it -- but only usable where the background actually
    differs in chroma from the border, so contrast is checked first and the
    luminance-based estimate is kept when it is not.
    """
    if "A" not in side:
        return None
    A, B, ds = side["A"], side["B"], side["ds"]
    base, normal = side["base"], side["normal"]
    chroma = np.hypot(A - 128.0, B - 128.0)
    idx = line["idx"]
    tt = side["t"][idx].astype(int)
    dd = side["d"][idx]

    med = np.median(chroma, axis=0)
    d0 = float(np.median(dd))
    outside = med[(ds > d0 - 1.2) & (ds < d0 - 0.35)]
    inside = med[(ds > d0 + 0.35) & (ds < d0 + 1.2)]
    if outside.size < 3 or inside.size < 3:
        return None
    if abs(float(np.median(inside)) - float(np.median(outside))) < min_contrast:
        return None                      # chroma cannot see this edge

    # Only override the luminance-based edge where a SHADOW is actually
    # present. A cast shadow darkens the background before any colour change,
    # so its signature is the L* transition sitting measurably OUTSIDE the
    # chroma transition. Applying chroma everywhere costs accuracy on clean,
    # shadow-free sides (measured: T/B mean error tripled, one low-resolution
    # card regressed 2 points), so the correction is gated on detecting the
    # thing it corrects.
    Lp = side.get("Lp")
    if Lp is None:
        return None
    def _cross(prof):
        m = np.median(prof, axis=0)
        lo = np.median(m[(ds > d0 - 1.2) & (ds < d0 - 0.5)])
        hi = np.median(m[(ds > d0 + 0.5) & (ds < d0 + 1.2)])
        if abs(hi - lo) < 1e-6:
            return None
        tgt = 0.5 * (lo + hi)
        sel = np.flatnonzero((m[:-1] - tgt) * (m[1:] - tgt) <= 0)
        return float(ds[sel[0]]) if len(sel) else None
    cL = _cross(Lp)
    cC = _cross(chroma)
    if cL is None or cC is None:
        return None
    if (cC - cL) < shadow_mm:            # luminance edge not outside chroma edge
        return None

    step = ds[1] - ds[0]
    g = np.abs(np.gradient(chroma, step, axis=1))
    new_t, new_d = [], []
    for row, d_old in zip(tt, dd):
        sel = np.flatnonzero((ds > d_old - win) & (ds < d_old + win))
        if len(sel) < 3:
            continue
        seg = g[row, sel]
        k = int(sel[int(np.argmax(seg))])
        k = int(np.clip(k, 1, g.shape[1] - 2))
        y0, y1, y2 = g[row, k - 1], g[row, k], g[row, k + 1]
        den = y0 - 2 * y1 + y2
        dl = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        new_t.append(row)
        new_d.append(ds[k] + float(np.clip(dl, -1, 1)) * step)
    if len(new_t) < 25:
        return None
    new_t = np.array(new_t, int)
    new_d = np.array(new_d, float)
    return base[new_t] + (new_d * ppm)[:, None] * (-normal)[None, :]


def _fit(pts):
    line = fit_line_huber(pts)
    r = line_residuals(line, pts)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 3 * 1.4826 * mad
    if keep.sum() >= 10:
        pts = pts[keep]
        line = fit_line_huber(pts)
        r = line_residuals(line, pts)
    return line, float(np.sqrt(np.mean(r ** 2)))


def _quad_from_pairs(sides, combo, key="outer"):
    """Build a quad by intersecting the chosen line on each of the four sides."""
    lines = []
    for i, sd in enumerate(sides):
        pair = sd["pairs"][combo[i]]
        ln = pair[key]
        # Convert the affine d(t) model back into a point+direction line.
        base, normal = sd["base"], sd["normal"]
        m, c0 = ln["model"]
        n = len(base)
        t0, t1 = 0.0, float(n - 1)
        p0 = base[0] + ((m * t0 + c0) * sd["ppm"]) * (-normal)
        p1 = base[-1] + ((m * t1 + c0) * sd["ppm"]) * (-normal)
        d = p1 - p0
        nd = np.linalg.norm(d)
        if nd < 1e-9:
            return None
        lines.append((p0, d / nd))
    pts = []
    for i in range(4):
        p = intersect(lines[i - 1], lines[i])
        if p is None:
            return None
        pts.append(p)
    return order_corners(np.array(pts))


def _shape_error(quad, img_shape):
    """How far the quad is from a true 63x88mm rectangle, as a fraction."""
    from .geom import zhang_aspect, naive_aspect, CARD_ASPECT
    a, _ = zhang_aspect(quad, img_shape)
    if a is None:
        a = naive_aspect(quad)
    best = min(abs(a - CARD_ASPECT) / CARD_ASPECT,
               abs(a - 1.0 / CARD_ASPECT) / (1.0 / CARD_ASPECT))
    return float(best)


def select_jointly(sides, img_shape, top_k=4, w_shape=6.0):
    """Choose one edge pair per side by scoring all four TOGETHER.

    Per-side selection cannot tell the die cut from a printed line running
    parallel just inside it -- both are real, straight and well supported, and
    the band inside either one still looks like border. What distinguishes them
    is global: a card is a rectangle of KNOWN proportions, so a combination that
    puts two edges a border width inside implies a 63x84mm card and is
    impossible. Measured on a 600dpi scan, per-side selection placed the two
    short edges 2.90mm and 0.89mm inside the card while the long edges were
    correct to 0.12mm; the resulting shape error is what this rejects.
    """
    cand = [list(range(min(top_k, len(sd["pairs"])))) for sd in sides]
    best, best_score = None, -1e9
    for c0 in cand[0]:
        for c1 in cand[1]:
            for c2 in cand[2]:
                for c3 in cand[3]:
                    combo = (c0, c1, c2, c3)
                    quad = _quad_from_pairs(sides, combo)
                    if quad is None:
                        continue
                    err = _shape_error(quad, img_shape)
                    colour = sum(sides[i]["pairs"][combo[i]]["score"]
                                 for i in range(4)) / 4.0
                    widths = np.array([sides[i]["pairs"][combo[i]]["width_mm"]
                                       for i in range(4)])
                    spread = float(widths.max() - widths.min())
                    sc = colour - w_shape * min(err, 0.5)
                    if sc > best_score:
                        best, best_score = combo, sc
    return best, best_score


def refine_by_enumeration(img, quad_approx, meta=None,
                          trim_mm=3.0 * 1.7, consistency_mm=0.8, chroma_tol=0.35,
                          out_mm=4.0, in_mm=7.0, l_weight=1.0, adaptive=True):
    """Select edge identities jointly, then fit their source-pixel transitions.

    Colour agreement helps reject non-border pairs; side widths are allowed
    to differ. Report validation evidence resampled on the final fitted quads.
    """
    q = order_corners(quad_approx)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    # Average each OPPOSITE pair first, then take the smaller average. The
    # previous form -- 0.5*(min(s0,s2) + min(s1,s3)) -- took the shorter of the
    # top/bottom pair and the shorter of the left/right pair and averaged those,
    # mixing one short side with one long side. That inflated px/mm by ~19% on
    # every card, miscalibrating search bands, trim distances and border_mm.
    # (Centering percentages were unaffected: they come from the homography,
    # which normalises to 63x88mm independently of this value.)
    short = min(0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]))
    ppm = short / 63.0
    trim_px = max(6.0, trim_mm * ppm)
    normals = _outward_normals(q)

    sides = []
    for i in range(4):
        sd = analyze_side(gray, hsv, q[i], q[(i + 1) % 4], normals[i], ppm,
                          trim_px, meta=meta, out_mm=out_mm, in_mm=in_mm,
                          lab_img=lab, l_weight=l_weight, adaptive=adaptive)
        if sd is None:
            # Relaxed retry for this side only: accept weaker peaks, shorter
            # consistent runs (glare can wipe out part of an edge) and a wider
            # width window. A single hard side should not fail the whole card.
            sd = analyze_side(gray, hsv, q[i], q[(i + 1) % 4], normals[i], ppm,
                              trim_px, meta=meta, rel_peak=0.04, min_frac=0.10,
                              lab_img=lab, l_weight=l_weight, adaptive=adaptive,
                              wmin=1.0, wmax=4.2, max_lines=9,
                              out_mm=out_mm + 1.0, in_mm=in_mm + 1.0)
        if sd is None:
            return None
        sd["ppm"] = ppm
        sides.append(sd)

    # Round 1: joint selection across all four sides (falls back to per-side).
    combo, _ = select_jointly(sides, img.shape)
    if combo is None:
        chosen = [sd["pairs"][0] for sd in sides]
    else:
        chosen = [sides[i]["pairs"][combo[i]] for i in range(4)]
    widths = np.array([c["width_mm"] for c in chosen])
    med = float(np.median(widths))

    # Do not re-pick a supported boundary to equalize opposite border widths:
    # their difference is the centering signal, not a fit error.

    # Round 3: cross-side colour agreement. A real border is one printed colour
    # all the way round, and all four sides share one illuminant, so the border
    # colour can be estimated from the image itself -- no reference value and no
    # colour constancy needed. A side whose band colour disagrees with the other
    # three is measuring something else (sleeve, holder, artwork, background).
    ch = np.array([c["chroma"] for c in chosen], dtype=float)
    ref = np.median(ch, axis=0)
    ref_mag = max(float(np.hypot(*ref)), 20.0)
    dev = np.hypot(ch[:, 0] - ref[0], ch[:, 1] - ref[1]) / ref_mag
    for i, sd in enumerate(sides):
        if dev[i] <= chroma_tol:
            continue
        cands = [p for p in sd["pairs"]
                 if np.hypot(*(p["chroma"] - ref)) / ref_mag <= chroma_tol
                 and abs(p["width_mm"] - med) <= consistency_mm * 1.5]
        if cands:
            chosen[i] = cands[0]
    ch = np.array([c["chroma"] for c in chosen], dtype=float)
    dev = np.hypot(ch[:, 0] - ref[0], ch[:, 1] - ref[1]) / ref_mag

    lo, li, rms_o, rms_i = [], [], [], []
    for i, sd in enumerate(sides):
        # RE-ENABLED. A soft cast shadow outside the card creates a luminance
        # ramp up to ~1.3mm wide BEFORE any colour change, so a luminance-
        # weighted gradient peaks outside the true die cut. Measured on two
        # eBay listings against hand-clicked corners, the left outer line sat
        # 2.7-4.6px outside the card while the right (shadow-free) side agreed
        # to under 1px -- inflating the smaller margin and compressing the
        # centering ratio by ~3 points. Chroma is immune: b* stays flat right
        # up to the edge on both sides. Previously disabled after it failed to
        # help on flatbed scans, where the effect was only ~0.18mm.
        # (old note retained below)
        # NOTE: chroma_outer_points() was implemented and tested but NOT used.
        # It corrects a real effect (shadow biases the L*-weighted edge outward
        # by up to 0.18mm) yet did not improve results: between-scan T/B stayed
        # at 1.70 while L/R agreement degraded from 0.00 to 0.24. Three separate
        # attacks -- fixed L* weighting, adaptive weighting, chroma relocation --
        # all left the between-scan T/B gap at ~1.7, which argues the residual
        # is not the shadow at all. Kept for a future test with scans taken at
        # different positions on the platen, which would separate scanner
        # geometry from edge estimation.
        po = chroma_outer_points(sd, chosen[i]["outer"], ppm)
        if po is None or len(po) < 25:
            po = _pts(sd, chosen[i]["outer"])
        pi = _pts(sd, chosen[i]["inner"], prior=sd.get("inner_prior"))
        l1, r1 = _fit(po); l2, r2 = _fit(pi)
        lo.append(l1); li.append(l2); rms_o.append(r1); rms_i.append(r2)

    def corners(lines):
        pts = []
        for i in range(4):
            p = intersect(lines[i - 1], lines[i])
            if p is None:
                return None
            pts.append(p)
        return order_corners(np.array(pts))

    outer, inner = corners(lo), corners(li)
    if outer is None or inner is None:
        return None

    from .yellow_precision import refine_boundaries, frame_evidence
    precise = refine_boundaries(img, outer, inner, ppm, capture=(meta or {}).get('capture','photo'))
    fit_diag = None
    if precise is not None:
        outer, inner, fit_diag = precise
        rms_o = [x['rms_px'] for x in fit_diag['outer']]
        rms_i = [x['rms_px'] for x in fit_diag['inner']]

    result = {"outer": outer, "inner": inner, "rms_outer": rms_o, "rms_inner": rms_i,
            "border_mm": [c["width_mm"] for c in chosen],
            "yellow_in": [c["yellow_in"] for c in chosen],
            "yellow_out": [c["yellow_out"] for c in chosen],
            "n_lines": [len(sd["lines"]) for sd in sides],
            "uniform": [c["uniform"] for c in chosen],
            "chroma_dev": dev.tolist(),
            "px_per_mm": ppm, "median_border_mm": med,
            "fit_diagnostics": fit_diag}
    recovered_widths=[dict(side=i,width_mm=p['width_mm'],yellow_in=p['yellow_in'],
        yellow_out=p['yellow_out'],support=p['support']) for i,p in enumerate(chosen) if p.get('width_recovery')]
    if recovered_widths:result['pair_width_recovery']=recovered_widths
    result.update(frame_evidence(img, outer, inner))
    # All diagnostics must describe the final lines, not the initial pair.
    from .centering import centering
    lengths=[np.linalg.norm(outer[(i+1)%4]-outer[i]) for i in range(4)]
    c=centering(outer,inner,portrait=(lengths[0]+lengths[2]<lengths[1]+lengths[3]))
    m=c['margins_in']
    result['border_mm']=[25.4*m[k] for k in ('top','right','bottom','left')]
    result['median_border_mm']=float(np.median(result['border_mm']))
    return result
