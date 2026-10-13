"""Sub-pixel edge localization and virtual-corner extrapolation.

This is the stage that actually MEASURES. Everything upstream only has to get
within a few pixels so the search band lands on the right edge.

Method, per card side:
  1. Walk along the approximate side, skipping a trim margin at each end so
     the rounded die-cut corners never contribute.
  2. At each sample, read an intensity profile along the edge NORMAL.
  3. Find the gradient peak in that profile and refine it to sub-pixel by
     fitting a parabola to the three samples around the peak.
  4. Robustly fit a line to the resulting edge points (Huber, so glare
     spikes and stray samples do not drag the fit).
  5. Intersect adjacent fitted lines to get virtual sharp corners.

No thresholding anywhere, so there is no threshold bias to leak into the
centering percentages. Note this deliberately does NOT use a line-segment
detector: the edge is already localized, so there is nothing to search for.
"""
import cv2
import numpy as np

from .geom import order_corners

MM_PER_IN = 25.4
CORNER_RADIUS_MM = 3.0


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _outward_normals(quad):
    """Unit normal for each side, pointing away from the quad centre."""
    q = np.asarray(quad, dtype=np.float64)
    c = q.mean(axis=0)
    normals = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = _unit(b - a)
        n = np.array([-d[1], d[0]])
        mid = 0.5 * (a + b)
        if np.dot(n, mid - c) < 0:
            n = -n
        normals.append(n)
    return np.array(normals)


def sample_edge_points(gray, a, b, normal, search_px, trim_px,
                       n_samples=160, step=0.25, polarity=None,
                       rel_peak=0.35):
    """Return sub-pixel edge points along the side a->b.

    search_px : half-width of the profile searched along the normal
    trim_px   : distance skipped at each end (keeps the fillet out of the fit)
    polarity  : None  -> strongest gradient of either sign
                +1/-1 -> require the signed gradient to have this sign,
                         which lets you pick the card->background step and
                         ignore a nearby shadow or sleeve edge
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    L = np.linalg.norm(b - a)
    if L <= 2 * trim_px + 4:
        return np.empty((0, 2)), np.empty(0)

    t0, t1 = trim_px / L, 1.0 - trim_px / L
    ts = np.linspace(t0, t1, n_samples)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]

    ds = np.arange(-search_px, search_px + 1e-9, step)
    coords = base[:, None, :] + ds[None, :, None] * normal[None, None, :]

    mapx = coords[:, :, 0].astype(np.float32)
    mapy = coords[:, :, 1].astype(np.float32)
    prof = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

    # Smooth along the profile a little to suppress sensor noise / JPEG ringing.
    prof = cv2.GaussianBlur(prof, (1, 5), 0)
    grad = np.gradient(prof, step, axis=1)

    score = grad if polarity is None else grad * polarity
    work = np.abs(grad) if polarity is None else np.where(score > 0, score, 0.0)

    # Choose the strong gradient peak NEAREST the current estimate, not the
    # globally strongest one. A cast shadow or a countertop seam a few mm out
    # is often a harder step than the card edge itself; but the incoming quad
    # is already approximately right, so proximity is the better discriminator.
    peak = np.zeros_like(work, dtype=bool)
    peak[:, 1:-1] = (work[:, 1:-1] >= work[:, :-2]) & (work[:, 1:-1] >= work[:, 2:])
    rowmax = work.max(axis=1, keepdims=True)
    strong = peak & (work >= rel_peak * np.maximum(rowmax, 1e-9))

    cost = np.where(strong, np.abs(ds)[None, :], np.inf)
    k = np.argmin(cost, axis=1)
    found = np.isfinite(cost[np.arange(len(k)), k])
    # Rows with no qualifying peak fall back to the global max.
    k = np.where(found, k, np.argmax(work, axis=1))
    strength = work[np.arange(len(k)), k]

    # Parabolic sub-pixel refinement on the gradient peak.
    kc = np.clip(k, 1, work.shape[1] - 2)
    idx = np.arange(len(kc))
    y0, y1, y2 = work[idx, kc - 1], work[idx, kc], work[idx, kc + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.zeros_like(denom)
    ok = np.abs(denom) > 1e-9
    delta[ok] = 0.5 * (y0[ok] - y2[ok]) / denom[ok]
    delta = np.clip(delta, -1.0, 1.0)

    d_edge = ds[kc] + delta * step
    pts = base + d_edge[:, None] * normal[None, :]

    # Drop samples that never found a real step.
    good = strength > max(1e-6, 0.15 * np.median(strength[strength > 0]) if np.any(strength > 0) else 0)
    return pts[good], strength[good]


def fit_line_huber(pts):
    """Orientation-agnostic robust line fit. Returns (point, direction)."""
    if len(pts) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER,
                                 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), _unit(np.array([vx, vy]))


def line_residuals(line, pts):
    p0, d = line
    n = np.array([-d[1], d[0]])
    return (pts - p0) @ n


def intersect(l1, l2):
    """Intersection of two point-direction lines, or None if near-parallel."""
    p1, d1 = l1
    p2, d2 = l2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def refine_quad(img, quad_approx, px_per_mm=None, search_mm=(3.0, 0.8),
                trim_mm=CORNER_RADIUS_MM * 1.6, polarity=None,
                gray=None):
    """Refine an approximate quad to sub-pixel virtual corners.

    Returns dict with 'quad', 'lines', 'rms' (per-side fit residual in px)
    and 'n_pts' (samples surviving per side).
    """
    q = order_corners(quad_approx)
    if gray is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if px_per_mm is None:
        # Short side of the quad is the card's 2.5in = 63.5mm dimension.
        s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        # Average each OPPOSITE pair first, then take the smaller average. The
    # previous form -- 0.5*(min(s0,s2) + min(s1,s3)) -- took the shorter of the
    # top/bottom pair and the shorter of the left/right pair and averaged those,
    # mixing one short side with one long side. That inflated px/mm by ~19% on
    # every card, miscalibrating search bands, trim distances and border_mm.
    # (Centering percentages were unaffected: they come from the homography,
    # which normalises to 63x88mm independently of this value.)
        short = min(0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]))
        px_per_mm = short / 63.0

    if np.isscalar(search_mm):
        search_mm = (search_mm,)
    trim_px = max(6.0, trim_mm * px_per_mm)
    normals = _outward_normals(q)

    lines, rms, npts, allpts = [], [], [], []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        # Coarse-to-fine: a wide band absorbs localizer error, then a narrow
        # band re-centred on the fitted line rejects nearby competing edges.
        for pass_i, smm in enumerate(search_mm):
            search_px = max(4.0, smm * px_per_mm)
            pts, _ = sample_edge_points(gray, a, b, normals[i], search_px,
                                        trim_px, polarity=polarity)
            if len(pts) < 10:
                return None
            line = fit_line_huber(pts)
            if pass_i < len(search_mm) - 1:
                # Re-project the side onto the fitted line for the next pass.
                p0, d = line
                a = p0 + np.dot(a - p0, d) * d
                b = p0 + np.dot(b - p0, d) * d
        r = line_residuals(line, pts)
        # One robust re-fit after discarding outliers beyond 3 MAD.
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r - np.median(r)) < 3 * 1.4826 * mad
        if keep.sum() >= 10:
            pts = pts[keep]
            line = fit_line_huber(pts)
            r = line_residuals(line, pts)
        lines.append(line)
        rms.append(float(np.sqrt(np.mean(r ** 2))))
        npts.append(int(len(pts)))
        allpts.append(pts)

    corners = []
    for i in range(4):
        p = intersect(lines[i - 1], lines[i])
        if p is None:
            return None
        corners.append(p)
    # lines[i] is side q[i]->q[i+1]; intersection of side i-1 and i meets at q[i]
    corners = np.array(corners)

    return {"quad": order_corners(corners), "lines": lines, "rms": rms,
            "n_pts": npts, "pts": allpts, "px_per_mm": px_per_mm,
            "search_px": search_px, "trim_px": trim_px}
