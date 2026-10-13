"""Pick edges by CONSENSUS rather than per-sample strength.

The per-sample failure mode: on a low-contrast side, wall speckle or JPEG
ringing outside the card can out-peak the die-cut edge in an individual
profile, so independent per-row picking scatters points across several
different features and the line fit lands between them.

The structure we can exploit: sample bases are collinear along the side, so a
straight physical edge makes the normal offset d an affine function of the
position t along that side. Noise has no such structure. RANSAC over (t, d)
therefore recovers real edges and rejects speckle regardless of relative
contrast -- which is what lets a weak cream-on-cream cut edge survive next to
a strong yellow-on-artwork border edge.
"""
import numpy as np


def _fit_td(t, d):
    A = np.stack([t, np.ones_like(t)], axis=1)
    sol, *_ = np.linalg.lstsq(A, d, rcond=None)
    return sol  # (slope, intercept)


def ransac_td(t, d, tol, iters=400, rng=None, unique_rows=False):
    """Best affine d(t) by RANSAC. Returns (model, inlier_mask) or (None, None)."""
    n = len(t)
    if n < 8:
        return None, None
    rng = rng or np.random.default_rng(0)
    best_model, best_inl = None, None
    best_count = 0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(t[i] - t[j]) < 1e-6:
            continue
        m = (d[j] - d[i]) / (t[j] - t[i])
        c = d[i] - m * t[i]
        res = np.abs(d - (m * t + c))
        inl = res < tol
        cnt = len(np.unique(t[inl])) if unique_rows else int(inl.sum())
        if cnt > best_count:
            best_count, best_model, best_inl = cnt, (m, c), inl
    if best_model is None or best_count < 8:
        return None, None
    if unique_rows:
        # Each independent profile gets one vote and one point. Nearby peaks
        # belonging to artwork must not pull the fitted line between edges.
        model = np.asarray(best_model)
        for _ in range(3):
            res = np.abs(d - (model[0] * t + model[1]))
            order = np.argsort(res)
            _, first = np.unique(t[order], return_index=True)
            idx = order[first]
            idx = idx[res[idx] < tol]
            if len(idx) < 8:
                return None, None
            model = _fit_td(t[idx], d[idx])
        inl = np.zeros(n, bool)
        inl[idx] = True
        return model, inl
    # Least-squares refit on the inliers.
    model = _fit_td(t[best_inl], d[best_inl])
    res = np.abs(d - (model[0] * t + model[1]))
    inl = res < tol
    return model, inl


def two_edges_by_consensus(t, d, strength, tol, min_gap_mm=0.6,
                           min_frac=0.25):
    """Find the two dominant consistent edges in a candidate-peak cloud.

    Returns (outer, inner), each a dict with 'model', 'mask', 'mean_d', or
    (None, None) if two separated edges could not be found.
    """
    n_rows = len(np.unique(t))
    m1, i1 = ransac_td(t, d, tol)
    if m1 is None or i1.sum() < min_frac * n_rows:
        return None, None

    # Mask out the first edge (plus a guard band) and search again.
    d1 = m1[0] * t + m1[1]
    away = np.abs(d - d1) > min_gap_mm
    if away.sum() < 8:
        return None, None
    m2, i2r = ransac_td(t[away], d[away], tol)
    if m2 is None:
        return None, None
    i2 = np.zeros_like(i1)
    idx = np.flatnonzero(away)
    i2[idx[i2r]] = True
    if i2.sum() < min_frac * n_rows:
        return None, None

    e1 = {"model": m1, "mask": i1, "mean_d": float(np.mean(d[i1]))}
    e2 = {"model": m2, "mask": i2, "mean_d": float(np.mean(d[i2]))}
    return (e1, e2) if e1["mean_d"] < e2["mean_d"] else (e2, e1)
