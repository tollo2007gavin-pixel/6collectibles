"""Pure geometry helpers. No detection logic here."""
import numpy as np

# Real Pokemon card stock is 63 x 88 mm. The commonly quoted 2.5 x 3.5 in
# (63.5 x 88.9 mm) is a rounding of it, and using it biased every homography
# slightly. Verified against a 600dpi flatbed scan measuring 63.4 x 88.0 mm.
CARD_W_MM = 63.0
CARD_H_MM = 88.0
CARD_W_IN = CARD_W_MM / 25.4   # 2.4803
CARD_H_IN = CARD_H_MM / 25.4   # 3.4646
CARD_ASPECT = CARD_W_IN / CARD_H_IN  # 0.71591


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL in image coordinates.

    Uses angle about the centroid, which is robust to rotation and perspective
    (unlike the common sum/difference trick, which breaks past ~30 degrees).
    Assumes the card is roughly upright; for a rotated card 'TL' just means the
    first corner going clockwise from upper-left-ish.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    # Clockwise in image coords (y down) = increasing angle.
    order = np.argsort(ang)
    pts = pts[order]
    # Rotate so the point closest to the top-left of the bounding box is first.
    start = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -start, axis=0)


def quad_side_lengths(quad):
    q = np.asarray(quad, dtype=np.float64)
    return np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])


def quad_area(quad):
    q = np.asarray(quad, dtype=np.float64)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def naive_aspect(quad):
    """width/height from mean opposite side lengths. Biased under perspective."""
    s = quad_side_lengths(quad)
    w = 0.5 * (s[0] + s[2])
    h = 0.5 * (s[1] + s[3])
    return w / h if h else np.inf


def zhang_aspect(quad, img_shape):
    """Recover the TRUE width/height of a rectangle from its projected quad.

    Zhang & He, 'Whiteboard scanning and image enhancement' (2007). Assumes
    square pixels, principal point at the image centre, and no lens distortion.
    Unlike naive_aspect this is perspective-correct, so comparing the result to
    2.5/3.5 is a real validity test on the quad rather than a rough guess.

    Returns (aspect, focal_px) or (None, None) if the configuration is
    degenerate (near-affine view, where focal length is unrecoverable).
    """
    q = order_corners(quad)
    # Zhang's indexing: m1=TL, m2=TR, m3=BL, m4=BR
    tl, tr, br, bl = q
    p1 = np.array([tl[0], tl[1], 1.0])
    p2 = np.array([tr[0], tr[1], 1.0])
    p3 = np.array([bl[0], bl[1], 1.0])
    p4 = np.array([br[0], br[1], 1.0])

    h, w = img_shape[:2]
    u0, v0 = w / 2.0, h / 2.0

    d2 = np.dot(np.cross(p2, p4), p3)
    d3 = np.dot(np.cross(p3, p4), p2)
    if abs(d2) < 1e-9 or abs(d3) < 1e-9:
        return None, None
    k2 = np.dot(np.cross(p1, p4), p3) / d2
    k3 = np.dot(np.cross(p1, p4), p2) / d3

    n2 = k2 * p2 - p1
    n3 = k3 * p3 - p1

    if abs(n2[2]) < 1e-9 or abs(n3[2]) < 1e-9:
        return None, None  # near-affine: focal length unrecoverable

    f_sq = -(
        (n2[0] - n2[2] * u0) * (n3[0] - n3[2] * u0)
        + (n2[1] - n2[2] * v0) * (n3[1] - n3[2] * v0)
    ) / (n2[2] * n3[2])
    if f_sq <= 0:
        return None, None
    f = float(np.sqrt(f_sq))

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
    Ai = np.linalg.inv(A)
    M = Ai.T @ Ai
    num = n2 @ M @ n2
    den = n3 @ M @ n3
    if den <= 0 or num <= 0:
        return None, None
    return float(np.sqrt(num / den)), f


def aspect_ok(quad, img_shape, tol=0.02):
    """True if the quad plausibly is a 63x88mm rectangle (either orientation).

    Tolerance is deliberately tight. At 8% a detection that placed two edges a
    full border width inside the card still passed, because 63x84mm is only
    4.3% off in aspect. Zhang recovery is perspective-corrected, so a real card
    should hold to a couple of percent even in an angled photo.
    """
    a, _ = zhang_aspect(quad, img_shape)
    if a is None:
        a = naive_aspect(quad)
    for target in (CARD_ASPECT, 1.0 / CARD_ASPECT):
        if abs(a - target) / target < tol:
            return True, a
    return False, a
