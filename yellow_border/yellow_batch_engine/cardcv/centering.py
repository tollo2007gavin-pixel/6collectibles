"""Rectify with a homography and compute centering percentages.

The homography is solved from the four refined OUTER corners against the known
2.5 x 3.5 in rectangle, then applied to the refined INNER corners. Because the
card is rigid and flat, the same transform that maps the cut edge to its true
position maps every other printed point to its true position too, so the border
corners rectify for free with no extra calibration.
"""
import cv2
import numpy as np

from .geom import order_corners, CARD_W_IN, CARD_H_IN

TARGET = np.array([[0.0, 0.0], [CARD_W_IN, 0.0],
                   [CARD_W_IN, CARD_H_IN], [0.0, CARD_H_IN]], dtype=np.float64)


def _inward_dirs(quad):
    """Unit inward direction at each corner, from the two adjacent edges."""
    q = np.asarray(quad, float)
    c = q.mean(axis=0)
    dirs = []
    for i in range(4):
        prev_e = q[i] - q[i - 1]
        next_e = q[(i + 1) % 4] - q[i]
        acc = np.zeros(2)
        for e in (prev_e, next_e):
            e = e / (np.linalg.norm(e) + 1e-12)
            n = np.array([-e[1], e[0]])
            if np.dot(n, c - q[i]) < 0:
                n = -n
            acc += n
        dirs.append(acc / (np.linalg.norm(acc) + 1e-12))
    return np.array(dirs)


def qc_inward(outer, inner, threshold=0.9):
    """Check A: outer->inner vector must point into the card at every corner."""
    o = order_corners(outer)
    i = order_corners(inner)
    dirs = _inward_dirs(o)
    dots = []
    for k in range(4):
        v = i[k] - o[k]
        n = np.linalg.norm(v)
        dots.append(float(np.dot(v / n, dirs[k])) if n > 1e-9 else 0.0)
    return np.array(dots), [d >= threshold for d in dots]


def qc_bounds(rect_inner, w=CARD_W_IN, h=CARD_H_IN, eps=0.0):
    """Check B: every rectified inner corner must lie inside the card.

    The bounds MUST follow the orientation actually used for the homography.
    Hardcoding 2.5 x 3.5 made every landscape card fail automatically, since
    its rectified x-coordinates run to ~3.4 against a 2.5 ceiling -- a false
    rejection that looked like a detection error.
    """
    ok = []
    for x, y in rect_inner:
        ok.append(bool(-eps <= x <= w + eps and -eps <= y <= h + eps))
    return ok


def centering(outer, inner, portrait=True):
    """Return margins in inches and centering percentages, plus QC results."""
    o = order_corners(outer).astype(np.float32)
    i = order_corners(inner).astype(np.float32)

    dots, dot_ok = qc_inward(o, i)

    target = TARGET.astype(np.float32)
    if not portrait:
        target = np.array([[0, 0], [CARD_H_IN, 0],
                           [CARD_H_IN, CARD_W_IN], [0, CARD_W_IN]], np.float32)

    H = cv2.getPerspectiveTransform(o, target)
    rect = cv2.perspectiveTransform(i.reshape(1, 4, 2), H).reshape(4, 2)

    W, Hh = (CARD_W_IN, CARD_H_IN) if portrait else (CARD_H_IN, CARD_W_IN)
    tl, tr, br, bl = rect
    left = 0.5 * (tl[0] + bl[0])
    right = W - 0.5 * (tr[0] + br[0])
    top = 0.5 * (tl[1] + tr[1])
    bottom = Hh - 0.5 * (bl[1] + br[1])

    lr = 100 * left / (left + right)
    tb = 100 * top / (top + bottom)

    return {
        "H": H, "rect_inner": rect,
        "margins_in": {"left": float(left), "right": float(right),
                       "top": float(top), "bottom": float(bottom)},
        "left_pct": float(lr), "right_pct": float(100 - lr),
        "top_pct": float(tb), "bottom_pct": float(100 - tb),
        "qc_inward_dots": dots, "qc_inward_ok": dot_ok,
        "qc_bounds_ok": qc_bounds(rect, W, Hh),
        "portrait": portrait,
    }
