"""Two-stage localization.

Stage 1 finds WHERE the card is on a downscaled image -- cheap, and only has
to be roughly right. Stage 2 re-runs the same segmentation at full resolution
inside a padded crop with a small morphology kernel.

Why this matters: the single-stage version thresholds at 1/8 scale with a 7px
kernel, so the returned quad can sit ~2mm off the true edge. That is not a
cosmetic error -- it forces the refinement search band to be wide, and a wide
band is exactly what lets a toploader or sleeve edge become a candidate. Tight
localization is what allows a narrow, unambiguous band downstream.
"""
import cv2
import numpy as np

from .geom import order_corners, aspect_ok
from .strategies import _clean, _fill, _largest_quad_from_mask


def _mask_saturation(bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    _, m = cv2.threshold(hsv[:, :, 1], 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _mask_yellow(bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([15, 60, 90]), np.array([40, 255, 255]))


def _mask_vivid_yellow(bgr):
    """Separate saturated border ink from warm wood or cream backgrounds."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([12, 90, 70]), np.array([42, 255, 255]))


def _grayworld(bgr):
    """Remove a global illuminant cast (gray-world assumption).

    Aimed at the warm-indoor-light failure, where a white countertop lands in
    the same hue range as the border and floods the yellow mask. This is a
    LOCALIZATION-only transform -- see the note below.
    """
    f = bgr.astype(np.float32)
    m = f.reshape(-1, 3).mean(0)
    return np.clip(f * (m.mean() / np.maximum(m, 1e-6)), 0, 255).astype(np.uint8)


def _mask_wb_yellow(bgr):
    return _mask_yellow(_grayworld(bgr))


def _mask_lab_b(bgr):
    """Otsu on the CIELAB b* axis, which IS the blue-yellow axis.

    Isolates the border largely independent of brightness, which is why it
    beats plain HSV on close-up shots with uneven illumination.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    _, m = cv2.threshold(lab[:, :, 2], 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _mask_wb_lab_b(bgr):
    return _mask_lab_b(_grayworld(bgr))


# NOTE: every transform here is applied ONLY to the segmentation input. The
# gradient profiles that actually measure edge position always read the
# original pixels. A non-affine point transform f() turns the profile into
# f(I(x)), whose gradient peak sits at a different x than I(x)'s -- and it
# shifts the outer (card/background) and inner (border/art) edges by different
# amounts, which would bias border width, which IS the centering measurement.
def _mask_luma(bgr):
    """Otsu on lightness, polarity chosen automatically.

    The chroma-based maskers all fail on silver/grey borders, which have no
    hue to key on. Luminance still separates a card from a light background,
    and for dark or metallic cards it is far stronger than any colour cue.
    Polarity is picked by whichever side yields the more card-like blob rather
    than assuming the card is darker or lighter than its surroundings.
    """
    g = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = g.shape
    best, best_score = None, -1.0
    for cand in (m, cv2.bitwise_not(m)):
        f = _fill(_clean(cand, k=5, iters=1))
        cnts, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c) / float(h * w)
        if not (0.05 < area < 0.95):
            continue
        # Prefer a blob that does not run off the frame edge.
        x, y, bw, bh = cv2.boundingRect(c)
        touch = (x <= 1) + (y <= 1) + (x + bw >= w - 1) + (y + bh >= h - 1)
        score = area - 0.3 * touch
        if score > best_score:
            best, best_score = cand, score
    return best if best is not None else m


MASKERS = {"saturation": _mask_saturation, "yellow": _mask_yellow,
           "vivid_yellow": _mask_vivid_yellow,
           "luma": _mask_luma,
           "wb_yellow": _mask_wb_yellow, "lab_b": _mask_lab_b,
           "wb_lab_b": _mask_wb_lab_b}


def localize(img, method="saturation", coarse_dim=1200, pad_mm=6.0):
    """Return a full-resolution quad, or None.

    The stage-2 crop is padded by pad_mm around the coarse quad, so the true
    edge is comfortably inside the crop even when stage 1 is off by a few mm,
    while the crop still excludes distant clutter (floor, bags, shadows) that
    caused the stage-1 failures.
    """
    masker = MASKERS[method]
    h, w = img.shape[:2]

    # --- stage 1: coarse ---
    s = max(h, w) / float(coarse_dim)
    small = cv2.resize(img, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
    m = _fill(_clean(masker(small), k=7, iters=2))
    quad_s, _ = _largest_quad_from_mask(m)
    if quad_s is None:
        return None
    quad = quad_s * s

    # px/mm implied by the coarse quad (short side = 2.5in = 63.5mm)
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    # Average each OPPOSITE pair first, then take the smaller average. The
    # previous form -- 0.5*(min(s0,s2) + min(s1,s3)) -- took the shorter of the
    # top/bottom pair and the shorter of the left/right pair and averaged those,
    # mixing one short side with one long side. That inflated px/mm by ~19% on
    # every card, miscalibrating search bands, trim distances and border_mm.
    # (Centering percentages were unaffected: they come from the homography,
    # which normalises to 63x88mm independently of this value.)
    short = min(0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3]))
    px_per_mm = short / 63.0
    pad = int(round(pad_mm * px_per_mm))

    x0 = max(0, int(quad[:, 0].min()) - pad)
    y0 = max(0, int(quad[:, 1].min()) - pad)
    x1 = min(w, int(quad[:, 0].max()) + pad)
    y1 = min(h, int(quad[:, 1].max()) + pad)
    if x1 - x0 < 50 or y1 - y0 < 50:
        return None

    # --- stage 2: full resolution inside the crop ---
    crop = img[y0:y1, x0:x1]
    m2 = masker(crop)
    # Kernel scaled to ~0.4mm: enough to close JPEG speckle, small enough that
    # the morphological bias on the edge stays well under a pixel.
    k = max(3, int(round(0.4 * px_per_mm)) | 1)
    m2 = _fill(_clean(m2, k=k, iters=1))
    quad2, _ = _largest_quad_from_mask(m2, min_area_frac=0.05)
    if quad2 is None:
        return None
    quad2 = quad2 + np.array([x0, y0], dtype=np.float64)
    return order_corners(quad2)


def localize_consensus(img, methods=("saturation", "yellow"), iou_min=0.9):
    """Run several maskers and accept only where they agree AND aspect holds.

    Returns (quad, info). Disagreement is the signal that one of them has
    latched onto background, which is the failure mode neither method can
    detect on its own.
    """
    quads = {}
    for meth in methods:
        try:
            q = localize(img, method=meth)
        except Exception:
            q = None
        if q is not None:
            quads[meth] = q

    scored = {}
    for meth, q in quads.items():
        ok, a = aspect_ok(q, img.shape)
        scored[meth] = {"quad": q, "aspect_ok": ok, "aspect": a}

    good = {m: v for m, v in scored.items() if v["aspect_ok"]}
    if not good:
        return None, {"reason": "no strategy passed aspect gate", "detail": scored}

    if len(good) == 1:
        m = next(iter(good))
        return good[m]["quad"], {"agreed": False, "used": m, "detail": scored}

    ms = list(good)
    iou = _quad_iou(good[ms[0]]["quad"], good[ms[1]]["quad"], img.shape)
    if iou < iou_min:
        return None, {"reason": f"strategies disagree (IoU={iou:.3f})",
                      "detail": scored}
    # Average the agreeing quads; they bracket the true edge.
    q = np.mean([good[m]["quad"] for m in ms], axis=0)
    return order_corners(q), {"agreed": True, "iou": iou, "detail": scored}


def _quad_iou(q1, q2, shape, s=4):
    h, w = shape[0] // s, shape[1] // s
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [(q1 / s).astype(np.int32)], 255)
    cv2.fillPoly(b, [(q2 / s).astype(np.int32)], 255)
    A, B = a > 0, b > 0
    return float((A & B).sum()) / max((A | B).sum(), 1)
