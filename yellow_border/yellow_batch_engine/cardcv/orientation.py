"""Report margins in CARD coordinates, not image coordinates.

Until now "left margin" meant left-of-frame. That makes a landscape scan and a
portrait photo of the same card produce non-comparable numbers, which blocked
validating the photos against the reference scan at all. PSA also distinguishes
top-heavy from bottom-heavy, so a 180-degree rotation changes the grade even
though the card stays portrait -- a landscape filter would never catch it.

Orientation is resolved in two steps:

  1. Long vs short axis is pure geometry: the 88mm sides are left/right, the
     63mm sides are top/bottom.

  2. Which short side is the BOTTOM comes from the card itself. Pokemon cards
     print the illustrator credit, set number and copyright line directly on
     the bottom border; the top border carries no dark print. So the short side
     whose border band contains dark ink is the bottom. (This is the same
     printing quirk that broke gradient-based inner-edge detection on that
     side -- here it is useful rather than harmful.)
"""
import cv2
import numpy as np

from .geom import order_corners


def _band_dark_fraction(img_gray, outer, inner, side, n=160, inset=0.2):
    """Fraction of dark-ink pixels in the border band along one side."""
    o0, o1 = outer[side], outer[(side + 1) % 4]
    i0, i1 = inner[side], inner[(side + 1) % 4]
    ts = np.linspace(0.12, 0.88, n)          # skip corners
    a = o0[None, :] + ts[:, None] * (o1 - o0)[None, :]
    b = i0[None, :] + ts[:, None] * (i1 - i0)[None, :]
    # sample across the band, avoiding both edges
    fs = np.linspace(inset, 1.0 - inset, 12)
    pts = a[:, None, :] * (1 - fs)[None, :, None] + b[:, None, :] * fs[None, :, None]
    mx = pts[:, :, 0].astype(np.float32)
    my = pts[:, :, 1].astype(np.float32)
    vals = cv2.remap(img_gray, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    med = float(np.median(vals))
    return float((vals < med - 45).mean()), med


def card_orientation(img, outer, inner):
    """Return (rot, info) where rot is how many 90-degree steps map quad side 0
    to the card's TOP. Applying `rot` reindexes sides into card order."""
    o = order_corners(outer)
    i = order_corners(inner)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lens = [np.linalg.norm(o[(k + 1) % 4] - o[k]) for k in range(4)]
    # Short sides (63mm) are the card's top and bottom.
    if 0.5 * (lens[0] + lens[2]) <= 0.5 * (lens[1] + lens[3]):
        short_sides = (0, 2)
    else:
        short_sides = (1, 3)

    dark = {}
    for sdi in short_sides:
        dark[sdi], _ = _band_dark_fraction(gray, o, i, sdi)

    bottom = max(short_sides, key=lambda k: dark[k])
    top = short_sides[0] if bottom == short_sides[1] else short_sides[1]
    # rot such that (0 + rot) % 4 == top  ->  rot = top
    return top, {"dark": dark, "short_sides": short_sides, "bottom_side": bottom}


def card_frame_margins(margins_by_side, top_side):
    """Relabel four per-side margins into card top/right/bottom/left."""
    idx = [(top_side + k) % 4 for k in range(4)]
    return {"top": margins_by_side[idx[0]], "right": margins_by_side[idx[1]],
            "bottom": margins_by_side[idx[2]], "left": margins_by_side[idx[3]]}


def centering_card_frame(img, outer, inner, rect_inner, W, H):
    """Compute margins per quad side, then relabel into card coordinates."""
    tl, tr, br, bl = rect_inner
    by_side = [
        float(0.5 * (tl[1] + tr[1])),          # side 0
        float(W - 0.5 * (tr[0] + br[0])),      # side 1
        float(H - 0.5 * (bl[1] + br[1])),      # side 2
        float(0.5 * (tl[0] + bl[0])),          # side 3
    ]
    top_side, info = card_orientation(img, outer, inner)
    m = card_frame_margins(by_side, top_side)
    tb = 100 * m["top"] / (m["top"] + m["bottom"])
    lr = 100 * m["left"] / (m["left"] + m["right"])
    return {"margins_card": m, "top_pct": tb, "bottom_pct": 100 - tb,
            "left_pct": lr, "right_pct": 100 - lr,
            "top_side": top_side, "orient_info": info}
