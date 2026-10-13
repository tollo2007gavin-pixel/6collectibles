"""
Shared OpenCV helpers: find + flatten the card, tell front from back,
score whitening for free, and build the zoomed edge/corner sheet the LLM sees.
"""
from __future__ import annotations

import cv2
import numpy as np

CARD_W, CARD_H = 1500, 2100         # 2.5 x 3.5 in at 600 dpi: keeps every pixel eBay gives us, so
                                    # tiny corner-tip whitening isn't blurred away before the AI sees it
PAD = 64                            # px of background kept around the card on every side, so the
                                    # true outer edge (where whitening shows) is never cropped off


def _geom(card: np.ndarray) -> tuple[int, int]:
    """(pad, card_width) in THIS image's pixels. Works for cards saved at any size
    (older 750-wide folders and current 1500-wide ones have the same proportions)."""
    w = card.shape[1]
    cw = w * CARD_W / (CARD_W + 2 * PAD)
    return int(round((w - cw) / 2)), int(round(cw))


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize with the right filter: AREA when shrinking (no aliasing), CUBIC when enlarging."""
    shrink = size[0] * size[1] < img.shape[0] * img.shape[1]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


# ------------------------------------------------------------- detection

def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


BLUE_LO, BLUE_HI = (95, 70, 40), (130, 255, 255)   # HSV range of the Pokémon back border


def _candidates(small: np.ndarray):
    """Yield (contour, method) pairs that might be the card, from several independent methods."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    k = np.ones((5, 5), np.uint8)

    # 1) Card backs: the blue border runs right to the card's edge, so the biggest
    #    blue blob IS the card, even inside sleeves/top-loaders or on busy backgrounds.
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=4)   # bridges glare streaks
    for c in cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        yield c, "blue"

    # 2) Edges at a few sensitivities (fronts, and backs the blue mask misses)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # RETR_LIST (not EXTERNAL) so a card nested inside a top-loader or sleeve outline
    # is still a candidate instead of being hidden by the outer rectangle.
    for lo, hi in ((30, 90), (50, 150), (80, 200)):
        e = cv2.dilate(cv2.Canny(gray, lo, hi), k, iterations=2)
        for c in cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            yield c, "edges"

    # 3) Brightness split both ways (card lighter or darker than background)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for m in (th, 255 - th):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            yield c, "brightness"


def _score(c, frame_area: float):
    """Card-likeness: big, rectangular, and close to the 2.5x3.5 (1.40) shape. Returns (score, box) or None.
    The shape term is sharp on purpose: top-loaders (3x4 in = 1.33) and inner artwork boxes
    are rectangles too, and must lose to the card itself."""
    area = cv2.contourArea(c)
    frac = area / frame_area
    if not 0.10 < frac < 0.97:
        return None
    rect = cv2.minAreaRect(c)
    (rw, rh) = rect[1]
    if min(rw, rh) < 1:
        return None
    rectangularity = area / (rw * rh)
    ratio = max(rw, rh) / min(rw, rh)
    if rectangularity < 0.85 or abs(ratio - 1.4) > 0.05:   # top-loaders are 1.33: rejected
        return None
    shape = float(np.exp(-((ratio - 1.4) / 0.035) ** 2))
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    box = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(rect)
    return frac * rectangularity ** 2 * shape, box, c


def _refine(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Snap each side to a straight line fitted through the contour's edge points
    (ignoring the rounded corners), then intersect neighbouring lines. This puts
    the corners where the straight edges would meet instead of on the rounded
    arc, which otherwise shaves a sliver off every edge.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1:
            return quad
        t = ((pts - a) @ ab) / (L * L)
        dist = np.abs(ab[0]*(pts[:,1]-a[1]) - ab[1]*(pts[:,0]-a[0])) / L
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.04 * L)]
        if len(sel) < 10:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):                      # corner i = side (i-1) x side i
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-6:
            return quad
        t1 = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append(p1 + t1 * d1)
    out = np.array(out, np.float32)
    # sanity: refinement should only nudge corners, never move them far
    if np.linalg.norm(out - quad, axis=1).max() > 0.08 * np.linalg.norm(quad[1] - quad[0]):
        return quad
    return out


def _bluish(hsv_px: np.ndarray) -> bool:
    """Loose 'some kind of blue' test that still holds for faded, pale or grey-ish backs."""
    h, sat, v = (float(x) for x in np.median(hsv_px, axis=0))
    return 88 <= h <= 140 and sat > 12 and 15 < v < 240


def _find_step(lab: np.ndarray, k0: int, thresh: float = 16.0) -> int | None:
    """
    First sharp colour change along a profile. The reference colour is a running
    average of the few pixels just behind the current one, so a border that fades
    gradually (wear, lighting, print variation) never triggers it; only a sudden
    jump (card edge -> background) does.
    """
    for k in range(k0 + 2, len(lab) - 3):
        ref = lab[max(k0, k - 6):k - 1].mean(0)     # only border pixels, never the panel behind us
        if (np.linalg.norm(lab[k:k + 3] - ref, axis=1) > thresh).all():
            return k
    return None


def _expand_to_outer_edge(small: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    A card back has a light swirl panel inside a blue border, and the panel's
    outline is often the strongest rectangle in the photo. If the quad we found
    is ringed by a bluish band of steady width on every side, it's that inner
    panel: walk outward from each side to the first sharp colour step, fit a
    line through those points, and return the true outer card edge.
    Works on colour CHANGE, not an exact shade, so faded borders are fine.
    """
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = hsv.shape[:2]
    ctr = quad.mean(0)
    width = float(np.linalg.norm(quad[1] - quad[0]))
    max_d = int(0.12 * width)
    k0 = max(3, int(0.015 * width))                   # start past the panel's own outline
    lines, side_widths, band_colors = [], [], []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = (b - a) / max(np.linalg.norm(b - a), 1)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (a + b) / 2 - ctr) < 0:
            n = -n                                            # make the normal point outward
        pts, ws, band_hsv, frame_pts = [], [], [], []
        for t in np.linspace(0.12, 0.88, 40):
            p = a + t * (b - a)
            xy_all = [(p + n * k).round().astype(int) for k in range(1, max_d)]
            xy = []
            for x, y in xy_all:
                if not (0 <= x < W and 0 <= y < H):
                    break
                xy.append((x, y))
            ran_off = len(xy) < len(xy_all)
            if len(xy) >= k0 + 4:
                ys, xs = np.array([y for _, y in xy]), np.array([x for x, _ in xy])
                band_hsv.append(np.median(hsv[ys[k0:k0 + 4], xs[k0:k0 + 4]], axis=0))
                k = _find_step(lab[ys, xs], k0)
                if k is not None:
                    pts.append(p + n * (k + 1))
                    ws.append(k + 1)
                    continue
            if ran_off:
                # Still inside the border when the photo ended: the card edge is off-frame.
                frame_pts.append(p + n * len(xy))
        if len(frame_pts) >= 16 and (not band_hsv or _bluish(np.array(band_hsv))):
            # This side of the card runs off the photo. Put the edge on the frame so
            # card_cut_off() flags the photo instead of grading an inner panel.
            vx, vy, x0, y0 = cv2.fitLine(np.array(frame_pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
            side_widths.append(None)
            band_colors.append(None)
            continue
        if len(pts) < 16 or not _bluish(np.array(band_hsv)):
            return quad                                       # no bluish band around it: not the inner panel
        ws = np.array(ws, float)
        med = np.median(ws)
        keep = np.abs(ws - med) <= max(3.0, 0.25 * med)       # a real border has a steady width
        if keep.mean() < 0.6:
            return quad
        side_widths.append(med)
        band_colors.append(float(np.median(np.array(band_hsv)[:, 0])))   # hue only: brightness may fade
        sel = np.array(pts, np.float32)[keep]
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    # Sanity checks, otherwise we walked into a sleeve, top-loader or blue background:
    #  - each border is thin (~1-9% of card width) and the same hue all the way round
    #    (brightness may fade);
    #  - opposite borders ADD UP: top+bottom ~= left+right. Individual sides can differ
    #    a lot on an off-centre card (that's what centering measures), but the totals can't.
    sw = [x for x in side_widths if x is not None]            # sides that ran off-frame have no width
    bc = [x for x in band_colors if x is not None]
    if len(sw) < 2 or not (0.01 * width < min(sw) and max(sw) < 0.09 * width and max(bc) - min(bc) < 20):
        return quad
    if None not in side_widths:                               # sides are top, right, bottom, left
        vert, horiz = side_widths[0] + side_widths[2], side_widths[1] + side_widths[3]
        if abs(vert - horiz) > 0.35 * (vert + horiz) / 2:
            return quad
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-6:
            return quad
        t1 = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append(p1 + t1 * d1)
    out = np.array(out, np.float32)
    grow = np.linalg.norm(out - quad, axis=1).max()
    return out if grow < 0.2 * width else quad


def find_card_quad(img: np.ndarray) -> np.ndarray | None:
    """Corner points (TL, TR, BR, BL) of the card in full-resolution pixels, or None."""
    h, w = img.shape[:2]
    scale = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    frame_area = small.shape[0] * small.shape[1]
    best = None
    for c, source in _candidates(small):
        r = _score(c, frame_area)
        if r and source == "blue":
            # The blue back border can't come from a sleeve, top-loader or background
            # outline, so a card-shaped blue blob beats any plain-outline candidate.
            r = (r[0] * 3, r[1], r[2])
        if r and (best is None or r[0] > best[0]):
            best = r
    if best is None:
        return None
    quad = _refine(best[2], _order(best[1].astype(np.float32)))
    return _expand_to_outer_edge(small, quad) / scale


def find_card(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return (portrait card warped to CARD_W x CARD_H, found).
    found=False means no card-shaped outline was detected; those photos are
    saved for reference but must NOT be used for edge/corner analysis.
    """
    pts = find_card_quad(img)
    if pts is None:
        return cv2.resize(img, (CARD_W + 2 * PAD, CARD_H + 2 * PAD), interpolation=cv2.INTER_AREA), False
    tl, tr, br, bl = pts
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width > height:                      # landscape photo of a portrait card: rotate corner order
        pts = np.array([bl, tl, tr, br], np.float32)
    dst = np.array([[PAD, PAD], [PAD + CARD_W - 1, PAD],
                    [PAD + CARD_W - 1, PAD + CARD_H - 1], [PAD, PAD + CARD_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (CARD_W + 2 * PAD, CARD_H + 2 * PAD),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), True


def card_cut_off(img_shape, pts: np.ndarray) -> bool:
    """
    True if part of the card is outside (or touching) the photo frame, so an
    edge can't be fully inspected. Straight edges must be fully inside; corners
    get a small allowance because the fitted corner point sits just outside a
    rounded corner even when the whole card is visible.
    """
    h, w = img_shape[:2]
    card_w = float(np.linalg.norm(pts[1] - pts[0]))
    edge_margin, corner_slack = 2.0, 0.02 * card_w
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        for t in np.linspace(0.08, 0.92, 25):
            x, y = a + t * (b - a)
            if x < edge_margin or y < edge_margin or x > w - 1 - edge_margin or y > h - 1 - edge_margin:
                return True
    for x, y in pts:
        if x < -corner_slack or y < -corner_slack or x > w - 1 + corner_slack or y > h - 1 + corner_slack:
            return True
    return False


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ------------------------------------------------------------ front / back

def _border_band(card: np.ndarray, inner: float = 0.06, outer: float = 0.012) -> np.ndarray:
    """Ring inside the card's printed border (fractions of card width, measured from the card edge)."""
    h, w = card.shape[:2]
    pad, cw = _geom(card)
    m = np.zeros((h, w), np.uint8)
    a, b = pad + int(cw * outer), pad + int(cw * inner)
    m[a:h - a, a:w - a] = 1
    m[b:h - b, b:w - b] = 0
    return m.astype(bool)


def classify_side(card: np.ndarray) -> tuple[str, float]:
    """Pokémon backs have a solid blue border; fronts don't. Returns (side, blue_ratio)."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 90) & (hsv[..., 0] <= 138) & (hsv[..., 1] > 30) & (hsv[..., 2] > 25)
    ratio = float(blue[_border_band(card)].mean())
    side = "back" if ratio > 0.45 else "front" if ratio < 0.20 else "unknown"
    return side, round(ratio, 3)


# --------------------------------------------------------------- whitening

def whitening_score(back: np.ndarray) -> dict:
    """
    Free first-pass score on a card BACK: share of near-white, low-saturation pixels
    along the outer blue border, per edge and per corner. Glare can inflate it.
    """
    h, w = back.shape[:2]
    hsv = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 55) & (hsv[..., 2] > 165)
    pad, cw = _geom(back)
    o, i = pad + max(4, cw // 190), pad + int(cw * 0.035)   # start just inside the card edge (skip background)
    c = pad + int(cw * 0.10)
    regions = {
        "top": white[o:i, c:w - c], "bottom": white[h - i:h - o, c:w - c],
        "left": white[c:h - c, o:i], "right": white[c:h - c, w - i:w - o],
        "top_left": white[o:c, o:c], "top_right": white[o:c, w - c:w - o],
        "bottom_left": white[h - c:h - o, o:c], "bottom_right": white[h - c:h - o, w - c:w - o],
    }
    # Corners: only count the rim, not the corner square's interior (that's artwork/border)
    for k in ("top_left", "top_right", "bottom_left", "bottom_right"):
        r = regions[k]
        rim = np.zeros_like(r)
        t = i - o
        rim[:t, :] = rim[:, :t] = True
        if "bottom" in k:
            rim = rim[::-1]
        if "right" in k:
            rim = rim[:, ::-1]
        regions[k] = r[rim]
    scores = {k: round(float(v.mean()), 4) for k, v in regions.items()}
    scores["max"] = max(scores.values())
    return scores


# -------------------------------------------------------------- crop sheet

def _label(img, text):
    """Caption in a bar ABOVE the crop so it never covers the edge being judged."""
    bar = np.full((20, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def edge_corner_sheet(card: np.ndarray, side: str, width: int = 1024) -> np.ndarray:
    """
    One compact image: 4 zoomed corners on top, then the 4 edges as long strips.
    Sending this instead of the whole photo gives the model more pixels where
    whitening actually happens, for fewer tokens.
    """
    h, w = card.shape[:2]
    pad, cw = _geom(card)
    cs = pad + int(cw * 0.16)                      # corner crops include background margin
    t = width // 4
    corners = [card[:cs, :cs], card[:cs, w - cs:], card[h - cs:, :cs], card[h - cs:, w - cs:]]
    names = ["TL", "TR", "BL", "BR"]
    gap = 8   # white gap so four adjacent corners don't read as one small card
    tiles = [_label(_fit(cr, (t - gap, t - gap)), f"{side} {n}")
             for cr, n in zip(corners, names)]
    sep = np.full((tiles[0].shape[0], gap, 3), 255, np.uint8)
    row = np.hstack([x for tile in tiles for x in (tile, sep)])[:, :width]

    et = pad + int(cw * 0.05)                      # edge strips: margin + outer border
    strips = {
        "TOP": card[:et, :], "BOTTOM": card[h - et:, :],
        "LEFT": cv2.rotate(card[:, :et], cv2.ROTATE_90_CLOCKWISE),
        "RIGHT": cv2.rotate(card[:, w - et:], cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    rows = [row]
    for name, s in strips.items():
        sh = max(40, int(s.shape[0] * width / s.shape[1]))
        rows.append(_label(_fit(s, (width, sh)), f"{side} {name}"))
        rows.append(np.full((4, width, 3), 255, np.uint8))  # separator
    return np.vstack(rows[:-1])


def corner_closeup_sheet(card: np.ndarray, side: str, tile: int = 500) -> np.ndarray:
    """
    The four corner TIPS, cropped tight and enlarged in a 2x2 grid. Whitening
    most often starts right at the tip, where it's only a few pixels in a
    listing photo; the regular sheet shows whole corners, this shows the tips.
    """
    h, w = card.shape[:2]
    pad, cw = _geom(card)
    out_ = int(cw * 0.02)                          # a little background beyond the tip
    in_ = int(cw * 0.07)                           # and a little way into the card
    a, b = max(0, pad - out_), pad + in_
    crops = {
        "TL": card[a:b, a:b],
        "TR": card[a:b, w - b:w - a],
        "BL": card[h - b:h - a, a:b],
        "BR": card[h - b:h - a, w - b:w - a],
    }
    gap = 10
    tiles = {k: _label(_fit(v, (tile, tile)), f"{side} {k} tip (close-up)") for k, v in crops.items()}
    vsep = np.full((tiles["TL"].shape[0], gap, 3), 255, np.uint8)
    top = np.hstack([tiles["TL"], vsep, tiles["TR"]])
    bottom = np.hstack([tiles["BL"], vsep, tiles["BR"]])
    return np.vstack([top, np.full((gap, top.shape[1], 3), 255, np.uint8), bottom])
