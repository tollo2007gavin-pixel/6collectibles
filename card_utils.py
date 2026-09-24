"""
Shared OpenCV helpers: find + flatten the card, tell front from back,
score whitening for free, and build the zoomed edge/corner sheet the LLM sees.
"""
from __future__ import annotations

import cv2
import numpy as np

CARD_W, CARD_H = 750, 1050          # 2.5 x 3.5 in at 300 dpi
WARP_MARGIN = 1.012                 # grow the detected quad ~1% so edge whitening isn't cropped off


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
    return frac * rectangularity ** 2 * shape, box


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
            r = (r[0] * 3, r[1])
        if r and (best is None or r[0] > best[0]):
            best = r
    return None if best is None else _order(best[1].astype(np.float32) / scale)


def find_card(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return (portrait card warped to CARD_W x CARD_H, found).
    found=False means no card-shaped outline was detected; those photos are
    saved for reference but must NOT be used for edge/corner analysis.
    """
    pts = find_card_quad(img)
    if pts is None:
        return cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA), False
    ctr = pts.mean(0)
    pts = ctr + (pts - ctr) * WARP_MARGIN
    tl, tr, br, bl = pts
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width <= height:
        dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (CARD_W, CARD_H)), True
    dst = np.array([[0, 0], [CARD_H - 1, 0], [CARD_H - 1, CARD_W - 1], [0, CARD_W - 1]], np.float32)
    land = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (CARD_H, CARD_W))
    return cv2.rotate(land, cv2.ROTATE_90_CLOCKWISE), True


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ------------------------------------------------------------ front / back

def _border_band(card: np.ndarray, inner: float = 0.06, outer: float = 0.015) -> np.ndarray:
    h, w = card.shape[:2]
    m = np.zeros((h, w), np.uint8)
    a, b = int(w * outer), int(w * inner)
    m[a:h - a, a:w - a] = 1
    m[b:h - b, b:w - b] = 0
    return m.astype(bool)


def classify_side(card: np.ndarray) -> tuple[str, float]:
    """Pokémon backs have a solid blue border; fronts don't. Returns (side, blue_ratio)."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 70) & (hsv[..., 2] > 40)
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
    o, i = int(w * 0.012), int(w * 0.035)          # skip the warp margin, stop inside the border
    c = int(w * 0.10)
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
    cs = int(w * 0.16)
    t = width // 4
    corners = [card[:cs, :cs], card[:cs, w - cs:], card[h - cs:, :cs], card[h - cs:, w - cs:]]
    names = ["TL", "TR", "BL", "BR"]
    row = np.hstack([_label(cv2.resize(cr, (t, t), interpolation=cv2.INTER_CUBIC), f"{side} {n}")
                     for cr, n in zip(corners, names)])

    et = int(w * 0.06)
    strips = {
        "TOP": card[:et, :], "BOTTOM": card[h - et:, :],
        "LEFT": cv2.rotate(card[:, :et], cv2.ROTATE_90_CLOCKWISE),
        "RIGHT": cv2.rotate(card[:, w - et:], cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    rows = [row]
    for name, s in strips.items():
        sh = max(40, int(s.shape[0] * width / s.shape[1]))
        rows.append(_label(cv2.resize(s, (width, sh), interpolation=cv2.INTER_CUBIC), f"{side} {name}"))
        rows.append(np.full((4, width, 3), 255, np.uint8))  # separator
    return np.vstack(rows[:-1])
