"""Candidate strategies for LOCALIZING the outer card quad.

None of these is meant to be precise. Their job is to get a quad that is
correct to a few pixels so the sub-pixel refinement stage has a place to look.
Precision comes later, from gradient-profile edge fitting.

Every strategy exposes: detect(img_bgr) -> dict with 'quad' (4x2 float32 in
FULL-resolution image coords, ordered TL/TR/BR/BL) or None, plus 'mask' and
'notes' for debugging.
"""
import cv2
import numpy as np

from .geom import order_corners, quad_area, aspect_ok
from .loader import downscale


def _largest_quad_from_mask(mask, min_area_frac=0.02):
    """Largest external contour in `mask`, reduced to a 4-point quad."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    h, w = mask.shape[:2]
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area_frac * h * w]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)

    # Try progressively looser polygon approximation for a clean 4-gon.
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            return order_corners(ap.reshape(-1, 2)), c
    # Fall back to the minimum-area rectangle.
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return order_corners(box), c


def _clean(mask, k=7, iters=2):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=iters)
    return mask


def _fill(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


class SaturationStrategy:
    """Card art is colourful; counters, walls and sleeves are near-neutral.

    Thresholds the HSV saturation channel with Otsu. Cheap and background-
    agnostic as long as the surface is grey/white/beige.
    """
    name = "saturation"

    def detect(self, img):
        small, s = downscale(img)
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = _fill(_clean(mask))
        quad, _ = _largest_quad_from_mask(mask)
        if quad is None:
            return {"quad": None, "mask": mask, "scale": s, "notes": "no contour"}
        return {"quad": (quad * s).astype(np.float64), "mask": mask, "scale": s,
                "notes": ""}


class YellowBorderStrategy:
    """Segment the yellow border ring directly.

    Attractive because the ring is bounded by BOTH edges we need. Fragile
    because the ring's apparent width is exactly the quantity being measured,
    so threshold bias maps straight into centering error. Localisation only.
    """
    name = "yellow"

    def detect(self, img):
        small, s = downscale(img)
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        # Broad yellow: hue 15-40 of 179, moderately saturated, not dark.
        mask = cv2.inRange(hsv, np.array([15, 60, 90]), np.array([40, 255, 255]))
        mask = _clean(mask, k=5)
        filled = _fill(mask)
        quad, _ = _largest_quad_from_mask(filled)
        if quad is None:
            return {"quad": None, "mask": mask, "scale": s, "notes": "no contour"}
        return {"quad": (quad * s).astype(np.float64), "mask": mask, "scale": s,
                "notes": ""}


class LSDQuadStrategy:
    """Find long straight segments and assemble the best quad from them.

    Skips segmentation entirely, so background clutter, glare patches and
    sleeve edges hurt it far less than they hurt threshold-based methods.
    """
    name = "lsd"

    def detect(self, img):
        small, s = downscale(img)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
        if lines is None:
            return {"quad": None, "mask": None, "scale": s, "notes": "no lines"}
        lines = lines.reshape(-1, 4)

        h, w = gray.shape
        seglen = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
        keep = seglen > 0.15 * min(h, w)
        lines, seglen = lines[keep], seglen[keep]
        if len(lines) < 4:
            return {"quad": None, "mask": None, "scale": s,
                    "notes": f"only {len(lines)} long segments"}

        # Group by orientation into two families (card edges are perpendicular).
        ang = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
        ang = np.mod(ang, np.pi)
        # Circular-ish split: rotate so the dominant direction sits mid-bin.
        dom = ang[np.argmax(seglen)]
        rel = np.mod(ang - dom + np.pi / 2, np.pi) - np.pi / 2
        famA = np.abs(rel) < np.pi / 4
        famB = ~famA
        if famA.sum() < 2 or famB.sum() < 2:
            return {"quad": None, "mask": None, "scale": s,
                    "notes": "could not split into two edge families"}

        def extremes(idx):
            """The two segments of a family that are furthest apart."""
            sub, sl = lines[idx], seglen[idx]
            n = np.array([-np.sin(dom), np.cos(dom)])  # normal of dominant dir
            mid = np.stack([(sub[:, 0] + sub[:, 2]) / 2,
                            (sub[:, 1] + sub[:, 3]) / 2], axis=1)
            proj = mid @ n
            return sub[np.argmin(proj)], sub[np.argmax(proj)], sl

        def line_from_seg(sg):
            p0 = np.array([sg[0], sg[1], 1.0])
            p1 = np.array([sg[2], sg[3], 1.0])
            return np.cross(p0, p1)

        a1, a2, _ = extremes(famA)
        # For family B project along the perpendicular instead.
        subB = lines[famB]
        nB = np.array([np.cos(dom), np.sin(dom)])
        midB = np.stack([(subB[:, 0] + subB[:, 2]) / 2,
                         (subB[:, 1] + subB[:, 3]) / 2], axis=1)
        projB = midB @ nB
        b1, b2 = subB[np.argmin(projB)], subB[np.argmax(projB)]

        L = [line_from_seg(x) for x in (a1, a2, b1, b2)]
        pts = []
        for i in (0, 1):
            for j in (2, 3):
                p = np.cross(L[i], L[j])
                if abs(p[2]) < 1e-9:
                    return {"quad": None, "mask": None, "scale": s,
                            "notes": "parallel intersection"}
                pts.append(p[:2] / p[2])
        quad = order_corners(np.array(pts))
        if quad_area(quad) < 0.01 * h * w:
            return {"quad": None, "mask": None, "scale": s, "notes": "quad too small"}
        return {"quad": (quad * s).astype(np.float64), "mask": None, "scale": s,
                "notes": ""}


ALL = [SaturationStrategy(), YellowBorderStrategy(), LSDQuadStrategy()]
