"""Recognize a printed Pokemon back, then measure the four physical card edges."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from card_utils import card_cut_off

WIDTH, HEIGHT = 360, 504
FRAME = np.float32([[0, 0], [WIDTH-1, 0], [WIDTH-1, HEIGHT-1], [0, HEIGHT-1]])
BANK = Path(__file__).parent / "references" / "back_landmarks.npz"


@lru_cache(maxsize=1)
def reference_features():
    with np.load(BANK, allow_pickle=False) as bank:
        return [(name, bank[name+"_points"], bank[name+"_features"].astype(np.float32))
                for name in ("classic", "alternate")]


def match_artwork(image):
    """Match features spread across the design, ignoring the border and background."""
    height, width = image.shape[:2]
    scale = min(1., 1000/max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2., (8, 8)).apply(gray)
    keys, features = cv2.SIFT_create(nfeatures=2200).detectAndCompute(gray, None)
    if features is None or len(keys) < 16:
        return None
    best = None
    for design, points, descriptors in reference_features():
        pairs = cv2.BFMatcher().knnMatch(descriptors, features, k=2)
        matches = [a for a,b in pairs if a.distance < .74*b.distance]
        if len(matches) < 16:
            continue
        source = np.float32([points[m.queryIdx] for m in matches])
        target = np.float32([keys[m.trainIdx].pt for m in matches])
        transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.)
        if transform is None or mask is None:
            continue
        valid = mask.ravel().astype(bool)
        count = len(np.unique(target[valid], axis=0))
        if count < 16 or valid.mean() < .45:
            continue
        coverage = cv2.contourArea(cv2.convexHull(source[valid])) / (WIDTH*HEIGHT)
        if coverage < .22:
            continue
        quad = cv2.perspectiveTransform(FRAME[None], transform)[0] / scale
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) < 2000:
            continue
        if best is None or count > best[1]["artwork_matches"]:
            best = quad, dict(design=design, artwork_matches=count, artwork_coverage=round(coverage,3))
    return best


def outlines_agree(candidate, reference):
    """Compare in card coordinates so the tolerance is independent of photo scale."""
    transform = cv2.getPerspectiveTransform(reference, FRAME)
    measured = cv2.perspectiveTransform(candidate[None], transform)[0]
    # Either detector can start at a different corner on rotated photographs.
    return any(np.linalg.norm(np.roll(measured, shift, axis=0)-FRAME, axis=1).max() < WIDTH*.04
               for shift in range(4))


def measure_side(rectified, side):
    """Vote for a continuous color transition near the predicted edge, allowing tilt."""
    # Rotate so the expected edge is horizontal with card interior below it.
    oriented = np.rot90(rectified, side)
    margin = 24
    length = oriented.shape[1] - 2*margin
    x = margin + np.rint(np.linspace(.10, .90, 65)*length).astype(int)
    t = (x - margin) / (length-1)
    best = None
    for tilt in range(-6, 7, 2):
        offsets = np.arange(-12, 13)[:, None]
        y = margin + np.rint(offsets + tilt*(t-.5)).astype(int)
        # Contrast across the edge, evaluated in three sections to resist glare spots.
        contrast = np.linalg.norm(oriented[y+2, x]-oriented[y-2, x], axis=2)
        support = np.min([np.mean(part > 13, axis=1) for part in np.array_split(contrast, 3, axis=1)], axis=0)
        score = np.median(np.minimum(contrast, 100),axis=1) + support*30 - 2.5*abs(offsets[:,0]) - .4*abs(tilt)
        score[support < .45] = -np.inf
        index = int(np.argmax(score))
        if np.isfinite(score[index]) and (best is None or score[index] > best[0]):
            best = score[index], float(offsets[index,0]), tilt, float(support[index])
    if best is None:
        return None
    _, offset, tilt, support = best
    a,b = FRAME[side], FRAME[(side+1)%4]
    along = (b-a)/np.linalg.norm(b-a)
    inward = np.array([-along[1],along[0]])
    return a+inward*(offset-tilt/2), b+inward*(offset+tilt/2), round(support,3)


def detect_printed_back(image):
    match = match_artwork(image)
    if match is None:
        return None, {"reason": "back_artwork_unconfirmed"}
    seed, info = match
    info.update(recovery="printed_back_landmarks", seed_quad=seed.tolist())
    # Printed artwork locates the search region. Only measured edges form the sheet.
    margin = 24
    transform = cv2.getPerspectiveTransform(seed, FRAME+margin)
    rectified = cv2.warpPerspective(image, transform, (WIDTH+2*margin, HEIGHT+2*margin),
                                    borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(cv2.GaussianBlur(rectified, (3,3), 0),cv2.COLOR_BGR2LAB).astype(np.float32)
    sides = [measure_side(lab,i) for i in range(4)]
    if any(line is None for line in sides):
        return None, dict(info, reason="back_edge_unconfirmed", unsupported_sides=[i for i,s in enumerate(sides) if s is None])
    corners = []
    for i in range(4):
        a,b,_ = sides[i-1]
        c,d,_ = sides[i]
        along = b-a
        coeff = np.linalg.solve(np.column_stack((along,c-d)), c-a)
        corners.append(a+along*coeff[0])
    quad = cv2.perspectiveTransform((np.float32(corners)+margin)[None], np.linalg.inv(transform))[0]
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or card_cut_off(image.shape, quad):
        return None, dict(info,reason="back_edges_cut_off")
    return quad, dict(info,status="ready",quad=quad.tolist(),sides=[{"support":s[2]} for s in sides])
