"""Sun & Moon full-art GX frame measurement.

Scans and bare listing photos use independently fitted physical edges.  The reference
feature is a printed, straight keyline: the first continuous grey/charcoal
frame inboard of the die-cut edge.  It never attempts to trace the artwork.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import cv2
import numpy as np


CARD_RATIO = 63.0 / 88.0


@dataclass(frozen=True)
class GXMeasurement:
    margins_rectified_px: dict[str, float]
    centering: dict[str, float]
    outer_quad: np.ndarray
    keyline_positions: dict[str, float]
    confidence: float
    api_keyline_disagreement_px: float | None = None
    api_edge_probe_contrast: dict[str, float] | None = None
    inner_quad: np.ndarray | None = None  # source-image extrapolated intersections
    capture: str = "scan"
    holder: str = "unknown"
    rectified_width: int = 1260
    fit_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class GXImageAssumptions:
    """Image eligibility and optional coarse geometry supplied by vision.

    These fields route the deterministic measurement. `card_quad_normalized`
    is a fallback ROI seed; keyline coordinates are diagnostic only.
    """

    front_card: bool = True
    capture: str = "auto"       # auto | scan | photo
    holder: str = "unknown"     # none | penny_sleeve | rigid_toploader | unknown
    background: str = "unknown" # plain | textured | busy | unknown
    outer_edge_visibility: str = "unknown"  # clear | partial | poor | unknown
    fully_in_frame: bool | None = None
    card_quad_normalized: tuple[tuple[float, float], ...] | None = None
    inner_keyline_quad_normalized: tuple[tuple[float, float], ...] | None = None
    # For each side, the API nominates pairs of points immediately outside and
    # inside the die-cut.  Pixel colours are sampled locally; the API never
    # supplies approximate RGB values under changing foil/lighting conditions.
    outer_edge_probe_pairs_normalized: dict[str, tuple[tuple[tuple[float, float],
                                                        tuple[float, float]], ...]] | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "GXImageAssumptions":
        """Parse a constrained vision response without trusting its geometry."""
        def parse_quad(name: str) -> tuple[tuple[float, float], ...] | None:
            quad = payload.get(name)
            if quad is None:
                return None
            if not isinstance(quad, (list, tuple)) or len(quad) != 4:
                raise ValueError(f"{name} must contain four [x, y] points")
            try:
                parsed = tuple(tuple(float(value) for value in point) for point in quad)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain numeric [x, y] points") from exc
            if any(len(point) != 2 or not all(0.0 <= value <= 1.0 for value in point)
                   for point in parsed):
                raise ValueError(f"{name} coordinates must be in [0, 1]")
            return parsed

        quad = parse_quad("card_quad_normalized")
        inner_quad = parse_quad("inner_keyline_quad_normalized")

        probes_raw = payload.get("outer_edge_probe_pairs_normalized")
        probes = None
        if probes_raw is not None:
            if not isinstance(probes_raw, dict) or set(probes_raw) != {"left", "right", "top", "bottom"}:
                raise ValueError("outer_edge_probe_pairs_normalized must contain left, right, top, bottom")
            probes = {}
            for side, pairs in probes_raw.items():
                if not isinstance(pairs, (list, tuple)) or not 1 <= len(pairs) <= 5:
                    raise ValueError(f"{side} probes must contain one to five outside/inside pairs")
                parsed_pairs = []
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError(f"{side} probe must be [[outside_x, outside_y], [inside_x, inside_y]]")
                    try:
                        outside = tuple(float(v) for v in pair[0])
                        inside = tuple(float(v) for v in pair[1])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{side} probe coordinates must be numeric") from exc
                    if (len(outside) != 2 or len(inside) != 2 or
                            not all(0.0 <= v <= 1.0 for v in outside + inside)):
                        raise ValueError(f"{side} probe coordinates must be in [0, 1]")
                    parsed_pairs.append((outside, inside))
                probes[side] = tuple(parsed_pairs)
        return cls(
            front_card=bool(payload.get("front_card", True)),
            capture=str(payload.get("capture", "auto")),
            holder=str(payload.get("holder", "unknown")),
            background=str(payload.get("background", "unknown")),
            outer_edge_visibility=str(payload.get("outer_edge_visibility", "unknown")),
            fully_in_frame=payload.get("fully_in_frame"),
            card_quad_normalized=quad,
            inner_keyline_quad_normalized=inner_quad,
            outer_edge_probe_pairs_normalized=probes,
        )


def assess_gx_measurement(result: GXMeasurement, target_width: int = 1260,
                          photo_mode: bool = False) -> dict:
    """Evidence-based quality gates for a GX centering measurement.

    These gates speak only to geometry and line support.  They do not claim a
    card is gradeable; their job is to prevent a wrong frame from being passed
    to the higher-level eBay pregrade decision.
    """
    target_width = result.rectified_width
    photo_mode = photo_mode or result.capture == "photo"
    m = result.margins_rectified_px
    reasons = []
    support_floor = .55 if photo_mode else .40
    if result.confidence < support_floor:
        reasons.append(f"weak keyline support ({result.confidence:.2f} < {support_floor:.2f})")
    ppm = target_width / 63.0
    if any(not np.isfinite(v) or v < 1.2 * ppm or v > 4.5 * ppm for v in m.values()):
        reasons.append("keyline inset outside plausible 1.2--4.5 mm range")
    horizontal_sum = m["left"] + m["right"]
    vertical_sum = m["top"] + m["bottom"]
    if abs(horizontal_sum - vertical_sum) > 1.0 * ppm:
        reasons.append("opposite border-pair totals disagree by more than 1 mm")
    d = result.fit_diagnostics or {}
    if not d.get("inner_contained", True):
        reasons.append("inner frame is not contained within the physical card")
    if photo_mode:
        if result.holder != "none":
            reasons.append("bare-card eligibility unconfirmed; use holder=none or vision classification")
        if d.get("source_short_side_px", 0) < 400:
            reasons.append("card is smaller than 400 source pixels across")
        if d.get("max_inner_inset_change_mm", float("inf")) > .65:
            reasons.append("inner/outer lines diverge by more than 0.65 mm across a side")
        outer = d.get("outer", {})
        if len(outer) != 4:
            reasons.append("photo outer edges lack independently fitted support")
        elif any(v["support"] < .55 or v["span"] < .60 or
                 v["residual_p90_rectified_px"] > 3.5 for v in outer.values()):
            reasons.append("outer-edge support, span, or residual is insufficient")
        inner = d.get("inner", {})
        if len(inner) != 4 or any(v["support"] < .55 or v["span"] < .60 for v in inner.values()):
            reasons.append("keyline support does not span enough of all four sides")
    return {"accepted": not reasons, "reasons": reasons,
            "min_line_support": result.confidence,
            "border_sum_delta_mm": abs(horizontal_sum - vertical_sum) / ppm,
            "validation_scope": "two-card calibration; listing-population accuracy unvalidated"}


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return four points as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = p.sum(axis=1)
    diffs = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(sums)], p[np.argmin(diffs)],
                     p[np.argmax(sums)], p[np.argmax(diffs)]], dtype=np.float32)


def _validate_frame_quad(quad: np.ndarray, shape: tuple, name: str) -> None:
    q = np.asarray(quad, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError(f"{name} quad must contain four finite [x, y] corners")
    if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) <= 0:
        raise ValueError(f"{name} quad must be convex in TL, TR, BR, BL order")
    if cv2.contourArea(q) < 100:
        raise ValueError(f"{name} quad is too small")
    h, w = shape[:2]
    # Extrapolated rounded corners may extend a few pixels beyond their edge.
    if np.any(q < 0) or np.any(q[:, 0] >= w) or np.any(q[:, 1] >= h):
        raise ValueError(f"{name} quad extends outside the image")


def locate_card_on_scan(bgr: np.ndarray) -> np.ndarray:
    """Locate the physical card on a mostly light, flatbed-scanned background.

    This is intentionally not used for listing photos.  A scanner gives us a
    reliable outer-edge reference while we calibrate the inner grey keyline.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Paper is close to white.  The card's foil/artwork remains connected after
    # a modest close, while isolated scanner dust is removed by area filtering.
    mask = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)[1]
    k = max(9, int(round(min(bgr.shape[:2]) / 250)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
                            iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No non-paper component found for scan card localization")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.01 * bgr.shape[0] * bgr.shape[1]:
        raise ValueError("Largest non-paper component is too small to be a card")
    rect = cv2.minAreaRect(contour)
    quad = _order_corners(cv2.boxPoints(rect))
    w = np.linalg.norm(quad[1] - quad[0])
    h = np.linalg.norm(quad[2] - quad[1])
    ratio = min(w, h) / max(w, h)
    if abs(ratio - CARD_RATIO) > 0.08:
        raise ValueError(f"Located component has non-card aspect ratio {ratio:.3f}")
    return quad


def _photo_rotation(bgr: np.ndarray) -> float:
    """Estimate roll from long segments; short background texture has no vote."""
    h, w = bgr.shape[:2]
    scale = min(1.0, 1200.0 / max(h,w))
    gray = cv2.cvtColor(cv2.resize(bgr, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    segments = cv2.createLineSegmentDetector().detect(gray)[0]
    if segments is None:
        return 0.0
    lines = segments.reshape(-1,4)
    dx,dy = lines[:,2]-lines[:,0], lines[:,3]-lines[:,1]
    lengths = np.hypot(dx,dy)
    angles = (np.degrees(np.arctan2(dy,dx))+45) % 90 -45
    keep = (lengths > .10 * min(gray.shape)) & (np.abs(angles) < 32)
    if np.count_nonzero(keep) < 4:
        return 0.0
    angles,lengths=angles[keep],lengths[keep]
    hist,bins=np.histogram(angles,bins=np.arange(-33,34,1),weights=lengths**2)
    mode=bins[np.argmax(hist)]+.5
    near=np.abs(angles-mode)<2.5
    return float(np.average(angles[near],weights=lengths[near]**2))


def locate_card_on_photo(bgr: np.ndarray) -> np.ndarray:
    """Undo modest roll for localization only, then restore source coordinates."""
    angle = _photo_rotation(bgr)
    if abs(angle) <= 2.0:
        return _locate_upright_photo(bgr)
    h,w=bgr.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    upright=cv2.warpAffine(bgr,M,(w,h),borderMode=cv2.BORDER_REPLICATE)
    seed=_locate_upright_photo(upright)
    return cv2.transform(seed[None],cv2.invertAffineTransform(M))[0].astype(np.float32)


def _locate_upright_photo(bgr: np.ndarray) -> np.ndarray:
    """Propose a card quad from luma/saturation masks on a listing photo.

    This only supplies a seed.  The die-cut is still established by
    `_refine_outer_scan`, which rejects the proposal when it lacks coherent
    edge support.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    outline = _profile_outline_seed(gray)
    if outline is not None:
        return outline
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    # Prefer the two-stage saturation seed used by the yellow-card localizer.
    # On a neutral/dark listing background it follows the colourful card face
    # without being drawn to the black textured surface.
    _, sat_mask = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, ker, iterations=2)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, ker, iterations=2)
    # Saturation is useful as one proposal, but a dark textured background can
    # produce an even larger saturation component.  Score it alongside luma
    # candidates rather than letting it short-circuit card localization.
    masks = [sat_mask]
    for channel in (gray, sat):
        _, hi = cv2.threshold(cv2.GaussianBlur(channel, (5, 5), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.extend((hi, cv2.bitwise_not(hi)))
    h, w = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(7, (min(h, w) // 80) | 1),) * 2)
    best, best_score = None, -1e9
    for mask in masks:
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            frac = area / (h * w)
            # The inverted background often encloses nearly the entire frame
            # and happens to have a card-like aspect.  A visible card may be
            # large, but should not occupy most of the photo's *connected*
            # area; retain room for close crops while excluding that failure.
            if not 0.04 <= frac <= 0.60:
                continue
            quad = _order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))
            a = np.linalg.norm(quad[1] - quad[0]); b = np.linalg.norm(quad[2] - quad[1])
            ratio = min(a, b) / max(a, b)
            if abs(ratio - CARD_RATIO) > 0.15:
                continue
            score = 3.0 * frac - 12.0 * abs(ratio - CARD_RATIO)
            if score > best_score:
                best, best_score = quad, score
    if best is None:
        raise ValueError("No card-like luma/saturation component found in photo")
    return best


def _profile_outline_seed(gray: np.ndarray) -> np.ndarray | None:
    """Seed a mostly upright photo from outline support, not card colour.

    Listing photos frequently place a foil card on a dark textured mat.  That
    texture can dominate both saturation and threshold masks, but it does not
    create a vertical/horizontal brightness transition that persists through
    most of the card's span.  Median Sobel profiles expose those transitions.
    This supplies an axis-aligned seed only; the subsequent edge fit recovers
    small perspective and rotation.
    """
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    px = np.median(gx[int(.12 * h):int(.88 * h)], axis=0)
    py = np.median(gy[:, int(.12 * w):int(.88 * w)], axis=1)

    def peaks(profile: np.ndarray, minimum_gap: int) -> list[int]:
        local = np.flatnonzero((profile[1:-1] >= profile[:-2]) &
                               (profile[1:-1] >= profile[2:])) + 1
        floor = float(np.median(profile))
        # The physical die-cut is often weaker than the adjacent foil stripe;
        # retain modest peaks here and let four-sided geometry reject texture.
        local = local[profile[local] >= max(floor * 1.25, floor + 5.0)]
        chosen: list[int] = []
        for index in local[np.argsort(profile[local])[::-1]]:
            if all(abs(int(index) - old) >= minimum_gap for old in chosen):
                chosen.append(int(index))
            if len(chosen) == 28:
                break
        # Foil cards commonly yield a bright inner stripe next to a weaker
        # physical die-cut transition.  Keep the *outermost* peak in each
        # nearby cluster: left/top clusters keep their smaller coordinate and
        # right/bottom clusters their larger coordinate.  This preserves the
        # actual card outline when its adjacent foil line is sharper.
        groups: list[list[int]] = []
        for index in sorted(chosen):
            if not groups or index - groups[-1][-1] > 3 * minimum_gap:
                groups.append([index])
            else:
                groups[-1].append(index)
        exterior = []
        midpoint = 0.5 * len(profile)
        for group in groups:
            exterior.append(group[0] if np.mean(group) < midpoint else group[-1])
        return exterior

    xs, ys = peaks(px, max(10, w // 80)), peaks(py, max(10, h // 80))
    best, best_score = None, -np.inf
    for left in xs:
        for right in xs:
            width = right - left
            if not 0.42 * w <= width <= 0.90 * w:
                continue
            for top in ys:
                for bottom in ys:
                    height = bottom - top
                    if not 0.45 * h <= height <= 0.92 * h:
                        continue
                    ratio = width / height
                    if abs(ratio - CARD_RATIO) > 0.11:
                        continue
                    # Log weights stop a very sharp internal rule-line from
                    # overruling the four-sided card geometry.
                    score = (np.log1p(px[left]) + np.log1p(px[right]) +
                             np.log1p(py[top]) + np.log1p(py[bottom]) -
                             18.0 * abs(ratio - CARD_RATIO))
                    if score > best_score:
                        best_score = score
                        best = (left, top, right, bottom)
    if best is None:
        return None
    left, top, right, bottom = best
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                    dtype=np.float32)


def _intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray:
    """Intersection of OpenCV fitLine vectors (vx, vy, x0, y0)."""
    va, pa = line_a[:2], line_a[2:]
    vb, pb = line_b[:2], line_b[2:]
    A = np.column_stack((va, -vb))
    s, _ = np.linalg.solve(A, pb - pa)
    return pa + s * va


def _refine_outer_scan(bgr: np.ndarray, seed: np.ndarray,
                       n_samples: int = 180, photo_mode: bool = False,
                       diagnostics: dict | None = None) -> np.ndarray:
    """Refine a scan's die-cut from supported perpendicular edge transitions."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    q = _order_corners(seed).astype(float)
    centre = q.mean(axis=0)
    short = min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))
    search = max(28.0, (0.09 if photo_mode else 0.045) * short)
    if photo_mode:
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), max(0.55, short / 1260.0))
    fit_tol = max(0.8, 2.5 * short / 1260.0) if photo_mode else 2.5
    offsets = np.linspace(-search, search, int(2 * search + 1))
    fitted = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        direction = b - a
        direction /= np.linalg.norm(direction)
        inward = np.array([-direction[1], direction[0]])
        if np.dot(inward, centre - 0.5 * (a + b)) < 0:
            inward = -inward
        ts_frac = np.linspace(0.10, 0.90, n_samples)
        ts = ts_frac * np.linalg.norm(b - a)  # RANSAC coordinates are pixels.
        base = a[None, :] + ts_frac[:, None] * (b - a)[None, :]
        points = base[:, None, :] + offsets[None, :, None] * inward[None, None, :]
        values = cv2.remap(gray, points[:, :, 0].astype(np.float32),
                           points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        gradients = np.abs(np.gradient(values.astype(float), axis=1))
        if photo_mode:
            signed = np.gradient(values.astype(float), axis=1)
            profile = np.median(values.astype(float), axis=0)
            background = np.median(values[:, :max(3, len(offsets)//8)])
            polarity = 1 if np.max(profile) - background >= background - np.min(profile) else -1
            gradients = np.maximum(0, polarity * signed)
            # An internal stripe has foil, rather than background, upstream.
            distance = max(3, int(round(.022 * short)))
            gap = max(1, int(round(.004 * short)))
            cumulative = np.pad(np.cumsum(values.astype(float), axis=1), ((0,0),(1,0)))
            ends = np.maximum(0, np.arange(len(offsets)) - gap)
            starts = np.maximum(0, np.arange(len(offsets)) - distance)
            exterior = (cumulative[:, ends] - cumulative[:, starts]) / np.maximum(1, ends-starts)
            bg = np.median(values[:, :max(3, len(offsets)//8)], axis=1)[:,None]
            mismatch = np.maximum(0, polarity * (exterior - bg))
            edge_weights = 1.0 / (1.0 + (mismatch / 30.0) ** 2)
        candidate_base = base
        if photo_mode:
            rows = []
            for row in range(n_samples):
                g = gradients[row]
                floor = float(np.median(g)) + 1e-6
                threshold = max(1.5 * floor, 0.10 * float(np.max(g)))
                peaks = np.flatnonzero((g[1:-1] >= g[:-2]) & (g[1:-1] >= g[2:]) &
                                       (g[1:-1] >= threshold)) + 1
                rows.extend((ts[row], offsets[i], g[i] * edge_weights[row, i]) for i in peaks)
            cand = np.asarray(rows, dtype=float)
            prior_tol = 0.8 * search
        else:
            # On a flatbed scan the exterior is reliably white.  The largest
            # edge gradient can instead be a high-contrast foil stripe just
            # inside the die-cut (notably along Guzzlord's right edge).  Walk
            # from paper into the card and use the first sustained darkening;
            # retain its local gradient only as a fitting weight.  This uses
            # a property of scans only, hence photo mode keeps its distinct
            # multi-candidate path above.
            gradient_pick = np.argmax(gradients, axis=1)
            dark = values < 200
            dark_pick = np.argmax(dark, axis=1)
            dark_valid = np.any(dark, axis=1)
            # Usually the strongest transition and the paper-to-card crossing
            # agree.  Switch only when the strongest transition is markedly
            # farther *inboard*; that is the foil-stripe failure mode rather
            # than ordinary edge blur or a slightly imperfect seed.
            use_dark = (np.count_nonzero(dark_valid) >= 25 and
                        np.median(offsets[gradient_pick[dark_valid]] -
                                  offsets[dark_pick[dark_valid]]) > 12.0)
            pick = dark_pick if use_dark else gradient_pick
            valid = dark_valid if use_dark else np.ones(n_samples, dtype=bool)
            cand = np.column_stack((ts[valid], offsets[pick[valid]],
                                    gradients[np.arange(n_samples)[valid], pick[valid]]))
            candidate_base = base[valid]
            prior_tol = search
        model, support = _ransac_keyline(cand, expected_inset=0.0,
                                         inset_tol_px=prior_tol, tol_px=fit_tol)
        t, d, weight = cand.T
        residual = np.abs(d - (model[0] * t + model[1]))
        inliers = residual <= fit_tol
        if photo_mode:
            keep = []
            for pos in np.unique(t[inliers]):
                choices = np.flatnonzero(inliers & (t == pos))
                keep.append(choices[np.argmax(weight[choices])])
            base_by_t = {float(value): point for value, point in zip(ts, base)}
            pts = np.asarray([base_by_t[float(t[i])] + d[i] * inward for i in keep])
        else:
            pts = candidate_base[inliers] + d[inliers, None] * inward[None, :]
        if len(pts) < 25 or support < 0.25:
            raise ValueError("Insufficient coherent support for outer die-cut edge")
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER,
                                     0, 0.01, 0.01).reshape(4)
        fitted.append(np.array([vx, vy, x0, y0], dtype=float))
        if diagnostics is not None:
            side = ("top", "right", "bottom", "left")[k]
            normal = np.array([-vy, vx])
            errors = np.abs((pts - [x0, y0]) @ normal)
            diagnostics[side] = {
                "support": float(len(pts) / n_samples),
                "span": float(np.ptp((pts - a) @ direction) / np.linalg.norm(b-a)),
                "residual_p90_px": float(np.percentile(errors, 90)),
                "residual_p90_rectified_px": float(np.percentile(errors, 90) * 1260 / short),
            }
    refined = _order_corners(np.array([_intersect_lines(fitted[3], fitted[0]),
                                       _intersect_lines(fitted[0], fitted[1]),
                                       _intersect_lines(fitted[1], fitted[2]),
                                       _intersect_lines(fitted[2], fitted[3])]))
    # Perspective is recovered by the independent line intersections. Never
    # blend them back toward an axis-aligned seed: that biases border ratios.
    if photo_mode:
        _validate_frame_quad(refined, bgr.shape, "outer")
    return refined.astype(np.float32)


def _refine_photo_outer_rectangle(bgr: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Jointly refine four photo edges as one plausible card rectangle.

    Unlike independent line fits, a candidate side must participate in a
    four-sided shape with the Pokémon card aspect ratio.  This is designed for
    listing photos where an interior foil stripe can be locally sharper than
    the physical die-cut.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    q = _order_corners(seed).astype(float)
    centre = q.mean(axis=0)
    short = min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))
    search = max(24.0, 0.040 * short)
    offsets = np.arange(-round(search), round(search) + 1, dtype=float)
    sides: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    choices: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        direction = b - a
        direction /= np.linalg.norm(direction)
        inward = np.array([-direction[1], direction[0]])
        if np.dot(inward, centre - .5 * (a + b)) < 0:
            inward = -inward
        frac = np.linspace(.10, .90, 180)
        base = a[None, :] + frac[:, None] * (b - a)[None, :]
        points = base[:, None, :] + offsets[None, :, None] * inward[None, None, :]
        values = cv2.remap(gray, points[:, :, 0].astype(np.float32),
                           points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        gradient = np.abs(np.gradient(values.astype(float), axis=1))
        # A die-cut is supported along most of a side.  Median support rejects
        # diagonal mat texture and isolated artwork transitions.
        support = np.median(gradient, axis=0)
        order = np.argsort(support)[::-1]
        picked: list[int] = []
        for index in order:
            if all(abs(int(index) - old) >= 3 for old in picked):
                picked.append(int(index))
            if len(picked) == 7:
                break
        # Preserve the profile-seeded outline as an explicit candidate.
        zero = int(np.argmin(np.abs(offsets)))
        if zero not in picked:
            picked.append(zero)
        choices.append(np.asarray(picked, int))
        supports.append(support)
        sides.append((a, b, direction, inward))

    best_quad, best_score = q, -np.inf
    for indices in product(*choices):
        lines = []
        score = 0.0
        for k, index in enumerate(indices):
            a, b, direction, inward = sides[k]
            point = .5 * (a + b) + offsets[index] * inward
            lines.append(np.r_[direction, point])
            score += float(supports[k][index])
        try:
            candidate = _order_corners(np.array([
                _intersect_lines(lines[3], lines[0]),
                _intersect_lines(lines[0], lines[1]),
                _intersect_lines(lines[1], lines[2]),
                _intersect_lines(lines[2], lines[3]),
            ]))
        except np.linalg.LinAlgError:
            continue
        width = .5 * (np.linalg.norm(candidate[1] - candidate[0]) +
                       np.linalg.norm(candidate[2] - candidate[3]))
        height = .5 * (np.linalg.norm(candidate[3] - candidate[0]) +
                        np.linalg.norm(candidate[2] - candidate[1]))
        ratio = width / max(height, 1e-6)
        if abs(ratio - CARD_RATIO) > .075:
            continue
        # Keep the global geometry near the profile outline unless all four
        # sides provide enough evidence to justify a move.
        shift = sum(abs(offsets[index]) for index in indices)
        score -= 110.0 * abs(ratio - CARD_RATIO) + .35 * shift
        if score > best_score:
            best_quad, best_score = candidate, score
    return best_quad.astype(np.float32)


def _profile_peak(gray: np.ndarray, axis: int, near_start: bool,
                  lo_frac: float = 0.020, hi_frac: float = 0.070) -> tuple[float, float]:
    """Find a continuous frame transition in the narrow expected border band.

    Median aggregation rejects local artwork details.  A blur along the side
    removes printed text, so a candidate needs support for most of that side.
    """
    # axis 1: vertical edge, varying x.  axis 0: horizontal edge, varying y.
    long_axis = 0 if axis == 1 else 1
    smooth = cv2.GaussianBlur(gray, (1, 61) if axis == 1 else (61, 1), 0)
    profile = np.median(smooth, axis=long_axis)
    gradient = np.abs(np.gradient(profile.astype(np.float32)))
    n = len(profile)
    lo, hi = int(n * lo_frac), int(n * hi_frac)
    if not near_start:
        lo, hi = n - hi, n - lo
    candidates = gradient[lo:hi]
    if len(candidates) < 4:
        raise ValueError("Image is too small for GX keyline search")
    # The foil rim contains very sharp transitions right at the die cut.  The
    # desired keyline is neither necessarily that rim nor the strongest inward
    # transition: it is a manufactured frame at a repeatable physical inset.
    # The two annotated GX cards place it near 4% of the short dimension and
    # 3% of the long dimension.  This is a broad prior, not a hard coordinate.
    floor = float(np.median(candidates)) + 1e-6
    threshold = max(1.5 * floor, 0.05 * float(np.max(candidates)))
    local = np.flatnonzero((candidates[1:-1] >= candidates[:-2]) &
                           (candidates[1:-1] >= candidates[2:]) &
                           (candidates[1:-1] >= threshold)) + 1
    if len(local):
        expected = (0.040 if axis == 1 else 0.030) * n
        positions = lo + local
        distances = np.abs((positions if near_start else (n - 1 - positions)) - expected)
        # Keep edge strength in the decision, but downweight peaks far from
        # the known frame inset.  This separates the keyline from its nearby
        # decorative foil stripes without assuming a particular card colour.
        scores = candidates[local] / (1.0 + (distances / (0.010 * n)) ** 2)
        offset = int(local[int(np.argmax(scores))])
    else:
        offset = int(np.argmax(candidates))
    peak = lo + offset
    confidence = float(gradient[peak] / floor)
    return float(peak), confidence


def _keyline_candidates(gray: np.ndarray, side: str, n_samples: int = 180):
    """Return plausible keyline transitions as (along, inset, strength).

    `inset` always increases from the physical die-cut edge into the card.
    Keeping several candidates per profile is essential: foil stripes and
    artwork can be locally stronger than the true keyline, while only the
    manufactured keyline remains straight across most of the side.
    """
    h, w = gray.shape
    vertical = side in {"left", "right"}
    dim, length = (w, h) if vertical else (h, w)
    expected = (0.040 if vertical else 0.030) * dim
    lo, hi = int(0.015 * dim), int(0.075 * dim)
    along = np.linspace(int(0.10 * length), int(0.90 * length), n_samples).astype(int)
    candidates = []
    for t in along:
        profile = gray[t, :] if vertical else gray[:, t]
        if side in {"right", "bottom"}:
            profile = profile[::-1]
        # A short blur stabilises scanner texture without moving a straight
        # transition by a meaningful amount.
        p = cv2.GaussianBlur(profile.astype(np.float32)[None, :], (9, 1), 0)[0]
        g = np.abs(np.gradient(p))
        band = g[lo:hi]
        floor = float(np.median(band)) + 1e-6
        threshold = max(1.5 * floor, 0.10 * float(np.max(band)))
        peaks = np.flatnonzero((band[1:-1] >= band[:-2]) &
                               (band[1:-1] >= band[2:]) &
                               (band[1:-1] >= threshold)) + 1 + lo
        for d in peaks:
            # A broad physical prior keeps the RANSAC cloud bounded to the
            # frame region rather than letting artwork far into the card win.
            strength = float(g[d] / (1.0 + ((d - expected) / (0.016 * dim)) ** 2))
            candidates.append((float(t), float(d), strength))
    return np.asarray(candidates, dtype=float)


def _ransac_keyline(candidates: np.ndarray, expected_inset: float,
                    inset_tol_px: float, tol_px: float = 2.5,
                    iters: int = 700) -> tuple[np.ndarray, float]:
    """Fit inset = slope * along + intercept, rewarding unique scan rows."""
    if len(candidates) < 20:
        raise ValueError("Too few supported transitions for GX keyline fit")
    rng = np.random.default_rng(0)
    t, d, weight = candidates.T
    best_model = None
    best_score = -1.0
    best_mask = None
    _, row_ids = np.unique(t, return_inverse=True)
    n_rows = int(row_ids.max()) + 1
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if abs(t[i] - t[j]) < 2:
            continue
        slope = (d[j] - d[i]) / (t[j] - t[i])
        intercept = d[i] - slope * t[i]
        residual = np.abs(d - (slope * t + intercept))
        mask = residual <= tol_px
        if not np.any(mask):
            continue
        # Multiple peaks from one profile must not count as multiple votes.
        votes = np.zeros(n_rows)
        np.maximum.at(votes, row_ids[mask], weight[mask])
        score = float(votes.sum())
        # Artwork/text can make a perfectly straight, well-supported line.
        # A real full-art GX keyline must additionally sit at the physical
        # frame inset.  This prior is deliberately soft so print variation is
        # tolerated, but it breaks ties with a footer or attack-rule line.
        mid_inset = slope * float(np.median(t)) + intercept
        score *= float(np.exp(-((mid_inset - expected_inset) / inset_tol_px) ** 2))
        if score > best_score:
            best_model, best_score, best_mask = np.array([slope, intercept]), score, mask
    if best_model is None:
        raise ValueError("No coherent GX keyline line found")
    # Iteratively select the strongest inlier per profile, then weighted-fit.
    model = best_model
    for _ in range(3):
        residual = np.abs(d - (model[0] * t + model[1]))
        mask = residual <= tol_px
        picked = []
        for row in np.unique(t[mask]):
            choices = np.flatnonzero(mask & (t == row))
            picked.append(choices[np.argmax(weight[choices])])
        picked = np.asarray(picked, int)
        if len(picked) < 20:
            break
        model = np.polyfit(t[picked], d[picked], 1, w=np.sqrt(weight[picked]))
    support = float(len(picked) / len(np.unique(t))) if len(picked) else 0.0
    return np.asarray(model, float), support


def _fit_keyline_quad(gray: np.ndarray, prior_quad: np.ndarray | None = None,
                      diagnostics: dict | None = None) -> tuple[np.ndarray, float]:
    """Fit all four straight keylines, optionally constrained by an API prior."""
    h, w = gray.shape
    models, supports = {}, []
    prior = None if prior_quad is None else _order_corners(prior_quad)
    for side in ("left", "right", "top", "bottom"):
        vertical = side in {"left", "right"}
        dim = w if vertical else h
        expected = (0.040 if vertical else 0.030) * dim
        inset_tol = 0.010 * dim
        if prior is not None:
            # Convert the API's normalized/source-image quadrilateral through
            # the same homography, then search locally for the actual printed
            # line.  The model supplies a geometric prior; pixels remain the
            # final authority for the line location.
            if side == "left":
                expected = float(np.mean(prior[[0, 3], 0]))
            elif side == "right":
                expected = float((w - 1) - np.mean(prior[[1, 2], 0]))
            elif side == "top":
                expected = float(np.mean(prior[[0, 1], 1]))
            else:
                expected = float((h - 1) - np.mean(prior[[2, 3], 1]))
            # Do not allow a coarse model box to select a distant artwork
            # line, while still tolerating meaningful vision-coordinate error.
            inset_tol = max(12.0, 0.012 * dim)
        candidates = _keyline_candidates(gray, side)
        model, support = _ransac_keyline(candidates, expected, inset_tol_px=inset_tol)
        if diagnostics is not None:
            t, d, _ = candidates.T
            residual = np.abs(d - (model[0] * t + model[1]))
            inliers = residual <= 2.5
            supported = np.unique(t[inliers])
            length = h if vertical else w
            diagnostics[side] = {
                "support": float(len(supported) / 180),
                "span": float(np.ptp(supported) / length) if len(supported) else 0.0,
                "end_to_end_inset_change_px": float(abs(model[0]) * length),
                "residual_p90_px": float(np.percentile(residual[inliers], 90)),
            }
        models[side] = model
        supports.append(support)
    # Lines in ordinary rectified x/y coordinates.
    # left/right: x = a*y+b; top/bottom: y = a*x+b
    al, bl = models["left"]
    ar, br = models["right"]
    at, bt = models["top"]
    ab, bb = models["bottom"]
    lines = {
        "left": (1.0, -al, -bl),
        "right": (1.0, ar, -(w - 1 - br)),
        "top": (-at, 1.0, -bt),
        "bottom": (ab, 1.0, -(h - 1 - bb)),
    }
    def meet(v, hline):
        a1, b1, c1 = lines[v]; a2, b2, c2 = lines[hline]
        return np.linalg.solve(np.array([[a1, b1], [a2, b2]]), -np.array([c1, c2]))
    inner = np.array([meet("left", "top"), meet("right", "top"),
                      meet("right", "bottom"), meet("left", "bottom")], dtype=np.float32)
    return inner, float(min(supports))


def _edge_probe_contrast(
    bgr: np.ndarray,
    probes: dict[str, tuple[tuple[tuple[float, float], tuple[float, float]], ...]],
) -> dict[str, float]:
    """Measure local outside-to-inside colour separation for API probe pairs.

    These values are evidence for a future edge-selection score, not an RGB
    instruction from the model.  Lab-distance makes the report comparable
    across sides while preserving the camera's actual lighting and foil hue.
    """
    h, w = bgr.shape[:2]
    radius = max(2, round(min(h, w) / 350))

    def sample(point: tuple[float, float]) -> np.ndarray:
        x = round(point[0] * (w - 1))
        y = round(point[1] * (h - 1))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        colour = np.median(bgr[y0:y1, x0:x1], axis=(0, 1)).astype(np.uint8)
        return cv2.cvtColor(colour.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(float)

    contrast = {}
    for side, pairs in probes.items():
        distances = [float(np.linalg.norm(sample(outside) - sample(inside)))
                     for outside, inside in pairs]
        contrast[side] = float(np.median(distances))
    return contrast


def measure_full_art_gx_scan(bgr: np.ndarray, target_width: int = 1260,
                             outer_quad: np.ndarray | None = None,
                             assumptions: GXImageAssumptions | None = None) -> GXMeasurement:
    """Measure a full-art GX scan or bare photo to the continuous inner keyline.

    The result is rectified before line searches, making horizontal and vertical
    margins comparable even if the scan is rotated a little.  `target_width`
    sets the sampling resolution; 1260 is the calibrated default. The homography
    geometry is scale invariant, but image resampling can affect fitted peaks.
    """
    assumptions = assumptions or GXImageAssumptions()
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 100:
        raise ValueError("expected a BGR image at least 100 pixels on each side")
    if target_width < 400:
        raise ValueError("target_width must be at least 400 pixels")
    if not assumptions.front_card:
        raise ValueError("GX centering requires a front-of-card image")
    if assumptions.fully_in_frame is False:
        raise ValueError("card is not fully in frame")
    if assumptions.holder in {"penny_sleeve", "rigid_toploader"}:
        raise ValueError(f"holder mode is not supported yet ({assumptions.holder})")
    if assumptions.outer_edge_visibility in {"partial", "poor"}:
        raise ValueError("API reports that the physical outer edge is not clear")
    if assumptions.capture not in {"auto", "scan", "photo"}:
        raise ValueError("capture must be auto, scan, or photo")

    api_seed = None
    api_inner_source = None
    api_probe_contrast = None
    if assumptions.card_quad_normalized is not None:
        h, w = bgr.shape[:2]
        api_seed = np.asarray(assumptions.card_quad_normalized, np.float32)
        api_seed *= np.array([w - 1, h - 1], np.float32)
    if assumptions.inner_keyline_quad_normalized is not None:
        h, w = bgr.shape[:2]
        api_inner_source = np.asarray(assumptions.inner_keyline_quad_normalized, np.float32)
        api_inner_source *= np.array([w - 1, h - 1], np.float32)
    if assumptions.outer_edge_probe_pairs_normalized is not None:
        api_probe_contrast = _edge_probe_contrast(bgr, assumptions.outer_edge_probe_pairs_normalized)

    photo_mode = (assumptions.capture == "photo" or
                  (assumptions.capture == "auto" and
                   float((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 245).mean()) <= .35))
    fit_diagnostics = {"outer": {}, "inner": {}}
    if outer_quad is None:
        try:
            outer = locate_card_on_photo(bgr) if photo_mode else locate_card_on_scan(bgr)
        except ValueError:
            if api_seed is None:
                raise
            _validate_frame_quad(api_seed, bgr.shape, "API seed")
            outer = api_seed
        # API localization is a fallback seed only. Measurement always requires
        # pixel-supported lines; API keylines are diagnostic, never an oracle.
        outer = _refine_outer_scan(bgr, outer, photo_mode=photo_mode,
                                   diagnostics=fit_diagnostics["outer"])
    else:
        outer = np.asarray(outer_quad, np.float32)
        _validate_frame_quad(outer, bgr.shape, "supplied outer")
    _validate_frame_quad(outer, bgr.shape, "outer")
    target_height = round(target_width / CARD_RATIO)
    dst = np.array([[0, 0], [target_width - 1, 0],
                    [target_width - 1, target_height - 1], [0, target_height - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(outer, dst)
    rectified = cv2.warpPerspective(bgr, H, (target_width, target_height))
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    api_inner_rectified = None
    if api_inner_source is not None:
        api_inner_rectified = cv2.perspectiveTransform(api_inner_source[None, :, :], H)[0]
    inner, support = _fit_keyline_quad(gray, diagnostics=fit_diagnostics["inner"])
    inner_source = cv2.perspectiveTransform(inner[None], np.linalg.inv(H))[0]
    _validate_frame_quad(inner_source, bgr.shape, "inner")
    fit_diagnostics["inner_contained"] = bool(
        np.all(inner >= 0) and np.all(inner[:, 0] < target_width) and
        np.all(inner[:, 1] < target_height))
    fit_diagnostics["source_short_side_px"] = float(min(
        np.linalg.norm(outer[(i+1)%4]-outer[i]) for i in range(4)))
    fit_diagnostics["max_inner_inset_change_mm"] = float(max(
        d["end_to_end_inset_change_px"] for d in fit_diagnostics["inner"].values()) * 63 / target_width)
    left = float(np.mean(inner[[0, 3], 0]))
    right_x = float(np.mean(inner[[1, 2], 0]))
    top = float(np.mean(inner[[0, 1], 1]))
    bottom_y = float(np.mean(inner[[2, 3], 1]))
    margins = {
        "left": left,
        "right": (target_width - 1) - right_x,
        "top": top,
        "bottom": (target_height - 1) - bottom_y,
    }
    horizontal = margins["left"] + margins["right"]
    vertical = margins["top"] + margins["bottom"]
    if min(horizontal, vertical) <= 0:
        raise ValueError("GX keyline search produced an invalid frame")
    centering = {
        "left_pct": 100.0 * margins["left"] / horizontal,
        "right_pct": 100.0 * margins["right"] / horizontal,
        "top_pct": 100.0 * margins["top"] / vertical,
        "bottom_pct": 100.0 * margins["bottom"] / vertical,
    }
    disagreement = None
    if api_inner_rectified is not None:
        disagreement = float(np.mean(np.linalg.norm(inner - api_inner_rectified, axis=1)))
    return GXMeasurement(margins, centering, outer,
                         {"left": left, "right": right_x, "top": top, "bottom": bottom_y},
                         support, disagreement, api_probe_contrast,
                         inner_quad=inner_source, capture="photo" if photo_mode else "scan",
                         holder=assumptions.holder, rectified_width=target_width,
                         fit_diagnostics=fit_diagnostics)


def estimate_full_art_gx(bgr: np.ndarray, assumptions: GXImageAssumptions | None = None,
                         target_width: int = 1260) -> tuple[GXMeasurement, dict]:
    """API-aware public entry point for the eBay pregrading flow.

    The API's classification controls eligibility and detector routing; OpenCV
    still establishes the outer frame, rectifies it, and fits the inner frame.
    """
    assumptions = assumptions or GXImageAssumptions()
    result = measure_full_art_gx_scan(bgr, target_width=target_width,
                                      assumptions=assumptions)
    return result, assess_gx_measurement(result, target_width=target_width)


def centering_from_frame_quads(outer: np.ndarray, inner: np.ndarray) -> dict[str, float]:
    """Compute the canonical perspective-corrected centering of annotated quads.

    This is the oracle used to test the detector.  It is also the same geometry
    a production result will use after robust outer and inner line fitting.
    """
    outer = np.asarray(outer, np.float32).reshape(4, 2)
    inner = np.asarray(inner, np.float32).reshape(4, 2)
    dst = np.float32([[0, 0], [63, 0], [63, 88], [0, 88]])
    H = cv2.getPerspectiveTransform(outer, dst)
    r = cv2.perspectiveTransform(inner[None, :, :], H)[0]
    left = float(np.mean(r[[0, 3], 0]))
    right = float(np.mean(63.0 - r[[1, 2], 0]))
    top = float(np.mean(r[[0, 1], 1]))
    bottom = float(np.mean(88.0 - r[[2, 3], 1]))
    return {
        "left_pct": 100.0 * left / (left + right),
        "right_pct": 100.0 * right / (left + right),
        "top_pct": 100.0 * top / (top + bottom),
        "bottom_pct": 100.0 * bottom / (top + bottom),
    }


def render_scan_diagnostic(bgr: np.ndarray, result: GXMeasurement,
                           target_width: int = 1260) -> np.ndarray:
    """Return a rectified scan with the selected outer/keyline frame overlaid."""
    target_height = round(target_width / CARD_RATIO)
    dst = np.array([[0, 0], [target_width - 1, 0],
                    [target_width - 1, target_height - 1], [0, target_height - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(result.outer_quad, dst)
    out = cv2.warpPerspective(bgr, H, (target_width, target_height))
    if result.inner_quad is not None:
        inner = cv2.perspectiveTransform(result.inner_quad[None].astype(np.float32), H)[0]
        cv2.polylines(out, [np.rint(inner).astype(np.int32)], True, (0, 0, 255), 2)
    else:
        p = result.keyline_positions
        cv2.rectangle(out, (round(p["left"]), round(p["top"])),
                      (round(p["right"]), round(p["bottom"])), (0, 0, 255), 2)
    return out


def render_source_diagnostic(bgr: np.ndarray, result: GXMeasurement) -> np.ndarray:
    """Green physical edge and red inner keyline in original image coordinates."""
    out = bgr.copy()
    for quad, colour in ((result.outer_quad, (0, 220, 0)), (result.inner_quad, (0, 0, 255))):
        if quad is not None:
            cv2.polylines(out, [np.rint(quad).astype(np.int32)], True, colour,
                          max(1, round(min(bgr.shape[:2]) / 700)))
    return out
