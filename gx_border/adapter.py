"""Adapt guarded GX reportable axes and overlays to the common batch schema."""
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from border_routing import validate_route
from .gx_guarded_engine.modern import analyze


def to_combined(result):
    reportable = result.get("status") in {"complete", "partial"} and result.get("margin_report_allowed", True)
    axes = {}
    blocked = result.get("reportability", {})
    for axis in ("lr", "tb"):
        source = result.get("axes", {}).get(axis, {})
        value = source.get("first_pct")
        valid = (reportable and source.get("measured") and not blocked.get("blocked")
                 and axis not in blocked.get("blocked_axes", []) and isinstance(value, (int, float))
                 and math.isfinite(value) and 0 <= value <= 100)
        axes[axis] = {"resolved": bool(valid), "percentage": float(value) if valid else None,
                      "reasons": [] if valid else [source.get("reason", result.get("status", "unresolved"))],
                      "measurement_source": "guarded_gx"}
    status = "Fail" if result.get("status") == "hard_failure" else "Measured" if all(a["resolved"] for a in axes.values()) else "Review / unresolved"
    return {"status": status, "axes": axes, "border_type": "gx", "reasons": result.get("reasons", []),
            "measurement_policy": {"axis_allowance_pp": {"lr": 0., "tb": 0.},
                                   "category": "pass" if status == "Measured" else "undetermined",
                                   "range_kind": "gx_point_estimate", "statistical_confidence_level": None,
                                   "accuracy_calibrated": False}}


def overlay(image, result, combined):
    """Same observed segments as the installed GX renderer; withhold unresolved axes."""
    drawn = image.copy()
    geometry = result.get("geometry")
    active = set()
    if combined["axes"]["lr"]["resolved"]:
        active.update(("left", "right"))
    if combined["axes"]["tb"]["resolved"]:
        active.update(("top", "bottom"))
    if geometry and active:
        tracks = result.get("tracks")
        if not tracks and result.get("default_method") == "source_straight_partial":
            tracks = result.get("methods", {}).get("source_straight", {}).get("tracks")
        for kind, color in (("outer", (0, 190, 0)), ("inner", (0, 0, 220))):
            if tracks:
                for side in active:
                    for segment in tracks[kind][side].get("segments", []):
                        cv2.polylines(drawn, [np.int32(np.rint(segment))], False, color, 1, cv2.LINE_AA)
            else:
                quad = np.int32(np.rint(geometry[kind]))
                for index, side in enumerate(("top", "right", "bottom", "left")):
                    if side in active:
                        cv2.line(drawn, tuple(quad[index]), tuple(quad[(index + 1) % 4]), color, 1, cv2.LINE_AA)
    scale = min(1., 1000 / drawn.shape[1])
    if scale < 1:
        drawn = cv2.resize(drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(drawn, 92, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    labels = ["GX | " + combined["status"] + " | Green: physical edge. Red: inner keyline."]
    for axis, name in (("lr", "L/R"), ("tb", "T/B")):
        value = combined["axes"][axis]["percentage"]
        labels.append(name + (f" {value:.3f}/{100-value:.3f}" if value is not None else " withheld"))
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (10, 24 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def process_front(content, metadata, root):
    validate_route(metadata)
    if metadata["border_type"] != "gx":
        raise ValueError("GX adapter refuses non-GX routes")
    root = Path(root)
    with Image.open(io.BytesIO(content)) as opened:
        if opened.width * opened.height > 50_000_000:
            raise ValueError("Image exceeds 50 million pixels")
        image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(opened).convert("RGB")), cv2.COLOR_RGB2BGR)
    result = analyze(image)  # Preserve current guarded defaults, quality gates and exact-image vetoes.
    combined = to_combined(result)
    stem = metadata["card_id"] + "_" + hashlib.sha256(metadata["item_id"].encode()).hexdigest()[:16]
    source, rendered, detail = "downloads/" + stem + ".png", "Combined/" + stem + ".png", "details/" + stem + ".json"
    if not cv2.imwrite(str(root / source), image) or not cv2.imwrite(str(root / rendered), overlay(image, result, combined)):
        raise OSError("Cannot save GX source or overlay")
    (root / detail).write_text(json.dumps({"metadata": metadata, "source_sha256": hashlib.sha256(content).hexdigest(),
                                         "analysis": result, "combined": combined}, indent=2), encoding="utf-8")
    return dict(metadata, combined=combined, source=source, overlays={"combined": rendered}, details=detail,
                gx_status=result["status"], accuracy_calibrated=False)
