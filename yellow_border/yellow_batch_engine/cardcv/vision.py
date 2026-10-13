"""OpenAI vision adapter for the GX centering pipeline.

This module deliberately returns *priors*, rather than a centering verdict.
The deterministic geometry in :mod:`cardcv.gx` remains responsible for
rectification, line fitting, and the final margin calculation.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from .gx import GXImageAssumptions


DEFAULT_VISION_MODEL = "gpt-5.4-mini"
_MAX_IMAGE_SIDE = 2048


class GXVisionError(RuntimeError):
    """A recoverable error while obtaining an upstream geometry prior."""


_POINT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "minItems": 2,
    "maxItems": 2,
}
_QUAD_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _POINT_SCHEMA,
    "minItems": 4,
    "maxItems": 4,
}
_PROBE_PAIR_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _POINT_SCHEMA,
    "minItems": 2,
    "maxItems": 2,
}
_PROBE_SIDE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _PROBE_PAIR_SCHEMA,
    "minItems": 1,
    "maxItems": 5,
}
_EDGE_PROBES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {side: _PROBE_SIDE_SCHEMA for side in ("left", "right", "top", "bottom")},
    "required": ["left", "right", "top", "bottom"],
}

# Strict Structured Outputs needs every declared object property to be required.
# A null quadrilateral is how the model explicitly reports that a frame is not
# visible enough to localise, rather than inventing a coordinate.
GX_VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "front_card": {"type": "boolean"},
        "capture": {"type": "string", "enum": ["scan", "photo", "auto"]},
        "holder": {
            "type": "string",
            "enum": ["none", "penny_sleeve", "rigid_toploader", "unknown"],
        },
        "background": {"type": "string", "enum": ["plain", "textured", "busy", "unknown"]},
        "outer_edge_visibility": {
            "type": "string", "enum": ["clear", "partial", "poor", "unknown"],
        },
        "fully_in_frame": {"type": "boolean"},
        "card_quad_normalized": {"anyOf": [_QUAD_SCHEMA, {"type": "null"}]},
        "inner_keyline_quad_normalized": {"anyOf": [_QUAD_SCHEMA, {"type": "null"}]},
        "outer_edge_probe_pairs_normalized": {"anyOf": [_EDGE_PROBES_SCHEMA, {"type": "null"}]},
        "geometry_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "front_card", "capture", "holder", "background", "outer_edge_visibility",
        "fully_in_frame", "card_quad_normalized", "inner_keyline_quad_normalized",
        "outer_edge_probe_pairs_normalized", "geometry_confidence",
    ],
}


GX_VISION_INSTRUCTIONS = """You are a precise image-localisation stage in a Pokémon-card centering system.
Inspect exactly one supplied image. Return only the schema fields.

The target is a front-facing Pokémon trading card. Determine whether it is a front card,
whether it is a scan or photograph, whether it is raw, sleeved, or in a rigid toploader,
whether the full physical card is in frame, and whether its physical outside die-cut edge
is clear.  Do not treat a toploader or sleeve as the card.

When the front card and its physical outline are visible, return card_quad_normalized as
four [x,y] points in this exact clockwise order: top-left, top-right, bottom-right,
bottom-left. Coordinates are fractions of the supplied image width and height, each from
0 to 1. Estimate the extrapolated rectangle behind rounded corners.

For Sun & Moon full-art GX cards, also return inner_keyline_quad_normalized using that
same ordering. It is the first continuous dark-grey/charcoal printed frame immediately
inside the metallic outer foil rim. It is not an artwork edge, title box, attack rule, or
bright foil stripe. If either frame cannot be identified confidently, return null for
that quadrilateral and lower geometry_confidence. Do not invent a quad.

When the physical die-cut is visible, also nominate one to five point pairs per
side in outer_edge_probe_pairs_normalized. Each pair is [[outside_x,outside_y],
[inside_x,inside_y]] across the *physical* card edge; choose uncluttered places
away from rounded corners, text, glare, or a holder. Do not estimate colours:
the caller samples the actual image pixels at these locations.
"""


def _data_url_from_bgr(bgr: np.ndarray) -> str:
    """JPEG-encode a bounded image; API coordinates remain normalized."""
    if bgr is None or bgr.size == 0:
        raise GXVisionError("cannot send an empty image to vision")
    h, w = bgr.shape[:2]
    scale = min(1.0, _MAX_IMAGE_SIDE / max(h, w))
    image = bgr
    if scale < 1.0:
        image = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise GXVisionError("could not JPEG-encode image for vision")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _focus_crop(bgr: np.ndarray, quad: np.ndarray, padding: float = 0.12) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop around a deterministic card proposal while retaining rim context."""
    h, w = bgr.shape[:2]
    q = np.asarray(quad, dtype=float).reshape(4, 2)
    x0, y0 = np.min(q, axis=0)
    x1, y1 = np.max(q, axis=0)
    pad = padding * min(x1 - x0, y1 - y0)
    left, top = max(0, int(np.floor(x0 - pad))), max(0, int(np.floor(y0 - pad)))
    right, bottom = min(w, int(np.ceil(x1 + pad))), min(h, int(np.ceil(y1 + pad)))
    if right - left < 32 or bottom - top < 32:
        raise GXVisionError("card focus crop is too small")
    return bgr[top:bottom, left:right], (left, top, right, bottom)


def _rebase_crop_payload(
    payload: dict[str, Any], crop: tuple[int, int, int, int], full_shape: tuple[int, int],
) -> dict[str, Any]:
    """Map crop-normalized API coordinates back to full-image normalization."""
    left, top, right, bottom = crop
    full_h, full_w = full_shape
    crop_w, crop_h = right - left, bottom - top

    def point(p: list[float] | tuple[float, float]) -> list[float]:
        return [
            (left + float(p[0]) * (crop_w - 1)) / (full_w - 1),
            (top + float(p[1]) * (crop_h - 1)) / (full_h - 1),
        ]

    rebased = dict(payload)
    for name in ("card_quad_normalized", "inner_keyline_quad_normalized"):
        if rebased.get(name) is not None:
            rebased[name] = [point(p) for p in rebased[name]]
    probes = rebased.get("outer_edge_probe_pairs_normalized")
    if probes is not None:
        rebased["outer_edge_probe_pairs_normalized"] = {
            side: [[point(outside), point(inside)] for outside, inside in pairs]
            for side, pairs in probes.items()
        }
    return rebased


def _response_json(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not text:
        raise GXVisionError("vision response did not contain structured output text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GXVisionError("vision response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise GXVisionError("vision response must be a JSON object")
    return payload


def analyze_gx_vision(
    bgr: np.ndarray,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    focus_quad: np.ndarray | None = None,
) -> tuple[GXImageAssumptions, dict[str, Any]]:
    """Call the Responses API and return validated assumptions plus raw metadata.

    Set ``OPENAI_API_KEY`` in the process environment (or supply ``api_key``)
    outside source control. ``client`` exists for deterministic tests and for
    callers that own an already configured OpenAI client. When ``focus_quad``
    is supplied, the request sees a padded card crop for better spatial
    precision; returned coordinates are transparently rebased to the original
    image's normalized coordinate system.
    """
    selected_model = model or os.environ.get("OPENAI_VISION_MODEL") or DEFAULT_VISION_MODEL
    if client is None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise GXVisionError("OPENAI_API_KEY is not set; vision prior was not requested")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GXVisionError("install the 'openai' package to use --vision") from exc
        client = OpenAI(api_key=key)
    source = bgr
    crop = None
    if focus_quad is not None:
        source, crop = _focus_crop(bgr, focus_quad)
    try:
        response = client.responses.create(
            model=selected_model,
            store=False,
            instructions=GX_VISION_INSTRUCTIONS,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Localise the card frames in this image."},
                    {"type": "input_image", "image_url": _data_url_from_bgr(source), "detail": "high"},
                ],
            }],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "gx_card_geometry",
                    "strict": True,
                    "schema": GX_VISION_SCHEMA,
                },
            },
        )
    except Exception as exc:  # SDK error classes may differ across SDK releases.
        raise GXVisionError(f"vision request failed: {exc}") from exc
    payload = _response_json(response)
    if crop is not None:
        payload = _rebase_crop_payload(payload, crop, bgr.shape[:2])
    try:
        assumptions = GXImageAssumptions.from_api(payload)
    except ValueError as exc:
        raise GXVisionError(f"vision response failed geometry validation: {exc}") from exc
    metadata = {
        "model": selected_model,
        "response_id": getattr(response, "id", None),
        "geometry_confidence": payload["geometry_confidence"],
        "focus_crop": crop,
        "assumptions": asdict(assumptions),
    }
    return assumptions, metadata
