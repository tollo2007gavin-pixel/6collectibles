"""
Thin, SDK-free adapters so switching vision providers is a one-flag change.
Each takes a LIST of JPEG bytes (one or more images per card) and returns
(text, input_tokens, output_tokens). Plain REST keeps dependencies
to `requests` and avoids SDK version churn.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import requests


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from .env into the environment (real env vars win)."""
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

# USD per 1M tokens (input, output), standard tier, checked 2026-09-23.
# Prices move often - re-check the provider pages before budgeting.
# Batch APIs on all three providers are ~50% off these rates.
PRICES = {
    "gpt-6-luna":            (0.10, 0.50),
    "gpt-6-sol":             (2.00, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),   # confirm exact model ID in AI Studio
    "claude-haiku-4-5":      (1.00, 5.00),
    "claude-sonnet-5":       (3.00, 15.00),
}

DEFAULT_MODEL = {"openai": "gpt-6-luna", "gemini": "gemini-3.1-flash-lite", "anthropic": "claude-haiku-4-5"}


def cost_usd(model: str, tin: int, tout: int) -> float | None:
    p = PRICES.get(model)
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


def _b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


# Requested reply shape. Refusals and truncated output still require handling,
# and locally validate all results, including JSON-mode fallback responses.
_LOC = {"type": ["integer", "null"], "minimum": 0, "maximum": 3}
WHITENING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["photo_usable", "corners", "edges", "overall", "confidence", "notes"],
    "properties": {
        "corners": {"type": "object", "additionalProperties": False, "required": ["TL", "TR", "BL", "BR"],
                    "properties": {k: _LOC for k in ("TL", "TR", "BL", "BR")}},
        "edges": {"type": "object", "additionalProperties": False, "required": ["TOP", "BOTTOM", "LEFT", "RIGHT"],
                  "properties": {k: _LOC for k in ("TOP", "BOTTOM", "LEFT", "RIGHT")}},
        "overall": {"type": "integer", "minimum": 0, "maximum": 3},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "photo_usable": {"type": "boolean"},
        "notes": {"type": "string"},
    },
}


class APIError(RuntimeError):
    pass


def redact_error(text):
    text = str(text)
    for name in ('OPENAI_API_KEY', 'EBAY_CLIENT_ID', 'EBAY_CLIENT_SECRET', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY'):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, '[REDACTED]')
    return re.sub(r'(?i)(sk-[\w*.-]+|Bearer\s+\S+)', '[REDACTED]', text)


def _post(url, body, headers):
    try:
        r = requests.post(url, json=body, timeout=120, headers=headers)
    except requests.RequestException as e:
        raise APIError(redact_error(e)) from None
    if not r.ok:
        # Include the provider's own explanation, not just the status code.
        raise APIError(f"{r.status_code}: {redact_error(r.text)[:400]}")
    return r.json()


# Which optional OpenAI features this model accepted; learned on the first call.
_openai_features = {}


def call_openai(model, system, user_text, jpegs, max_out=4000, extra=None):
    """
    max_out is generous on purpose: reasoning models spend part of it 'thinking'
    before they answer, and a small limit gets used up entirely -> empty reply.
    reasoning_effort=low keeps that thinking (which is billed) short.
    """
    content = [{"type": "text", "text": user_text},
               *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(j)}", "detail": "high"}}
                 for j in jpegs]]
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    features = _openai_features.setdefault(model, {"schema": True, "reasoning": True})
    while True:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_completion_tokens": max_out,
            "response_format": ({"type": "json_schema", "json_schema": {
                                    "name": "whitening", "strict": True, "schema": WHITENING_SCHEMA}}
                                if features["schema"] else {"type": "json_object"}),
            **({"reasoning_effort": "low"} if features["reasoning"] else {}),
            **(extra or {}),
        }
        try:
            d = _post("https://api.openai.com/v1/chat/completions", body, headers)
            break
        except APIError as e:
            msg = str(e)
            # Older/other models may not support one of the optional features:
            # switch it off once and retry, instead of failing every call.
            unsupported = any(word in msg.lower() for word in ('unsupported', 'not supported', 'does not support', 'unknown parameter'))
            if msg.startswith("400") and unsupported and "reasoning_effort" in msg and features["reasoning"]:
                features["reasoning"] = False
                continue
            if msg.startswith("400") and unsupported and ("json_schema" in msg or "response_format" in msg) and features["schema"]:
                features["schema"] = False
                continue
            raise
    choice = d["choices"][0]
    u = d.get("usage", {})
    text = choice["message"].get("content") or ""
    if choice['message'].get('refusal'):
        text = '[refused]'
    elif choice.get('finish_reason') != 'stop':
        text = '[incomplete: ' + str(choice.get('finish_reason')) + ']'
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_gemini(model, system, user_text, jpegs, max_out=4000, extra=None):
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [
            {"text": user_text},
            *[{"inline_data": {"mime_type": "image/jpeg", "data": _b64(j)}} for j in jpegs],
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_out, **(extra or {})},
    }
    d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
              body, {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    u = d.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)  # thinking bills as output
    return text, u.get("promptTokenCount", 0), out


def call_anthropic(model, system, user_text, jpegs, max_out=1500, extra=None):
    body = {
        "model": model, "max_tokens": max_out, "system": system,
        "messages": [{"role": "user", "content": [
            *[{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(j)}}
              for j in jpegs],
            {"type": "text", "text": user_text},
        ]}],
        **(extra or {}),
    }
    d = _post("https://api.anthropic.com/v1/messages", body, {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    u = d.get("usage", {})
    text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


CALLERS = {"openai": call_openai, "gemini": call_gemini, "anthropic": call_anthropic}
