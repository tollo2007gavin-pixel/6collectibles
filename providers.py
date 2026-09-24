"""
Thin, SDK-free adapters so switching vision providers is a one-flag change.
Each takes a LIST of JPEG bytes (one or more images per card) and returns
(text, input_tokens, output_tokens). Plain REST keeps dependencies
to `requests` and avoids SDK version churn.
"""
from __future__ import annotations

import base64
import os
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


def call_openai(model, system, user_text, jpegs, max_out=600, extra=None):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(j)}", "detail": "high"}}
                  for j in jpegs],
            ]},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_out,
        **(extra or {}),
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=90,
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_gemini(model, system, user_text, jpegs, max_out=600, extra=None):
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [
            {"text": user_text},
            *[{"inline_data": {"mime_type": "image/jpeg", "data": _b64(j)}} for j in jpegs],
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_out, **(extra or {})},
    }
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      json=body, timeout=90, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    r.raise_for_status()
    d = r.json()
    u = d.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)  # thinking bills as output
    return text, u.get("promptTokenCount", 0), out


def call_anthropic(model, system, user_text, jpegs, max_out=600, extra=None):
    body = {
        "model": model, "max_tokens": max_out, "system": system,
        "messages": [{"role": "user", "content": [
            *[{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(j)}}
              for j in jpegs],
            {"type": "text", "text": user_text},
        ]}],
        **(extra or {}),
    }
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=90, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


CALLERS = {"openai": call_openai, "gemini": call_gemini, "anthropic": call_anthropic}
