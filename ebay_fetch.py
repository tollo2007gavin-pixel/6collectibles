#!/usr/bin/env python3
"""
Pull eBay listings for configured searches and hand usable card images to the
centering pipeline.

Designed to run stateless (GitHub Actions): the only state is a small JSON file
of already-processed item IDs / image IDs, persisted between runs via
actions/cache. Every run only pays for what's new.

Cost controls, in the order they save money:
  1. Official Browse API (free tier, 5k calls/day) - no scraping, no proxies.
  2. 200 results per call (API max), sorted newest-first, with a configured
     maximum page count. Known items are skipped individually.
  3. Title keyword filtering BEFORE any image is downloaded.
  4. Image-ID dedupe: relisted cards reusing the same eBay-hosted photo are skipped.
  5. One image per listing (the primary photo), fetched at eBay's s-l1600 size.
  6. Cheap resolution + sharpness gate BEFORE the (more expensive) pipeline runs.
  7. Downloads run in parallel but in bounded batches to keep memory flat.

Pipeline contract (wrap your friend's code to match):
    def grade(image_bgr: np.ndarray, meta: dict) -> dict
Pass it as --pipeline some_module:grade. With no pipeline set, images and a
manifest are written to the output folder instead.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import providers  # noqa: F401  (importing it loads .env for local runs)

log = logging.getLogger("ebay_fetch")

EBAY_API = "https://api.ebay.com"
TOKEN_URL = f"{EBAY_API}/identity/v1/oauth2/token"
SEARCH_URL = f"{EBAY_API}/buy/browse/v1/item_summary/search"
SCOPE = "https://api.ebay.com/oauth/api_scope"
PAGE_SIZE = 200  # API maximum -> fewest calls per listing
IMG_SIZE_RE = re.compile(r"s-l\d+")
IMG_ID_RE = re.compile(r"/images/g/([^/]+)/")


@dataclass
class Listing:
    item_id: str
    title: str
    price: str | None
    currency: str | None
    url: str
    image_url: str
    search: str
    created: str | None = None
    additional_images: list[str] = field(default_factory=list)


# ----------------------------------------------------------------- HTTP / auth

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.headers["User-Agent"] = "six-collectibles-centering/1.0"
    return s


def get_app_token(session: requests.Session, client_id: str, client_secret: str) -> str:
    """Client-credentials token (valid ~2h) - one per run is plenty."""
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = session.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {creds}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": SCOPE},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ----------------------------------------------------------------------- state

class State:
    """Item IDs and image IDs already handled, pruned after ttl_days."""

    def __init__(self, path: Path, ttl_days: int = 30):
        self.path = path
        self.ttl = ttl_days * 86400
        self.items: dict[str, float] = {}
        self.images: dict[str, float] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.items = data.get("items", {})
                self.images = data.get("images", {})
            except json.JSONDecodeError:
                log.warning("State file corrupt; starting fresh")
        cutoff = time.time() - self.ttl
        self.items = {k: v for k, v in self.items.items() if v >= cutoff}
        self.images = {k: v for k, v in self.images.items() if v >= cutoff}

    def seen_item(self, item_id: str) -> bool:
        return item_id in self.items

    def seen_image(self, img_key: str) -> bool:
        return img_key in self.images

    def mark(self, item_id: str, img_key: str | None = None) -> None:
        now = time.time()
        self.items[item_id] = now
        if img_key:
            self.images[img_key] = now

    def save(self) -> None:
        cutoff = time.time() - self.ttl
        self.items = {k: v for k, v in self.items.items() if v >= cutoff}
        self.images = {k: v for k, v in self.images.items() if v >= cutoff}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"items": self.items, "images": self.images}))
        tmp.replace(self.path)


# ---------------------------------------------------------------------- search

def build_filter(s: dict) -> str:
    parts = []
    if "min_price" in s or "max_price" in s:
        parts.append(f"price:[{s.get('min_price', '')}..{s.get('max_price', '')}]")
        parts.append(f"priceCurrency:{s.get('currency', 'USD')}")
    if s.get("condition_ids"):
        parts.append("conditionIds:{" + "|".join(map(str, s["condition_ids"])) + "}")
    if s.get("buying_options"):
        parts.append("buyingOptions:{" + "|".join(s["buying_options"]) + "}")
    if s.get("item_location_country"):
        parts.append(f"itemLocationCountry:{s['item_location_country']}")
    parts.extend(s.get("extra_filters", []))
    return ",".join(parts)


def search_listings(session, token, s: dict, state: State, marketplace: str):
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace}
    params = {"q": s["q"], "limit": PAGE_SIZE, "sort": "newlyListed"}
    if s.get("category_ids"):
        params["category_ids"] = str(s["category_ids"])
    if s.get("aspect_filter"):
        params["aspect_filter"] = s["aspect_filter"]
    f = build_filter(s)
    if f:
        params["filter"] = f

    exclude = [k.lower() for k in s.get("exclude_keywords", [])]
    require = [k.lower() for k in s.get("require_keywords", [])]
    found: list[Listing] = []
    calls = 0
    url, page_params = SEARCH_URL, params

    for _ in range(int(s.get("max_pages", 2))):
        r = session.get(url, headers=headers, params=page_params, timeout=30)
        calls += 1
        r.raise_for_status()
        data = r.json()

        for it in data.get("itemSummaries", []):
            iid = it["itemId"]
            if state.seen_item(iid):
                continue
            title = it.get("title", "")
            tl = title.lower()
            img = (it.get("image") or {}).get("imageUrl")
            if any(k in tl for k in exclude) or not all(k in tl for k in require) or not img:
                # Filters are search-specific; another search may accept this listing.
                continue
            price = it.get("price") or it.get("currentBidPrice") or {}
            found.append(Listing(
                item_id=iid, title=title,
                price=price.get("value"), currency=price.get("currency"),
                url=it.get("itemWebUrl", ""), image_url=img,
                search=s.get("name", s["q"]), created=it.get("itemCreationDate"),
                additional_images=[a["imageUrl"] for a in it.get("additionalImages", []) if a.get("imageUrl")],
            ))

        nxt = data.get("next")
        # A known item does not prove later pages were handled (overlapping
        # searches and transient download failures leave holes).
        if not nxt:
            break
        url, page_params = nxt, None

    return found, calls


# ---------------------------------------------------------------------- images

def full_res(url: str) -> str:
    """eBay serves any size from the same path; s-l1600 is the largest standard one."""
    return IMG_SIZE_RE.sub("s-l1600", url)


def image_key(url: str) -> str:
    m = IMG_ID_RE.search(url)
    return m.group(1) if m else hashlib.sha1(url.encode()).hexdigest()


def quality_check(img: np.ndarray, q: dict) -> tuple[bool, str]:
    h, w = img.shape[:2]
    if min(h, w) < q.get("min_short_side", 700):
        return False, f"low_res {w}x{h}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1000 / max(h, w)  # normalize so the sharpness score is size-independent
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharp < q.get("min_sharpness", 60):
        return False, f"blurry {sharp:.0f}"
    return True, f"ok sharp={sharp:.0f}"


def fetch_image(session, listing: Listing, q: dict):
    try:
        r = session.get(full_res(listing.image_url), timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        return listing, None, f"download_error {e.__class__.__name__}", False
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return listing, None, "decode_failed", True
    ok, reason = quality_check(img, q)
    return listing, (img if ok else None), reason, True


# -------------------------------------------------------------------- pipeline

def load_pipeline(spec: str | None):
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "grade")


def short_id(item_id: str) -> str:
    parts = item_id.split("|")
    if len(parts) > 1:
        raw = parts[1] + ('_' + parts[2] if len(parts) > 2 and parts[2] != '0' else '')
        return re.sub(r'\W', '_', raw)
    return re.sub(r"\W", "_", item_id)


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="searches.yaml")
    ap.add_argument("--state", default="state/seen.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--pipeline", help="module:function, e.g. centering:grade")
    ap.add_argument("--dry-run", action="store_true", help="search only; no downloads, state untouched")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    q = cfg.get("quality", {})
    workers = int(cfg.get("download_workers", 8))

    session = make_session()
    token = get_app_token(session, os.environ["EBAY_CLIENT_ID"], os.environ["EBAY_CLIENT_SECRET"])
    state = State(Path(args.state), cfg.get("state_ttl_days", 30))
    pipeline = load_pipeline(args.pipeline or cfg.get("pipeline"))

    # 1. Search (cheap: metadata only)
    listings: dict[str, Listing] = {}
    api_calls = 0
    for s in cfg["searches"]:
        found, calls = search_listings(session, token, s, state, cfg.get("marketplace", "EBAY_US"))
        api_calls += calls
        for l in found:
            listings.setdefault(l.item_id, l)
        log.info("search %-20s calls=%d new=%d", s.get("name", s["q"]), calls, len(found))

    # 2. Drop relists that reuse a photo we've already processed
    todo = []
    for l in listings.values():
        k = image_key(l.image_url)
        if state.seen_image(k):
            state.mark(l.item_id)
        else:
            todo.append(l)
    log.info("api_calls=%d candidates=%d after_image_dedupe=%d", api_calls, len(listings), len(todo))

    if args.dry_run:
        for l in todo[:25]:
            log.info("  %s | %s %s | %s", l.title[:70], l.price, l.currency, l.url)
        return 0

    # 3. Download + quality gate + pipeline, in bounded batches
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = {"graded": 0, "saved": 0, "rejected": 0, "errors": 0}

    with open(out / "results.jsonl", "a") as results, ThreadPoolExecutor(workers) as pool:
        batch = workers * 3
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for listing, img, reason, final in pool.map(lambda l: fetch_image(session, l, q), chunk):
                rec = {**asdict(listing), "quality": reason, "ts": int(time.time())}
                key = image_key(listing.image_url)

                if img is None:
                    rec["status"] = "rejected" if final else "error"
                    stats["rejected" if final else "errors"] += 1
                    if final:  # transient download errors get retried next run
                        state.mark(listing.item_id, key)
                elif pipeline:
                    try:
                        rec["result"] = pipeline(img, asdict(listing))
                        rec["status"] = "graded"
                        stats["graded"] += 1
                        state.mark(listing.item_id, key)
                    except Exception as e:  # keep the run alive; retry next time
                        log.exception("pipeline failed on %s", listing.item_id)
                        rec["status"], rec["error"] = "pipeline_error", repr(e)
                        stats["errors"] += 1
                else:
                    path = img_dir / f"{short_id(listing.item_id)}.jpg"
                    if cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        rec["status"], rec["image_path"] = "saved", str(path)
                        stats["saved"] += 1
                        state.mark(listing.item_id, key)
                    else:
                        rec['status'], rec['error'] = 'error', 'image_write_failed'
                        stats['errors'] += 1

                results.write(json.dumps(rec, default=str) + "\n")

    state.save()
    log.info("done api_calls=%d %s", api_calls, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
