#!/usr/bin/env python3
"""
Turn listings into one folder per card ("batch"), with every photo flattened
and labelled front/back:

    cards/<card_id>/
        front_1.jpg  back_1.jpg  unknown_1.jpg ...
        manifest.json   <- source URLs, side detection, sharpness, listing info

Two ways in:
  A) --from-results out/results.jsonl   listings from ebay_fetch.py (after centering)
  B) --from-dir uploads/                 one sub-folder per card, any image files inside
                                         (for manually uploaded batches)

Idempotent: cards that already have a manifest are skipped, so re-running is free.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

from card_utils import classify_side, find_card, sharpness
from providers import load_dotenv  # noqa: F401  (importing it loads .env)
from ebay_fetch import EBAY_API, full_res, get_app_token, make_session, short_id

log = logging.getLogger("organize")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
MAX_PHOTOS = 8   # listings rarely need more; caps download cost on 24-photo listings


def listing_image_urls(session, token, rec: dict) -> list[str]:
    urls = [rec["image_url"], *rec.get("additional_images", [])]
    if len(urls) == 1 and token:
        # Search results didn't include extra photos: one getItem call fetches them all
        r = session.get(f"{EBAY_API}/buy/browse/v1/item/{rec['item_id']}",
                        headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if r.ok:
            urls += [a["imageUrl"] for a in r.json().get("additionalImages", []) if a.get("imageUrl")]
    seen, out = set(), []
    for u in urls:                     # de-dupe, keep order (primary first)
        if u not in seen:
            seen.add(u)
            out.append(full_res(u))
    return out[:MAX_PHOTOS]


def download(session, url: str) -> np.ndarray | None:
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except requests.RequestException:
        return None


def process_card(card_id: str, images: list[tuple[str, np.ndarray | None]], out_root: Path, meta: dict) -> dict:
    d = out_root / card_id
    d.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    photos = []
    for src, img in images:
        if img is None:
            photos.append({"source": src, "status": "download_failed"})
            continue
        card, found = find_card(img)
        side, blue = classify_side(card)
        counts[side] = counts.get(side, 0) + 1
        name = f"{side}_{counts[side]}.jpg"
        cv2.imwrite(str(d / name), card, [cv2.IMWRITE_JPEG_QUALITY, 95])
        photos.append({"source": src, "file": name, "side": side, "card_found": found,
                       "blue_ratio": blue, "sharpness": round(sharpness(card), 1)})
    manifest = {"card_id": card_id, **meta, "photos": photos,
                "has_front": counts.get("front", 0) > 0, "has_back": counts.get("back", 0) > 0}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-results", help="results.jsonl from ebay_fetch.py")
    src.add_argument("--from-dir", help="folder of per-card sub-folders")
    ap.add_argument("--out", default="cards")
    ap.add_argument("--statuses", default="graded,saved", help="which fetcher statuses to organize")
    ap.add_argument("--require-result-key", help="only cards whose pipeline result has this key truthy, e.g. passed")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root = Path(args.out)
    done = skipped = 0

    if args.from_dir:
        for folder in sorted(p for p in Path(args.from_dir).iterdir() if p.is_dir()):
            if (out_root / folder.name / "manifest.json").exists():
                skipped += 1
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
            imgs = [(str(f), cv2.imread(str(f))) for f in files[:MAX_PHOTOS]]
            m = process_card(folder.name, imgs, out_root, {"source": "upload"})
            done += 1
            log.info("%s front=%s back=%s", folder.name, m["has_front"], m["has_back"])
        log.info("organized=%d skipped_existing=%d", done, skipped)
        return 0

    statuses = set(args.statuses.split(","))
    recs = {}
    for line in Path(args.from_results).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("status") not in statuses:
            continue
        if args.require_result_key and not (rec.get("result") or {}).get(args.require_result_key):
            continue
        recs[rec["item_id"]] = rec          # last record per listing wins

    session = make_session()
    token = None
    if os.environ.get("EBAY_CLIENT_ID"):
        token = get_app_token(session, os.environ["EBAY_CLIENT_ID"], os.environ["EBAY_CLIENT_SECRET"])

    with ThreadPoolExecutor(args.workers) as pool:
        for rec in recs.values():
            cid = short_id(rec["item_id"])
            if (out_root / cid / "manifest.json").exists():
                skipped += 1
                continue
            urls = listing_image_urls(session, token, rec)
            imgs = list(zip(urls, pool.map(lambda u: download(session, u), urls)))
            meta = {k: rec.get(k) for k in ("item_id", "title", "price", "currency", "url", "search", "result")}
            m = process_card(cid, imgs, out_root, {"source": "ebay", **meta})
            done += 1
            log.info("%s photos=%d front=%s back=%s", cid, len(urls), m["has_front"], m["has_back"])

    log.info("organized=%d skipped_existing=%d", done, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
