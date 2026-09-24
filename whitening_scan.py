#!/usr/bin/env python3
"""
Scan organized card batches (from organize_listings.py) for edge/corner whitening.

Per card:
  1. Build a zoomed corner+edge sheet from the card's photos
     (--photos best = sharpest back only; --photos all = every usable photo).
  2. Free OpenCV whitening score on the back's blue border.
  3. Optional gate: obvious clean / obvious damage decided without an API call.
  4. Vision call(s): all of the card's sheets go in ONE request. --repeats N runs
     that request N times to measure how consistent the model is.
  5. Results -> cards/<id>/whitening.json, whitening_results.jsonl and, with
     repeats > 1, consistency_report.json.

Examples:
  python whitening_scan.py --dry-run                                  # sheets + OpenCV only, $0
  python whitening_scan.py --provider openai --photos all --min-photos 2 --repeats 3 --max-cards 10
  python whitening_scan.py --provider gemini --gate                   # production-style run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from card_utils import edge_corner_sheet, whitening_score
from providers import CALLERS, DEFAULT_MODEL, cost_usd

log = logging.getLogger("whitening")
MAX_IMAGES_PER_CALL = 6   # caps cost on listings with many photos


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        ✏️  EDIT THE PROMPT HERE  ✏️                        ║
# ║  SYSTEM_PROMPT = instructions to the model.                              ║
# ║  USER_PROMPT   = sent with each card; {placeholders} are filled in.      ║
# ║  If you change the JSON keys, update REQUIRED_KEYS below to match.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """\
You are a trading card condition inspector. You only assess WHITENING: white or
light wear showing through the printed border at a card's edges and corners.
Ignore centering, surface scratches, print lines and glare reflections.

You will receive one or more images. Every image is a sheet of zoomed crops
from the SAME card, taken from different listing photos. In each sheet, row 1
shows the four corners (TL, TR, BL, BR); the strips below are the TOP, BOTTOM,
LEFT and RIGHT edges. Red captions above each crop say front or back.

Combine the evidence across all sheets. Wear that is clearly visible in any
sharp photo counts. If photos disagree because of glare, blur or a sleeve,
trust the clearest one.

Rate each location 0-3 (use the worst side, front or back):
  0 = none visible
  1 = light (a few specks, visible only on close inspection)
  2 = moderate (clear chipping or a continuous light line)
  3 = heavy (large chips, frayed or peeling edges)

If no photo lets you judge a location, use null and lower your confidence.
Do not guess.

Reply with JSON only, no prose, exactly these keys:
{
  "corners": {"TL": int|null, "TR": int|null, "BL": int|null, "BR": int|null},
  "edges":   {"TOP": int|null, "BOTTOM": int|null, "LEFT": int|null, "RIGHT": int|null},
  "overall": int,            // worst credible severity, 0-3
  "confidence": float,       // 0-1
  "notes": str               // one short sentence
}"""

USER_PROMPT = "Card: {title}\nPhotos attached: {n}\nAssess whitening."

REQUIRED_KEYS = {"corners", "edges", "overall", "confidence"}

# ═══════════════════════════ end of prompt section ══════════════════════════

LOCATIONS = [("corners", k) for k in ("TL", "TR", "BL", "BR")] + \
            [("edges", k) for k in ("TOP", "BOTTOM", "LEFT", "RIGHT")]


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if REQUIRED_KEYS <= d.keys() else None


def usable_photos(manifest: dict, sides: list[str], mode: str) -> list[dict]:
    photos = [p for p in manifest["photos"] if p.get("file") and p.get("side") in sides]
    photos.sort(key=lambda p: (sides.index(p["side"]), -p["sharpness"]))
    if mode == "best":
        return [next(p for p in photos if p["side"] == s) for s in sides if any(p["side"] == s for p in photos)]
    return photos[:MAX_IMAGES_PER_CALL]


def build_sheets(card_dir: Path, manifest: dict, sides: list[str], mode: str):
    """One JPEG sheet per photo. Returns (jpegs, cv_score of sharpest back)."""
    jpegs, cv_score = [], None
    for i, p in enumerate(usable_photos(manifest, sides, mode), 1):
        img = cv2.imread(str(card_dir / p["file"]))
        if img is None:
            continue
        if p["side"] == "back" and cv_score is None:
            cv_score = whitening_score(img)
        sheet = edge_corner_sheet(img, f'{p["side"]} #{i}')
        cv2.imwrite(str(card_dir / f"sheet_{i}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            jpegs.append(buf.tobytes())
    return jpegs, cv_score


def consensus(verdicts: list[dict]) -> dict:
    """Median (rounded toward worse) per location across repeats, plus how much runs disagreed."""
    out = {"corners": {}, "edges": {}}
    spreads = {}
    for grp, k in LOCATIONS:
        vals = [v[grp].get(k) for v in verdicts if isinstance(v.get(grp), dict)]
        vals = [x for x in vals if isinstance(x, (int, float))]
        out[grp][k] = statistics.median_high(vals) if vals else None
        spreads[f"{grp}.{k}"] = (max(vals) - min(vals)) if vals else None
    overalls = [v["overall"] for v in verdicts if isinstance(v.get("overall"), (int, float))]
    out["overall"] = statistics.median_high(overalls) if overalls else None
    out["confidence"] = round(statistics.mean(v.get("confidence", 0) or 0 for v in verdicts), 2)
    out["runs_overall"] = overalls
    out["overall_agree"] = len(set(overalls)) == 1
    out["spread"] = spreads
    return out


def consistency_report(recs: list[dict]) -> dict:
    multi = [r for r in recs if len(r.get("runs", [])) > 1 and r.get("verdict")]
    if not multi:
        return {}
    loc_disagree = {f"{g}.{k}": 0 for g, k in LOCATIONS}
    for r in multi:
        for loc, s in r["verdict"]["spread"].items():
            if s:
                loc_disagree[loc] += 1
    return {
        "cards": len(multi),
        "overall_full_agreement_pct": round(100 * sum(r["verdict"]["overall_agree"] for r in multi) / len(multi), 1),
        "mean_overall_spread": round(statistics.mean(
            max(r["verdict"]["runs_overall"]) - min(r["verdict"]["runs_overall"]) for r in multi), 2),
        "location_disagreement_pct": {k: round(100 * v / len(multi), 1) for k, v in loc_disagree.items()},
        "least_consistent_cards": [
            {"card_id": r["card_id"], "runs_overall": r["verdict"]["runs_overall"], "url": r.get("url")}
            for r in sorted(multi, key=lambda r: -(max(r["verdict"]["runs_overall"]) - min(r["verdict"]["runs_overall"])))[:5]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", default="cards")
    ap.add_argument("--provider", choices=list(CALLERS), default="openai")
    ap.add_argument("--model", help="defaults to the provider's cheapest vision model")
    ap.add_argument("--sides", choices=["back", "both"], default="back")
    ap.add_argument("--photos", choices=["best", "all"], default="best",
                    help="best = one photo per side (cheapest); all = every usable photo in one call")
    ap.add_argument("--min-photos", type=int, default=1, help="only scan cards with at least this many usable photos")
    ap.add_argument("--repeats", type=int, default=1, help="calls per card, to measure consistency")
    ap.add_argument("--gate", action="store_true", help="skip the API when the OpenCV score is decisive")
    ap.add_argument("--gate-clean", type=float, default=0.003)
    ap.add_argument("--gate-heavy", type=float, default=0.08)
    ap.add_argument("--max-cards", type=int, default=0, help="0 = no limit")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="sheets + OpenCV scores only; no API calls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = args.model or DEFAULT_MODEL[args.provider]
    call = CALLERS[args.provider]
    sides = ["back"] if args.sides == "back" else ["back", "front"]
    root = Path(args.cards)
    stats = {"scanned": 0, "api_calls": 0, "gated": 0, "too_few_photos": 0, "bad_json": 0, "errors": 0, "skipped": 0}
    spend, all_recs = 0.0, []

    with open(root / "whitening_results.jsonl", "a") as results:
        for card_dir in sorted(p for p in root.iterdir() if (p / "manifest.json").exists()):
            if args.max_cards and stats["scanned"] >= args.max_cards:
                break
            if (card_dir / "whitening.json").exists() and not args.rescan:
                stats["skipped"] += 1
                continue
            manifest = json.loads((card_dir / "manifest.json").read_text())
            if len(usable_photos(manifest, sides, "all")) < args.min_photos:
                stats["too_few_photos"] += 1
                continue
            stats["scanned"] += 1

            jpegs, cv_score = build_sheets(card_dir, manifest, sides, args.photos)
            rec = {"card_id": manifest["card_id"], "title": manifest.get("title"), "url": manifest.get("url"),
                   "images_sent": len(jpegs), "cv_score": cv_score, "ts": int(time.time())}

            if not jpegs:
                rec["status"] = "no_usable_photos"
            elif args.dry_run:
                rec["status"] = "dry_run"
            elif args.gate and cv_score and cv_score["max"] <= args.gate_clean:
                rec.update(status="gated", verdict={"overall": 0, "source": "opencv"})
                stats["gated"] += 1
            elif args.gate and cv_score and cv_score["max"] >= args.gate_heavy:
                rec.update(status="gated", verdict={"overall": 3, "source": "opencv"})
                stats["gated"] += 1
            else:
                runs, card_cost = [], 0.0
                prompt = USER_PROMPT.format(title=manifest.get("title") or "unknown", n=len(jpegs))
                for _ in range(args.repeats):
                    try:
                        text, tin, tout = call(model, SYSTEM_PROMPT, prompt, jpegs)
                        stats["api_calls"] += 1
                        card_cost += cost_usd(model, tin, tout) or 0
                        v = parse_json(text)
                        if v:
                            runs.append(v)
                        else:
                            stats["bad_json"] += 1
                            rec.setdefault("bad_raw", []).append(text[:300])
                    except Exception as e:  # network / auth / quota: record and keep going
                        stats["errors"] += 1
                        rec.setdefault("errors", []).append(repr(e)[:300])
                spend += card_cost
                rec.update(provider=args.provider, model=model, runs=runs, cost_usd=round(card_cost, 6),
                           verdict=consensus(runs) if runs else None, status="ok" if runs else "failed")

            if rec["status"] not in ("dry_run", "failed"):
                (card_dir / "whitening.json").write_text(json.dumps(rec, indent=2))
            results.write(json.dumps(rec) + "\n")
            all_recs.append(rec)
            v = rec.get("verdict") or {}
            log.info("%-14s %-8s imgs=%d cv_max=%s overall=%s runs=%s", card_dir.name, rec["status"],
                     rec["images_sent"], cv_score and cv_score["max"], v.get("overall"), v.get("runs_overall"))

    report = consistency_report(all_recs)
    if report:
        (root / "consistency_report.json").write_text(json.dumps(report, indent=2))
        log.info("consistency: %s%% of cards got the same overall score on every run (mean spread %s)",
                 report["overall_full_agreement_pct"], report["mean_overall_spread"])
    log.info("done %s spend=$%.4f", stats, spend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
