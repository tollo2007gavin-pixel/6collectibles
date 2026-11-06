"""Configurable coordinator for the unchanged yellow and reviewed Luna pipelines."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yellow_border"))
import ebay_line_fit_test as yellow
import luna_whitening as luna
import providers
import border_routing
from yellow_batch_engine.measurement_screening import screen_axis


@dataclass
class Settings:
    pipeline: str = "both"
    detection_method: str = "validated_whitening"
    replay_folder: str = ""
    border_type: str = "yellow"
    output_mode: str = "testing"
    source: str = "ebay"
    check_only: bool = False
    max_listings: int = 45
    max_listing_age_hours: float | None = None
    max_listings_per_card: int = 3
    max_candidates_per_card: int = 100
    card_ids: list[str] = field(default_factory=list)
    cards_csv: str = "yellow_border/data/cards.csv"
    gx_cards_csv: str = "gx_border/data/cards.csv"
    shuffle_cards: bool = True
    random_seed: int | None = 53
    skip_previous_listings: bool = False
    centering_max_side_pct: float = 53.0
    allow_angular_variance: bool = True
    target_passes: int = 0
    max_whitening_cards: int = 15
    prepare_api_only: bool = False
    max_back_photos: int = 8
    references_json: str = "luna_references.json"
    centering_workers: int = 2
    output_root: str = "outputs/card_runs"
    run_label: str = ""
    resume_run: str | None = None
    local_input_dir: str = "uploads"
    env_file: str = ".env"
    open_report: bool = False


def path_at_root(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def validate(s):
    for name, choices in {"pipeline": {"both", "centering", "api", "detection"}, "detection_method": {"shared_baseline", "validated_whitening"},
                          "border_type": {"yellow", "gx", "all"}, "source": {"ebay", "local"}, "output_mode": {"testing", "production"}}.items():
        if getattr(s, name) not in choices:
            raise ValueError(f"{name} must be one of {sorted(choices)}")
    if s.replay_folder and s.pipeline != "detection":
        raise ValueError("Saved batch replay requires detection mode")
    for name in ("max_listings", "max_listings_per_card", "max_candidates_per_card", "centering_workers", "max_back_photos"):
        if type(getattr(s, name)) is not int or getattr(s, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("target_passes", "max_whitening_cards"):
        if type(getattr(s, name)) is not int or getattr(s, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if s.centering_workers > 8 or s.max_back_photos > 24 or s.max_candidates_per_card > 10000:
        raise ValueError("Limits: workers <= 8, back photos <= 24, candidates per card <= 10000")
    if s.target_passes > s.max_listings:
        raise ValueError("target_passes cannot exceed max_listings")
    if type(s.centering_max_side_pct) not in (int, float) or not math.isfinite(s.centering_max_side_pct) or not 50 <= s.centering_max_side_pct <= 100:
        raise ValueError("centering_max_side_pct must be between 50 and 100")
    if s.max_listing_age_hours is not None and (type(s.max_listing_age_hours) not in (int, float) or not math.isfinite(s.max_listing_age_hours) or s.max_listing_age_hours <= 0):
        raise ValueError("max_listing_age_hours must be positive, or None for no age limit")
    if not isinstance(s.card_ids, list) or not all(isinstance(v, (str, int)) for v in s.card_ids):
        raise ValueError('card_ids must be a list, e.g. ["22", "23"]')
    for name in ("check_only", "shuffle_cards", "skip_previous_listings", "allow_angular_variance", "prepare_api_only", "open_report"):
        if type(getattr(s, name)) is not bool:
            raise ValueError(f"{name} must be True or False")
    if s.random_seed is not None and type(s.random_seed) is not int:
        raise ValueError("random_seed must be an integer or None")
    for name in ("replay_folder", "cards_csv", "gx_cards_csv", "references_json", "output_root", "local_input_dir", "env_file", "run_label"):
        if not isinstance(getattr(s, name), str):
            raise ValueError(f"{name} must be text")
    if not re.fullmatch(r"[A-Za-z0-9_-]{0,60}", s.run_label):
        raise ValueError("run_label may contain up to 60 letters, numbers, underscores or hyphens")
    if s.resume_run is not None and not isinstance(s.resume_run, str):
        raise ValueError("resume_run must be a folder path or None")


def selected_cards(s):
    cards = []
    seen = set()
    for family, filename in (("yellow", s.cards_csv), ("gx", s.gx_cards_csv)):
        if s.border_type not in {family, "all"}:
            continue
        for row in yellow.load_cards(path_at_root(filename)):
            entry = border_routing.approved(row["id"], family)
            if row["id"] in seen or not border_routing.listing_eligible(row["search_query"], entry):
                raise ValueError("Duplicate or mismatched approved card query: " + row["id"])
            seen.add(row["id"])
            cards.append(dict(row, border_type=family))
    if s.card_ids:
        wanted = {str(v) for v in s.card_ids}
        missing = wanted - {c["id"] for c in cards}
        if missing:
            raise ValueError("Card IDs not active in the selected approved border lists: " + ", ".join(sorted(missing)))
        cards = [c for c in cards if c["id"] in wanted]
    if s.shuffle_cards:
        random.Random(s.random_seed).shuffle(cards)
    return cards


def load_bank(filename):
    path = path_at_root(filename)
    bank = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(bank) != 16 or len({r["card_id"] for r in bank}) != 16:
        raise ValueError("Luna requires the 16 uniquely identified human-reviewed reference cards")
    for row in bank:
        p = Path(row["path"])
        row["path"] = str(p if p.is_absolute() else path.parent / p)
    luna.references(bank, bank[0]["card_id"])
    luna.references(bank, bank[1]["card_id"])
    return bank


def centering_gate(combined, s):
    if combined.get("status") == "Fail":
        return {"accepted": False, "status": "fail", "reason": "gx_hard_failure" if combined.get("border_type") == "gx" else "exterior_background_gate", "axes": {}}
    axes = combined.get("axes", {})
    if any(not axes.get(a, {}).get("resolved") for a in ("lr", "tb")):
        return {"accepted": False, "status": "undetermined", "reason": "centering_unresolved", "axes": {}}
    allowances = combined.get("measurement_policy", {}).get("axis_allowance_pp", {})
    screened = {a: screen_axis(axes[a].get("percentage"), allowances.get(a) if s.allow_angular_variance else 0.,
                              max_side_pct=s.centering_max_side_pct) for a in ("lr", "tb")}
    accepted = all(v["accepted"] for v in screened.values())
    unknown = any(v["status"].startswith("withheld") for v in screened.values())
    return {"accepted": accepted, "status": "pass" if accepted else "undetermined" if unknown else "reject",
            "reason": "centering_pass" if accepted else "centering_uncertainty_unavailable" if unknown else "outside_centering_gate",
            "axes": screened}


def front_worker(content, metadata, audit):
    import cv2
    cv2.setNumThreads(1)  # One per PROCESS; shared legacy gates must not run in threads.
    started = time.perf_counter()
    border_routing.validate_route(metadata)
    if metadata["border_type"] == "gx":
        from gx_border.adapter import process_front
    else:
        process_front = yellow.process_front
    row = process_front(content, metadata, Path(audit))
    row["timings"] = {"centering_seconds": time.perf_counter() - started}
    return row


def prepare_back(client, detail, row, audit, s):
    import cv2
    import numpy as np
    from PIL import Image
    from organize_listings import process_card
    from card_utils import edge_corner_sheet, corner_closeup_sheet
    images, seen = [], set()
    # Search thumbnails are not guaranteed to be fronts. Keep the primary too,
    # without displacing the existing additional-photo budget.
    urls = [extra.get("imageUrl") for extra in (detail.get("additionalImages") or [])]
    primary = (detail.get("image") or {}).get("imageUrl") or row.get("image_url")
    extras = []
    for url in urls:
        if url and url != primary and url not in extras:
            extras.append(url)
    for url in extras[:s.max_back_photos] + [primary]:
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            raw = client.download(url)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > yellow.MAX_IMAGE_PIXELS:
                    raise ValueError("Back image exceeds pixel limit")
            pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            images.append((url, pixels, raw))
        except Exception:
            images.append((url, None, None))
    # Keep eBay card identity for reference exclusion; use a hash for local files.
    cid = row["item_id"].split("|")[1] if "|" in row["item_id"] else hashlib.sha256(row["item_id"].encode()).hexdigest()[:16]
    # Variations have separate folders even when sharing an eBay numeric item ID.
    folder_id = hashlib.sha256(row["item_id"].encode()).hexdigest()[:16]
    folder = audit / "backs" / folder_id
    manifest = process_card(folder_id, images, audit / "backs", {"item_id": row["item_id"]}, whitening_boundaries=s.pipeline != "detection" or s.detection_method == "validated_whitening")
    usable = [p for p in manifest["photos"] if p.get("side") == "back" and p.get("card_found")
              and not p.get("cut_off") and (s.pipeline == "detection" and s.detection_method == "shared_baseline" or p.get("boundary_review", {}).get("status") == "ready")]
    if not usable:
        originals = [str(folder / p["original_file"]) for p in manifest["photos"] if p.get("original_file")]
        if not originals:
            return {"ready": False, "reason": "no_readable_listing_photos", "manifest": str(folder / "manifest.json")}
        reviews = [p.get("boundary_review", {}) for p in manifest["photos"]]
        reason = ("back_photo_found_edges_unconfirmed" if any(r.get("design") for r in reviews)
                  else "back_corner_sheet_unavailable")
        return {"ready": False, "card_id": cid, "context": originals,
                "sheet": None, "tips": None, "input_mode": "sheet_preparation_failed",
                "reason": reason,
                "manifest": str(folder / "manifest.json")}
    photo = max(usable, key=lambda p: p.get("sharpness", 0))
    card = cv2.imread(str(folder / photo["file"]))
    sheet, tips = folder / "luna_sheet.jpg", folder / "luna_tips.png"
    if not cv2.imwrite(str(sheet), edge_corner_sheet(card, "back"), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not save back sheet")
    if not cv2.imwrite(str(tips), corner_closeup_sheet(card, "back")):
        raise OSError("Could not save corner tips")
    return {"ready": True, "card_id": cid, "context": str(folder / photo["original_file"]),
            "sheet": str(sheet), "tips": str(tips), "input_mode": "baseline_crops" if s.pipeline == "detection" and s.detection_method == "shared_baseline" else "validated_crops", "manifest": str(folder / "manifest.json")}


def finish_listing(row, detail, client, audit, s, bank, state, save, *, preparer=prepare_back, classifier=luna.classify):
    border_routing.validate_route(row)
    row.update(status="undetermined", accepted=False, reason="not_finished")
    row["whitening"] = {"decision": "not_run"}
    if s.pipeline not in {"api", "detection"}:
        gate = centering_gate(row["combined"], s)
        row["centering_gate"] = gate
        if not gate["accepted"]:
            row.update(status=gate["status"], reason=gate["reason"])
            return
        if s.pipeline == "centering":
            row.update(status="pass", accepted=True, reason="centering_pass")
            return
    if s.pipeline != "detection" and not s.prepare_api_only and state["whitening_attempts"] >= s.max_whitening_cards:
        row["reason"] = "whitening_limit_reached"
        return
    started = time.perf_counter()
    packet = preparer(client, detail, row, audit, s)
    row.setdefault("timings", {})["back_preparation_seconds"] = time.perf_counter() - started
    row["back_packet"] = packet
    if s.pipeline == "detection":
        row.update(reason="detection_sheet_ready" if packet["ready"] else packet["reason"],
                   detection={"method": s.detection_method, "code_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("card_utils.py", "whitening_geometry.py", "back_landmarks.py", "references/back_landmarks.npz")}, "sheet_ready": packet["ready"], "selected_original": packet.get("context") if packet["ready"] else None},
                   whitening={"decision": "not_run"})
        return
    if not packet["ready"]:
        row.update(reason=packet["reason"], whitening={"decision": "needs_review", "reason": packet["reason"]})
        return
    if s.prepare_api_only:
        row.update(reason="api_prepared_not_called", whitening={"decision": "not_run", "reason": "prepare_api_only"})
        return
    refs = luna.references(bank, packet["card_id"])
    state["whitening_attempts"] += 1
    row["reason"] = "api_in_flight"
    save()  # Reserve the attempt before the paid call; interruption must not repeat it.
    started = time.perf_counter()
    result = classifier(packet["context"], packet["sheet"], packet["tips"], refs)
    row["timings"]["whitening_seconds"] = time.perf_counter() - started
    row["whitening"] = result
    invalid = result.get("error") or any(v.get("invalid") for v in result.get("stages", []))
    clean = result.get("decision") == "clean" and not invalid
    row.update(accepted=clean, status="pass" if clean else "reject" if result.get("decision") == "not_clean" and not invalid else "undetermined",
               reason="clean_back" if clean else "whitening_api_error" if result.get("error") else "whitening_" + result.get("decision", "invalid"))
    if result.get("error"):
        raise RuntimeError("Whitening API failed: " + providers.redact_error(result["error"]))


class LocalClient:
    def download(self, filename):
        p = Path(filename)
        if p.stat().st_size > yellow.MAX_IMAGE_BYTES:
            raise ValueError("Local image exceeds size limit")
        return p.read_bytes()

    def close(self):
        pass


def local_candidates(s, seen, state):
    base = path_at_root(s.local_input_dir)
    if not base.is_dir():
        raise ValueError(f"Local input directory not found: {base}")
    folders = [p for p in sorted(base.iterdir()) if p.is_dir()]
    counts = Counter(str(r.get("card_id")) for r in state["records"])
    allowed = {r["id"]: r for r in selected_cards(s)}
    for folder in folders:
        itemid = "local:" + str(folder.resolve())
        if itemid in seen:
            continue
        metadata_file = folder / "metadata.json"
        meta = json.loads(metadata_file.read_text(encoding="utf-8-sig")) if metadata_file.exists() else {}
        card_id = str(meta.get("card_id", folder.name))
        entry = border_routing.approved(card_id)
        if s.border_type not in {entry["border_type"], "all"} or (s.card_ids and card_id not in {str(v) for v in s.card_ids}):
            continue
        if card_id not in allowed:
            raise ValueError(f"{folder}: metadata.card_id must belong to the selected active approved border list")
        family = allowed[card_id]["border_type"]
        if meta.get("border_type", family) != family:
            raise ValueError(f"{folder}: border_type conflicts with its approved card_id")
        if s.card_ids and card_id not in {str(v) for v in s.card_ids}:
            continue
        if counts[card_id] >= s.max_listings_per_card:
            continue
        photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        fronts = [p for p in photos if p.stem.lower() == "front"]
        if s.pipeline not in {"api", "detection"} and len(fronts) != 1:
            raise ValueError(f"{folder}: provide exactly one front.jpg/png/webp")
        front = fronts[0] if fronts else None
        seen.add(itemid)
        counts[card_id] += 1
        yield ({"item_id": itemid, "card_id": card_id, "border_type": family, "title": meta.get("title", folder.name), "url": meta.get("url", ""),
                "image_url": str(front) if front else "", "query": "local"},
               {"additionalImages": [{"imageUrl": str(p)} for p in photos if p != front]})


def ebay_candidates(s, client, cards, seen, state):
    counts = Counter(str(r.get("card_id")) for r in state["records"])
    for card in cards:
        entry = border_routing.validate_route(dict(card_id=card["id"], border_type=card.get("border_type")))
        if counts[card["id"]] >= s.max_listings_per_card:
            continue
        for offset in range(0, s.max_candidates_per_card, 200):
            limit = min(200, s.max_candidates_per_card - offset)
            page = client.search(card["search_query"], offset=offset, limit=limit,
                                 age_hours=s.max_listing_age_hours, card_id=card["id"] if card["border_type"] == "yellow" else None)
            for summary in page.get("itemSummaries", []):
                if counts[card["id"]] >= s.max_listings_per_card:
                    break
                itemid = summary.get("itemId")
                if not itemid or itemid in seen:
                    continue
                seen.add(itemid)
                try:
                    detail = client.details(itemid)
                    if not yellow.is_near_mint_ungraded(detail):
                        state["fetch_counts"]["not_near_mint_ungraded"] += 1
                        continue
                    if not border_routing.listing_eligible(detail.get("title", ""), entry):
                        state["fetch_counts"]["identity_not_approved"] = state["fetch_counts"].get("identity_not_approved", 0) + 1
                        continue
                    image = (detail.get("image") or {}).get("imageUrl") or (summary.get("image") or {}).get("imageUrl")
                    if not image:
                        state["fetch_counts"]["missing_front_image"] += 1
                        continue
                    counts[card["id"]] += 1
                    yield ({"item_id": itemid, "card_id": card["id"], "border_type": card["border_type"], "query": card["search_query"],
                            "title": detail.get("title", card["search_query"]), "url": detail.get("itemWebUrl", ""), "image_url": image}, detail)
                except Exception as error:
                    state["errors"].append({"item_id": itemid, "stage": "listing_details", "error": providers.redact_error(error)})
                    response = getattr(error, "response", None)
                    if response is not None and response.status_code in {401, 403, 429}:
                        raise
            if counts[card["id"]] >= s.max_listings_per_card or not page.get("next"):
                break


def report_row(row):
    axes = row.get("combined", {}).get("axes", {})
    lr, tb = (axes.get(a, {}).get("percentage") for a in ("lr", "tb"))
    whitening = row.get("whitening", {})
    stages = whitening.get("stages", [])
    evidence = stages[-1].get("evidence", "") if stages else whitening.get("reason", "")
    gate_axes = row.get("centering_gate", {}).get("axes", {})
    return {"title": row["title"], "url": row.get("url", ""), "item_id": row["item_id"], "border_type": row.get("border_type", "unknown"), "status": row["status"],
            "reason": row["reason"], "left_pct": lr, "right_pct": None if lr is None else 100 - lr,
            "top_pct": tb, "bottom_pct": None if tb is None else 100 - tb,
            "angular_allowance_pp": max((v.get("uncertainty_pp") or 0 for v in gate_axes.values()), default=0),
            "depends_on_allowance": any(v.get("status") == "accepted_due_to_uncertainty" for v in gate_axes.values()),
            "whitening": whitening.get("decision", "not_run"), "evidence": evidence,
            "whitening_input_mode": row.get("back_packet", {}).get("input_mode", ""),
            "inspection_mode": whitening.get("timing", {}).get("inspection_mode", ""),
            "estimated_api_cost_usd": whitening.get("cost_usd", 0),
            **row.get("timings", {})}


def write_csv(path, rows):
    fields = ["title", "url", "item_id", "border_type", "status", "reason", "left_pct", "right_pct", "top_pct", "bottom_pct",
              "angular_allowance_pp", "depends_on_allowance", "whitening", "evidence", "estimated_api_cost_usd",
              "download_seconds", "centering_seconds", "back_preparation_seconds", "whitening_seconds",
              "whitening_input_mode", "inspection_mode"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Listing titles are untrusted text, never spreadsheet formulas.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})


def safe_link(url):
    return html.escape(url, quote=True) if url.startswith(("https://", "http://")) else ""


def checkpoint(out, state, s):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    yellow.write_json(out / "_audit" / "run.json", state)
    accepted = [r for r in state["records"] if r.get("accepted")]
    shown = state["records"] if s.output_mode == "testing" else accepted
    summary = {"state": state["state"], "pipeline": s.pipeline, "output_mode": s.output_mode,
               "border_type": s.border_type, "border_counts": dict(Counter(r.get("border_type", "unknown") for r in state["records"])),
               "processed": len(state["records"]), "accepted": len(accepted),
               "outcomes": dict(Counter(r["status"] for r in state["records"])),
               "whitening_attempts": state["whitening_attempts"],
               "whitening_input_modes": dict(Counter(r.get("back_packet", {}).get("input_mode", "not_prepared") for r in state["records"])),
               "completed_api_responses": sum(len(r.get("whitening", {}).get("stages", [])) for r in state["records"]),
               "estimated_api_cost_usd": sum(r.get("whitening", {}).get("cost_usd", 0) for r in state["records"]),
               "fetch_counts": state["fetch_counts"], "errors": len(state["errors"])}
    yellow.write_json(out / "summary.json", summary)
    yellow.write_json(out / "accepted.json", [report_row(r) for r in accepted])
    write_csv(out / "accepted.csv", [report_row(r) for r in accepted])
    if s.output_mode == "testing":
        write_csv(out / "results.csv", [report_row(r) for r in shown])
    from card_review import write_review
    write_review(out, state, production=s.output_mode == "production", embed=state["state"] != "running")


def identity_settings(s):
    values = asdict(s)
    for key in ("resume_run", "output_root", "open_report", "max_listings", "max_listings_per_card", "max_candidates_per_card", "target_passes", "max_whitening_cards", "centering_workers", "skip_previous_listings"):
        values.pop(key)
    return values


def new_state(s):
    return {"version": 2, "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
            "approved_catalog_sha256": hashlib.sha256(border_routing.CATALOG_PATH.read_bytes()).hexdigest(),
            "settings": asdict(s), "records": [], "errors": [], "whitening_attempts": 0,
            "fetch_counts": {"not_near_mint_ungraded": 0, "missing_front_image": 0}}


def previous_ids(root):
    seen = set()
    for file in root.glob("*/_audit/run.json"):
        data = json.loads(file.read_text(encoding="utf-8-sig"))
        seen.update(r["item_id"] for r in data.get("records", []))
    return seen


def run(s, *, client_factory=None, processor=front_worker, preparer=prepare_back, classifier=luna.classify):
    validate(s)
    if s.pipeline == "detection":
        s = replace(s, output_mode="testing", target_passes=0, prepare_api_only=True)
    cards = selected_cards(s)
    bank = load_bank(s.references_json) if s.pipeline not in {"centering", "detection"} and not s.prepare_api_only else None
    root = path_at_root(s.output_root)
    if s.resume_run:
        out = path_at_root(s.resume_run)
        state = json.loads((out / "_audit/run.json").read_text(encoding="utf-8-sig"))
        if (state.get("version") != 2 or identity_settings(Settings(**state["settings"])) != identity_settings(s)
                or state.get("approved_catalog_sha256") != hashlib.sha256(border_routing.CATALOG_PATH.read_bytes()).hexdigest()):
            raise ValueError("Resume settings differ. Start a new run to change pipeline, gates, inputs or output mode.")
        for row in state["records"]:
            if row.get("reason") in {"api_in_flight", "not_finished"}:
                row.update(status="undetermined", accepted=False, reason="interrupted_review_required")
        state.update(state="running", settings=asdict(s))
    else:
        name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ("_" + s.run_label if s.run_label else "")
        out = root / name
        out.mkdir(parents=True, exist_ok=False)
        state = new_state(s)
    audit = out / "_audit"
    for folder in (audit, audit / "downloads", audit / "details", audit / "Combined"):
        folder.mkdir(parents=True, exist_ok=True)
    save = lambda: checkpoint(out, state, s)
    save()
    print(f"Results: {out}\nMode: {s.pipeline}; maximum listings: {s.max_listings}; paid card cap: {s.max_whitening_cards}", flush=True)
    seen = {r["item_id"] for r in state["records"]}
    if s.skip_previous_listings:
        seen |= previous_ids(root)
    client, pool, pending = None, None, deque()
    def target_reached():
        return s.target_passes > 0 and sum(r.get("accepted", False) for r in state["records"]) >= s.target_passes
    try:
        client = client_factory() if client_factory else LocalClient() if s.source == "local" else LocalClient() if s.replay_folder else yellow.EbayClient()
        candidates = iter(__import__("detection_replay").candidates(s.replay_folder) if s.replay_folder else local_candidates(s, seen, state) if s.source == "local" else ebay_candidates(s, client, cards, seen, state))
        capacity = s.centering_workers if s.pipeline not in {"api", "detection"} else 1
        if s.pipeline not in {"api", "detection"} and capacity > 1:
            pool = ProcessPoolExecutor(max_workers=capacity)
        exhausted = False
        while not target_reached() and len(state["records"]) < s.max_listings:
            while not exhausted and len(pending) < capacity and len(state["records"]) + len(pending) < s.max_listings:
                try:
                    meta, detail = next(candidates)
                except StopIteration:
                    exhausted = True
                    break
                started = time.perf_counter()
                try:
                    if s.pipeline in {"api", "detection"}:
                        future, row = None, dict(meta)
                    else:
                        content = client.download(meta["image_url"])
                        seconds = time.perf_counter() - started
                        future = pool.submit(processor, content, meta, str(audit)) if pool else None
                        row = None if future else processor(content, meta, str(audit))
                    pending.append((meta, detail, future, row, 0. if s.pipeline in {"api", "detection"} else seconds, None))
                except Exception as error:
                    pending.append((meta, detail, None, None, time.perf_counter() - started, providers.redact_error(error)))
            if not pending:
                break
            meta, detail, future, row, seconds, error = pending.popleft()
            if future:
                try:
                    row = future.result()
                except Exception as exc:
                    error = providers.redact_error(exc)
            if error:
                row = dict(meta, status="undetermined", accepted=False, reason="front_processing_error", error=error)
                state["records"].append(row)
            else:
                row.setdefault("timings", {})["download_seconds"] = seconds
                row.update(status="undetermined", accepted=False, reason="not_finished")
                state["records"].append(row)
                try:
                    finish_listing(row, detail, client, audit, s, bank, state, save, preparer=preparer, classifier=classifier)
                except Exception as exc:
                    row.update(status="undetermined", accepted=False, error=providers.redact_error(exc))
                    if row["reason"] != "whitening_api_error":
                        row["reason"] = "processing_error"
                    save()
                    if row["reason"] == "whitening_api_error":
                        raise
            save()
            print(f'{len(state["records"]):03d}/{s.max_listings} {row["item_id"]}: {row["status"]} ({row["reason"]})', flush=True)
        state["state"] = "target_reached" if target_reached() else "listing_limit_reached" if len(state["records"]) >= s.max_listings else "search_exhausted"
    except KeyboardInterrupt:
        state["state"] = "stopped_by_user"
    except Exception as error:
        state["state"] = "stopped_error"
        state["errors"].append({"stage": "run", "error": providers.redact_error(error)})
        raise
    finally:
        save()
        if pool:
            for _, _, future, _, _, _ in pending:
                if future:
                    future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        if client:
            client.close()
    print(f'Finished: {state["state"]}; {sum(r.get("accepted", False) for r in state["records"])} accepted. Open {out / "review.html"}', flush=True)
    if s.open_report and not os.getenv("GITHUB_ACTIONS"):
        webbrowser.open((out / "review.html").as_uri())
    return out


def apply_overrides(s, values):
    updates = {}
    for value in values:
        name, sep, raw = value.partition("=")
        if not sep or name not in asdict(s):
            raise ValueError(f"Unknown setting or missing '=': {value}")
        try:
            updates[name] = json.loads(raw)
        except json.JSONDecodeError:
            updates[name] = raw
    return replace(s, **updates)


def main(settings, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Run cards with the SETTINGS in run_cards.py. No automatic scheduled runs.")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Override any setting; use JSON for lists, numbers, true/false/null")
    parser.add_argument("--check", action="store_true", help="Validate settings, files and reference hashes only; no network or paid calls")
    args = parser.parse_args(argv)
    try:
        settings = apply_overrides(settings, args.set)
        # GitHub supplies a JSON object through env, never interpolated into shell code.
        if os.getenv("CARD_RUN_INPUTS"):
            inputs = json.loads(os.environ["CARD_RUN_INPUTS"])
            settings = apply_overrides(settings, [f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in inputs.items() if v != ""])
        validate(settings)
        selected_cards(settings)
        if settings.source == "local" and not path_at_root(settings.local_input_dir).is_dir():
            raise ValueError("local_input_dir does not exist")
        if settings.pipeline not in {"centering", "detection"} and not settings.prepare_api_only:
            load_bank(settings.references_json)
        from dotenv import load_dotenv
        load_dotenv(path_at_root(settings.env_file))
        needed = (["EBAY_CLIENT_ID", "EBAY_CLIENT_SECRET"] if settings.source == "ebay" and not settings.replay_folder else [])
        if settings.pipeline not in {"centering", "detection"} and not settings.prepare_api_only and settings.max_whitening_cards > 0:
            needed.append("OPENAI_API_KEY")
        missing = [key for key in needed if not os.getenv(key)]
        if args.check or settings.check_only:
            print(json.dumps({"settings_valid": True, "missing_credentials_for_live_run": missing, "settings": asdict(settings)}, indent=2))
            return 0
        if missing:
            raise ValueError("Set these keys in .env or GitHub Actions secrets: " + ", ".join(missing))
        run(settings)
        return 0
    except Exception as error:
        print("Stopped: " + providers.redact_error(error), file=sys.stderr)
        return 1
