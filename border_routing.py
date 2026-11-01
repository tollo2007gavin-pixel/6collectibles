"""Fail-closed routing by explicit approved card design. Never try another fitter."""
import json
from pathlib import Path
import re

CATALOG_PATH = Path(__file__).with_name("approved_borders.json")


def catalog():
    rows = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["cards"]
    by_id = {}
    identities = set()
    for row in rows:
        identity = (row["name"].casefold(), row["number"].upper())
        if row["id"] in by_id or identity in identities or row["border_type"] not in {"yellow", "gx"}:
            raise ValueError("Approved border catalog has duplicate or invalid routing entries")
        if identity == ("zekrom", "SM138"):
            raise ValueError("Zekrom SM138 remains explicitly excluded")
        by_id[row["id"]] = row
        identities.add(identity)
    return by_id


def approved(card_id, family=None):
    entry = catalog().get(str(card_id))
    if not entry or (family is not None and entry["border_type"] != family):
        raise ValueError(f"Unapproved or mismatched border route: card_id={card_id}, border_type={family}")
    return entry


def numbers(text):
    return {prefix.upper() + str(int(number)) for prefix, number in
            re.findall(r"\b(SM|GG|TG)[\s_/-]*0*(\d+)\b", text, re.I)}


def expected_number(entry):
    return next(iter(numbers(entry["number"])))


def matches_identity(text, entry):
    text = str(text)
    if not re.search(r"\b" + re.escape(entry["name"]) + r"\b", text, re.I):
        return False
    if expected_number(entry) not in numbers(text):
        return False
    has_gx = bool(re.search(r"\bgx\b", text, re.I))
    return has_gx == (entry["border_type"] == "gx")


def listing_eligible(title, entry):
    """Seller identity evidence, not an OCR/visual authenticity classifier."""
    title = str(title)
    if re.search(r"\b(lot|bundle|proxy|custom|replica|jumbo|oversized|random|digital|metal)\b|\b(?:choose|pick)\b|\bset\s+of\b", title, re.I):
        return False
    if not matches_identity(title, entry):
        return False
    # GG70/TG30 after a slash are set denominators, not a second card.
    cleaned = re.sub(r"/\s*(?:GG\s*70|TG\s*30)\b", "", title, flags=re.I)
    if numbers(cleaned) != {expected_number(entry)}:
        return False
    matches = [r["id"] for r in catalog().values() if matches_identity(title, r)]
    return matches == [entry["id"]]


def validate_route(metadata):
    family = metadata.get("border_type")
    if family not in {"yellow", "gx"}:
        raise ValueError("An explicit approved border_type is required before processing")
    return approved(metadata.get("card_id"), family)
