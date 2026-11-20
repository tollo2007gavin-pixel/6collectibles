"""Read the approved thresholds used by the pixel inspection and the learning workflow."""
import json
import math
from pathlib import Path

from edge_classifier import DEFAULTS

ACTIVE_PATH = Path(__file__).with_name("pixel_active_settings.json")


def checked_thresholds(values):
    if not isinstance(values, dict) or set(values) != set(DEFAULTS):
        raise ValueError("Pixel settings must include exactly the classifier's threshold names.")
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid pixel setting for {name}.")
    return dict(values)


def active_thresholds(path=ACTIVE_PATH):
    if not path.exists():
        return dict(DEFAULTS)
    record = json.loads(path.read_text(encoding="utf-8"))
    return checked_thresholds(record["core_thresholds"])
