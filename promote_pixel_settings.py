"""Apply settings from an eligible comparison made by the improvement workflow."""
import argparse
import json
from pathlib import Path

from pixel_active_settings import ACTIVE_PATH, active_thresholds, checked_thresholds
from pixel_training import CONFIG_SCHEMA, EXPOSURE_SCHEMA, code_fingerprint, digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def promote(artifact, run_id, root=ACTIVE_PATH.parent):
    if not run_id.isascii() or not run_id.isdigit():
        raise ValueError("Enter the numeric improvement run ID.")
    reports = list(artifact.rglob("report.json"))
    if len(reports) != 1:
        raise ValueError("Expected one comparison report. Select a completed improvement run.")
    folder = reports[0].parent
    report = read(reports[0])
    if report.get("mode") != "improve" or report.get("decision") != "recommend_update":
        raise ValueError("This comparison did not recommend a settings update.")

    config = read(folder / "frozen-config.json")
    fingerprint = digest({k: v for k, v in config.items() if k != "config_sha256"})
    if (config.get("schema") != CONFIG_SCHEMA or fingerprint != config.get("config_sha256")
            or fingerprint != report.get("config_sha256")):
        raise ValueError("The saved settings do not match the comparison.")
    if config.get("code_fingerprint") != code_fingerprint():
        raise ValueError("The detector or scoring code has changed since this comparison.")
    exposure = read(root / "review_labels" / f"validation-exposure-{run_id}.json")
    if (exposure.get("schema") != EXPOSURE_SCHEMA or exposure.get("config_sha256") != fingerprint
            or not exposure.get("exposed_aliases") or exposure != read(folder / "validation-exposure.json")):
        raise ValueError("The recorded validation cards do not match this comparison.")

    selected = checked_thresholds(read(folder / "recommended-settings.json"))
    if selected != config.get("core_thresholds"):
        raise ValueError("The proposed settings differ from the settings used for validation.")
    path = root / ACTIVE_PATH.name
    if path.exists() and read(path).get("comparison_run_id") == run_id and active_thresholds(path) == selected:
        print("These settings are already active.")
        return
    previous = active_thresholds(path)
    if previous != report.get("active_thresholds") or selected == previous:
        raise ValueError("The live settings have changed or already match this proposal. Use a new comparison.")
    active = dict(core_thresholds=selected, comparison_run_id=run_id, config_sha256=fingerprint)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(active, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()
    try:
        promote(args.artifact, args.run_id)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Could not apply pixel settings. {exc}\n")
