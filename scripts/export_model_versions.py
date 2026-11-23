"""
Writes config/model_versions.json: the model versions the API and the chat can offer besides the standard
(live) model, each with its one-line legend and the fingerprints of its registered files.

The fingerprints come from the forecast log's registry (models/versions.json in the private forecasting-log
repository); only hashes and labels are copied, no model data. The API checks every model file against them
before using it, and the build checks the downloaded files the same way, so a version is either exactly what
the log registered or it is not served at all.

Run after registering a new version, then add its legend to CHOICES below:

    python scripts/export_model_versions.py --registry <path-to>/forecasting-log/models/versions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "model_versions.json"

UNDER_TEST = "Under test in the weekly forecast log since {since}; formal results after 26 to 52 weeks."

# id -> (registered version whose files it uses, which arm, legend)
CHOICES = {
    "v1-post": ("v1", "post", "The weekend package: separate models for Friday, Saturday and Sunday on top of the main model, and a Saturday and Sunday correction from recent errors (Series 23-27)."),
    "v1-hydro": ("v1", "hydro", "The weekend package plus a flag for Thursdays with a show at the OVO Hydro (Series 36-41)."),
    "v1.11": ("v1.11", "hydro", "The Hydro model plus a payday intensity, how close the day is to payday, in the Saturday models (Series 70-73)."),
    "v1.12": ("v1.12", "hydro", "v1.11 with Christmas Day and New Year's Day treated as closed days in the payday intensity (Series 76-78)."),
    "v1.13": ("v1.13", "hydro", "v1.12 plus specialist models for 26 December to 8 January, when the usual recent-sales inputs mislead (Series 80-89)."),
    "v1.14": ("v1.14", "hydro", "v1.13 plus last year's sales on the same weekday, Monday to Saturday (Series 92-94)."),
    "v1.131": ("v1.131", "hydro", "v1.13 with a better stand-in for the manager's own forecast around the holidays (Series 88-89)."),
    "v1.141": ("v1.141", "hydro", "v1.14 with the same better stand-in for the manager's own forecast (Series 88-89)."),
}
STANDARD = {
    "id": "standard",
    "label": "Standard model (live)",
    "legend": "The live model: CatBoost averaged over 25 seeds with 38 inputs, chosen after 18 series of backtests.",
    "status": "The deployed model.",
}


def build(registry: dict) -> dict:
    by_version = {v["version"]: v for v in registry["versions"]}
    choices = [STANDARD]
    for choice_id, (version, arm, legend) in CHOICES.items():
        entry = by_version[version]
        since = entry["fitted_at"][:10]
        choices.append({
            "id": choice_id,
            "label": choice_id,
            "legend": legend,
            "status": UNDER_TEST.format(since=since),
            "version": version,
            "arm": arm,
            "standin": entry.get("standin", "S0"),
            "registered": since,
            "pieces": {name: spec["fingerprint"] for name, spec in sorted(entry["pieces"].items())},
            "correction_history_sha256": entry["correction_history_sha256"],
        })
    return {"source": "forecasting-log models/versions.json", "choices": choices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    missing = sorted({v for v, _, _ in CHOICES.values()} - {v["version"] for v in registry["versions"]})
    if missing:
        sys.exit(f"not registered: {missing}")
    OUT.write_text(json.dumps(build(registry), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
