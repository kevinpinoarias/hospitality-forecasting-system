"""
Fail the build if the freshly rebuilt model is not as good as expected.

Reads the rebuilt model's backtest error from
reports/results/final_model_metrics.csv and compares it with the figure in
config/quality_gate.json. A rebuild is never bit-identical (the weather
service revises old days, and CatBoost differs slightly between machines), so
the check allows a tolerance - last measured drift was 0.005%, the limit is
1%. Too good is also a failure: a sudden jump in accuracy usually means
something leaked, so a person should look before the new figure is accepted.

If a change is meant to move the error, update "expected" in
config/quality_gate.json in the same pull request.
Run:  python scripts/quality_gate.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "quality_gate.json"
METRICS = ROOT / "reports" / "results" / "final_model_metrics.csv"


def main() -> int:
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))
    with METRICS.open(newline="", encoding="utf-8") as f:
        rows = {row["model"]: row for row in csv.DictReader(f)}
    row = rows[gate["model"]]

    actual = float(row[gate["metric"]])
    expected = float(gate["expected"])
    drift_pct = (actual - expected) / expected * 100
    days = int(float(row["days"]))

    print(f"{gate['metric']}: rebuilt {actual:.2f}, expected {expected:.2f} "
          f"({drift_pct:+.3f}%, allowed +/-{gate['tolerance_pct']}%); {days} days scored")

    problems = []
    if abs(drift_pct) > gate["tolerance_pct"]:
        direction = "worse" if drift_pct > 0 else "suspiciously better"
        problems.append(f"error is {direction} than expected by more than {gate['tolerance_pct']}%")
    if days != gate["expected_days"]:
        problems.append(f"scored {days} days, expected {gate['expected_days']}")

    for problem in problems:
        print(f"QUALITY GATE FAILED: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
