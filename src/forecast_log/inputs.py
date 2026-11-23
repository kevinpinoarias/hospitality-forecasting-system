"""
The log's hand-supplied input: the Hydro programme (the curated list and how far it reaches).

The manager's forecast is not an input: it reaches us only in the weekly export, after the day, so every forecast is made without it (the stand-in takes its place, see predict.py) and the
manager's number is scored afterwards as a benchmark (score_forecasts.py).

Hydro programme: `config/events/concert_days.csv` and `config/events/concert_coverage.json` ({"hydro_list_through": "YYYY-MM-DD"}). A Thursday after the date the list reaches is UNKNOWN, not "no
show": the hydro arm's forecast for it is skipped rather than made with a zero flag.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.features.events import EVENTS_DIR

COVERAGE_FILE = EVENTS_DIR / "concert_coverage.json"
THURSDAY = 3


def hydro_list_through(path: Path = COVERAGE_FILE) -> date:
    return date.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["hydro_list_through"])


def unknown_hydro_thursdays(targets: list[date], through: date) -> set[date]:
    """Thursdays among `targets` that lie beyond the date the Hydro list reaches: their Hydro status is unknown."""
    return {d for d in targets if d.weekday() == THURSDAY and d > through}
