"""
The log's hand-supplied inputs: the manager's forecasts (a folder of CSV files) and the Hydro programme (the curated list and how far it reaches).

Manager files: any `*.csv` in the folder with a `date` column (YYYY-MM-DD or DD/MM/YYYY) and a `forecast` column (pounds), one row per day. The file's modified time is when the number
arrived; if a date appears in several files the most recently modified wins. Numbers that arrive after a run's deadline are not used for that run (the run uses what exists when it runs, and a
late run is labelled late).

Hydro programme: `config/events/concert_days.csv` and `config/events/concert_coverage.json` ({"hydro_list_through": "YYYY-MM-DD"}). A Thursday after the date the list reaches is UNKNOWN, not "no
show": the hydro arm's forecast for it is skipped rather than made with a zero flag.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.features.events import EVENTS_DIR

COVERAGE_FILE = EVENTS_DIR / "concert_coverage.json"
THURSDAY = 3


def parse_dates(values: pd.Series) -> pd.Series:
    """YYYY-MM-DD read strictly as year-month-day (pandas' day-first switch would read 2026-11-09 as 9 September); anything else, e.g. 09/11/2026, read day first."""
    text = values.astype(str).str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    rest = iso.isna()
    out = iso.copy()
    if rest.any():  # only the non-ISO values, so the format guessed from an ISO first value does not turn the rest into NaT
        out[rest] = pd.to_datetime(text[rest], dayfirst=True, errors="coerce")
    return out


def read_manager_forecasts(folder: Path | None) -> pd.DataFrame:
    """Columns date, forecast, arrived_at (timezone-aware local time); empty if the folder is missing or holds nothing readable."""
    empty = pd.DataFrame(columns=["date", "forecast", "arrived_at", "file"])
    if folder is None or not Path(folder).is_dir():
        return empty
    frames = []
    for path in sorted(Path(folder).glob("*.csv")):
        try:
            raw = pd.read_csv(path)
        except Exception:  # an unreadable file must not stop the run; it is simply not used
            continue
        cols = {c.strip().lower(): c for c in raw.columns}
        if "date" not in cols or "forecast" not in cols:
            continue
        frame = pd.DataFrame({"date": parse_dates(raw[cols["date"]]), "forecast": pd.to_numeric(raw[cols["forecast"]], errors="coerce")})
        frame = frame.dropna()
        frame["arrived_at"] = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        frame["file"] = path.name
        frames.append(frame)
    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("arrived_at", kind="stable").drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def manager_for(forecasts: pd.DataFrame, targets: list[date], not_after: datetime | None = None) -> pd.DataFrame:
    """The manager's numbers for `targets` that had arrived by `not_after`."""
    if forecasts.empty:
        return forecasts
    sel = forecasts[forecasts["date"].dt.date.isin(targets)]
    if not_after is not None:
        sel = sel[sel["arrived_at"] <= not_after]
    return sel.reset_index(drop=True)


def hydro_list_through(path: Path = COVERAGE_FILE) -> date:
    return date.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["hydro_list_through"])


def unknown_hydro_thursdays(targets: list[date], through: date) -> set[date]:
    """Thursdays among `targets` that lie beyond the date the Hydro list reaches: their Hydro status is unknown."""
    return {d for d in targets if d.weekday() == THURSDAY and d > through}
