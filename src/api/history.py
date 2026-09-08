"""
In-memory cache of the historical model-ready dataset, used for two purposes:

1. Fallback lookups when a caller doesn't supply `forecast_sales`, or when
   live rain data isn't available (see `weekday_seasonal_lookup`).
2. Reporting how far a requested date sits beyond the training data, so
   callers can judge how much to trust a far-future prediction.

The cache is loaded once at API startup and reloaded on a fixed interval
(see src/api/main.py) rather than re-read from disk on every request, since
the underlying file only changes when the pipeline is re-run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "model_features.csv"

DATE_COL = "date"
DAYS_IN_YEAR = 365.25


class HistoryCache:
    """Holds the historical feature dataset and its last refresh time."""

    def __init__(self, path: Path = MODEL_FEATURES_PATH) -> None:
        self.path = path
        self.df: pd.DataFrame = pd.DataFrame()
        self.last_refreshed: dt.datetime | None = None

    def refresh(self) -> None:
        df = pd.read_csv(self.path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
        self.df = df
        self.last_refreshed = dt.datetime.now(dt.timezone.utc)

    @property
    def max_date(self) -> dt.date:
        return self.df[DATE_COL].max().date()


def _circular_day_of_year_distance(day_of_year: pd.Series, target_day_of_year: int) -> pd.Series:
    """Distance in days between two days-of-year, wrapping around year end."""
    diff = (day_of_year - target_day_of_year).abs()
    return pd.concat([diff, DAYS_IN_YEAR - diff], axis=1).min(axis=1)


def weekday_seasonal_lookup(
    df: pd.DataFrame,
    target_date: dt.date,
    column: str,
    window_days: int = 10,
) -> float:
    """
    Median historical value of `column` for rows matching the target date's
    weekday, within `window_days` of its day-of-year (wrapping across the
    year boundary). Falls back to progressively broader matches if the
    initial window has too few historical rows to be meaningful.

    Median (not mean) is used deliberately: a single unusual historical day
    (a closure, a one-off event) shouldn't swing the estimate as much as it
    would in a mean of only a handful of points.
    """
    target_dow = target_date.weekday()
    target_doy = target_date.timetuple().tm_yday

    dow_match = df["day_of_week"] == target_dow
    doy_distance = _circular_day_of_year_distance(df["day_of_year"], target_doy)

    for window in (window_days, window_days * 2, window_days * 4):
        subset = df[dow_match & (doy_distance <= window)]
        if len(subset) >= 3:
            return float(subset[column].median())

    weekday_only = df[dow_match]
    if len(weekday_only) > 0:
        return float(weekday_only[column].median())

    return float(df[column].median())


def days_beyond_training_data(df: pd.DataFrame, target_date: dt.date) -> int:
    max_date = df[DATE_COL].max().date()
    delta = (target_date - max_date).days
    return max(0, delta)


def equivalent_weekday_last_year(target_date: dt.date) -> dt.date:
    """
    The date last year with the same weekday AND the same position within
    the month (e.g. "the 2nd Saturday of September") - not a naive 365-day
    offset, which almost always lands on a different weekday (a normal
    year is 52 weeks + 1 day). Demand at a hospitality venue is driven far
    more by day-of-week than by the exact calendar date, so this is the
    meaningful comparison, not "exactly 365 days ago".

    Falls back to the last matching weekday in the month if the same
    occurrence (e.g. a 5th Saturday) doesn't exist last year.
    """
    weekday = target_date.weekday()
    occurrence = (target_date.day - 1) // 7 + 1

    prev_year = target_date.year - 1
    first_of_month = dt.date(prev_year, target_date.month, 1)
    days_until_weekday = (weekday - first_of_month.weekday()) % 7
    first_occurrence = first_of_month + dt.timedelta(days=days_until_weekday)
    result = first_occurrence + dt.timedelta(weeks=occurrence - 1)

    if result.month != target_date.month:
        result -= dt.timedelta(weeks=1)

    return result


def actual_sales_on(df: pd.DataFrame, target_date: dt.date) -> float | None:
    """
    Real realised sales for an exact past date, if it exists in the
    historical data. Returns None if that date isn't in the dataset
    (e.g. it's a future date, or falls in one of the small number of
    genuine gaps) - never estimated or interpolated, only a real figure
    or nothing.
    """
    match = df[df[DATE_COL].dt.date == target_date]
    if match.empty:
        return None
    return float(match.iloc[0]["total_sales"])
