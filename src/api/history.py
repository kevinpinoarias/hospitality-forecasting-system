"""
In-memory cache of the historical engineered-feature dataset, used for:

1. The real inputs behind a forecast - every feature for dates inside the
   data, and the recent sales and weather history a date beyond it is
   built from (see src/features/serving_features.py).
2. Fallback lookups when a caller doesn't supply `forecast_sales`, or when
   live weather isn't available (see `seasonal_medians` in
   src/features/serving_features.py).
3. Reporting how far a requested date sits beyond the training data, so
   callers can judge how much to trust a far-future prediction.

The cache is loaded once at API startup and reloaded on a fixed interval
(see src/api/main.py) rather than re-read from disk on every request, since
the underlying file only changes when the pipeline is re-run.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.preprocessing.build_daily_sales import KNOWN_CLOSURE_MONTH_DAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINEERED_FEATURES_PATH = CONFIG.path("engineered_features")

DATE_COL = "date"


class HistoryCache:
    """Holds the historical feature dataset and its last refresh time."""

    def __init__(self, path: Path = ENGINEERED_FEATURES_PATH) -> None:
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


def comparison_sales_on(df: pd.DataFrame, comparison_date: dt.date) -> float | None:
    """
    Sales on a date offered as a comparison (the same day last week or last
    year): the real figure, or None when that day was a known closure day.
    The venue is shut on Christmas Day and New Year's Day, so their 0 is not
    a comparison - for 24 and 31 December and about 7 January the equivalent
    weekday last year is one of them, and for 1 and 8 January the same day
    last week is, and the assistant would otherwise quote a 0 as if it were a
    real day's sales. None is what the assistant already treats as "no
    comparison to mention".
    """
    if (comparison_date.month, comparison_date.day) in KNOWN_CLOSURE_MONTH_DAYS:
        return None
    return actual_sales_on(df, comparison_date)


def actual_sales_on(df: pd.DataFrame, target_date: dt.date) -> float | None:
    """
    Real realised sales for an exact past date, if it exists in the
    historical data. Returns None if that date isn't in the dataset, or its
    sales figure is missing (e.g. it's a future date, or one of the small
    number of genuine gaps in the export) - never estimated or
    interpolated, only a real figure or nothing.
    """
    match = df[df[DATE_COL].dt.date == target_date]
    if match.empty:
        return None
    value = float(match.iloc[0]["total_sales"])
    return None if math.isnan(value) else value
