"""
The stand-in for the manager's number, and the hybrid one (S6) that versions v1.131 and v1.141 are served with.

The manager's number for a day reaches the forecast log only after the day (Series 42-43), so every version is given a stand-in in its place. Until v1.131 the stand-in was S0: the median of the manager's past
numbers on the same weekday within ten days of the same day of the year (src/features/serving_features.py, `seasonal_medians`), closed days included. S0 takes 25 December's zeros as the number for the
Wednesday or Thursday before it, which can make the models forecast a few pounds for Christmas Eve.

The hybrid (S6, Series 89, 9 November 2026):
  * 24 December to 2 January: the manager's number on the SAME DATE one year earlier (365 days), when there is one;
  * every other day, and a day with no year-ago number: S1, the closed-day-aware median: the same recipe as S0 with the closed days (manager number 0) left out of the history, and 0 on 25 December and 1 January.
Only the number fed to the models changes; the fitted pieces are the parent's own files. The history is the days known when the run is issued, so nothing is read from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.serving_features import seasonal_medians
from src.models.final_model import DATE_COL

MANAGER_COL = "forecast_sales"
CHRISTMAS_WEEK = ((12, 24), (1, 2))  # (month, day) first and last, inclusive, wrapping over the new year
CLOSED_DAYS = ((12, 25), (1, 1))
YEAR_AGO_DAYS = 365
NAME = "S6"


def _in_window(dates: pd.Series, first: tuple[int, int], last: tuple[int, int]) -> np.ndarray:
    md = (dates.dt.month * 100 + dates.dt.day).to_numpy()
    a, b = first[0] * 100 + first[1], last[0] * 100 + last[1]
    return (md >= a) | (md <= b) if a > b else (md >= a) & (md <= b)


def hybrid_standin(full: pd.DataFrame, last_known_day: pd.Timestamp, targets) -> np.ndarray:
    """The hybrid stand-in (S6) for each target date, from the manager's numbers of the days up to `last_known_day` only."""
    dates = pd.to_datetime(pd.Series(list(targets))).dt.normalize()
    history = full[pd.to_datetime(full[DATE_COL]) <= last_known_day]
    history = history[history[MANAGER_COL].notna()]
    open_days = history[history[MANAGER_COL] > 0]  # closed days (manager number 0) left out
    s1 = seasonal_medians(open_days, dates, MANAGER_COL)
    closed = np.zeros(len(dates), dtype=bool)
    for month, day in CLOSED_DAYS:
        closed |= ((dates.dt.month == month) & (dates.dt.day == day)).to_numpy()
    s1 = np.where(closed, 0.0, s1)
    manager = history.assign(**{DATE_COL: pd.to_datetime(history[DATE_COL]).dt.normalize()}).set_index(DATE_COL)[MANAGER_COL].astype(float)
    year_ago = manager.reindex(dates - pd.Timedelta(days=YEAR_AGO_DAYS)).to_numpy()
    use = _in_window(dates, *CHRISTMAS_WEEK) & ~np.isnan(year_ago)
    return np.where(use, year_ago, s1)
