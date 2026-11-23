"""
The holiday rule: a forecast that does not come from a model, for the days the venue is closed or nearly so.

  25 December and 1 January  the venue is shut (zero sales in 2024-25 and 2025-26): the rule forecasts 0.
  26 December (Boxing Day)   open but slow (4,321 in 2024, 2,919 in 2025): the rule forecasts the average of the earlier Boxing Days' sales, taken from the sales history the run reads, so it updates by itself
                             each year (3,620 from the two years so far). It does not know the weekday: Boxing Day 2026 is a Saturday and no Saturday Boxing Day is in the data, so the level is a judgement
                             that the first one will test.

Why a rule and not a better model: the models have seen two Christmases and cannot learn from two examples that these days are different (Series 76-78 in docs/EXPERIMENT_LOG.md: the closed-day heat fix
helped a little, a stronger Boxing Day penalty changed nothing, because a tree only sees the order of a column's values). The log issues the rule beside the models, as rows of the model name
`holiday_rule` on those days only, so the models' own rows stay exactly what the models said, and the scores (Q5) put the rule next to each of them on those days. Off by default (setting `holiday_rule`).
"""

from __future__ import annotations

import hashlib

import pandas as pd

RULE_MODEL = "holiday_rule"
RULE_VERSION = "r1"
CLOSED_DAYS = ((12, 25), (1, 1))  # (month, day): the forecast is 0
BOXING_DAY = (12, 26)  # the forecast is the mean of the earlier Boxing Days' sales
DESCRIPTION = f"{RULE_VERSION}|closed={CLOSED_DAYS}|boxing_day={BOXING_DAY}:mean_of_earlier_years"


def fingerprint() -> str:
    """A short hash of the rule's definition, written to the log's model_fingerprint column."""
    return hashlib.sha256(DESCRIPTION.encode()).hexdigest()[:32]


def is_closed(day: pd.Timestamp) -> bool:
    return (day.month, day.day) in CLOSED_DAYS


def is_boxing_day(day: pd.Timestamp) -> bool:
    return (day.month, day.day) == BOXING_DAY


def boxing_day_level(history: pd.DataFrame, before: pd.Timestamp, date_col: str = "date", sales_col: str = "total_sales") -> float | None:
    """The mean sales of the Boxing Days (26 December) that fall before `before` and have a known, positive sales figure; None if there is none."""
    d = pd.to_datetime(history[date_col])
    sales = pd.to_numeric(history[sales_col], errors="coerce")
    keep = (d.dt.month == BOXING_DAY[0]) & (d.dt.day == BOXING_DAY[1]) & (d < before) & (sales > 0)
    return float(sales[keep].mean()) if keep.any() else None


def rule_forecast(day: pd.Timestamp, level: float | None) -> float | None:
    """0 on a closed day, `level` on Boxing Day, None on every other day (and on Boxing Day when there is no earlier one to average)."""
    if is_closed(day):
        return 0.0
    if is_boxing_day(day):
        return level
    return None
