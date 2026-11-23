"""
The triangle payday intensity (`intensity_sym`): one number per day for how much pay is arriving around it, from four pay rules.

Each rule has a payday every month: the last working day, the 25th, the 28th and the 15th; a date that falls on a weekend or a Scottish bank holiday is paid on the working day before it. Each rule adds a
triangle of heat around its payday, 1 - k / 4 for the nearest payday at k = 0 to 3 days either side and nothing from 4 days out, and the day's number is the sum of the four, each weighted by the share of
UK payroll teams that use that date (CIPP Payslip Statistics Survey 2023: 0.34, 0.30, 0.30, 0.18). It is computed from the date and the bank-holiday calendar alone, so a forecast for a future day can build it.

Experiment record: Series 70 (the variant `intensity_sym`), 71 (its amplification), 72 (the placebo test) in docs/EXPERIMENT_LOG.md. The experiments' own copy is `run_series61_payday_rules.days_around` and
`run_series71_amplify_intensity.triangle_intensity`; tests/test_payday_intensity.py checks that this module agrees with it on every day.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG

COLUMN = "intensity_sym"
RULES = ("last_working_day", 25, 28, 15)
SHARES = {"last_working_day": 0.34, 25: 0.30, 28: 0.30, 15: 0.18}
WIDTH = 4.0  # days: the triangle reaches 3 days either side of a payday and is 0 from 4 days out


def month_payday(period: pd.Period, rule, calendar) -> pd.Timestamp:
    """The payday of one calendar month: the date (the last day of the month for `last_working_day`), or the working day before it when it falls on a weekend or a bank holiday."""
    day = period.to_timestamp(how="end").normalize() if rule == "last_working_day" else pd.Timestamp(year=period.year, month=period.month, day=rule)
    while day.weekday() >= 5 or day in calendar:
        day -= pd.Timedelta(days=1)
    return day


def distance_to_nearest_payday(dates: pd.Series | pd.DatetimeIndex, rule, calendar) -> np.ndarray:
    """Whole days from each date to the nearest payday of one rule, before or after (0 on a payday)."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).normalize()
    periods = pd.period_range(d.min().to_period("M") - 2, d.max().to_period("M") + 2, freq="M")
    paydays = np.sort(np.array([month_payday(p, rule, calendar) for p in periods], dtype="datetime64[ns]"))
    values = d.to_numpy().astype("datetime64[ns]")
    latest = paydays[np.searchsorted(paydays, values, side="right") - 1]
    upcoming = paydays[np.searchsorted(paydays, values, side="left")]
    day = np.timedelta64(1, "D")
    return np.minimum((values - latest) // day, (upcoming - values) // day).astype(int)


def payday_intensity(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """`intensity_sym` for each date, in the order given (0 to 1.12; 0.79 is the highest seen in the data)."""
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    total = np.zeros(len(dates))
    for rule in RULES:
        k = distance_to_nearest_payday(dates, rule, calendar)
        total += SHARES[rule] * np.clip(1.0 - k / WIDTH, 0.0, None)
    return total


def add_payday_intensity(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """A copy of `frame` with the `intensity_sym` column appended (last, after every existing column)."""
    out = frame.copy()
    out[COLUMN] = payday_intensity(out[date_col])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The closed-day version (forecast-log version v1.12): the same intensity, but 0 on the days the venue is closed and a penalty on Boxing Day. Series 77 in docs/EXPERIMENT_LOG.md is the evidence: the column
# `xmas_only` there is this one (the fix alone, December's pay dates left as they are); Series 78 found nothing better on top of it (the early pay dates, the 27th, the 15th removed, a larger penalty).
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

COLUMN_FIXED = "intensity_fixed"
CLOSED_DAYS = ((12, 25), (1, 1))  # (month, day): zero sales in the data (25 December 2024 and 2025, 1 January 2025 and 2026): no pay can arrive on a shut day, so the intensity is 0
BOXING_DAY = (12, 26)
BOXING_DAY_PENALTY = -0.5  # an untuned placeholder: below every other value of the column (0 to about 1.1); a tree model only sees the order of the values (Series 78: -2.0 gave identical forecasts)


def payday_intensity_fixed(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """`intensity_sym` with 25 December and 1 January set to 0 and 26 December set to the penalty; every other day keeps its value."""
    values = payday_intensity(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))
    month, day = np.asarray(idx.month), np.asarray(idx.day)
    for m, d in CLOSED_DAYS:
        values[(month == m) & (day == d)] = 0.0
    values[(month == BOXING_DAY[0]) & (day == BOXING_DAY[1])] = BOXING_DAY_PENALTY
    return values


def add_payday_intensity_fixed(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """A copy of `frame` with the `intensity_fixed` column appended (last, after every existing column)."""
    out = frame.copy()
    out[COLUMN_FIXED] = payday_intensity_fixed(out[date_col])
    return out
