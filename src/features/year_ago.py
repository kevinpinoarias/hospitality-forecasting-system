"""
The year-ago sales input of forecast-log version v1.14: `sales_364`, the sales on the same weekday 364 days earlier.

364 days is exactly 52 weeks, so the day a year ago is the same weekday (Series 90 to 94 in docs/EXPERIMENT_LOG.md tried the same calendar date, a three-week average of that weekday, the centred seven-day average
around last year's date, the ratio of the day to its week and two inputs together: none beat the plain same weekday). The value is the ACTUAL sales of that day, known at forecast time for any day less than
364 days beyond the last known one; a day with no value a year back (the first 364 days of the data) is missing, which a tree model reads as its own value. Closed days keep their zero.
"""

from __future__ import annotations

import pandas as pd

COLUMN = "sales_364"
LAG_DAYS = 364


def year_ago_sales(dates, sales_by_date: pd.Series) -> pd.Series:
    """The sales 364 days before each of `dates`, from a series of actual sales indexed by date (gaps and unknown days stay missing). Nothing dated after a date is ever read for it."""
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))
    shifted = sales_by_date.astype(float).shift(LAG_DAYS, freq="D")  # the label of each sale moves 364 days later, so a date finds the sale 364 days before it
    return pd.Series(shifted.reindex(index).to_numpy(), index=range(len(index)), name=COLUMN)


def add_year_ago_sales(frame: pd.DataFrame, sales_by_date: pd.Series, date_col: str = "date") -> pd.DataFrame:
    """A copy of `frame` with `sales_364` appended last."""
    out = frame.copy()
    out[COLUMN] = year_ago_sales(out[date_col], sales_by_date).to_numpy()
    return out
