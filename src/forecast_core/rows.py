"""
The feature rows a forecast is made from: the school calendars, the weekend inputs, the Thursday-Hydro flag and
the rows for days beyond the data. Moved unchanged from src/experiments/run_series29_school_calendar.py
(with_calendar), run_series26_weekend_features.py (the weekend inputs), run_series35_holdout.py (the calendar
switch, weekday names, sales known), src/forecast_log/v1.py (the Hydro flag) and src/forecast_log/predict.py
(served_rows, lags_known).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.features.build_features import (
    LONG_BREAK_TYPES, SCHOOL_BREAK_TYPES, SCHOOL_BREAKS, SCHOOL_BREAKS_V1, add_school_holiday_features,
)
from src.features.events import concert_calendar
from src.features.serving_features import (
    WEATHER_COLS, build_feature_frame, extend_daily_frame, fill_future_sales_seasonally, with_history_features,
)
from src.models.final_model import DATE_COL, TARGET_COL

# --------------------------------------------------------------------------- the school calendars (Series 29)

SCHOOL_COLUMNS = ["is_christmas_break", "is_summer_break", "is_easter_break", "is_school_holiday"]


def with_calendar(frame: pd.DataFrame, breaks: pd.DataFrame = SCHOOL_BREAKS, break_types: tuple[str, ...] = SCHOOL_BREAK_TYPES) -> pd.DataFrame:
    """The model frame with its four school columns recomputed from `breaks` (only the given break types count);
    everything else untouched."""
    out = add_school_holiday_features(frame[["date"]].copy(), breaks_df=breaks, council=CONFIG.project.council, break_types=break_types)
    new = frame.copy()
    for col in SCHOOL_COLUMNS:
        new[col] = out[col].to_numpy()
    return new


def v1_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with its four school columns recomputed from the ORIGINAL school table (what the deployed model was trained on)."""
    return with_calendar(frame, SCHOOL_BREAKS_V1, LONG_BREAK_TYPES)


def build_frame(daily: pd.DataFrame, calendar: str) -> pd.DataFrame:
    frame = build_feature_frame(daily)
    return v1_calendar(frame) if calendar == "v1" else frame


def weekday_names(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).dt.day_name().to_numpy()


# --------------------------------------------------------------------------- the weekend inputs (Series 26)

FAMILY_A = ["lag_14_sales", "lag_21_sales", "same_wd_mean4", "same_wd_max4"]
FAMILY_B = ["wk_payday", "wk_dry", "wk_temp", "wk_bh_near"]
WEEKEND_VARIANTS = {
    "A_all": {"extra": FAMILY_A, "label": "A: same-weekday memory (4)"},
    "A_summary": {"extra": ["same_wd_mean4", "same_wd_max4"], "label": "A: mean and max of last 4 (2)"},
    "A_lags": {"extra": ["lag_14_sales", "lag_21_sales"], "label": "A: sales 2 and 3 weeks ago (2)"},
    "B": {"extra": FAMILY_B, "label": "B: weekend interactions (4)"},
    "AB": {"extra": FAMILY_A + FAMILY_B, "label": "A + B (8)"},
}
# The inputs each weekend piece adds, as fixed in the holdout's pre-registration (Series 27 / 31).
EXTRA = {"Friday": WEEKEND_VARIANTS["A_all"]["extra"], "Saturday": WEEKEND_VARIANTS["A_lags"]["extra"], "Sunday": WEEKEND_VARIANTS["B"]["extra"]}


def add_weekend_features(frame: pd.DataFrame, daily_sales: pd.Series) -> pd.DataFrame:
    """Append the family A and B columns to `frame` (one row per day). `daily_sales` is the sales series
    indexed by calendar date over ALL days of the data (including the warm-up rows the model frame drops), so
    a lag is the sales on that calendar date, never a neighbouring row, and uses only days before the row."""
    s = daily_sales.asfreq("D")
    lags = pd.DataFrame({k: s.shift(k) for k in (7, 14, 21, 28)})
    extra = pd.DataFrame({"lag_14_sales": lags[14], "lag_21_sales": lags[21],
                          "same_wd_mean4": lags.mean(axis=1), "same_wd_max4": lags.max(axis=1)})
    out = frame.copy()
    for col in FAMILY_A:
        out[col] = extra[col].reindex(pd.DatetimeIndex(out[DATE_COL])).to_numpy()
    fri_sun = (out[DATE_COL].dt.dayofweek >= 4).astype(float)
    near_bank_holiday = (np.minimum(out["days_to_bank_holiday"], out["days_since_bank_holiday"]) <= 3).astype(float)
    out["wk_payday"] = fri_sun * out["is_payday_window_pm3"]
    out["wk_dry"] = fri_sun * out["is_dry_day"]
    out["wk_temp"] = fri_sun * out["max_temp"]
    out["wk_bh_near"] = fri_sun * near_bank_holiday
    return out


def with_weekend(rows: pd.DataFrame, sales_by_date: pd.Series) -> pd.DataFrame:
    return add_weekend_features(rows, sales_by_date)


def sales_known_series(full: pd.DataFrame, last_known_day: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = full[full[DATE_COL] <= last_known_day].set_index(DATE_COL)[TARGET_COL]
    return s.reindex(pd.date_range(s.index.min(), end, freq="D"))


# --------------------------------------------------------------------------- the Thursday-Hydro flag (forecast log v1)

THURSDAY = 3


def add_hydro_thu(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `hydro_thu` (a Hydro show on a Thursday) from the curated concert list. The list must cover the dates asked for."""
    cal = concert_calendar(str(frame[DATE_COL].min().date()), str(frame[DATE_COL].max().date()))
    flag = (cal["hydro_show"] * (cal[DATE_COL].dt.dayofweek == THURSDAY)).rename("hydro_thu")
    out = frame.drop(columns=["hydro_thu"], errors="ignore").merge(pd.concat([cal[DATE_COL], flag], axis=1), on=DATE_COL, how="left")
    out["hydro_thu"] = out["hydro_thu"].fillna(0).astype(int)
    return out


# --------------------------------------------------------------------------- rows for days beyond the data (forecast log)


def served_rows(full: pd.DataFrame, last_known_day: pd.Timestamp, target_dates, calendar: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for `target_dates`: real history up to `last_known_day`, the days after it filled with weekday-and-season medians (sales, the manager's number, weather), then the
    live weather laid over those days where Open-Meteo has a figure. Nothing dated after `last_known_day` is read from `full`."""
    history = full[full[DATE_COL] <= last_known_day]
    daily = extend_daily_frame(history, max(pd.Timestamp(d) for d in target_dates).date())
    if len(weather):
        live = weather.set_index(DATE_COL)
        later = (daily[DATE_COL] > last_known_day) & daily[DATE_COL].isin(live.index)
        for col in WEATHER_COLS:
            daily.loc[later, col] = live.loc[daily.loc[later, DATE_COL], col].to_numpy()
    frame = build_frame(daily, calendar)
    position = {d: i for i, d in enumerate(frame[DATE_COL])}
    sales = fill_future_sales_seasonally(frame, len(history), history)
    return with_history_features(frame, sales, [position[pd.Timestamp(d)] for d in target_dates])


def lags_known(sales_known: pd.Series, dates: pd.Series, lags: tuple[int, ...]) -> np.ndarray:
    """True for each date whose sales exactly `lag` days earlier are all known, for every lag given."""
    ok = pd.Series(True, index=sales_known.index)
    for k in lags:
        ok &= sales_known.shift(k).notna()
    return ok.reindex(pd.DatetimeIndex(dates)).fillna(False).to_numpy()
