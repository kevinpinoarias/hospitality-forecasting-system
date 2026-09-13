"""
Builds the final model's features for dates the historical data does not
reach yet.

The 15-feature XGBoost could forecast any date from the date itself plus
`forecast_sales` and a rain flag. The final model (see
src/models/train_final_model.py) also uses:

- recent sales and forecast-error history (`HISTORY_FEATURES`), which
  needs real sales up to the day before the date being forecast;
- raw weather and weather history (`max_temp`, `rain_mm`, `sun_hours`,
  `temp_anomaly_14d`, `warm_streak_len`), which needs the weather on the
  date and on the 14 days before it.

For a date beyond the end of the data, those inputs do not exist yet. This
module fills the gap the way the API does it, and the same code is used by
the Series 18 backtest (src/experiments/run_series18_serving_horizon.py),
so what was tested is what is served:

1. `extend_daily_frame` lays out one contiguous daily frame: real history,
   then every day up to the target date with `forecast_sales` and weather
   set to the historical weekday/season median. Callers overwrite the days
   they have real values for (a live weather forecast, a manager's
   forecast).
2. `build_feature_frame` computes every calendar, holiday and weather
   feature by running the training pipeline's own functions from
   src/features/build_features.py over that frame.
3. `fill_future_sales_seasonally` sets the future days' sales to their
   weekday/season median, and `with_history_features` recomputes the lag
   and rolling features from that series - real sales where they exist, a
   typical day after that.

`fill_future_sales_recursively` (feeding the model's own predictions
forward instead) is kept for Series 18, which compared the two: they were
within £6 MAE of each other over 180 days, and the seasonal fill was
slightly better at the short horizons most requests are for.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import numpy as np
import pandas as pd

from src.features.build_features import (
    SCHOOL_BREAKS,
    add_bank_holiday_features,
    add_payday_features,
    add_school_holiday_features,
    add_time_features,
    add_weather_features,
)

DATE_COL = "date"
TARGET_COL = "total_sales"
WEATHER_COLS = ["max_temp", "rain_mm", "sun_hours"]
DAYS_IN_YEAR = 365.25

# (source series, lag, rolling window or None) - the exact shift()/rolling()
# definitions in build_features.add_time_features. tests/test_serving_features.py
# checks history_features_at() against that function on the real data.
HISTORY_FEATURES: dict[str, tuple[str, int, int | None]] = {
    "lag_7_sales": ("sales", 7, None),
    "rolling_7_sales": ("sales", 1, 7),
    "rolling_14_sales": ("sales", 1, 14),
    "lag_1_fe": ("fe", 1, None),
    "lag_7_fe": ("fe", 7, None),
    "rolling_7_fe": ("fe", 1, 7),
}


# ---------------------------------------------------------------------
# Historical fallbacks
# ---------------------------------------------------------------------

def seasonal_medians(
    history: pd.DataFrame,
    dates,
    column: str,
    window_days: int = 10,
) -> np.ndarray:
    """
    For each date, the median historical value of `column` on the same
    weekday within `window_days` of the same day of the year (wrapping
    across the year boundary). Widens to 2x and 4x the window if fewer
    than 3 historical values match, then to the weekday alone, then to
    every row.

    Median (not mean) is used deliberately: a single unusual historical day
    (a closure, a one-off event) shouldn't swing the estimate as much as it
    would in a mean of only a handful of points.
    """
    hist_dates = pd.to_datetime(history[DATE_COL])
    values = history[column].to_numpy(dtype=float)
    valid = ~np.isnan(values)
    hist_dow = hist_dates.dt.dayofweek.to_numpy()
    hist_doy = hist_dates.dt.dayofyear.to_numpy()

    targets = pd.to_datetime(pd.Series(list(dates)))
    out = np.empty(len(targets))
    for i, (dow, doy) in enumerate(zip(targets.dt.dayofweek, targets.dt.dayofyear)):
        diff = np.abs(hist_doy - doy)
        distance = np.minimum(diff, DAYS_IN_YEAR - diff)
        same_weekday = (hist_dow == dow) & valid
        for window in (window_days, window_days * 2, window_days * 4):
            match = same_weekday & (distance <= window)
            if match.sum() >= 3:
                out[i] = np.median(values[match])
                break
        else:
            out[i] = np.median(values[same_weekday]) if same_weekday.any() else np.median(values[valid])
    return out


def weekday_seasonal_lookup(
    df: pd.DataFrame,
    target_date: dt.date,
    column: str,
    window_days: int = 10,
) -> float:
    """seasonal_medians() for a single date."""
    return float(seasonal_medians(df, [target_date], column, window_days)[0])


def typical_heavy_rain_mm(history: pd.DataFrame, heavy_rain_mm: float = 5.0) -> float:
    """Median rainfall on the historical days that counted as heavy rain -
    the rain amount used for the heavy-rain scenario when no real forecast
    exists for the date."""
    heavy = history.loc[history["rain_mm"] > heavy_rain_mm, "rain_mm"]
    return float(heavy.median())


def extend_daily_frame(history: pd.DataFrame, end_date: dt.date) -> pd.DataFrame:
    """
    The history's daily columns followed by one row per day up to and
    including `end_date`, with total_sales left empty and forecast_sales
    and weather set to each day's weekday/season median. Callers overwrite
    any day they have a real value for (a live weather forecast, a
    manager's forecast).
    """
    columns = [DATE_COL, TARGET_COL, "forecast_sales"] + WEATHER_COLS
    base = history[columns].copy()
    base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
    future_dates = pd.date_range(base[DATE_COL].max() + pd.Timedelta(days=1), pd.Timestamp(end_date), freq="D")
    future = pd.DataFrame({DATE_COL: future_dates, TARGET_COL: np.nan})
    for col in ["forecast_sales"] + WEATHER_COLS:
        future[col] = seasonal_medians(history, future_dates, col)
    return pd.concat([base, future], ignore_index=True)


def fill_future_sales_seasonally(frame: pd.DataFrame, first_future_idx: int, history: pd.DataFrame) -> np.ndarray:
    """
    The sales series with every day from `first_future_idx` onward set to
    its weekday/season median, so a date's lag and rolling features read
    real sales where they exist and a typical day after that. Series 18
    found this as accurate as feeding the model's own predictions forward,
    and slightly better at short horizons (see docs/EXPERIMENT_LOG.md).
    """
    sales = frame[TARGET_COL].to_numpy(dtype=float).copy()
    future_dates = frame[DATE_COL].iloc[first_future_idx:]
    sales[first_future_idx:] = seasonal_medians(history, future_dates, TARGET_COL)
    return sales


# ---------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------

def build_feature_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Every engineered feature for a contiguous daily frame with columns
    date, total_sales, forecast_sales, max_temp, rain_mm, sun_hours -
    computed by the training pipeline's own functions, in the same order as
    build_features.build_feature_dataset(), so a served feature can never
    be defined differently from the one the model was trained on.

    Future days can have total_sales = NaN; their history features are
    then filled by fill_future_sales_recursively().
    """
    frame = daily.copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL]).dt.normalize()
    frame = frame.sort_values(DATE_COL).reset_index(drop=True)
    if not (frame[DATE_COL].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("build_feature_frame needs one row per calendar day, with no gaps.")

    weather = frame[[DATE_COL] + WEATHER_COLS]
    out = add_time_features(frame.drop(columns=WEATHER_COLS))
    out = add_payday_features(out)
    out = add_bank_holiday_features(out)
    out = add_school_holiday_features(
        out, breaks_df=SCHOOL_BREAKS, council="Glasgow", break_types=("christmas", "summer", "easter"),
    )
    return add_weather_features(out, weather)


def history_features_at(sales: np.ndarray, forecast: np.ndarray, i: int) -> dict[str, float]:
    """The six history features for row i of a contiguous daily series,
    reading only rows before i."""
    series = {"sales": sales, "fe": sales - forecast}
    values = {}
    for name, (source, lag, window) in HISTORY_FEATURES.items():
        s = series[source]
        if window is None:
            values[name] = float(s[i - lag]) if i - lag >= 0 else np.nan
        else:
            start, end = i - lag - window + 1, i - lag + 1
            values[name] = float(np.mean(s[start:end])) if start >= 0 else np.nan
    return values


def fill_future_sales_recursively(
    frame: pd.DataFrame,
    first_future_idx: int,
    features: list[str],
    predict: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Walk forward from `first_future_idx`, predicting each day from features
    that read real sales where they exist and the model's own earlier
    predictions after that, and write each prediction back as that day's
    sales. Returns the filled sales series (real history unchanged).

    `frame` comes from build_feature_frame(); `predict` maps a 2-D feature
    array (columns in `features` order) to predicted sales.
    """
    sales = frame[TARGET_COL].to_numpy(dtype=float).copy()
    forecast = frame["forecast_sales"].to_numpy(dtype=float)
    base = frame[features].to_numpy(dtype=float)
    positions = {name: features.index(name) for name in HISTORY_FEATURES if name in features}

    for i in range(first_future_idx, len(frame)):
        row = base[i].copy()
        hist = history_features_at(sales, forecast, i)
        for name, pos in positions.items():
            row[pos] = hist[name]
        sales[i] = float(predict(row.reshape(1, -1))[0])
    return sales


def with_history_features(frame: pd.DataFrame, sales: np.ndarray, rows: list[int]) -> pd.DataFrame:
    """Copies of `rows` from `frame` with their history features recomputed
    from `sales` (for example a recursively filled series)."""
    forecast = frame["forecast_sales"].to_numpy(dtype=float)
    out = frame.iloc[rows].copy()
    for idx, i in zip(out.index, rows):
        for name, value in history_features_at(sales, forecast, i).items():
            out.at[idx, name] = value
    return out
