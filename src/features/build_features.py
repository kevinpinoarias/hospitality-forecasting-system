"""
Builds forecasting features from the processed daily sales dataset.

Purpose
-------
Creates time-based, holiday-based, payday-based, school-break, and weather
features for the hospitality forecasting project.

Outputs
-------
- data/features/engineered_features.csv
- data/features/model_features.csv

Notes
-----
- This public portfolio version uses anonymised local data.
- The file builds a broad engineered feature set, then exports a narrower
  model-ready feature subset based on the final selected features used in the
  project.
"""

from __future__ import annotations

from pathlib import Path
import os
import json
import time
import hashlib

import numpy as np
import pandas as pd
import holidays
import requests


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sales" / "daily_sales_totals_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
CACHE_DIR = PROJECT_ROOT / "data_cache" / "open_meteo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATE_COL = "date"
TARGET_COL = "total_sales"

# Final selected features used in the model
FINAL_MODEL_FEATURES = [
    "forecast_sales",
    "month_sin",
    "day_of_week",
    "day_of_year_cos",
    "day_of_year_sin",
    "day_of_year",
    "month_cos",
    "month",
    "day_of_week_sin",
    "is_bank_holiday",
    "days_to_bank_holiday",
    "days_since_payday",
    "is_payday_window_pm3",
    "is_long_weekend",
    "is_heavy_rain",
]

# Glasgow area coordinates used in the original project
LAT = 55.8612
LON = -4.2502
TIMEZONE = "Europe/London"


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def load_daily_sales(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load processed daily sales data."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Expected '{DATE_COL}' column in input data.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def add_cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Add cyclical sine/cosine encoding for a column."""
    angle = 2 * np.pi * (df[col] - 1) / period
    df[f"{col}_sin"] = np.sin(angle)
    df[f"{col}_cos"] = np.cos(angle)
    return df


# ---------------------------------------------------------------------
# Core time-based features
# ---------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Add core calendar and lag-based forecasting features."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col).reset_index(drop=True)

    # Calendar
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 4).astype(int)
    out["day_of_year"] = out[date_col].dt.dayofyear

    # Forecast error
    if {"total_sales", "forecast_sales"}.issubset(out.columns):
        out["forecast_error"] = out["total_sales"] - out["forecast_sales"]

    # Sales lag / rolling
    #
    # DEPRECATED - DO NOT USE AS MODEL FEATURES: lag_1_sales, lag_2_sales,
    # lag_3_sales, rolling_28_sales (and, below, lag_2_fe/lag_3_fe).
    # Tested in Series 4 (see docs/EXPERIMENT_LOG.md) added individually and
    # all together on top of the add_all_candidates winning configuration -
    # every one made MAE worse (£907.77 -> £914-927), most likely because
    # the short-term/weekly signal they'd add is already captured by
    # lag_7_sales / rolling_7_sales / rolling_14_sales / lag_1_fe /
    # rolling_7_fe, which are already in that configuration. Left computed
    # here for reference/reproducibility and in case a future feature set
    # that excludes those five ever needs re-testing them, but they must
    # not be added to any production or recommended feature list on
    # current evidence.
    if "total_sales" in out.columns:
        out["lag_1_sales"] = out["total_sales"].shift(1)
        out["lag_2_sales"] = out["total_sales"].shift(2)
        out["lag_3_sales"] = out["total_sales"].shift(3)
        out["lag_7_sales"] = out["total_sales"].shift(7)
        out["rolling_7_sales"] = out["total_sales"].shift(1).rolling(window=7).mean()
        out["rolling_14_sales"] = out["total_sales"].shift(1).rolling(window=14).mean()
        out["rolling_28_sales"] = out["total_sales"].shift(1).rolling(window=28).mean()  # DEPRECATED - see above

    # Forecast error lag / rolling. Note: forecast_error itself
    # (= total_sales - forecast_sales) is never a candidate model feature -
    # it directly encodes the target and would be leakage. Only its lagged
    # values (yesterday's, or earlier, forecast-vs-actual gap - already
    # known history by the time a new day is being predicted) are safe.
    if "forecast_error" in out.columns:
        out["lag_1_fe"] = out["forecast_error"].shift(1)
        out["lag_2_fe"] = out["forecast_error"].shift(2)  # DEPRECATED - see note above
        out["lag_3_fe"] = out["forecast_error"].shift(3)  # DEPRECATED - see note above
        out["lag_7_fe"] = out["forecast_error"].shift(7)
        out["rolling_7_fe"] = out["forecast_error"].shift(1).rolling(7).mean()

    # Cyclical encodings
    out = add_cyclical(out, "month", 12)
    out = add_cyclical(out, "day_of_week", 7)
    out = add_cyclical(out, "day_of_year", 365)

    return out


# ---------------------------------------------------------------------
# Payday features
# ---------------------------------------------------------------------

def add_payday_features(df: pd.DataFrame, date_col: str = DATE_COL, uk_subdiv: str = "SCT") -> pd.DataFrame:
    """Add payday proximity features based on last working day of month."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    d = out[date_col].dt.normalize()

    uk_holidays = holidays.UK(subdiv=uk_subdiv)

    def last_working_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
        month_end = (date + pd.offsets.MonthEnd(0)).normalize()
        while month_end.weekday() >= 5 or month_end in uk_holidays:
            month_end -= pd.Timedelta(days=1)
        return month_end

    out["payday"] = d.apply(last_working_day_of_month)
    out["is_payday"] = (d == out["payday"]).astype(int)

    out["prev_payday"] = (d + pd.offsets.MonthEnd(-1)).apply(last_working_day_of_month)

    next_month_anchor = d + pd.offsets.MonthEnd(1)
    out["next_payday"] = next_month_anchor.apply(last_working_day_of_month)

    out["next_payday"] = np.where(d <= out["payday"], out["payday"], out["next_payday"])
    out["next_payday"] = pd.to_datetime(out["next_payday"])

    out["days_to_payday"] = (out["next_payday"] - d).dt.days
    out["days_since_payday"] = (d - out["prev_payday"]).dt.days

    out["is_payday_window_pm3"] = (
        out["days_to_payday"].between(0, 3) |
        out["days_since_payday"].between(0, 3)
    ).astype(int)

    return out


# ---------------------------------------------------------------------
# Bank holiday features
# ---------------------------------------------------------------------

def add_bank_holiday_features(df: pd.DataFrame, date_col: str = DATE_COL, uk_subdiv: str = "SCT") -> pd.DataFrame:
    """Add bank holiday proximity and long-weekend features."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    d = out[date_col].dt.normalize()

    min_year = int(d.dt.year.min())
    max_year = int(d.dt.year.max())
    years = list(range(min_year - 1, max_year + 2))

    uk_holidays = holidays.UK(subdiv=uk_subdiv, years=years)
    holiday_dates = sorted(pd.to_datetime(list(uk_holidays.keys())).normalize().unique())
    holiday_idx = pd.DatetimeIndex(holiday_dates)

    out["is_bank_holiday"] = d.isin(holiday_idx).astype(int)

    hol_vals = holiday_idx.values
    pos_next = np.searchsorted(hol_vals, d.values, side="left")
    pos_prev = np.searchsorted(hol_vals, d.values, side="right") - 1

    next_hol = np.where(pos_next < len(hol_vals), hol_vals[pos_next], np.datetime64("NaT"))
    prev_hol = np.where(pos_prev >= 0, hol_vals[pos_prev], np.datetime64("NaT"))

    out["days_to_bank_holiday"] = (pd.to_datetime(next_hol) - d).dt.days
    out["days_since_bank_holiday"] = (d - pd.to_datetime(prev_hol)).dt.days

    long_weekend_dates = set()
    for h in holiday_idx:
        wd = h.weekday()
        if wd == 0:  # Monday
            window = pd.date_range(h - pd.Timedelta(days=3), h, freq="D")
            long_weekend_dates.update(window)
        elif wd == 4:  # Friday
            window = pd.date_range(h, h + pd.Timedelta(days=3), freq="D")
            long_weekend_dates.update(window)

    out["is_long_weekend"] = d.isin(pd.DatetimeIndex(sorted(long_weekend_dates))).astype(int)

    return out


# ---------------------------------------------------------------------
# School holiday features
# ---------------------------------------------------------------------

SCHOOL_BREAKS = pd.DataFrame([
    {"council": "Glasgow", "break_type": "christmas", "start": "2024-12-23", "end": "2025-01-03"},
    {"council": "Glasgow", "break_type": "easter",    "start": "2025-04-07", "end": "2025-04-18"},
    {"council": "Glasgow", "break_type": "summer",    "start": "2025-06-30", "end": "2025-08-15"},
    {"council": "Glasgow", "break_type": "christmas", "start": "2025-12-22", "end": "2026-01-05"},
    {"council": "Glasgow", "break_type": "easter",    "start": "2026-03-30", "end": "2026-04-10"},
    {"council": "Glasgow", "break_type": "summer",    "start": "2026-06-29", "end": "2026-08-14"},
])


def add_school_holiday_features(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    breaks_df: pd.DataFrame | None = None,
    council: str = "Glasgow",
    break_types: tuple[str, ...] = ("christmas", "summer", "easter"),
) -> pd.DataFrame:
    """Add school holiday flags from a council-specific break table."""
    if breaks_df is None:
        raise ValueError("breaks_df is required.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    d = out[date_col].dt.normalize()

    cal = breaks_df.copy()
    needed_cols = {"council", "break_type", "start", "end"}
    missing = needed_cols - set(cal.columns)
    if missing:
        raise ValueError(f"breaks_df is missing columns: {sorted(missing)}")

    cal["start"] = pd.to_datetime(cal["start"]).dt.normalize()
    cal["end"] = pd.to_datetime(cal["end"]).dt.normalize()
    cal = cal[cal["council"].astype(str).str.lower() == council.lower()].copy()

    if cal.empty:
        raise ValueError(f"No school break rows found for council='{council}'")

    any_break = np.zeros(len(out), dtype=bool)

    for bt in break_types:
        rows = cal[cal["break_type"].astype(str).str.lower() == bt.lower()]
        mask = np.zeros(len(out), dtype=bool)

        for _, r in rows.iterrows():
            mask |= (d >= r["start"]) & (d <= r["end"])

        out[f"is_{bt.lower()}_break"] = mask.astype(int)
        any_break |= mask

    out["is_school_holiday"] = any_break.astype(int)
    return out


# ---------------------------------------------------------------------
# Fixed-calendar Christmas period
# ---------------------------------------------------------------------

def add_christmas_period_feature(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """
    Binary flag for the fixed calendar month of December (1st-31st).

    Deliberately calendar-fixed (not council-specific school-break dates,
    see SCHOOL_BREAKS/is_christmas_break above) and deliberately whole-
    month rather than a narrower window around Christmas Day itself - the
    working assumption tested here is that once December starts, it is
    already "Christmas" from a hospitality-demand perspective (works
    parties, festive bookings), not just the days immediately around the
    25th.

    DEPRECATED - DO NOT USE AS A MODEL FEATURE: tested in Series 5 (see
    docs/EXPERIMENT_LOG.md), added to the add_all_candidates winning
    configuration - made MAE worse (£907.77 -> £919.33), most likely
    because is_christmas_break/is_school_holiday/is_long_weekend/the
    month cyclical encodings already capture most of this signal. Left
    computed here for reference; see docs/EXPERIMENT_LOG.md's discussion of a
    graded (non-binary) alternative, which has not yet been tried.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["is_christmas_period"] = (out[date_col].dt.month == 12).astype(int)  # DEPRECATED - see docstring
    return out


# Named December sub-periods, directly from the real 2-year pattern (see
# docs/EXPERIMENT_LOG.md's Series 5/6 discussion): a "party season" build-up,
# a pre-Christmas peak, a dip on Christmas Eve itself, a Boxing Day lull
# (25th is already covered by is_bank_holiday - no separate flag needed),
# a short recovery, a second and larger Hogmanay peak, then New Year's
# Eve itself easing off slightly from that peak. Fixed calendar rules,
# not derived from historical outcomes - unlike december_intensity_index,
# this needs no walk-forward machinery: CatBoost learns the appropriate
# level for each phase from whatever training data is available at fit
# time, exactly as it already does for month/day_of_week, so there is no
# risk of a phase's own future value leaking into itself.
DECEMBER_PHASES: dict[str, tuple[int, int]] = {
    "is_party_season": (1, 21),
    "is_pre_christmas_peak": (22, 23),
    "is_christmas_eve_dip": (24, 24),
    "is_boxing_day_lull": (26, 26),
    "is_post_christmas_recovery": (27, 28),
    "is_hogmanay_peak": (29, 30),
    "is_new_years_eve": (31, 31),
}


def add_december_phase_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """
    Binary flags for named December sub-periods, directly encoding the
    real empirical pattern found in the 2 years of history (see
    DECEMBER_PHASES above and docs/EXPERIMENT_LOG.md's Series 7) rather than
    asking a model to infer it from a single continuous index.

    Tested as a direct alternative to the deprecated is_christmas_period
    (Series 5) and december_intensity_index (Series 6): both prior
    attempts made add_all_candidates worse. The hypothesis here was that
    grouping days into a handful of named phases - each backed by more
    training rows than a single day-of-month ever could be with only two
    Decembers of data - would be more robust than either a single flat
    window or a 31-point empirical curve estimated from just two
    observations per day.

    DEPRECATED - DO NOT USE AS MODEL FEATURES: tested in Series 7 (see
    docs/EXPERIMENT_LOG.md), added to add_all_candidates individually and all
    together - every one made MAE worse (£907.77 -> £916-923), the same
    direction as Series 4/5/6. This is now the fourth independent
    December/Christmas encoding to fail; see Series 7's conclusions for
    an important open question this raises about whether CatBoost's
    fixed hyperparameters (tuned for a smaller feature space) are
    penalising any added feature, not specifically these ones. Left
    computed here for reference.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    month = out[date_col].dt.month
    day = out[date_col].dt.day

    for name, (start_day, end_day) in DECEMBER_PHASES.items():
        out[name] = ((month == 12) & (day >= start_day) & (day <= end_day)).astype(int)  # DEPRECATED - see docstring

    return out


def add_december_intensity_feature(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Empirically-derived, graded December demand signal - built to test
    whether getting the real (twin-peak) December shape right would
    succeed where the deprecated binary is_christmas_period did not.

    DEPRECATED - DO NOT USE AS A MODEL FEATURE: tested in Series 6 (see
    docs/EXPERIMENT_LOG.md), added to add_all_candidates - still made MAE
    worse on the expanding window (£907.77 -> £916.20), though more
    narrowly than the binary flag or Series 4's short-lag features; near
    -neutral on the sliding-365d window (+£0.52). Most likely the same
    story as Series 4/5: is_christmas_break/is_school_holiday/
    is_long_weekend/the month cyclical encodings already capture most of
    what's extractable from this signal family with the current data.
    Left computed here for reference.

    Continuous index built from the real 2-year pattern: weekday-adjusted
    relative sales intensity (actual / typical-for-that-weekday) averaged
    by day-of-December, computed separately for each year using only
    PRIOR years' December data.

    Real shape found (see docs/EXPERIMENT_LOG.md's Series 5 discussion): NOT a
    single peak centred on the 25th. There are two peaks - one around the
    22nd-23rd (pre-Christmas), a real trough on the 25th-26th (closure +
    Boxing Day lull), then a SECOND, LARGER peak around the 29th-30th
    (New Year/Hogmanay run-up, consistent with this being a Glasgow
    venue) - before easing off on the 31st itself. A flat binary flag or
    an assumed single-peak decay curve would both mis-model this.

    Walk-forward safety is the reason this needs its own function rather
    than a simple lookup: naively pooling both Decembers together to
    build one shared per-day table would let a later year's own actual
    December outcomes leak into that same year's feature value (the
    per-day pattern is computed from data that would include the row
    being predicted). Instead, for each December in the dataset, only
    data strictly before that December 1st is used - both for the
    weekday baseline and for the prior-December pattern itself. With only
    two Decembers currently in the data, this means the first one
    (2024) has no prior December to learn from and is left NaN (the same
    edge behaviour as this project's lag features at the start of the
    series) - only the second (2025) gets a real, non-leaked value.
    Non-December rows get the neutral value 1.0 (an "ordinary day" has no
    December effect by construction, not an estimate).

    The weekday baseline itself is a plain historical mean (not further
    smoothed/expanding day-by-day) computed once per December's cutoff -
    a documented simplification: it is a slowly-changing, stable quantity
    compared to the December-specific pattern (which does vary
    meaningfully year to year, exactly the leakage this function exists
    to avoid), so computing it once per cutoff rather than expanding
    row-by-row is a much lower-risk simplification than pooling the
    December pattern itself would have been.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    day_of_week = out[date_col].dt.dayofweek
    day_of_month = out[date_col].dt.day
    year = out[date_col].dt.year
    is_december = out[date_col].dt.month == 12

    index_values = pd.Series(1.0, index=out.index)
    index_values.loc[is_december] = np.nan

    for y in sorted(year[is_december].unique()):
        cutoff = pd.Timestamp(year=int(y), month=12, day=1)
        prior_mask = out[date_col] < cutoff  # boolean Series, same index as out
        if not prior_mask.any():
            continue

        # Every slice below is taken with .loc against prior_mask's own
        # index, so it stays label-aligned with prior_relative_intensity
        # rather than a full-length mask being applied to a
        # already-reduced-length Series (the original bug here).
        prior_dow = day_of_week.loc[prior_mask]
        prior_weekday_avg = out.loc[prior_mask].groupby(prior_dow)[target_col].mean()
        prior_relative_intensity = out.loc[prior_mask, target_col] / prior_dow.map(prior_weekday_avg)

        prior_is_dec = is_december.loc[prior_mask]
        if not prior_is_dec.any():
            continue  # no PRIOR December exists yet - stays NaN for this year

        prior_dom = day_of_month.loc[prior_mask]
        prior_dec_by_day = prior_relative_intensity[prior_is_dec].groupby(prior_dom[prior_is_dec]).mean()

        this_year_dec_mask = is_december & (year == y)
        index_values.loc[this_year_dec_mask] = day_of_month.loc[this_year_dec_mask].map(prior_dec_by_day)

    out["december_intensity_index"] = index_values  # DEPRECATED - see docstring
    return out


# ---------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _open_meteo_daily_to_df(api_json: dict) -> pd.DataFrame:
    daily = api_json.get("daily", {})
    if not daily:
        raise ValueError("Open-Meteo response missing 'daily' block.")

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).normalize(),
        "max_temp": daily.get("temperature_2m_max"),
        "rain_mm": daily.get("precipitation_sum"),
        "sun_seconds": daily.get("sunshine_duration"),
    })

    df["sun_hours"] = df["sun_seconds"] / 3600.0
    return df.drop(columns=["sun_seconds"])


def fetch_open_meteo_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = TIMEZONE,
    cache_dir: Path = CACHE_DIR,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo.

    Returns columns:
    - date
    - max_temp
    - rain_mm
    - sun_hours
    """
    os.makedirs(cache_dir, exist_ok=True)

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = pd.Timestamp.now(tz=timezone).date()

    daily_vars = ["temperature_2m_max", "precipitation_sum", "sunshine_duration"]

    def _get(url: str, params: dict) -> dict:
        key = _cache_key({"url": url, **params})
        path = cache_dir / f"{key}.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        time.sleep(0.2)
        return data

    frames = []

    hist_end = min(end, today)
    if start <= hist_end:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start),
            "end_date": str(hist_end),
            "daily": ",".join(daily_vars),
            "timezone": timezone,
        }
        frames.append(_open_meteo_daily_to_df(_get(ARCHIVE_URL, params)))

    fut_start = max(start, today)
    if fut_start <= end:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(fut_start),
            "end_date": str(end),
            "daily": ",".join(daily_vars),
            "timezone": timezone,
        }
        frames.append(_open_meteo_daily_to_df(_get(FORECAST_URL, params)))

    if not frames:
        return pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"])

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    date_col: str = DATE_COL,
    warm_temp_c: float = 18.0,
    hot_scotland_c: float = 20.0,
    heavy_rain_mm: float = 5.0,
    anomaly_window: int = 14,
) -> pd.DataFrame:
    """Merge weather data and add weather-derived features."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()

    w = weather_df.copy()
    w["weather_date"] = pd.to_datetime(w["date"]).dt.normalize()
    w = w.drop(columns=["date"])

    out = out.merge(w, left_on=date_col, right_on="weather_date", how="left")
    out = out.drop(columns=["weather_date"])
    out = out.sort_values(date_col).reset_index(drop=True)

    out["is_hot_for_scotland"] = (out["max_temp"] > hot_scotland_c).astype(int)
    out["is_heavy_rain"] = (out["rain_mm"] > heavy_rain_mm).astype(int)
    out["is_dry_day"] = (out["rain_mm"] < 1.0).astype(int)

    past_mean = (
        out["max_temp"]
        .rolling(window=anomaly_window, min_periods=anomaly_window)
        .mean()
        .shift(1)
    )
    out[f"temp_anomaly_{anomaly_window}d"] = out["max_temp"] - past_mean

    warm_cond = out["max_temp"] >= warm_temp_c
    grp = (~warm_cond).cumsum()
    out["warm_streak_len"] = warm_cond.groupby(grp).cumsum().astype(int)

    return out


# ---------------------------------------------------------------------
# Final dataset preparation
# ---------------------------------------------------------------------

def build_feature_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both:
    1. full engineered feature dataset
    2. final model-ready dataset with selected features only
    """
    df = load_daily_sales()

    df = add_time_features(df)
    df = add_payday_features(df)
    df = add_bank_holiday_features(df)
    df = add_school_holiday_features(
        df,
        date_col=DATE_COL,
        breaks_df=SCHOOL_BREAKS,
        council="Glasgow",
        break_types=("christmas", "summer", "easter"),
    )
    df = add_christmas_period_feature(df, date_col=DATE_COL)  # is_christmas_period: DEPRECATED, see function docstring
    df = add_december_intensity_feature(df, date_col=DATE_COL, target_col=TARGET_COL)  # december_intensity_index: DEPRECATED, see function docstring
    df = add_december_phase_features(df, date_col=DATE_COL)  # DECEMBER_PHASES flags: DEPRECATED, see function docstring

    start_date = str(df[DATE_COL].min().date())
    end_date = str(df[DATE_COL].max().date())

    weather_df = fetch_open_meteo_daily(
        lat=LAT,
        lon=LON,
        start_date=start_date,
        end_date=end_date,
        timezone=TIMEZONE,
    )

    df = add_weather_features(df, weather_df, date_col=DATE_COL)

    # Save full engineered dataset
    engineered_path = OUTPUT_DIR / "engineered_features.csv"
    df.to_csv(engineered_path, index=False)
    print(f"Engineered feature dataset written to: {engineered_path}")

    # Build final model-ready dataset
    required_cols = [DATE_COL, TARGET_COL] + FINAL_MODEL_FEATURES
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")

    model_df = df[required_cols].copy()

    # Drop rows with NA values created by lag/rolling/weather edges
    model_df = model_df.dropna().reset_index(drop=True)

    model_path = OUTPUT_DIR / "model_features.csv"
    model_df.to_csv(model_path, index=False)
    print(f"Model-ready dataset written to: {model_path}")

    return df, model_df


if __name__ == "__main__":
    build_feature_dataset()