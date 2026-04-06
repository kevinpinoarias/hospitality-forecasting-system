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
    if "total_sales" in out.columns:
        out["lag_7_sales"] = out["total_sales"].shift(7)
        out["rolling_7_sales"] = out["total_sales"].shift(1).rolling(window=7).mean()
        out["rolling_14_sales"] = out["total_sales"].shift(1).rolling(window=14).mean()

    # Forecast error lag / rolling
    if "forecast_error" in out.columns:
        out["lag_1_fe"] = out["forecast_error"].shift(1)
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