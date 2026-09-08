"""
Resolves the `is_heavy_rain` feature for a single target date.

Three possible sources, in order of preference:
1. "forecast"           - live Open-Meteo forecast, for dates within its
                           ~16-day forecast horizon.
2. "historical_average" - the date is beyond the forecast horizon, so we use
                           the historical probability of heavy rain for that
                           weekday/season instead.
3. "historical_average_api_unavailable" - a live forecast should have been
                           available, but the API call failed after retries;
                           we fall back to the same historical estimate
                           rather than maintaining a second, rain-blind model.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from src.features.build_features import (
    CACHE_DIR,
    LAT,
    LON,
    TIMEZONE,
    fetch_open_meteo_daily,
)
from src.api.history import weekday_seasonal_lookup

logger = logging.getLogger("hospitality_api")

FORECAST_HORIZON_DAYS = 16
HEAVY_RAIN_MM = 5.0
DRY_MM = 1.0  # matches is_dry_day's threshold in src/features/build_features.py
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)


def categorize_rain(rain_mm: float) -> str:
    """Plain-language rain description, not a raw mm figure nobody
    actually reasons in day to day. Reuses the exact same thresholds
    already used elsewhere for is_dry_day / is_heavy_rain, so this stays
    consistent with what the model itself treats as a meaningful cutoff."""
    if rain_mm < DRY_MM:
        return "dry"
    if rain_mm > HEAVY_RAIN_MM:
        return "heavy rain"
    return "light rain"


def _fetch_weather(target_date: dt.date) -> tuple[float | None, float | None, bool]:
    """
    Attempt a live weather fetch with retries.

    Returns (rain_mm, max_temp_c, errored). Both values are None either when
    the call succeeded but the provider has no data yet for that date
    (normal, for dates near the edge of its real forecast range), or when
    every retry raised an error - `errored` distinguishes the two so the
    caller can report an honest source label instead of calling a null
    response an "API failure".
    """
    date_str = target_date.isoformat()

    for attempt in range(MAX_RETRIES):
        try:
            weather_df = fetch_open_meteo_daily(
                lat=LAT,
                lon=LON,
                start_date=date_str,
                end_date=date_str,
                timezone=TIMEZONE,
                cache_dir=CACHE_DIR,
            )
            if not weather_df.empty and pd.notna(weather_df.loc[0, "rain_mm"]):
                rain_mm = float(weather_df.loc[0, "rain_mm"])
                max_temp_val = weather_df.loc[0, "max_temp"]
                max_temp = float(max_temp_val) if pd.notna(max_temp_val) else None
                return rain_mm, max_temp, False
            return None, None, False
        except Exception as exc:  # network error, timeout, bad response
            logger.warning(
                "Open-Meteo fetch failed for %s (attempt %d/%d): %s",
                date_str, attempt + 1, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    return None, None, True


def resolve_is_heavy_rain(
    target_date: dt.date,
    history_df: pd.DataFrame,
    today: dt.date | None = None,
) -> dict:
    """
    Returns a dict:
        is_heavy_rain: int
        source: str
        rain_mm / max_temp_c: float | None - the actual weather figures
            behind the decision, only populated when a live forecast or
            observed value was used (not for the historical-average
            fallback, where there's no single real figure to report).
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    within_horizon = target_date <= today + dt.timedelta(days=FORECAST_HORIZON_DAYS)

    if within_horizon:
        rain_mm, max_temp_c, errored = _fetch_weather(target_date)
        if rain_mm is not None:
            live_source = "observed" if target_date <= today else "forecast"
            return {
                "is_heavy_rain": int(rain_mm > HEAVY_RAIN_MM),
                "source": live_source,
                "rain_mm": rain_mm,
                "max_temp_c": max_temp_c,
            }
        source = "historical_average_api_error" if errored else "historical_average"
    else:
        source = "historical_average"

    rain_probability = weekday_seasonal_lookup(history_df, target_date, "is_heavy_rain")
    return {
        "is_heavy_rain": int(rain_probability >= 0.5),
        "source": source,
        "rain_mm": None,
        "max_temp_c": None,
    }
