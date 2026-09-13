"""
Live weather for the dates around a forecast.

The final model reads the weather on the forecast date (maximum
temperature, rain, sunshine) and on the 14 days before it. Where Open-Meteo
has a real figure - observed for past days, forecast for roughly the next
16 days - that figure is used; every other day keeps its historical
weekday/season median (see src/features/serving_features.py).

A failed call is retried, then reported through an `errored` flag so the
response can honestly say the live source was unavailable rather than
calling a normal gap in the provider's data an API failure.
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

logger = logging.getLogger("hospitality_api")

# Open-Meteo forecasts 16 days including today, so the last forecastable
# date is today + 15 (asking for more makes the whole request fail).
FORECAST_HORIZON_DAYS = 16
HEAVY_RAIN_MM = 5.0  # matches is_heavy_rain in src/features/build_features.py
DRY_MM = 1.0  # matches is_dry_day in src/features/build_features.py
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


def live_weather_end(today: dt.date) -> dt.date:
    """The last date Open-Meteo can give a real forecast for. Uses the
    earlier of the caller's date and the venue's local date, since the
    provider counts days in the venue's timezone."""
    local_today = pd.Timestamp.now(tz=TIMEZONE).date()
    return min(today, local_today) + dt.timedelta(days=FORECAST_HORIZON_DAYS - 1)


def fetch_live_weather(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, bool]:
    """
    Daily max_temp, rain_mm and sun_hours from Open-Meteo for a date range,
    with retries.

    Returns (weather, errored). Rows the provider has no figure for yet are
    dropped. `errored` is True only when every retry raised.
    """
    empty = pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"])
    if start_date > end_date:
        return empty, False

    for attempt in range(MAX_RETRIES):
        try:
            weather = fetch_open_meteo_daily(
                lat=LAT,
                lon=LON,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                timezone=TIMEZONE,
                cache_dir=CACHE_DIR,
            )
            return weather.dropna(subset=["max_temp", "rain_mm", "sun_hours"]), False
        except Exception as exc:  # network error, timeout, bad response
            logger.warning(
                "Open-Meteo fetch failed for %s to %s (attempt %d/%d): %s",
                start_date, end_date, attempt + 1, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    return empty, True
