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
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)


def _fetch_rain_mm(target_date: dt.date) -> tuple[float | None, bool]:
    """
    Attempt a live weather fetch with retries.

    Returns (rain_mm, errored). `rain_mm` is None either when the call
    succeeded but the provider has no data yet for that date (normal, for
    dates near the edge of its real forecast range), or when every retry
    raised an error - `errored` distinguishes the two so the caller can
    report an honest source label instead of calling a null response
    an "API failure".
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
                return float(weather_df.loc[0, "rain_mm"]), False
            return None, False
        except Exception as exc:  # network error, timeout, bad response
            logger.warning(
                "Open-Meteo fetch failed for %s (attempt %d/%d): %s",
                date_str, attempt + 1, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    return None, True


def resolve_is_heavy_rain(
    target_date: dt.date,
    history_df: pd.DataFrame,
    today: dt.date | None = None,
) -> tuple[int, str]:
    """Returns (is_heavy_rain flag, source label)."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    within_horizon = target_date <= today + dt.timedelta(days=FORECAST_HORIZON_DAYS)

    if within_horizon:
        rain_mm, errored = _fetch_rain_mm(target_date)
        if rain_mm is not None:
            live_source = "observed" if target_date <= today else "forecast"
            return int(rain_mm > HEAVY_RAIN_MM), live_source
        source = "historical_average_api_error" if errored else "historical_average"
    else:
        source = "historical_average"

    rain_probability = weekday_seasonal_lookup(history_df, target_date, "is_heavy_rain")
    return int(rain_probability >= 0.5), source
