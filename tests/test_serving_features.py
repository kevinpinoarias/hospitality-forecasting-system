"""
Tests for src/features/serving_features.py - the code that builds the final
model's features for dates beyond the end of the data. A silent difference
from the training pipeline's own feature definitions would mean the API
serves a model on inputs it was never trained on.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_time_features
from src.features.serving_features import (
    HISTORY_FEATURES,
    build_feature_frame,
    fill_future_sales_recursively,
    history_features_at,
    weekday_seasonal_lookup,
    with_history_features,
)


def make_daily(days: int = 60, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "total_sales": rng.uniform(4000, 16000, days),
        "forecast_sales": rng.uniform(4000, 16000, days),
        "max_temp": rng.uniform(0, 25, days),
        "rain_mm": rng.uniform(0, 12, days),
        "sun_hours": rng.uniform(0, 12, days),
    })


def test_history_features_match_the_training_pipeline():
    daily = make_daily()
    expected = add_time_features(daily)
    sales = daily["total_sales"].to_numpy()
    forecast = daily["forecast_sales"].to_numpy()

    for i in range(len(daily)):
        got = history_features_at(sales, forecast, i)
        for name in HISTORY_FEATURES:
            assert got[name] == pytest.approx(expected.at[i, name], nan_ok=True), (name, i)


def test_history_features_never_read_the_day_itself():
    daily = make_daily()
    sales = daily["total_sales"].to_numpy().copy()
    forecast = daily["forecast_sales"].to_numpy()
    before = history_features_at(sales, forecast, 30)
    sales[30] = 1e9
    assert history_features_at(sales, forecast, 30) == before


def test_build_feature_frame_rejects_missing_days():
    daily = make_daily().drop(index=10)
    with pytest.raises(ValueError, match="no gaps"):
        build_feature_frame(daily)


def test_recursive_fill_feeds_each_prediction_forward():
    """Day i's history features must read the prediction written for day i-1,
    not the NaN the frame started with."""
    daily = make_daily(40)
    daily.loc[30:, "total_sales"] = np.nan
    frame = build_feature_frame(daily)
    features = ["lag_7_sales", "rolling_7_sales", "day_of_week"]
    rolling_7 = features.index("rolling_7_sales")

    seen = []

    def predict(row: np.ndarray) -> np.ndarray:
        seen.append(row[0, rolling_7])
        return np.array([1000.0])

    sales = fill_future_sales_recursively(frame, 30, features, predict)

    assert np.isfinite(seen).all()
    assert np.allclose(sales[30:], 1000.0)
    assert np.allclose(sales[:30], daily["total_sales"].to_numpy()[:30])
    # After seven predicted days, the 7-day rolling mean is entirely predictions.
    assert seen[7] == pytest.approx(1000.0)


def test_with_history_features_uses_the_supplied_series():
    daily = make_daily(40)
    frame = build_feature_frame(daily)
    sales = daily["total_sales"].to_numpy().copy()
    sales[25:32] = 500.0
    out = with_history_features(frame, sales, [32])
    assert out["rolling_7_sales"].iloc[0] == pytest.approx(500.0)
    assert out["lag_7_sales"].iloc[0] == pytest.approx(500.0)


def test_weekday_seasonal_lookup_ignores_missing_values():
    daily = make_daily(400, start="2024-01-01")
    target = dt.date(2025, 2, 3)
    baseline = weekday_seasonal_lookup(daily, target, "total_sales")

    with_gap = daily.copy()
    same_weekday = pd.to_datetime(with_gap["date"]).dt.dayofweek == target.weekday()
    with_gap.loc[same_weekday & (with_gap.index < 5), "total_sales"] = np.nan
    assert np.isfinite(weekday_seasonal_lookup(with_gap, target, "total_sales"))
    assert np.isfinite(baseline)
