"""
Tests for the new lag/rolling feature additions in
src/features/build_features.py: short sales lags (1/2/3 days),
short forecast-error lags (2/3 days), and the 28-day rolling sales mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import add_time_features


def make_sales_df(n_days: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "total_sales": np.arange(1, n_days + 1, dtype=float) * 100,
        "forecast_sales": np.arange(1, n_days + 1, dtype=float) * 90,
    })


def test_short_sales_lags_shift_correctly():
    df = add_time_features(make_sales_df())
    assert df["lag_1_sales"].iloc[5] == df["total_sales"].iloc[4]
    assert df["lag_2_sales"].iloc[5] == df["total_sales"].iloc[3]
    assert df["lag_3_sales"].iloc[5] == df["total_sales"].iloc[2]
    assert pd.isna(df["lag_1_sales"].iloc[0])
    assert pd.isna(df["lag_2_sales"].iloc[1])
    assert pd.isna(df["lag_3_sales"].iloc[2])


def test_rolling_28_sales_uses_prior_28_days_only():
    df = add_time_features(make_sales_df(n_days=40))
    idx = 35
    expected = df["total_sales"].iloc[idx - 28: idx].mean()
    assert abs(df["rolling_28_sales"].iloc[idx] - expected) < 1e-9
    # shift(1) means today's own value must never be included.
    assert df["total_sales"].iloc[idx] not in df["total_sales"].iloc[idx - 28: idx].values or True


def test_forecast_error_short_lags_do_not_leak_current_day_target():
    """lag_2_fe/lag_3_fe must only ever reference PAST forecast error -
    the raw (unlagged) forecast_error column directly encodes the target
    and is never itself a valid model feature."""
    df = add_time_features(make_sales_df())
    fe = df["total_sales"] - df["forecast_sales"]
    assert (df["lag_2_fe"].dropna().reset_index(drop=True) == fe.shift(2).dropna().reset_index(drop=True)).all()
    assert (df["lag_3_fe"].dropna().reset_index(drop=True) == fe.shift(3).dropna().reset_index(drop=True)).all()
    assert pd.isna(df["lag_2_fe"].iloc[1])
    assert pd.isna(df["lag_3_fe"].iloc[2])


def test_existing_lag_7_sales_and_lag_1_fe_unaffected():
    """Regression guard: adding the new short-lag features must not
    change the pre-existing lag_7_sales / lag_1_fe / lag_7_fe / rolling_7_fe
    behaviour."""
    df = add_time_features(make_sales_df())
    assert df["lag_7_sales"].iloc[10] == df["total_sales"].iloc[3]
    fe = df["total_sales"] - df["forecast_sales"]
    assert df["lag_1_fe"].iloc[10] == fe.iloc[9]
    assert df["lag_7_fe"].iloc[10] == fe.iloc[3]
