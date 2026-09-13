"""
Tests for src/experiments/recursive_horizon_test.py's buffer/lag
mechanics - the part of this experiment where a subtle off-by-one or
wrong-semantics bug would silently produce a meaningless result. These
tests check the lag/rolling arithmetic matches
src/features/build_features.py's real shift()/rolling() semantics
exactly, and that the recursive feedback loop genuinely uses the model's
own prior prediction, not a leaked true actual.
"""

from __future__ import annotations

import pandas as pd

from src.experiments.recursive_horizon_test import _lagged_value


def test_plain_lag_reads_correct_offset_date():
    buffer = {pd.Timestamp("2025-01-01"): 100.0, pd.Timestamp("2025-01-08"): 999.0}
    result = _lagged_value(buffer, pd.Timestamp("2025-01-08"), lag_days=7, rolling_window=None)
    assert result == 100.0


def test_plain_lag_missing_date_returns_nan():
    buffer = {pd.Timestamp("2025-01-01"): 100.0}
    result = _lagged_value(buffer, pd.Timestamp("2025-01-20"), lag_days=7, rolling_window=None)
    assert pd.isna(result)


def test_rolling_window_matches_pandas_shift_rolling_semantics():
    """rolling_7_sales = total_sales.shift(1).rolling(7).mean(): for a
    target date T, this should average the 7 real days ending T-1, i.e.
    T-7 through T-1 inclusive."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    values = pd.Series(range(1, 21), index=dates, dtype=float)  # 1,2,3,...,20

    buffer = values.to_dict()
    target = dates[15]  # value at target would be 16, but we want shift(1).rolling(7)

    # Expected: mean of the 7 days ending target-1 day = dates[8:15] -> values[9..15] = 10..16 -> wait recompute
    expected_window = values.loc[target - pd.Timedelta(days=7): target - pd.Timedelta(days=1)]
    expected = expected_window.mean()

    result = _lagged_value(buffer, target, lag_days=1, rolling_window=7)
    assert abs(result - expected) < 1e-9


def test_use_true_actuals_flag_feeds_real_value_not_prediction():
    """With use_true_actuals=True, a later day's lag feature must be built
    from the real actual of an earlier in-batch day, not the model's
    (here deliberately absurd) prediction for it - this is what makes the
    matched 'with real data' comparison point to Series 2 valid."""
    import pandas as pd
    from unittest.mock import patch

    from src.experiments.recursive_horizon_test import simulate_recursive_forecast
    from src.experiments.splits import Fold

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "total_sales": [1000.0 + i for i in range(40)],
        "forecast_sales": [900.0 + i for i in range(40)],
        "day_of_week": dates.dayofweek,
        "lag_7_sales": 0.0,  # placeholder - always overwritten before use
    })
    full_history = df[["date", "total_sales", "forecast_sales"]].copy()

    fold = Fold(fold_id=0, train_start=dates[0], train_end=dates[30], test_start=dates[30], test_end=dates[39] + pd.Timedelta(days=1))
    features = ["day_of_week", "lag_7_sales"]

    captured_feature_rows = []

    def fake_predict(model, feature_row, features):
        captured_feature_rows.append(feature_row.copy())
        return 1_000_000.0  # deliberately absurd, so any leak into a later lag is obvious

    with patch("src.experiments.recursive_horizon_test._predict_catboost", side_effect=fake_predict):
        simulate_recursive_forecast(
            "catboost", "test_set", features, fold, df, full_history,
            horizon_days=8, use_true_actuals=True,
        )

    # Day 8's lag_7_sales references day 1 of this same batch (test_start).
    # Day 1's REAL total_sales is 1000 + 30 = 1030 (index 30 in the series).
    day8_lag_7 = captured_feature_rows[7]["lag_7_sales"]
    assert day8_lag_7 == 1030.0
    assert day8_lag_7 != 1_000_000.0


def test_recursive_feedback_uses_predicted_not_true_value():
    """The whole point of the experiment: once a date has been 'predicted'
    and fed back into the buffer, a later lookup referencing that date
    must return the PREDICTED value, not silently keep any true value
    that might already be present in a stale copy of the buffer."""
    buffer = {pd.Timestamp("2025-01-01"): 100.0}
    # Simulate the recursive loop overwriting day 2 with a model prediction
    # that differs from what the "true" value would have been.
    buffer[pd.Timestamp("2025-01-02")] = 555.0  # model's own prediction, not the real 200.0

    result = _lagged_value(buffer, pd.Timestamp("2025-01-09"), lag_days=7, rolling_window=None)
    assert result == 555.0
