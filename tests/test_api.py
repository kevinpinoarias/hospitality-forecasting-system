"""
Tests for the forecasting API against the trained final model.

These need the pipeline's outputs (models/final_model/ and
data/features/engineered_features.csv, both gitignored), so they are
skipped on a fresh clone until `python main.py` has been run. Every date
used is far enough from today that no live weather call is made.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.api.history import ENGINEERED_FEATURES_PATH
from src.models.final_model import FINAL_FEATURES, MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "metadata.json").exists() or not ENGINEERED_FEATURES_PATH.exists(),
    reason="needs a trained final model - run `python main.py` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def predict(client, **item) -> dict:
    response = client.post("/predict", json={"requests": [item]})
    assert response.status_code == 200, response.text
    return response.json()["results"][0]


def test_health_reports_the_final_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "bagged_log1p_catboost"


def test_backtest_date_returns_the_out_of_sample_prediction(client):
    from src.models.final_model import BACKTEST_PREDICTIONS_FILENAME

    backtest = pd.read_csv(MODEL_DIR / BACKTEST_PREDICTIONS_FILENAME, parse_dates=["date"])
    row = backtest.iloc[100]
    result = predict(client, date=row["date"].date().isoformat())

    assert result["prediction_source"] == "made_before_the_day"
    assert result["forecast_sales_source"] == "manager_forecast_on_record"
    assert result["predictions"]["best_estimate"] == pytest.approx(row["final_model_prediction"])
    assert result["predictions"]["dry_scenario"] == result["predictions"]["best_estimate"]
    assert result["actual_sales"] is not None


def test_caller_forecast_on_a_past_date_is_labelled_as_seen(client):
    backtest_day = pd.read_csv(MODEL_DIR / "backtest_predictions.csv")["date"].iloc[100]
    result = predict(client, date=backtest_day, forecast_sales=9000)
    assert result["prediction_source"] == "model_had_seen_the_day"
    assert result["forecast_sales_used"] == 9000


def test_far_future_date_uses_historical_weather_and_both_scenarios(client):
    result = predict(client, date="2030-03-02")
    assert result["prediction_source"] == "forecast"
    assert result["rain_data_source"] == "historical_average"
    assert result["weather"] is None
    assert result["days_beyond_training_data"] > 0
    assert result["predictions"]["dry_scenario"] != result["predictions"]["heavy_rain_scenario"]


def test_day_context_is_returned(client):
    past = predict(client, date="2025-06-28")
    assert past["day_context"]["recent_sales"] is not None
    assert past["day_context"]["recent_forecast_accuracy"] is not None
    assert past["weather"]["expected_sunshine_hours"] is not None

    future = predict(client, date="2030-12-24")
    assert future["day_context"]["school_holiday"] is None  # beyond the council dates on file
    assert future["day_context"]["bank_holidays"]["next_bank_holiday_name"] == "Christmas Day"
    assert future["day_context"]["recent_sales"] is None  # never estimated sales presented as real
    assert future["day_context"]["recent_forecast_accuracy"] is None


def test_date_before_the_data_is_rejected(client):
    response = client.post("/predict", json={"requests": [{"date": "2020-01-01"}]})
    assert response.status_code == 422


def test_served_features_match_the_series_18_construction(client):
    """The API's precomputed-frame shortcut must build exactly the features
    that serving_features builds from scratch - the construction Series 18
    validated."""
    from src.api.main import model_service
    from src.features.serving_features import (
        build_feature_frame,
        extend_daily_frame,
        fill_future_sales_seasonally,
        with_history_features,
    )

    history = model_service.history.df
    target = model_service.history.max_date + dt.timedelta(days=200)
    far_today = target + dt.timedelta(days=365)  # no live weather anywhere near the target

    model_service._ensure_frame(target)
    served, _, _ = model_service._future_row(target, far_today)

    frame = build_feature_frame(extend_daily_frame(history, target))
    sales = fill_future_sales_seasonally(frame, len(history), history)
    expected = with_history_features(frame, sales, [len(frame) - 1])

    assert np.allclose(
        served[FINAL_FEATURES].to_numpy(dtype=float),
        expected[FINAL_FEATURES].to_numpy(dtype=float),
        equal_nan=True,
    )
