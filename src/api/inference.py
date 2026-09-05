"""
Builds features for a single requested date and produces the dry/heavy-rain
scenario predictions from the trained XGBoost model.

Feature construction deliberately reuses the exact functions from
src/features/build_features.py that the training pipeline uses, so serving
and training can't silently drift apart (the same calendar/holiday/payday
logic runs in both places).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src.features.build_features import (
    add_bank_holiday_features,
    add_payday_features,
    add_time_features,
)
from src.api.history import HistoryCache, days_beyond_training_data, weekday_seasonal_lookup
from src.api.schemas import PredictResult, ScenarioPredictions
from src.api.weather import resolve_is_heavy_rain

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "xgboost_model_metadata.json"

DATE_COL = "date"


def _build_feature_row(target_date: dt.date) -> pd.DataFrame:
    """Compute every date-derived feature for a single target date."""
    row = pd.DataFrame({DATE_COL: [pd.Timestamp(target_date)]})
    row = add_time_features(row, date_col=DATE_COL)
    row = add_payday_features(row, date_col=DATE_COL)
    row = add_bank_holiday_features(row, date_col=DATE_COL)
    return row


class ModelService:
    """Loads the trained model once and serves predictions from it."""

    def __init__(self, history: HistoryCache) -> None:
        self.history = history

        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.features: list[str] = metadata["features"]

        self.model = XGBRegressor()
        self.model.load_model(MODEL_PATH)

    def _resolve_forecast_sales(
        self, target_date: dt.date, user_value: float | None
    ) -> tuple[float, str]:
        if user_value is not None:
            return user_value, "user_provided"

        value = weekday_seasonal_lookup(self.history.df, target_date, "forecast_sales")
        return value, "historical_weekday_seasonal_average"

    def predict_one(self, target_date: dt.date, forecast_sales: float | None) -> PredictResult:
        forecast_sales_used, forecast_sales_source = self._resolve_forecast_sales(
            target_date, forecast_sales
        )
        resolved_rain, rain_source = resolve_is_heavy_rain(target_date, self.history.df)

        feature_row = _build_feature_row(target_date)
        feature_row["forecast_sales"] = forecast_sales_used

        dry_row = feature_row.copy()
        dry_row["is_heavy_rain"] = 0

        rain_row = feature_row.copy()
        rain_row["is_heavy_rain"] = 1

        dry_pred = float(self.model.predict(dry_row[self.features])[0])
        rain_pred = float(self.model.predict(rain_row[self.features])[0])
        best_estimate = rain_pred if resolved_rain == 1 else dry_pred

        return PredictResult(
            date=target_date,
            forecast_sales_used=forecast_sales_used,
            forecast_sales_source=forecast_sales_source,
            rain_data_source=rain_source,
            days_beyond_training_data=days_beyond_training_data(self.history.df, target_date),
            predictions=ScenarioPredictions(
                dry_scenario=dry_pred,
                heavy_rain_scenario=rain_pred,
                best_estimate=best_estimate,
            ),
        )
