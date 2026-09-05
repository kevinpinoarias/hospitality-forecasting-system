"""
Pydantic request/response models for the forecasting API.
"""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel, Field, field_validator


class PredictItem(BaseModel):
    date: date_type
    forecast_sales: float | None = Field(default=None, ge=0)


class PredictRequest(BaseModel):
    requests: list[PredictItem]

    @field_validator("requests")
    @classmethod
    def non_empty(cls, value: list[PredictItem]) -> list[PredictItem]:
        if not value:
            raise ValueError("requests must contain at least one item")
        return value


class ScenarioPredictions(BaseModel):
    dry_scenario: float
    heavy_rain_scenario: float
    best_estimate: float


class PredictResult(BaseModel):
    date: date_type
    forecast_sales_used: float
    forecast_sales_source: str
    rain_data_source: str
    days_beyond_training_data: int
    predictions: ScenarioPredictions


class PredictResponse(BaseModel):
    results: list[PredictResult]


class HealthResponse(BaseModel):
    status: str
    historical_rows: int
    history_last_refreshed: str
