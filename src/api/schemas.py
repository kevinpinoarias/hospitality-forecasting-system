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


class WeatherDetails(BaseModel):
    """The actual weather figures behind the rain scenario, only present
    when rain_data_source is "forecast" or "observed" (a real, specific
    value) - null for the historical-average fallback, where there's no
    single real figure to report."""
    expected_rain_mm: float
    expected_rain_description: str
    expected_max_temp_c: float | None = None


class HistoricalComparison(BaseModel):
    """Real realised sales for the same weekday in recent history, if that
    exact date exists in the historical data - null otherwise, never
    estimated."""
    same_day_last_week: float | None = None
    same_day_last_year: float | None = None


class PredictResult(BaseModel):
    date: date_type
    # Computed and returned explicitly rather than left for a caller (or an
    # LLM) to work out from the date itself - day-of-week arithmetic is a
    # known weak spot for LLMs, and this project doesn't leave real facts
    # to be guessed when they can just be provided.
    day_of_week: str
    # True for Christmas Day / New Year's Day (any year) - the venue is
    # confirmed always closed on these dates (see KNOWN_CLOSURE_MONTH_DAYS
    # in src/preprocessing/build_daily_sales.py, the same definition used
    # to zero-fill these dates in the training data). The model itself has
    # no "closed" feature and will still return a normal-looking numeric
    # prediction for these dates - this flag lets a caller (or the LLM
    # assistant) recognise that figure isn't meaningful, rather than
    # presenting it as an ordinary day's forecast.
    is_known_closure_day: bool
    forecast_sales_used: float
    forecast_sales_source: str
    rain_data_source: str
    days_beyond_training_data: int
    # Where best_estimate comes from:
    # - "forecast": a date beyond the historical data.
    # - "out_of_sample_backtest": a past date, predicted by the pipeline's
    #   rolling-origin backtest before the model saw that day, so comparing
    #   it with actual_sales is a fair test of accuracy. The weather is
    #   already known, so both scenarios equal best_estimate.
    # - "in_sample": a past date the served model was trained on (or a past
    #   date with a caller-supplied forecast_sales) - not a fair accuracy test.
    prediction_source: str
    predictions: ScenarioPredictions
    weather: WeatherDetails | None = None
    historical_comparison: HistoricalComparison
    # Real realised sales for this exact date - only populated when the
    # date is in the past and exists in the historical data (null for
    # future dates or genuine gaps). Lets a caller compare a past
    # prediction against what actually happened.
    actual_sales: float | None = None


class PredictResponse(BaseModel):
    results: list[PredictResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_trained_through: str
    historical_rows: int
    history_last_refreshed: str
