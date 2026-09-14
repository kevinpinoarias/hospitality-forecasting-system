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
    expected_sunshine_hours: float | None = None
    # Above 20C, the model's own "hot for Scotland" threshold.
    hot_for_scotland: bool | None = None
    # Maximum temperature minus the average of the previous 14 days' maximums.
    temperature_vs_previous_fortnight_c: float | None = None
    # Consecutive days up to and including this one with a maximum of 18C or more.
    warm_streak_days: int | None = None


class HistoricalComparison(BaseModel):
    """Real realised sales for the same weekday in recent history, if that
    exact date exists in the historical data - null otherwise, never
    estimated."""
    same_day_last_week: float | None = None
    same_day_last_year: float | None = None


class PaydayContext(BaseModel):
    """Payday is the last working day of the month."""
    is_payday: bool
    last_payday: date_type
    days_since_last_payday: int
    next_payday: date_type
    days_until_next_payday: int
    within_3_days_of_payday: bool


class BankHolidayContext(BaseModel):
    """Scottish bank holidays."""
    is_bank_holiday: bool
    bank_holiday_name: str | None = None
    next_bank_holiday: date_type
    next_bank_holiday_name: str
    days_until_next_bank_holiday: int
    previous_bank_holiday: date_type
    previous_bank_holiday_name: str
    # The weekend next to a Monday or Friday bank holiday, including the holiday.
    is_long_weekend: bool


class RecentSalesContext(BaseModel):
    """Real average daily sales before the date. Only present when those days
    are in the historical data - never estimated."""
    average_daily_sales_previous_7_days: float
    average_daily_sales_previous_14_days: float


class RecentForecastAccuracyContext(BaseModel):
    """How far actual sales have recently run from the manager's own sales
    forecast (not the model's forecast) - positive means actual sales came in
    ABOVE that forecast, negative means BELOW. Only present when the days
    behind it are in the historical data - never estimated."""
    actual_vs_forecast_yesterday_gbp: float
    actual_vs_forecast_same_day_last_week_gbp: float
    average_actual_vs_forecast_previous_7_days_gbp: float


class DayContext(BaseModel):
    """What the model took into account about the date, as plain facts - see
    src/api/day_context.py."""
    # Friday, Saturday or Sunday - the days the model treats as the weekend.
    part_of_weekend_trading: bool
    payday: PaydayContext
    bank_holidays: BankHolidayContext
    # "Christmas holidays", "Easter holidays" or "summer holidays" for a
    # Glasgow school break, else null. Only those three breaks are tracked,
    # so null does not rule out another school break (e.g. October week).
    school_holiday: str | None = None
    recent_sales: RecentSalesContext | None = None
    recent_forecast_accuracy: RecentForecastAccuracyContext | None = None


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
    # Where best_estimate comes from. Plain-English values on purpose: the
    # chat assistant reads them, and technical labels leaked into its answers.
    # - "forecast": a date beyond the historical data.
    # - "made_before_the_day": a past date, predicted by the pipeline's
    #   rolling-origin backtest before the model saw that day (out of
    #   sample), so comparing it with actual_sales is a fair test of
    #   accuracy. The weather is already known, so both scenarios equal
    #   best_estimate.
    # - "model_had_seen_the_day": a past date the served model was trained
    #   on (in sample), or a past date with a caller-supplied forecast_sales -
    #   not a fair accuracy test.
    prediction_source: str
    predictions: ScenarioPredictions
    day_context: DayContext
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
