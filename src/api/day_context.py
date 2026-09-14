"""
Plain facts about a forecast date, taken from the same feature row the model
predicted from - payday timing, bank holidays, school holidays, whether it's
a weekend trading day, and the recent sales behind it - so a caller (or the
chat assistant) can explain what the model took into account without
reverse-engineering the raw features.

Payday dates come from the pipeline's own payday columns (the last working
day of the month, see build_features.add_payday_features), but the day
counts are worked out here from the actual last and next payday. The
model's own `days_since_payday` feature always counts from the previous
month's payday, so in the few days between an early payday and the end of
the month (e.g. 31 October 2026, the day after a Friday payday) it reads
about 30 days rather than 1. That quirk is part of how the model was
trained and is left unchanged; it just isn't repeated to people here.
"""

from __future__ import annotations

import datetime as dt
import math

import holidays
import pandas as pd

from src.api.schemas import (
    BankHolidayContext,
    DayContext,
    ForecastGap,
    PaydayContext,
    RecentForecastAccuracyContext,
    RecentSalesContext,
)

UK_SUBDIVISION = "SCT"  # matches add_bank_holiday_features / add_payday_features
PAYDAY_WINDOW_DAYS = 3
SCHOOL_BREAK_NAMES = {"christmas": "Christmas holidays", "easter": "Easter holidays", "summer": "summer holidays"}


def _as_date(value) -> dt.date:
    return pd.Timestamp(value).date()


def _optional_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 2)


def payday_context(row: pd.Series, target_date: dt.date) -> PaydayContext:
    this_month = _as_date(row["payday"])
    last = this_month if target_date >= this_month else _as_date(row["prev_payday"])
    upcoming = this_month if target_date <= this_month else _as_date(row["next_payday"])
    days_since = (target_date - last).days
    days_until = (upcoming - target_date).days
    return PaydayContext(
        is_payday=target_date == this_month,
        last_payday=last,
        days_since_last_payday=days_since,
        next_payday=upcoming,
        days_until_next_payday=days_until,
        within_3_days_of_payday=min(days_since, days_until) <= PAYDAY_WINDOW_DAYS,
    )


def bank_holiday_context(row: pd.Series, target_date: dt.date) -> BankHolidayContext:
    calendar = holidays.UK(subdiv=UK_SUBDIVISION, years=range(target_date.year - 1, target_date.year + 2))
    dates = sorted(calendar)
    upcoming = next(d for d in dates if d > target_date)
    previous = next(d for d in reversed(dates) if d < target_date)
    return BankHolidayContext(
        is_bank_holiday=bool(row["is_bank_holiday"]),
        bank_holiday_name=calendar.get(target_date),
        next_bank_holiday=upcoming,
        next_bank_holiday_name=calendar[upcoming],
        days_until_next_bank_holiday=(upcoming - target_date).days,
        previous_bank_holiday=previous,
        previous_bank_holiday_name=calendar[previous],
        is_long_weekend=bool(row["is_long_weekend"]),
    )


def school_holiday_name(row: pd.Series) -> str | None:
    """The school break the model flagged for this date, if any."""
    for break_type, name in SCHOOL_BREAK_NAMES.items():
        if row[f"is_{break_type}_break"]:
            return name
    return None


def recent_sales_context(row: pd.Series) -> RecentSalesContext | None:
    week, fortnight = _optional_float(row["rolling_7_sales"]), _optional_float(row["rolling_14_sales"])
    if week is None or fortnight is None:
        return None
    return RecentSalesContext(
        average_daily_sales_previous_7_days=week,
        average_daily_sales_previous_14_days=fortnight,
    )


def _forecast_gap(forecast_error: float | None) -> ForecastGap | None:
    """forecast_error = actual sales - forecast sales (see
    build_features.add_time_features). A positive forecast_error means
    actual sales came in above the forecast, i.e. the forecast under-shot -
    reported here as a positive gap with an explicit direction, so nothing
    downstream has to interpret a sign."""
    if forecast_error is None:
        return None
    direction = "under_forecast" if forecast_error >= 0 else "over_forecast"
    return ForecastGap(gap_gbp=round(abs(forecast_error), 2), direction=direction)


def recent_forecast_accuracy_context(row: pd.Series) -> RecentForecastAccuracyContext | None:
    """How far actual sales have recently run from the manager's own forecast
    - built from the model's forecast-error history features (`lag_1_fe`,
    `lag_7_fe`, `rolling_7_fe`)."""
    yesterday = _forecast_gap(_optional_float(row["lag_1_fe"]))
    last_week = _forecast_gap(_optional_float(row["lag_7_fe"]))
    average = _forecast_gap(_optional_float(row["rolling_7_fe"]))
    if yesterday is None or last_week is None or average is None:
        return None
    return RecentForecastAccuracyContext(
        yesterday=yesterday,
        same_day_last_week=last_week,
        average_previous_7_days=average,
    )


def build_day_context(row: pd.Series, target_date: dt.date, sales_history_is_real: bool) -> DayContext:
    """`row` is the feature row the prediction used. `sales_history_is_real`
    is False when the previous fortnight's sales/forecast-accuracy history
    were filled from historical averages (a date beyond the data), in which
    case they are not reported as if they were real."""
    recent, recent_accuracy = None, None
    if sales_history_is_real:
        recent = recent_sales_context(row)
        recent_accuracy = recent_forecast_accuracy_context(row)

    return DayContext(
        part_of_weekend_trading=bool(row["is_weekend"]),
        payday=payday_context(row, target_date),
        bank_holidays=bank_holiday_context(row, target_date),
        school_holiday=school_holiday_name(row),
        recent_sales=recent,
        recent_forecast_accuracy=recent_accuracy,
    )
