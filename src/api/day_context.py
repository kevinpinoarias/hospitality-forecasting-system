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

from src.api.schemas import BankHolidayContext, DayContext, PaydayContext, RecentSalesContext

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


def build_day_context(row: pd.Series, target_date: dt.date, sales_history_is_real: bool) -> DayContext:
    """`row` is the feature row the prediction used. `sales_history_is_real`
    is False when the previous fortnight's sales were filled from
    historical averages (a date beyond the data), in which case they are not
    reported as if they were real sales."""
    recent = None
    if sales_history_is_real:
        week, fortnight = _optional_float(row["rolling_7_sales"]), _optional_float(row["rolling_14_sales"])
        if week is not None and fortnight is not None:
            recent = RecentSalesContext(
                average_daily_sales_previous_7_days=week,
                average_daily_sales_previous_14_days=fortnight,
            )

    return DayContext(
        part_of_weekend_trading=bool(row["is_weekend"]),
        payday=payday_context(row, target_date),
        bank_holidays=bank_holiday_context(row, target_date),
        school_holiday=school_holiday_name(row),
        recent_sales=recent,
    )
