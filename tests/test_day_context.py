"""
Tests for src/api/day_context.py - the plain facts about a forecast date
that the API returns and the chat assistant explains. Built from the same
feature rows the model predicts from.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.api.day_context import build_day_context
from src.features.serving_features import build_feature_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    dates = pd.date_range("2026-09-01", "2027-01-10", freq="D")
    rng = np.random.default_rng(0)
    daily = pd.DataFrame({
        "date": dates,
        "total_sales": rng.uniform(5000, 15000, len(dates)),
        "forecast_sales": rng.uniform(5000, 15000, len(dates)),
        "max_temp": rng.uniform(0, 15, len(dates)),
        "rain_mm": rng.uniform(0, 8, len(dates)),
        "sun_hours": rng.uniform(0, 8, len(dates)),
    })
    return build_feature_frame(daily).set_index("date", drop=False)


def context(frame: pd.DataFrame, day: str, real_sales: bool = True):
    return build_day_context(frame.loc[pd.Timestamp(day)], dt.date.fromisoformat(day), real_sales)


def test_day_after_an_early_payday_counts_from_that_payday(frame):
    """31 October 2026 is a Saturday, the day after a Friday payday. The
    model's own feature counts from September's payday; the context must
    report the real 1 day."""
    row = frame.loc[pd.Timestamp("2026-10-31")]
    assert row["days_since_payday"] == 31  # the model's feature, unchanged

    payday = context(frame, "2026-10-31").payday
    assert payday.last_payday == dt.date(2026, 10, 30)
    assert payday.days_since_last_payday == 1
    assert payday.next_payday == dt.date(2026, 11, 27)
    assert payday.within_3_days_of_payday


def test_payday_itself(frame):
    payday = context(frame, "2026-10-30").payday
    assert payday.is_payday
    assert payday.days_since_last_payday == 0 and payday.days_until_next_payday == 0


def test_mid_month_is_not_near_payday(frame):
    payday = context(frame, "2026-11-12").payday
    assert not payday.is_payday
    assert not payday.within_3_days_of_payday
    assert payday.days_until_next_payday == 15


def test_bank_holidays_are_named(frame):
    christmas_eve = context(frame, "2026-12-24").bank_holidays
    assert not christmas_eve.is_bank_holiday
    assert christmas_eve.next_bank_holiday_name == "Christmas Day"
    assert christmas_eve.days_until_next_bank_holiday == 1

    christmas = context(frame, "2026-12-25").bank_holidays
    assert christmas.is_bank_holiday and christmas.bank_holiday_name == "Christmas Day"
    assert christmas.is_long_weekend  # a Friday bank holiday


def test_school_holidays_follow_the_model_flags(frame):
    assert context(frame, "2026-12-24").school_holiday == "Christmas holidays"
    assert context(frame, "2026-11-12").school_holiday is None


def test_weekend_trading_is_friday_to_sunday(frame):
    assert context(frame, "2026-10-30").part_of_weekend_trading  # Friday
    assert context(frame, "2026-11-01").part_of_weekend_trading  # Sunday
    assert not context(frame, "2026-10-29").part_of_weekend_trading  # Thursday


def test_recent_sales_only_when_real(frame):
    day = pd.Timestamp("2026-11-12")
    expected = frame.loc[day - pd.Timedelta(days=7): day - pd.Timedelta(days=1), "total_sales"].mean()

    real = context(frame, "2026-11-12", real_sales=True).recent_sales
    assert real.average_daily_sales_previous_7_days == pytest.approx(expected, abs=0.01)
    assert context(frame, "2026-11-12", real_sales=False).recent_sales is None


def _expected_gap(forecast_error: float) -> tuple[float, str]:
    return abs(forecast_error), ("under_forecast" if forecast_error >= 0 else "over_forecast")


def test_recent_forecast_accuracy_reads_the_forecast_error_history(frame):
    """The direction is pre-computed in code, not left for a reader to work
    out from a sign - a real bug found in live testing had the assistant
    inverting "high"/"low" when it had to interpret the raw signed number
    itself."""
    day = pd.Timestamp("2026-11-12")
    fe = frame["total_sales"] - frame["forecast_sales"]
    expected_yesterday = _expected_gap(fe.loc[day - pd.Timedelta(days=1)])
    expected_last_week = _expected_gap(fe.loc[day - pd.Timedelta(days=7)])
    expected_average = _expected_gap(fe.loc[day - pd.Timedelta(days=7): day - pd.Timedelta(days=1)].mean())

    real = context(frame, "2026-11-12", real_sales=True).recent_forecast_accuracy
    assert real.yesterday.direction == expected_yesterday[1]
    assert real.yesterday.gap_gbp == pytest.approx(expected_yesterday[0], abs=0.01)
    assert real.same_day_last_week.direction == expected_last_week[1]
    assert real.same_day_last_week.gap_gbp == pytest.approx(expected_last_week[0], abs=0.01)
    assert real.average_previous_7_days.direction == expected_average[1]
    assert real.average_previous_7_days.gap_gbp == pytest.approx(expected_average[0], abs=0.01)
    assert context(frame, "2026-11-12", real_sales=False).recent_forecast_accuracy is None


def test_forecast_gap_direction_for_both_signs(frame):
    from src.api.day_context import _forecast_gap

    positive = _forecast_gap(150.0)  # actual beat the forecast
    assert positive.direction == "under_forecast" and positive.gap_gbp == 150.0

    negative = _forecast_gap(-150.0)  # actual fell short of the forecast
    assert negative.direction == "over_forecast" and negative.gap_gbp == 150.0
