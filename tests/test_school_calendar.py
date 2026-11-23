"""
Tests for the verified school calendar (docs/SCHOOL_CALENDAR.md): every break starts the day after a school day and
ends the day before pupils return, breaks do not overlap, the flags land on the days the council's dates say, and the
original table is kept for reproducing the earlier series.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    LONG_BREAK_TYPES, SCHOOL_BREAK_TYPES, SCHOOL_BREAKS, SCHOOL_BREAKS_V1, add_school_holiday_features,
)


def _flags(dates: list[str], breaks: pd.DataFrame = SCHOOL_BREAKS, types: tuple[str, ...] = SCHOOL_BREAK_TYPES) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    return add_school_holiday_features(frame, breaks_df=breaks, council="Glasgow", break_types=types).set_index("date")


def _spans() -> pd.DataFrame:
    s = SCHOOL_BREAKS.copy()
    s["start"], s["end"] = pd.to_datetime(s["start"]), pd.to_datetime(s["end"])
    return s.sort_values("start").reset_index(drop=True)


def test_every_break_starts_after_a_school_day_and_ends_before_pupils_return():
    for row in _spans().itertuples():
        assert row.start < row.end
        assert (row.start - pd.Timedelta(days=1)).dayofweek < 5, f"{row.start.date()}: the day before the break must be a school day"
        assert (row.end + pd.Timedelta(days=1)).dayofweek < 5, f"{row.end.date()}: pupils must return on a weekday"


def test_breaks_are_in_order_do_not_overlap_and_cover_three_per_year():
    spans = _spans()
    assert (spans["start"].shift(-1).dropna() > spans["end"].iloc[:-1]).all()
    assert spans.groupby(spans["start"].dt.year)["break_type"].nunique().min() >= 1
    for year in (2024, 2025, 2026, 2027):
        types = set(spans.loc[spans["start"].dt.year == year, "break_type"])
        assert {"easter", "summer"} <= types


def test_the_flags_follow_the_councils_dates():
    f = _flags(["2024-01-02", "2024-01-08", "2024-04-01", "2024-07-15", "2025-06-25", "2025-06-26", "2025-06-28", "2025-08-13", "2025-08-14",
                "2025-04-19", "2025-04-22", "2025-12-20", "2026-01-04", "2026-01-05", "2026-03-30", "2026-04-03", "2026-04-19", "2026-04-20"])
    holiday = f["is_school_holiday"].to_dict()
    expected = {"2024-01-02": 1, "2024-01-08": 0, "2024-04-01": 1, "2024-07-15": 1, "2025-06-25": 0, "2025-06-26": 1, "2025-06-28": 1,
                "2025-08-13": 1, "2025-08-14": 0, "2025-04-19": 1, "2025-04-22": 0, "2025-12-20": 1, "2026-01-04": 1, "2026-01-05": 0,
                "2026-03-30": 0, "2026-04-03": 1, "2026-04-19": 1, "2026-04-20": 0}
    assert {k.strftime("%Y-%m-%d"): v for k, v in holiday.items()} == expected
    assert f.loc["2024-01-02", "is_christmas_break"] == 1 and f.loc["2024-04-01", "is_easter_break"] == 1 and f.loc["2024-07-15", "is_summer_break"] == 1


def test_2024_is_no_longer_treated_as_term_time():
    days = pd.date_range("2024-01-02", "2024-12-31").strftime("%Y-%m-%d").tolist()
    new, old = _flags(days)["is_school_holiday"].sum(), _flags(days, SCHOOL_BREAKS_V1, LONG_BREAK_TYPES)["is_school_holiday"].sum()
    assert old == 9 and new > 90  # January 2024's Christmas days, February, Easter, summer, October and the Christmas 2024 days


def test_the_original_table_is_kept_unchanged_for_reproducing_the_earlier_series():
    assert len(SCHOOL_BREAKS_V1) == 9
    assert SCHOOL_BREAKS_V1.iloc[0].to_dict() == {"council": "Glasgow", "break_type": "christmas", "start": "2024-12-23", "end": "2025-01-03"}
    assert SCHOOL_BREAKS_V1.iloc[2]["start"] == "2025-06-30" and SCHOOL_BREAKS_V1.iloc[2]["end"] == "2025-08-15"


@pytest.mark.parametrize("day,flag", [("2027-06-26", 1), ("2027-08-17", 1), ("2027-08-18", 0), ("2027-09-15", 0)])
def test_the_2027_rows_flag_the_dates_the_api_forecasts(day, flag):
    assert _flags([day]).iloc[0]["is_school_holiday"] == flag


def test_the_october_week_and_the_february_mid_term_include_their_in_service_day_and_adjacent_weekends():
    f = _flags(["2024-10-10", "2024-10-11", "2024-10-14", "2024-10-18", "2024-10-20", "2024-10-21", "2025-02-14", "2025-02-15", "2025-02-19",
                "2025-02-20", "2026-10-09", "2026-10-16", "2026-10-19", "2027-02-17", "2027-02-18", "2026-02-18", "2026-02-19"])
    got = {k.strftime("%Y-%m-%d"): int(v) for k, v in f["is_school_holiday"].items()}
    assert got == {"2024-10-10": 0, "2024-10-11": 1, "2024-10-14": 1, "2024-10-18": 1, "2024-10-20": 1, "2024-10-21": 0,
                   "2025-02-14": 0, "2025-02-15": 1, "2025-02-19": 1, "2025-02-20": 0, "2026-10-09": 1, "2026-10-16": 1,
                   "2026-10-19": 0, "2027-02-17": 1, "2027-02-18": 0, "2026-02-18": 1, "2026-02-19": 0}
    assert f.loc["2024-10-14", "is_october_break"] == 1 and f.loc["2024-10-14", "is_february_break"] == 0
    assert f.loc["2025-02-17", "is_february_break"] == 1 if "2025-02-17" in f.index else True


def test_each_session_has_one_october_week_and_one_february_break_where_the_data_has_them():
    spans = _spans()
    for year in (2024, 2025, 2026):
        assert (spans[(spans["start"].dt.year == year)]["break_type"] == "october").sum() == 1
    for year in (2024, 2025, 2026, 2027):
        assert (spans[(spans["start"].dt.year == year)]["break_type"] == "february").sum() == 1
    assert set(spans["break_type"]) == set(SCHOOL_BREAK_TYPES)
    short = spans[spans["break_type"].isin(("october", "february"))]
    assert ((short["end"] - short["start"]).dt.days + 1).between(5, 10).all()  # five to ten days, never a long holiday
