"""
Tests for the walk-forward-safe December intensity feature in
src/features/build_features.py. The single most important property here
is that no year's December value is ever informed by that same year's
own outcomes - these tests exist to catch exactly that kind of leakage
directly, not just check the happy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import add_december_intensity_feature


def make_two_year_df() -> pd.DataFrame:
    """Two full years, Mondays always sell 100, and December always spikes
    to 200 on Mondays (day-of-month deliberately irrelevant here - single
    weekday, single value, so the expected index is exactly computable by
    hand)."""
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    dow = df["date"].dt.dayofweek
    is_dec = df["date"].dt.month == 12
    # Base level: 100 on every day regardless of weekday, except December
    # Mondays specifically spike to 200 (a real, decidable pattern).
    df["total_sales"] = 100.0
    df.loc[is_dec & (dow == 0), "total_sales"] = 200.0
    return df


def test_first_december_has_no_prior_year_and_is_nan():
    df = make_two_year_df()
    out = add_december_intensity_feature(df)
    first_dec = out[(out["date"].dt.year == 2024) & (out["date"].dt.month == 12)]
    assert first_dec["december_intensity_index"].isna().all()


def test_non_december_rows_get_neutral_value():
    df = make_two_year_df()
    out = add_december_intensity_feature(df)
    non_dec = out[out["date"].dt.month != 12]
    assert (non_dec["december_intensity_index"] == 1.0).all()


def test_second_december_uses_first_decembers_pattern_not_its_own():
    """The core leakage guard: 2025's December values must come from 2024's
    December pattern only - never from 2025's own December sales, even
    though those exist in the same input dataframe."""
    df = make_two_year_df()
    # Sabotage 2025's December data with an absurd, distinctive value -
    # if this leaks into the computed index, the test will catch it.
    mask_2025_dec = (df["date"].dt.year == 2025) & (df["date"].dt.month == 12)
    df.loc[mask_2025_dec, "total_sales"] = 999_999.0

    out = add_december_intensity_feature(df)
    second_dec = out[(out["date"].dt.year == 2025) & (out["date"].dt.month == 12)]

    # None of the computed index values should reflect the sabotaged
    # 999,999 figure in any way (which would show up as an enormous index).
    assert (second_dec["december_intensity_index"] < 100).all()
    assert not second_dec["december_intensity_index"].isna().all()


def test_index_reflects_a_day_of_month_pattern_independent_of_weekday():
    """The index is a DAY-OF-MONTH lookup (weekday-adjusted first, then
    grouped by day-of-month) - not a weekday-tied pattern. December 1st
    falls on a different weekday in 2024 vs. 2025, so this test spikes a
    specific day-of-month (the 23rd) by a fixed multiple of whatever that
    weekday's normal level is, and checks the 2025 index picks up
    2024's 23rd specifically - not whatever weekday the 23rd happens to
    be in 2025."""
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    df["total_sales"] = 100.0
    is_dec_23_2024 = (df["date"].dt.year == 2024) & (df["date"].dt.month == 12) & (df["date"].dt.day == 23)
    df.loc[is_dec_23_2024, "total_sales"] = 300.0  # 3x the normal 100 level for that specific date

    out = add_december_intensity_feature(df)
    dec_23_2025 = out[(out["date"].dt.year == 2025) & (out["date"].dt.month == 12) & (out["date"].dt.day == 23)]
    # Not exactly 3.0: the weekday baseline is a plain historical mean (a
    # documented simplification) that includes 2024-12-23's own spiked
    # value among that weekday's prior occurrences, diluting it slightly
    # (~2.94, not 3.0, for this ~2-year sample). Still clearly reflects
    # the real spike, nowhere near the neutral 1.0.
    assert 2.8 < dec_23_2025["december_intensity_index"].iloc[0] < 3.0

    # A different December day, never spiked, should stay at the neutral ~1.0.
    dec_10_2025 = out[(out["date"].dt.year == 2025) & (out["date"].dt.month == 12) & (out["date"].dt.day == 10)]
    assert np.allclose(dec_10_2025["december_intensity_index"], 1.0, atol=1e-6)
