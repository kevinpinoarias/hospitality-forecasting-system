"""
Tests for the fixed-calendar Christmas period feature in
src/features/build_features.py: binary flag for December 1st-31st.
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features import add_christmas_period_feature


def test_flags_every_day_in_december():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-12-01", "2025-12-15", "2025-12-31"])})
    out = add_christmas_period_feature(df)
    assert list(out["is_christmas_period"]) == [1, 1, 1]


def test_excludes_days_outside_december():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-11-30", "2026-01-01", "2025-06-15"])})
    out = add_christmas_period_feature(df)
    assert list(out["is_christmas_period"]) == [0, 0, 0]


def test_boundary_days_correct():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-11-30", "2025-12-01", "2025-12-31", "2026-01-01"])})
    out = add_christmas_period_feature(df)
    assert list(out["is_christmas_period"]) == [0, 1, 1, 0]


def test_applies_consistently_across_different_years():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-12-10", "2025-12-10", "2026-12-10"])})
    out = add_christmas_period_feature(df)
    assert list(out["is_christmas_period"]) == [1, 1, 1]
