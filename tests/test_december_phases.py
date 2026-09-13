"""
Tests for the named December phase flags in
src/features/build_features.py (DECEMBER_PHASES / add_december_phase_features).
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features import DECEMBER_PHASES, add_december_phase_features


def flags_for(date_str: str) -> dict:
    df = pd.DataFrame({"date": pd.to_datetime([date_str])})
    out = add_december_phase_features(df)
    return {name: int(out[name].iloc[0]) for name in DECEMBER_PHASES}


def test_party_season_boundaries():
    assert flags_for("2025-11-30")["is_party_season"] == 0
    assert flags_for("2025-12-01")["is_party_season"] == 1
    assert flags_for("2025-12-21")["is_party_season"] == 1
    assert flags_for("2025-12-22")["is_party_season"] == 0


def test_pre_christmas_peak_is_22_and_23_only():
    assert flags_for("2025-12-21")["is_pre_christmas_peak"] == 0
    assert flags_for("2025-12-22")["is_pre_christmas_peak"] == 1
    assert flags_for("2025-12-23")["is_pre_christmas_peak"] == 1
    assert flags_for("2025-12-24")["is_pre_christmas_peak"] == 0


def test_christmas_eve_dip_is_24_only():
    flags = flags_for("2025-12-24")
    assert flags["is_christmas_eve_dip"] == 1
    assert flags["is_pre_christmas_peak"] == 0
    assert flags["is_boxing_day_lull"] == 0


def test_boxing_day_lull_is_26_only_not_25():
    assert flags_for("2025-12-25")["is_boxing_day_lull"] == 0  # covered by is_bank_holiday elsewhere
    assert flags_for("2025-12-26")["is_boxing_day_lull"] == 1
    assert flags_for("2025-12-27")["is_boxing_day_lull"] == 0


def test_post_christmas_recovery_is_27_28():
    assert flags_for("2025-12-27")["is_post_christmas_recovery"] == 1
    assert flags_for("2025-12-28")["is_post_christmas_recovery"] == 1
    assert flags_for("2025-12-26")["is_post_christmas_recovery"] == 0
    assert flags_for("2025-12-29")["is_post_christmas_recovery"] == 0


def test_hogmanay_peak_is_29_30():
    assert flags_for("2025-12-29")["is_hogmanay_peak"] == 1
    assert flags_for("2025-12-30")["is_hogmanay_peak"] == 1
    assert flags_for("2025-12-28")["is_hogmanay_peak"] == 0
    assert flags_for("2025-12-31")["is_hogmanay_peak"] == 0


def test_new_years_eve_is_31_only():
    assert flags_for("2025-12-31")["is_new_years_eve"] == 1
    assert flags_for("2026-01-01")["is_new_years_eve"] == 0


def test_non_december_dates_have_all_flags_zero():
    flags = flags_for("2025-06-15")
    assert all(v == 0 for v in flags.values())


def test_every_december_day_has_at_most_one_active_phase():
    """No two phases should ever overlap - each December day belongs to
    at most one named phase (day 25 belongs to none, by design)."""
    dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    out = add_december_phase_features(df)
    phase_cols = list(DECEMBER_PHASES)
    active_count = out[phase_cols].sum(axis=1)
    assert (active_count <= 1).all()


def test_applies_consistently_across_different_years():
    for year in (2024, 2025, 2026):
        assert flags_for(f"{year}-12-30")["is_hogmanay_peak"] == 1
