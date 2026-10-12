"""
Tests for the Series 30 transition features: counters and flags land on the right days of a break, they depend only on the
date and the calendar, the capped counters never exceed 14, and the variants append their columns to the final model's 38.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series30_transition_features as s30
from src.models.final_model import FINAL_FEATURES


def _features(dates: list[str]) -> pd.DataFrame:
    return s30.transition_features(pd.Series(pd.to_datetime(dates))).set_axis(pd.to_datetime(dates))


def test_inside_a_long_break_the_counters_count_from_the_start_and_down_to_the_end():
    f = _features(["2025-06-26", "2025-06-28", "2025-07-02", "2025-07-03", "2025-08-06", "2025-08-07", "2025-08-13"])
    assert f["tf_days_since_start"].tolist() == [0, 2, 6, 7, 41, 42, 48]
    assert f["tf_days_to_end"].tolist() == [48, 46, 42, 41, 7, 6, 0]
    assert f["tf_first_week"].tolist() == [1, 1, 1, 0, 0, 0, 0]       # the first 7 days: 26 Jun to 2 Jul
    assert f["tf_last_week_break"].tolist() == [0, 0, 0, 0, 0, 1, 1]  # the last 7 days: 7 to 13 Aug
    assert f["tf_days_to_start"].isna().all() and (f["tf_last_week_term"] == 0).all()


def test_before_a_long_break_the_days_to_start_and_the_last_week_of_term_flag_are_set():
    f = _features(["2025-06-18", "2025-06-19", "2025-06-23", "2025-06-25", "2025-09-05"])
    assert f["tf_days_to_start"].tolist()[:4] == [8, 7, 3, 1]
    assert f["tf_last_week_term"].tolist() == [0, 1, 1, 1, 0]
    assert f["tf_days_since_start"].isna().all() and f["tf_days_to_end"].isna().all()
    assert f.loc["2025-09-05", "tf_days_to_start"] == 35  # the October week, 10 Oct 2025


def test_a_short_break_gets_counters_but_no_week_flags():
    f = _features(["2025-10-09", "2025-10-10", "2025-10-14", "2025-10-19", "2025-10-20"])
    assert f["tf_days_since_start"].tolist()[1:4] == [0, 4, 9] and f["tf_days_to_end"].tolist()[1:4] == [9, 5, 0]
    assert f.loc["2025-10-09", "tf_days_to_start"] == 1
    assert (f[["tf_first_week", "tf_last_week_term", "tf_last_week_break"]].iloc[1:4] == 0).all().all()
    assert f.loc["2025-10-09", "tf_last_week_term"] == 0  # the next break is a short one, so the week flag stays off


def test_capped_counters_never_exceed_14_and_equal_the_plain_ones_below_it():
    f = _features(pd.date_range("2025-05-01", "2025-09-30").strftime("%Y-%m-%d").tolist())
    for plain, capped in zip(s30.COUNTERS, s30.CAPPED):
        assert f[capped].max() == 14 and (f[capped].dropna() <= 14).all()
        small = f[plain] <= 14
        assert (f.loc[small, plain] == f.loc[small, capped]).all()
        assert (f.loc[f[plain] > 14, capped] == 14).all()


def test_the_flags_agree_with_the_autopsys_phase_labels_for_the_long_holidays():
    from src.experiments.run_error_autopsy import school_phase
    from src.experiments.run_series29_school_calendar import LONG_BREAKS

    days = pd.date_range("2025-05-01", "2025-09-30")
    f = _features(days.strftime("%Y-%m-%d").tolist())
    for d in days:
        label = school_phase(d, LONG_BREAKS)
        assert f.loc[d, "tf_first_week"] == int(label == "first week of a break")
        assert f.loc[d, "tf_last_week_term"] == int(label == "last week of term")
        assert f.loc[d, "tf_last_week_break"] == int(label == "last week of a break")


def test_variants_append_their_columns_after_the_final_models_38():
    assert {v: len(spec["extra"]) for v, spec in s30.VARIANTS.items()} == {"TF_all": 6, "TF_flags": 3, "TF_counters": 3, "TF_capped": 3}
    assert not set(sum((spec["extra"] for spec in s30.VARIANTS.values()), [])) & set(FINAL_FEATURES)
    assert len(s30.all_variant_targets()) == 8 and len(s30.make_tasks(s30.all_variant_targets(), list(range(25)))) == 40


def test_with_transition_adds_columns_without_touching_the_rest():
    frame = pd.DataFrame({"date": pd.date_range("2025-06-20", periods=10), "x": np.arange(10.0)})
    out = s30.with_transition(frame)
    assert out[["date", "x"]].equals(frame)
    assert set(s30.COUNTERS + s30.FLAGS + s30.CAPPED) <= set(out.columns)


def test_the_bar_needs_an_interval_below_zero_on_the_transition_weeks_and_nothing_worse():
    def rep(trans, overall, phases):
        rows = [("Transition weeks (3 phases)", *trans), ("All 585 days", *overall)] + [(p, *phases) for p in s30.TRANSITION_PHASES + s30.OTHER_PHASES]
        return pd.DataFrame(rows, columns=["segment", "change", "lo", "hi"])

    assert s30.passes_bar(rep((-40, -80, -5), (-5, -12, 2), (-3, -20, 10))) == (True, True, True)
    assert s30.passes_bar(rep((-40, -80, 5), (-5, -12, 2), (-3, -20, 10)))[0] is False
    assert s30.passes_bar(rep((-40, -80, -5), (6, 1, 12), (-3, -20, 10)))[1] is False
    assert s30.passes_bar(rep((-40, -80, -5), (-5, -12, 2), (8, 2, 14)))[2] is False
