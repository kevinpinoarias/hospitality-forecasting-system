"""
Tests for the Series 28 manager diagnostic: the correlation interval brackets a known relationship, the blend table
starts from the package itself, and the manager-wins list ranks by how much closer the manager was, keeps the
weekend filter and leaves closure days out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series28_manager_diagnostic as s28


def _days(n: int = 56) -> pd.DataFrame:
    dates = pd.date_range("2025-03-03", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"date": dates, "weekday": dates.day_name(), "actual": rng.uniform(5000, 15000, n),
                         "manual": rng.uniform(5000, 15000, n), "closure": False, "holiday": "", "is_school_holiday": 0,
                         "rain_mm": 0.0, "max_temp": 12.0, "same_weekday_4wk_avg": 9000.0})


def test_correlation_interval_brackets_a_strong_relationship_and_a_zero_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    r, lo, hi = s28.corr_with_interval(x, x + rng.normal(0, 0.3, 200))
    assert lo <= r <= hi and lo > 0.8
    r0, lo0, hi0 = s28.corr_with_interval(x, rng.normal(size=200))
    assert lo0 < 0 < hi0


def test_blend_table_starts_from_the_package_and_a_perfect_manager_pulls_the_error_down():
    days = _days()
    package = days["actual"].to_numpy() + 500.0
    days["manual"] = days["actual"]  # a perfect manager
    t = s28.blend_table(days, package).set_index("manager_share_%")
    assert t.loc[0, "all_days_MAE"] == 500.0
    assert t["all_days_MAE"].is_monotonic_decreasing and abs(t.loc[30, "all_days_MAE"] - 350.0) < 1e-6


def test_manager_wins_ranks_by_advantage_filters_weekends_and_skips_closures():
    days = _days()
    package = days["actual"].to_numpy() + 1000.0
    days["manual"] = days["actual"] + 100.0  # the manager is closer everywhere
    friday = days.index[days["weekday"] == "Friday"][0]
    monday = days.index[days["weekday"] == "Monday"][0]
    package[friday] = days.loc[friday, "actual"] + 3000.0   # the biggest weekend advantage
    package[monday] = days.loc[monday, "actual"] + 5000.0   # bigger, but a Monday
    days.loc[days.index[days["weekday"] == "Saturday"][0], "closure"] = True
    wk = s28.manager_wins(days, package, weekend_only=True)
    assert wk.iloc[0]["Manager advantage"] == 2900.0 and wk["Date"].str[:3].isin(["Fri", "Sat", "Sun"]).all()
    assert len(wk) == s28.LIST_LENGTH and (wk["What happened?"] == "").all()
    overall = s28.manager_wins(days, package, weekend_only=False)
    assert overall.iloc[0]["Manager advantage"] == 4900.0
    closed = days.loc[days["closure"], "date"].dt.strftime("%a %d %b %Y").iloc[0]
    assert closed not in set(wk["Date"]) | set(overall["Date"])


def test_the_package_uses_the_series_27_choices():
    assert s28.CHOSEN == {"Friday": "A_summary", "Saturday": "A_lags", "Sunday": "B"}
