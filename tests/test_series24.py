"""
Tests for the Series 24 weekend correction: it only ever learns from EARLIER blocks, leaves the first block
and the other weekdays alone, removes a steady bias, and weights recent history more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series24_weekend_correction as s24


def _days(blocks: int = 5, per_block: int = 28) -> pd.DataFrame:
    n = blocks * per_block
    dates = pd.date_range("2025-01-06", periods=n)  # a Monday
    rng = np.random.default_rng(0)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "fold_id": np.arange(n) // per_block,
                         "actual": rng.uniform(5000, 15000, n), "weekday": dates.day_name()})


def test_hybrid_takes_the_raw_forecast_on_saturdays_and_the_log_forecast_elsewhere():
    wd = np.array(["Friday", "Saturday", "Sunday", "Saturday"])
    out = s24.hybrid(wd, np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))
    assert out.tolist() == [1.0, 20.0, 3.0, 40.0]


def test_a_steady_saturday_and_friday_bias_is_removed_from_the_second_block_onwards():
    days = _days()
    pred = days["actual"].to_numpy().copy()
    for day, bias in (("Saturday", -700.0), ("Friday", 300.0)):
        pred[(days["weekday"] == day).to_numpy()] += bias
    out = s24.correct(days, pred, half_life_weeks=8, strength=1.0)
    for day in ("Saturday", "Friday"):
        m = ((days["weekday"] == day) & (days["fold_id"] >= 1)).to_numpy()
        assert np.allclose(out[m], days["actual"].to_numpy()[m], atol=1e-6)


def test_block_zero_other_weekdays_and_unscoped_days_are_left_alone():
    days = _days()
    pred = days["actual"].to_numpy() + 500.0
    out = s24.correct(days, pred, 8, 1.0)
    assert np.array_equal(out[(days["fold_id"] == 0).to_numpy()], pred[(days["fold_id"] == 0).to_numpy()])
    other = ~days["weekday"].isin(s24.SCOPE).to_numpy()
    assert np.array_equal(out[other], pred[other])
    only_sat = s24.correct(days, pred, 8, 1.0, scope=("Saturday",))
    fri = ((days["weekday"] == "Friday") & (days["fold_id"] >= 1)).to_numpy()
    assert np.array_equal(only_sat[fri], pred[fri])


def test_a_block_is_corrected_only_from_earlier_blocks_never_its_own_or_later_data():
    days = _days()
    pred = days["actual"].to_numpy() - 400.0
    before = s24.correct(days, pred, 8, 0.5)
    changed = days["actual"].to_numpy().copy()
    changed[(days["fold_id"] >= 3).to_numpy()] += 5000.0  # the future changes completely
    days2 = days.copy()  # the forecasts are fixed; only the later actuals move
    days2["actual"] = changed
    after = s24.correct(days2, pred, 8, 0.5)
    early = (days["fold_id"] <= 3).to_numpy()  # block 3's history is blocks 0-2, which did not change
    assert np.array_equal(before[early], after[early])
    assert not np.array_equal(before[(days["fold_id"] == 4).to_numpy()], after[(days["fold_id"] == 4).to_numpy()])


def test_a_shorter_memory_follows_a_recent_change_in_bias_more_closely():
    days = _days(blocks=6)
    wd = days["weekday"].to_numpy()
    sat = wd == "Saturday"
    pred = days["actual"].to_numpy().copy()
    pred[sat & (days["fold_id"].to_numpy() <= 3)] -= 1000.0  # old: too low
    pred[sat & (days["fold_id"].to_numpy() >= 4)] += 600.0   # recent: too high
    target = ((days["fold_id"] == 5).to_numpy()) & sat
    short = s24.correct(days, pred, half_life_weeks=2, strength=1.0)
    long = s24.correct(days, pred, half_life_weeks=64, strength=1.0)
    err_short = np.abs(short[target] - days["actual"].to_numpy()[target]).mean()
    err_long = np.abs(long[target] - days["actual"].to_numpy()[target]).mean()
    assert err_short < err_long


def test_grid_is_nine_variants_and_changes_are_measured_against_the_uncorrected_base():
    days = _days()
    base = days["actual"].to_numpy() * 0.9
    g = s24.grid_table(days, base)
    assert len(g) == 9 and set(g["half_life_weeks"]) == {4, 8, 16} and set(g["strength"]) == {0.25, 0.5, 0.75}
    assert g["uncorrected_MAE"].nunique() == 1
    assert (g["change"] == g["MAE"] - g["uncorrected_MAE"]).all()
    assert (g["change"] < 0).all()  # a steady 10% under-forecast is corrected


def test_fresh_tasks_cover_25_new_seeds_for_both_bases():
    todo = s24.fresh_tasks()
    assert len(todo) == 10 and len({t["id"] for t in todo}) == 10
    for variant in s24.BASE_VARIANTS:
        seeds = sorted(s for t in todo if t["variant"] == variant for s in t["seeds"])
        assert seeds == list(range(100, 125))


def test_change_report_covers_all_days_and_each_weekday():
    days = _days()
    base = days["actual"].to_numpy() * 0.9
    rep = s24.change_report(days, base, s24.correct(days, base, 8, 0.5))
    assert rep["segment"].tolist()[0] == "All scored days" and len(rep) == 8
    assert rep.loc[0, "days"] == int((days["fold_id"] >= 1).sum())
    by = rep.set_index("segment")["change"]
    assert (rep["lo"] <= rep["hi"]).all()
    assert (by[list(s24.SCOPE)] < 0).all() and by["All scored days"] < 0  # a steady under-forecast is corrected on the scoped weekdays
    assert (by[["Monday", "Tuesday", "Wednesday", "Thursday"]] == 0).all()  # and the other weekdays are untouched
