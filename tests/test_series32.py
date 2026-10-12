"""
Tests for Series 32: the breaks in the data, the correction arithmetic, that forward-chaining only ever learns from EARLIER breaks,
that leave-one-out never learns from the break being corrected, and the training rows left out of each leave-one-break-out fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series32_first_week_correction as s32


def _days(start: str = "2025-03-01", n: int = 400) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "actual": np.full(n, 10000.0)})


def test_the_breaks_with_data_are_the_easter_and_summer_ones_with_all_their_days():
    b = s32.breaks_in_data(pd.Series(pd.date_range("2024-01-16", "2026-09-06")))
    assert b["name"].tolist() == ["Easter 2024", "Summer 2024", "Easter 2025", "Summer 2025", "Easter 2026", "Summer 2026"]
    short = s32.breaks_in_data(pd.Series(pd.date_range("2025-01-01", "2025-06-30")))
    assert short["name"].tolist() == ["Easter 2025"]  # summer 2025 starts on 26 June, so its first week is not all there


def test_scope_dates_are_the_first_week_and_optionally_the_week_before():
    start = pd.Timestamp("2025-06-26")
    first = s32.scope_dates(start, "first week")
    assert first[0] == start and first[-1] == pd.Timestamp("2025-07-02") and len(first) == 7
    both = s32.scope_dates(start, "first week + last week of term")
    assert both[0] == pd.Timestamp("2025-06-19") and both[-1] == pd.Timestamp("2025-07-02") and len(both) == 14


def test_the_correction_arithmetic_for_both_forms():
    pred = np.array([100.0, 200.0])
    hist_pred, hist_actual = np.array([100.0, 100.0]), np.array([110.0, 130.0])  # forecasts 10 and 30 too low
    assert np.allclose(s32.shifted(pred, hist_pred, hist_actual, "additive", 1.0), [120.0, 220.0])
    assert np.allclose(s32.shifted(pred, hist_pred, hist_actual, "additive", 0.5), [110.0, 210.0])
    assert np.allclose(s32.shifted(pred, hist_pred, hist_actual, "ratio", 1.0), [120.0, 240.0])  # the ratio is 1.2
    assert np.allclose(s32.shifted(pred, hist_pred, hist_actual, "ratio", 0.5), [110.0, 220.0])


def test_forward_chaining_learns_only_from_earlier_breaks_and_leaves_the_first_one_alone():
    days = _days()
    breaks = pd.DataFrame({"name": ["A", "B", "C"], "break_type": ["easter"] * 3,
                           "start": pd.to_datetime(["2025-04-05", "2025-06-26", "2025-12-01"])})
    pred = np.full(len(days), 9000.0)
    first_dates = [d.strftime("%Y-%m-%d") for d in s32.scope_dates(breaks["start"][0], "first week")]
    first_idx = days.index[days["date"].isin(first_dates)].to_numpy()
    pred[first_idx] = 8000.0  # 2,000 too low in the first break
    out, hit = s32.forward_chained(days, pred, breaks, "additive", 1.0, "first week")
    assert "A" not in hit and set(hit) == {"B", "C"}
    assert np.array_equal(out[first_idx], pred[first_idx])  # the first break has no history to learn from
    assert np.allclose(out[hit["B"]], pred[hit["B"]] + 2000.0)  # it learned the first break's shortfall
    days2 = days.copy()
    days2.loc[hit["C"], "actual"] = 99999.0  # change the last break completely
    out2, _ = s32.forward_chained(days2, pred, breaks, "additive", 1.0, "first week")
    assert np.array_equal(out2[hit["B"]], out[hit["B"]])  # the earlier break's correction does not move


def test_leave_one_out_never_learns_from_the_break_being_corrected():
    rows = []
    for name, start, gap in [("A", "2025-04-05", 2000.0), ("B", "2025-06-26", 1000.0), ("C", "2026-04-03", 500.0)]:
        for k in range(7):
            d = pd.Timestamp(start) + pd.Timedelta(days=k)
            rows.append({"break": name, "start": pd.Timestamp(start), "date": d, "actual": 10000.0 + gap, "pred": 10000.0})
    frame = pd.DataFrame(rows)
    out = s32.leave_one_out(frame, "additive", 1.0, "first week")
    corrected = out.groupby("break")["corrected"].first()
    assert np.isclose(corrected["A"], 10000.0 + (1000.0 + 500.0) / 2)  # shifted down by minus the others' average shortfall
    assert np.isclose(corrected["C"], 10000.0 + (2000.0 + 1000.0) / 2)


def test_each_leave_one_break_out_fit_leaves_out_42_days_and_predicts_the_14_around_the_start():
    purge_lo, purge_hi, test_lo, test_hi = s32.lobo_windows(pd.Timestamp("2025-06-26"))
    assert (purge_hi - purge_lo).days == 42 and (test_hi - test_lo).days == 14
    assert purge_lo <= test_lo and test_hi <= purge_hi and purge_lo == pd.Timestamp("2025-06-05")


def test_the_cluster_bootstrap_centres_on_the_day_weighted_mean_and_the_tasks_cover_every_break():
    change, days = np.array([-10.0, -20.0, 5.0]), np.array([7.0, 7.0, 14.0])
    mean, lo, hi = s32.cluster_bootstrap(change, days)
    assert np.isclose(mean, (-70 - 140 + 70) / 28) and lo <= hi
    tasks = s32.lobo_tasks(s32.breaks_in_data(pd.Series(pd.date_range("2024-01-16", "2026-09-06"))))
    assert len(tasks) == 30 and len({t["id"] for t in tasks}) == 30
