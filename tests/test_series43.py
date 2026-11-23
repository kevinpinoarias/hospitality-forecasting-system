"""Tests for the Series 43 machinery (src/experiments/run_series43_post_model_horizon.py): origins, the lag checks and the weekday routing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.experiments import run_series43_post_model_horizon as s43


def test_origins_need_180_days_of_data_after_them():
    last = pd.Timestamp("2026-09-06")
    folds = [SimpleNamespace(test_start=last - pd.Timedelta(days=d)) for d in (400, 179, 178)]
    kept = s43.origins(folds, last)
    assert [f.test_start for f in kept] == [last - pd.Timedelta(days=400), last - pd.Timedelta(days=179)]


def known_until(last_known: str) -> pd.Series:
    days = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    return pd.Series(np.where(days <= pd.Timestamp(last_known), 5000.0, np.nan), index=days)


def test_lags_known_counts_calendar_days():
    known = known_until("2025-03-02")
    dates = pd.Series(pd.to_datetime(["2025-03-03", "2025-03-16", "2025-03-17"]))
    # 14 days back from 16 March is 2 March (known); from 17 March it is 3 March (not known)
    assert list(s43.h_lags_known(known, dates, (14,))) == [True, True, False]


def test_route_falls_back_when_recent_sales_are_unknown():
    known = known_until("2025-03-02")  # a Sunday; the forecast starts Monday 3 March
    dates = pd.Series(pd.date_range("2025-03-03", "2025-03-30", freq="D"))
    preds = {name: np.full(len(dates), float(i)) for i, name in enumerate(["log", "raw", "fri_A_all", "sat_A_lags", "sun_B"])}
    out = pd.Series(s43.route(dates, preds, known), index=dates.dt.strftime("%Y-%m-%d"))
    assert out["2025-03-03"] == 0  # Monday: log model
    assert out["2025-03-07"] == 2  # Friday 4 days ahead: 7, 14, 21 and 28 days back are all known -> A_all
    assert out["2025-03-14"] == 0  # Friday 11 days ahead: 7 days back unknown -> log model
    assert out["2025-03-08"] == 3 and out["2025-03-15"] == 3  # Saturdays 5 and 12 days ahead: 14 and 21 days back known -> A_lags
    assert out["2025-03-22"] == 1  # Saturday 19 days ahead: 14 days back unknown -> raw model
    assert out["2025-03-09"] == 4 and out["2025-03-30"] == 4  # Sundays: always B


def test_every_piece_is_a_task_for_every_origin():
    tasks = s43.build_tasks([0, 3])
    assert len(tasks) == 2 * len(s43.PIECES) and len({t["id"] for t in tasks}) == len(tasks)
