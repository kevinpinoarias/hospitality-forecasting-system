"""
Tests for forecast-log versions v1.131 and v1.141 (v1.13 and v1.14 served with the hybrid stand-in for the manager's number): the stand-in is built only from days known when the run is issued, it is the
same date a year earlier inside 24 December to 2 January and the closed-day-aware median elsewhere, the registration copies the parent's files and refuses what the log's rules forbid, and a version with no
stand-in named keeps the old one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.features.serving_features import seasonal_medians  # noqa: E402
from src.forecast_log import predict, standin, v1_131_141  # noqa: E402


def manager_history(first="2024-01-01", last="2026-11-10") -> pd.DataFrame:
    """Two years of manager numbers: 100 + weekday (Monday 100 ... Sunday 106), 0 on 25 December and 1 January."""
    days = pd.date_range(first, last)
    frame = pd.DataFrame({"date": days, "forecast_sales": 100.0 + days.dayofweek.to_numpy()})
    closed = ((days.month == 12) & (days.day == 25)) | ((days.month == 1) & (days.day == 1))
    frame.loc[closed, "forecast_sales"] = 0.0
    return frame


def test_outside_the_christmas_week_the_hybrid_is_the_closed_day_aware_median():
    full = manager_history()
    targets = [pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-20"), pd.Timestamp("2026-01-10") + pd.DateOffset(years=0)]
    got = standin.hybrid_standin(full, pd.Timestamp("2026-11-10"), targets)
    assert got[0] == 100.0 and got[1] == 104.0  # a Monday and a Friday in November: the weekday's number
    assert got[2] == 100.0 + pd.Timestamp("2026-01-10").dayofweek  # 10 January: no closed day in its window, so the plain weekday median


def test_inside_the_christmas_week_it_is_the_manager_number_of_the_same_date_a_year_earlier():
    full = manager_history()
    full.loc[full["date"] == pd.Timestamp("2025-12-24"), "forecast_sales"] = 777.0
    full.loc[full["date"] == pd.Timestamp("2025-12-28"), "forecast_sales"] = 555.0
    targets = [pd.Timestamp("2026-12-24"), pd.Timestamp("2026-12-25"), pd.Timestamp("2026-12-28"), pd.Timestamp("2027-01-01"), pd.Timestamp("2027-01-02"), pd.Timestamp("2027-01-03")]
    got = standin.hybrid_standin(full, pd.Timestamp("2026-11-10"), targets)
    assert got[0] == 777.0 and got[1] == 0.0 and got[2] == 555.0  # the same dates of 2025, the closed day at zero
    assert got[3] == 0.0  # 1 January 2026 was closed
    assert got[4] == full.loc[full["date"] == pd.Timestamp("2026-01-02"), "forecast_sales"].iloc[0]
    assert got[5] == 100.0 + pd.Timestamp("2027-01-03").dayofweek  # 3 January is outside the week: the closed-day-aware median


def test_a_christmas_week_day_with_no_year_ago_number_falls_back_to_the_closed_day_aware_median():
    full = manager_history(first="2025-06-01")  # no 2024 numbers: 24 December 2025 has no 365-day-earlier number
    got = standin.hybrid_standin(full, pd.Timestamp("2026-11-10"), [pd.Timestamp("2025-12-24"), pd.Timestamp("2026-12-24")])
    assert got[1] == full.loc[full["date"] == pd.Timestamp("2025-12-24"), "forecast_sales"].iloc[0]
    assert not np.isnan(got).any()


def test_the_closed_days_are_left_out_of_the_median_and_the_closed_dates_themselves_forecast_zero():
    full = manager_history()
    days = pd.Timestamp("2026-12-23")  # a Wednesday just before Christmas: S0 would pull the 25 December zero in, S1 does not
    history = full[full["date"] <= pd.Timestamp("2026-11-10")]
    s0 = seasonal_medians(history, [days], "forecast_sales")[0]
    s1 = standin.hybrid_standin(full, pd.Timestamp("2026-11-10"), [days])[0]
    assert s1 == 102.0 and s0 <= s1
    # 25 December and 1 January are zero even outside a year-ago number
    out = standin.hybrid_standin(manager_history(first="2025-06-01"), pd.Timestamp("2026-11-10"), [pd.Timestamp("2026-12-25"), pd.Timestamp("2027-01-01")])
    assert out[0] == 0.0 and out[1] == 0.0


def test_nothing_after_the_last_known_day_is_read():
    full = manager_history(last="2027-02-01")
    full.loc[full["date"] > pd.Timestamp("2026-11-10"), "forecast_sales"] = 9999.0  # numbers that arrive later must not matter
    base = standin.hybrid_standin(manager_history(last="2026-11-10"), pd.Timestamp("2026-11-10"), [pd.Timestamp("2026-11-16"), pd.Timestamp("2026-12-24")])
    got = standin.hybrid_standin(full, pd.Timestamp("2026-11-10"), [pd.Timestamp("2026-11-16"), pd.Timestamp("2026-12-24")])
    assert (got == base).all()


def test_the_versions_are_named_and_pointed_at_their_parents():
    assert v1_131_141.PARENTS == {"v1.131": "v1.13", "v1.141": "v1.14"}
    assert v1_131_141.arm_name("v1.131") == predict.extra_arm_name("v1.131") == "hydro_v1.131"
    assert predict.Models("v1.99", None, {}, None, {}).standin == "S0"  # a version with no stand-in named keeps the weekday-and-season median


def test_the_registration_refuses_an_unknown_version_a_registered_one_and_a_missing_parent(tmp_path: Path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "versions.json").write_text(json.dumps({"versions": [{"version": "v1.131", "parent": "v1.13"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not one of"):
        v1_131_141.register(tmp_path, "v1.15")
    with pytest.raises(RuntimeError, match="already registered"):
        v1_131_141.register(tmp_path, "v1.131")
    with pytest.raises(RuntimeError, match="parent version v1.14 is not registered"):
        v1_131_141.register(tmp_path, "v1.141")


# --------------------------------------------------------------------------- Q9 and Q10 in the scores


def _scored(models: dict[str, float]) -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2026-12-21", "2027-01-10"):
        for m, bias in models.items():
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": m, "variant": "no_manager", "on_time": "on_time", "forecast": 1000.0 + bias, "actual_sales": 1000.0})
    frame = pd.DataFrame(rows)
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def test_q9_and_q10_appear_only_once_their_rows_exist_and_are_read_on_the_christmas_week():
    from src.forecast_log import score_forecasts

    base = {"portfolio": 200.0, "post": 200.0, "hydro": 200.0, "hydro_v1.13": 200.0, "hydro_v1.14": 200.0}
    without, _ = score_forecasts.build_report(_scored(base))
    assert "Q9 v1.131" not in without and "Q10 v1.141" not in without
    with_both, _ = score_forecasts.build_report(_scored({**base, "hydro_v1.131": 100.0, "hydro_v1.141": 100.0}))
    assert "Q9 v1.131 hydro vs v1.13 hydro, 24 December to 2 January" in with_both and "Q10 v1.141 hydro vs v1.14 hydro, all other days" in with_both
    groups = dict(score_forecasts.comparison_groups("Q10", sorted(_scored(base)["target_date"].unique())))
    assert len(groups["24 December to 2 January"]) == 10  # 24-31 December and 1-2 January
    assert score_forecasts.comparison_groups("Q4", ["2026-07-04"])[1][0] == "Saturdays"  # the older questions keep their groups
