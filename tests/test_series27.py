"""
Tests for the Series 27 per-weekday choice: each weekend day picks its own best option on the selection rows only,
ties keep the current setup, and the assembly changes only the chosen weekdays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series27_weekend_per_weekday as s27


def _days() -> pd.DataFrame:
    dates = pd.date_range("2025-03-03", periods=28)  # four weeks from a Monday
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "fold_id": np.repeat([0, 0, 1, 1], 7), "actual": 1000.0 + np.arange(28),
                         "weekday": dates.day_name()})


def _options(days: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    a = days["actual"].to_numpy()
    out = {}
    for day in ("Friday", "Saturday", "Sunday"):
        out[day] = {"none": a + 100.0, "A_lags": a + 10.0 if day == "Saturday" else a + 500.0, "B": a + 100.0}
    return out


def test_each_weekday_picks_its_own_best_option_and_ties_keep_none():
    days = _days()
    chosen = s27.choose_options(days, _options(days), np.ones(len(days), bool))
    assert chosen == {"Friday": "none", "Saturday": "A_lags", "Sunday": "none"}  # B ties with none, none wins ties


def test_the_choice_uses_only_the_selection_rows():
    days = _days()
    a = days["actual"].to_numpy()
    opts = _options(days)
    # make A_lags look best on the evaluation block only (blocks 1) for Friday
    late = days["fold_id"].to_numpy() == 1
    opts["Friday"]["A_lags"] = np.where(late, a, a + 900.0)
    chosen = s27.choose_options(days, opts, ~late)
    assert chosen["Friday"] == "none"
    assert s27.choose_options(days, opts, late)["Friday"] == "A_lags"


def test_assembly_changes_only_the_chosen_weekdays_and_leaves_monday_to_thursday_alone():
    days = _days()
    hybrid = days["actual"].to_numpy() + 100.0
    out = s27.assemble_weekend(days, hybrid, _options(days), {"Friday": "none", "Saturday": "A_lags", "Sunday": "none"})
    wd = days["weekday"].to_numpy()
    assert np.array_equal(out[wd != "Saturday"], hybrid[wd != "Saturday"])
    assert np.allclose(out[wd == "Saturday"], days["actual"].to_numpy()[wd == "Saturday"] + 10.0)


def test_the_tables_compare_the_assembly_with_the_hybrid_on_the_rows_given():
    days = _days()
    hybrid = days["actual"].to_numpy() + 100.0
    assembled = s27.assemble_weekend(days, hybrid, _options(days), {"Friday": "none", "Saturday": "A_lags", "Sunday": "none"})
    t = s27.table_for(days, np.ones(len(days), bool), hybrid, assembled).set_index("segment")
    assert t.loc["Saturday", "change"] == -90.0 and t.loc["Friday", "change"] == 0.0 and t.loc["Sunday", "change"] == 0.0
    assert t.loc["Friday to Sunday", "days"] == 12
    rep = s27.evaluation_report(days, days["fold_id"].to_numpy() == 1, hybrid, assembled)
    assert rep["segment"].tolist() == ["Friday to Sunday", "Friday", "Saturday", "Sunday"]
    assert rep.set_index("segment").loc["Saturday", "change"] == -90.0


def test_the_split_and_targets_follow_the_plan():
    assert s27.SELECTION_BLOCKS == 5 and s27.OPTIONS == ["none", "A_all", "A_summary", "A_lags", "B", "AB"]
    assert s27.DAY_TARGET == {"Friday": "log1p", "Saturday": "raw", "Sunday": "log1p"}
