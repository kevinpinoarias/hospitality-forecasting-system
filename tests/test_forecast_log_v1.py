"""
Tests for the forecast log's version v1 (src/forecast_log/v1.py): what the pieces are, that the Hydro-Thursday flag marks only Hydro Thursdays,
and the rule that a version is fitted once and never overwritten.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.forecast_log import v1  # noqa: E402


def test_the_pieces_are_the_ones_in_the_pre_registration():
    assert set(v1.PIECES) == {"log", "log_hydro_thu", "raw", "fri_A_all", "sat_A_lags", "sun_B"}
    assert v1.PIECES["log"] == ([], "log1p") and v1.PIECES["raw"] == ([], "raw")
    assert v1.PIECES["log_hydro_thu"] == (["hydro_thu"], "log1p")
    assert v1.PIECES["sat_A_lags"][1] == "raw" and v1.PIECES["fri_A_all"][1] == "log1p" and v1.PIECES["sun_B"][1] == "log1p"
    assert set(v1.WEEKEND_PIECES) == {"fri_A_all", "sat_A_lags", "sun_B"}


def test_the_cut_off_is_the_day_after_the_export_of_1_november():
    assert v1.FIRST_EXCLUDED_DAY == pd.Timestamp("2026-11-02")
    assert v1.BACKTEST_END == pd.Timestamp("2026-09-07")  # the withdrawn fit's forecasts fill the history from here to the cut-off


def test_the_flag_marks_a_hydro_thursday_and_nothing_else():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-08", "2026-10-03"])})  # Wed (Hydro), Thu (Hydro: Westlife), Thu (none), Sat (Hydro: Kevin Bridges)
    out = v1.add_hydro_thu(frame).set_index("date")["hydro_thu"]
    assert out["2026-10-01"] == 1
    assert out["2026-09-30"] == 0 and out["2026-10-08"] == 0 and out["2026-10-03"] == 0


def test_a_registered_version_is_never_refitted(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "versions.json").write_text(json.dumps({"versions": [{"version": "v1"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already registered"):
        v1.fit_v1(tmp_path)


def test_the_correction_history_loads_in_the_shape_the_correction_needs(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"date": ["2026-05-02", "2026-05-09"], "fold_id": [0, 0], "actual": [1000.0, 1100.0], "weekday": ["Saturday", "Saturday"],
                  "package_weekend_forecast": [1100.0, 1200.0]}).to_csv(path, index=False)
    days, forecast = v1.load_correction_history(path)
    assert list(days.columns) == ["date", "fold_id", "actual", "weekday"]
    assert list(forecast) == [1100.0, 1200.0]
