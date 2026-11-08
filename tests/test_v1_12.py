"""
Tests for forecast-log version v1.12: the closed-day intensity equals the current one except on 25 and 26 December and 1 January (and equals Series 77's `xmas_only` column), the version's pieces and arm name
are as decided, the registration refuses what the log's rules forbid, and Q6 appears in the scores only once v1.12 rows exist.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.experiments import run_series77_christmas_closed_days as s77  # noqa: E402
from src.features import payday_intensity as pi  # noqa: E402
from src.forecast_log import predict, score_forecasts, v1, v1_11, v1_12  # noqa: E402


def _dates() -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", "2027-12-31"))


def test_the_closed_day_intensity_differs_from_the_current_one_only_on_three_days_a_year():
    d = _dates()
    current, fixed = pi.payday_intensity(d), pi.payday_intensity_fixed(d)
    on = lambda m, day: ((d.dt.month == m) & (d.dt.day == day)).to_numpy()  # noqa: E731
    special = on(12, 25) | on(12, 26) | on(1, 1)
    assert np.allclose(current[~special], fixed[~special])
    assert (fixed[on(12, 25)] == 0).all() and (fixed[on(1, 1)] == 0).all() and (fixed[on(12, 26)] == pi.BOXING_DAY_PENALTY).all()
    assert pi.BOXING_DAY_PENALTY < 0 <= fixed[~special].min()  # the penalty sits below every other value, as the reasoning for its size says it must


def test_the_closed_day_intensity_is_series_77s_xmas_only_column():
    frame = pd.DataFrame({"date": _dates()})
    assert np.allclose(pi.add_payday_intensity_fixed(frame)[pi.COLUMN_FIXED], s77.add_columns(frame)["xmas_only"])


def test_the_fixed_column_is_appended_last_and_the_input_is_not_changed():
    frame = pd.DataFrame({"date": pd.date_range("2025-12-20", periods=10), "x": 1})
    out = pi.add_payday_intensity_fixed(frame)
    assert list(out.columns) == ["date", "x", pi.COLUMN_FIXED] and list(frame.columns) == ["date", "x"]


def test_v1_12_changes_only_the_saturday_pieces_and_names_its_arm():
    assert v1_12.PARENT == "v1.11" and v1_12.ARM == predict.extra_arm_name("v1.12") == "hydro_v1.12"
    for name, (extra, target) in v1_12.PIECES.items():
        if name in v1_12.SATURDAY_PIECES:
            assert extra == v1.PIECES[name][0] + [pi.COLUMN_FIXED]  # the closed-day column takes the intensity's place, last
        else:
            assert extra == v1.PIECES[name][0]  # v1's own inputs: the piece is copied from v1.11 unchanged
        assert target == v1.PIECES[name][1]
    assert v1_12.HYDRO_ARM_PIECES == v1_11.HYDRO_ARM_PIECES
    assert v1_12.MODELS_DIR.name == "v1.12"


def test_registration_refuses_what_the_log_rules_forbid(tmp_path, monkeypatch):
    log_repo = tmp_path / "log"
    (log_repo / "models").mkdir(parents=True)
    versions = log_repo / "models" / "versions.json"
    versions.write_text(json.dumps({"versions": [{"version": "v1.12"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already registered"):
        v1_12.fit_v1_12(log_repo)
    versions.write_text(json.dumps({"versions": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent version v1.11 is not registered"):
        v1_12.fit_v1_12(log_repo)
    versions.write_text(json.dumps({"versions": [{"version": "v1.11", "correction_history_sha256": "x", "pieces": {}}]}), encoding="utf-8")
    monkeypatch.setattr(v1_12, "MODELS_DIR", tmp_path / "models_v1_12")
    (tmp_path / "models_v1_12").mkdir()
    (tmp_path / "models_v1_12" / "seed_00.cbm").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="never overwritten"):
        v1_12.fit_v1_12(log_repo)
    (tmp_path / "models_v1_12" / "seed_00.cbm").unlink()
    monkeypatch.setattr(v1, "correction_history_path", lambda version: tmp_path / f"{version}_missing.csv")
    with pytest.raises(RuntimeError, match="is missing"):
        v1_12.fit_v1_12(log_repo)
    history = tmp_path / "v1.12_history.csv"
    history.write_text("not the file v1.11 recorded", encoding="utf-8")
    monkeypatch.setattr(v1, "correction_history_path", lambda version: history)
    with pytest.raises(RuntimeError, match="is not the correction history v1.11 recorded"):
        v1_12.fit_v1_12(log_repo)


def _scored(models: list[str], start: str = "2026-11-02") -> pd.DataFrame:
    rows = []
    for d in pd.date_range(start, periods=28, freq="D"):
        for m in models:
            bias = {"hydro": 100.0, "hydro_v1.11": 100.0, "hydro_v1.12": 100.0 if d.dayofweek != 5 else 50.0, "post": 100.0, "portfolio": 100.0}[m]
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": m, "variant": "no_manager", "on_time": "on_time", "forecast": 1000.0 + bias, "actual_sales": 1000.0})
    frame = pd.DataFrame(rows)
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def test_q6_appears_only_once_v1_12_rows_exist_and_compares_with_v1_11():
    without, _ = score_forecasts.build_report(_scored(["portfolio", "post", "hydro", "hydro_v1.11"]))
    assert "Q6 v1.12 hydro vs v1.11 hydro" not in without and "Q4 v1.11 hydro vs v1 hydro, Saturdays" in without
    with_rows, _ = score_forecasts.build_report(_scored(["portfolio", "post", "hydro", "hydro_v1.11", "hydro_v1.12"]))
    assert "Q6 v1.12 hydro vs v1.11 hydro, Saturdays" in with_rows and "Q6 v1.12 hydro vs v1.11 hydro, all days" in with_rows
    frame = _scored(["hydro_v1.11", "hydro_v1.12"])
    sat = {d for d in frame["target_date"].unique() if pd.Timestamp(d).dayofweek == 5}
    q6 = score_forecasts.paired_difference(frame, ("hydro_v1.12", "no_manager"), ("hydro_v1.11", "no_manager"), "TW", sat)
    assert q6["days"] == 4 and q6["change"] == pytest.approx(-50.0) and q6["better_days"] == 4  # 50 off against 100 off on every Saturday
