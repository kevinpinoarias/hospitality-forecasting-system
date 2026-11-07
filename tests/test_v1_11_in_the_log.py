"""
Tests for running forecast-log version v1.11 beside v1: the version's rows get a model name of their own (the log's de-duplication key has no version), a registered version without a
portfolio arm loads, the registration refuses what the log's rules forbid, Q4 appears in the scores only once v1.11 rows exist, and v1.11 is off in the tracked configuration.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.experiments import run_series35_holdout as h  # noqa: E402
from src.forecast_log import logio, predict, run_forecasts, score_forecasts, v1, v1_11  # noqa: E402


def test_the_extra_arm_has_a_model_name_of_its_own_and_is_not_dropped_as_a_repeat(tmp_path):
    assert predict.extra_arm_name("v1.11") == v1_11.ARM == "hydro_v1.11"
    log = tmp_path / "forecasts.csv"
    base = {"forecast_type": "NW", "target_date": "2026-11-14", "variant": "no_manager"}
    logio._append(log, logio.FORECAST_COLUMNS, [{**{c: "" for c in logio.FORECAST_COLUMNS}, **base, "model": "hydro", "model_version": "v1", "forecast": "100"}])
    new = logio.new_rows_only([{**base, "model": "hydro", "model_version": "v1", "forecast": 101}, {**base, "model": "hydro_v1.11", "model_version": "v1.11", "forecast": 102}], log)
    assert [r["model"] for r in new] == ["hydro_v1.11"]  # the v1 hydro row for that day is already there; the v1.11 row is new


def test_the_tracked_configuration_does_not_run_v1_11():
    settings = run_forecasts.load_settings()
    assert settings["model_version"] == "v1"
    assert not settings.get("also_run")


def test_a_registered_version_without_a_portfolio_arm_loads_with_no_portfolio(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40), "total_sales": rng.uniform(1000, 2000, size=40)})
    models_root, log_repo = tmp_path / "models", tmp_path / "log"
    pieces = {}
    for name in v1_11.HYDRO_ARM_PIECES:
        bag = h.Bag.fit(train, ["a", "b"], "log1p", seeds=[0], thread_count=1)
        bag.save(models_root / "v1.11" / name)
        pieces[name] = {"fingerprint": h.fingerprint(models_root / "v1.11" / name)}
    (log_repo / "models").mkdir(parents=True)
    (log_repo / "models" / "versions.json").write_text(json.dumps({"versions": [{"version": "v1.11", "pieces": pieces}]}), encoding="utf-8")
    models = predict.load_models(log_repo, "v1.11", models_root=models_root)
    assert models.portfolio is None and models.portfolio_fingerprint is None
    assert set(models.pieces) == set(v1_11.HYDRO_ARM_PIECES)


def test_registration_refuses_what_the_log_rules_forbid(tmp_path, monkeypatch):
    log_repo = tmp_path / "log"
    (log_repo / "models").mkdir(parents=True)
    versions = log_repo / "models" / "versions.json"
    versions.write_text(json.dumps({"versions": [{"version": "v1.11"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already registered"):
        v1_11.fit_v1_11(log_repo)
    versions.write_text(json.dumps({"versions": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent version v1 is not registered"):
        v1_11.fit_v1_11(log_repo)
    versions.write_text(json.dumps({"versions": [{"version": "v1", "correction_history_sha256": "x", "pieces": {}}]}), encoding="utf-8")
    monkeypatch.setattr(v1_11, "MODELS_DIR", tmp_path / "models_v1_11")
    (tmp_path / "models_v1_11").mkdir()
    (tmp_path / "models_v1_11" / "seed_00.cbm").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="never overwritten"):
        v1_11.fit_v1_11(log_repo)
    (tmp_path / "models_v1_11" / "seed_00.cbm").unlink()
    monkeypatch.setattr(v1, "correction_history_path", lambda version: tmp_path / f"{version}_missing.csv")
    with pytest.raises(RuntimeError, match="is missing: v1.11 has v1's cut-off"):
        v1_11.fit_v1_11(log_repo)  # v1.11 uses v1's correction history, copied to its own file name
    history = tmp_path / "v1.11_history.csv"
    history.write_text("not the file v1 recorded", encoding="utf-8")
    monkeypatch.setattr(v1, "correction_history_path", lambda version: history)
    with pytest.raises(RuntimeError, match="is not the correction history v1 recorded"):
        v1_11.fit_v1_11(log_repo)


def _scored(models: list[str]) -> pd.DataFrame:
    days = pd.date_range("2026-11-02", periods=28, freq="D")
    rows = []
    for d in days:
        for m in models:
            bias = {"hydro": 100.0, "hydro_v1.11": 100.0 if d.dayofweek != 5 else 60.0, "post": 100.0, "portfolio": 100.0}[m]
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": m, "variant": "no_manager", "on_time": "on_time", "forecast": 1000.0 + bias, "actual_sales": 1000.0})
    frame = pd.DataFrame(rows)
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def test_q4_is_scored_only_once_v1_11_rows_exist_and_is_zero_off_saturdays():
    without, _ = score_forecasts.build_report(_scored(["portfolio", "post", "hydro"]))
    assert "Q4 v1.11 hydro vs v1 hydro" not in without
    with_rows, _ = score_forecasts.build_report(_scored(["portfolio", "post", "hydro", "hydro_v1.11"]))
    assert "Q4 v1.11 hydro vs v1 hydro, Saturdays" in with_rows and "Q4 v1.11 hydro vs v1 hydro, all days" in with_rows
    frame = _scored(["hydro", "hydro_v1.11"])
    sat = {d for d in frame["target_date"].unique() if pd.Timestamp(d).dayofweek == 5}
    q4 = score_forecasts.paired_difference(frame, ("hydro_v1.11", "no_manager"), ("hydro", "no_manager"), "TW", sat)
    assert q4["days"] == 4 and q4["change"] == pytest.approx(-40.0) and q4["better_days"] == 4  # 60 off against 100 off on every Saturday
    every = score_forecasts.paired_difference(frame, ("hydro_v1.11", "no_manager"), ("hydro", "no_manager"), "TW")
    assert every["days"] == 28 and every["better_days"] == 4  # identical on the other 24 days, as v1.11's other pieces are v1's own files
