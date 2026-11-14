"""
Tests for replaying a logged run with another version (src/forecast_log/replay_run.py): the run's own weather is read back as recorded, an unknown run is refused,
the check that the replayed v1 numbers equal the logged ones to the penny, and the back-run of a newly registered version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.forecast_log import replay_run


def _log(tmp_path):
    (tmp_path / "log").mkdir()
    pd.DataFrame({"run_id": ["a", "a", "b"], "issue_time": ["2026-11-02T16:49:32+01:00"] * 2 + ["2026-11-09T12:30:00+01:00"], "trigger": ["monday"] * 3, "status": ["ok"] * 3,
                  "reason": [""] * 3, "data_through": ["2026-11-01", "2026-11-01", "2026-11-08"], "forecast_types_made": ["NW+TW"] * 3, "code_commit": ["x"] * 3}).drop_duplicates("run_id").to_csv(
        tmp_path / "log" / "runs.csv", index=False)
    pd.DataFrame({"run_id": ["a", "a", "b"], "date": ["2026-11-02", "2026-11-03", "2026-11-09"], "max_temp": [8.5, 10.3, 9.0], "rain_mm": [0.0, 0.0, 1.2],
                  "sun_hours": [8.1, 0.0, 2.0], "fetched_at": ["t"] * 3}).to_csv(tmp_path / "log" / "weather_used.csv", index=False)
    return tmp_path


def test_the_weather_is_the_one_the_run_recorded(tmp_path):
    weather = replay_run.run_weather(_log(tmp_path), "a")
    assert list(weather.columns) == ["date", "max_temp", "rain_mm", "sun_hours"]
    assert list(weather["date"]) == [pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03")]  # run b's weather is not mixed in
    assert weather["max_temp"].tolist() == [8.5, 10.3]


def test_a_run_that_is_not_in_the_log_is_refused(tmp_path):
    log = _log(tmp_path)
    assert replay_run.run_row(log, "a")["data_through"] == "2026-11-01"
    with pytest.raises(RuntimeError, match="is not in"):
        replay_run.run_row(log, "zzz")


def _replayed(offset: float) -> pd.DataFrame:
    out = pd.DataFrame({"portfolio": [4789.391, 5086.2], "post": [4825.36, 5061.0], "hydro": [4814.06, 5065.6]})
    for model in ("portfolio", "post", "hydro"):
        out[f"{model}_logged"] = out[model].round(2) + offset
    return out


def test_the_replay_must_equal_the_logged_numbers_to_the_penny():
    assert replay_run.reproduces_the_log(_replayed(0.0))
    assert not replay_run.reproduces_the_log(_replayed(0.05))
    partly = _replayed(0.0)
    partly.loc[1, ["portfolio_logged", "post_logged", "hydro_logged"]] = np.nan  # a day the run did not log (an unknown Hydro Thursday) is not compared
    assert replay_run.reproduces_the_log(partly)


def test_features_are_cut_at_the_day_the_run_knew_and_a_shorter_file_is_refused():
    full = pd.DataFrame({"date": pd.date_range("2026-10-30", periods=10), "total_sales": [1000.0] * 10})
    cut = replay_run.features_as_of(full, pd.Timestamp("2026-11-01"))
    assert cut["date"].max() == pd.Timestamp("2026-11-01") and len(cut) == 3
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        replay_run.features_as_of(full, pd.Timestamp("2026-11-30"))


COLUMNS = {"issue_time": "2026-11-02T16:49:32+01:00", "lead_days": "0", "model_version": "v1", "variant": "no_manager", "forecast": "100.0", "manager_forecast_used": "5000.0", "manager_arrived_at": "",
           "data_through": "2026-11-01", "weather_source": "forecast", "hydro_list_through": "2026-12-31", "input_hash": "h", "model_fingerprint": "old", "code_commit": "c", "on_time": "on_time"}


def _backfill_log(tmp_path, already_logged=False):
    (tmp_path / "log").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "versions.json").write_text(json.dumps({"versions": [{"version": "early", "last_training_day": "2026-11-01"}, {"version": "later", "last_training_day": "2026-11-08"}]}), encoding="utf-8")
    run_lines = [("a", "monday", "ok", "2026-11-01"), ("gone", "monday", "withdrawn", "2026-11-08"), ("b", "monday", "ok", "2026-11-08")]
    pd.DataFrame([{"run_id": r, "issue_time": "t", "trigger": t, "status": s, "reason": "", "data_through": d, "forecast_types_made": "TW", "code_commit": "c"} for r, t, s, d in run_lines]).to_csv(tmp_path / "log" / "runs.csv", index=False)
    rows = [{**COLUMNS, "run_id": "a", "target_date": "2026-11-02", "forecast_type": "TW", "model": "hydro"},
            {**COLUMNS, "run_id": "b", "target_date": "2026-11-09", "forecast_type": "TW", "model": "hydro"}]
    if already_logged:
        rows.append({**COLUMNS, "run_id": "b", "target_date": "2026-11-09", "forecast_type": "TW", "model": "hydro_early"})
    pd.DataFrame(rows).to_csv(tmp_path / "log" / "forecasts.csv", index=False)
    return tmp_path


def _fake_replay(monkeypatch, offset=0.0):
    calls = []

    def fake(log_repo, run_id, also, version="v1"):
        calls.append((run_id, also))
        day = {"a": "2026-11-02", "b": "2026-11-09"}[run_id]
        out = pd.DataFrame({"forecast_type": ["TW"], "target_date": [day], "portfolio": [100.0], "post": [100.0], "hydro": [100.0]})
        for model in ("portfolio", "post", "hydro"):
            out[f"{model}_logged"] = 100.0 + offset
        for v in also:
            out[f"hydro_{v}"] = 123.456
            out[f"hydro_{v}{replay_run.predict.STANDIN_SUFFIX}"] = 5000.0  # the manager's-number input the version was given (the same as the run's own for a version on the old stand-in)
        return out

    monkeypatch.setattr(replay_run, "replay", fake)
    monkeypatch.setattr(replay_run.predict, "load_models", lambda log_repo, version: SimpleNamespace(combined_fingerprint=f"print-{version}"))
    return calls


NOW = datetime(2026, 11, 9, 14, 0, tzinfo=timezone.utc)


def test_a_withdrawn_run_is_not_replayed_and_a_version_never_forecasts_days_it_was_trained_on(tmp_path, monkeypatch):
    calls = _fake_replay(monkeypatch)
    batches = replay_run.backfill(_backfill_log(tmp_path), ("early", "later"), NOW, "new")
    assert calls == [("a", ("early",)), ("b", ("early", "later"))]
    assert {r["model"] for r in batches[0][0]} == {"hydro_early"}  # "later" trained to 8 November, so it has no 2 November forecast


def test_replayed_rows_are_labelled_and_copy_the_runs_own_row(tmp_path, monkeypatch):
    _fake_replay(monkeypatch)
    rows, run = replay_run.backfill(_backfill_log(tmp_path), ("early",), NOW, "new")[0]
    row = rows[0]
    assert row["on_time"] == "replay" and row["forecast"] == 123.46 and row["model_version"] == "early" and row["model_fingerprint"] == "print-early" and row["code_commit"] == "new"
    assert row["issue_time"] == NOW.isoformat(timespec="seconds")  # when it was really made
    assert row["data_through"] == "2026-11-01" and row["weather_source"] == "forecast" and row["manager_forecast_used"] == 5000.0  # as of that Monday; the stand-in the version itself was given
    assert run["trigger"] == "replay" and run["reason"] == "replay of a with early" and row["run_id"] == run["run_id"]


def test_rows_already_in_the_log_are_not_repeated(tmp_path, monkeypatch):
    _fake_replay(monkeypatch)
    batches = replay_run.backfill(_backfill_log(tmp_path, already_logged=True), ("early",), NOW, "new")
    assert [r["target_date"] for rows, _ in batches for r in rows] == ["2026-11-02"]


def test_nothing_is_returned_when_a_replay_does_not_reproduce_the_log(tmp_path, monkeypatch):
    _fake_replay(monkeypatch, offset=5.0)
    with pytest.raises(RuntimeError, match="nothing is written"):
        replay_run.backfill(_backfill_log(tmp_path), ("early",), NOW, "new")
