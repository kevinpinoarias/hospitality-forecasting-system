"""
Tests for replaying a logged run with another version (src/forecast_log/replay_run.py): the run's own weather is read back as recorded, an unknown run is refused,
and the check that the replayed v1 numbers equal the logged ones to the penny.
"""

from __future__ import annotations

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
