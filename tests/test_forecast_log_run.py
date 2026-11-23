"""
Tests for the forecast log's run logic (src/forecast_log: schedule, inputs, logio, predict.lags_known). The models and the full run are exercised by a dry run in the runner folder;
these guard the rules that make the log honest: which days are forecast, when a run is late, that unknown Hydro Thursdays are skipped, that the live weather only covers the days after the
last known one, that the manager's number is scored as a benchmark once it arrives, and that rows are appended once and never repeated.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.forecast_log import inputs, logio, predict, schedule

TZ = timezone.utc


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


def test_the_weekly_run_forecasts_next_week_and_all_of_this_week():
    issue = at("2026-11-02T12:30:00")  # Monday 12:30: the one scheduled run of the week
    assert schedule.forecast_types_for("monday") == ("NW", "TW")  # tomorrow would repeat a day this week's forecast already covers
    assert schedule.target_dates(schedule.NEXT_WEEK, issue) == [date(2026, 11, 9) + timedelta(days=i) for i in range(7)]
    assert schedule.target_dates(schedule.THIS_WEEK, issue) == [date(2026, 11, 2) + timedelta(days=i) for i in range(7)]  # Monday too: the data ends on Sunday
    assert schedule.on_time(schedule.NEXT_WEEK, issue) and schedule.on_time(schedule.THIS_WEEK, issue)


def test_an_evening_run_forecasts_tomorrow_only():
    assert schedule.target_dates(schedule.TOMORROW, at("2026-11-04T18:00:00")) == [date(2026, 11, 5)]


def test_a_late_this_week_run_forecasts_from_its_issue_date_on():
    issue = at("2026-11-04T15:00:00")  # a catch-up on Wednesday afternoon
    assert schedule.target_dates(schedule.THIS_WEEK, issue) == [date(2026, 11, 4) + timedelta(days=i) for i in range(5)]  # Wednesday to Sunday
    assert not schedule.on_time(schedule.THIS_WEEK, issue)


def test_deadlines_and_lateness():
    assert schedule.on_time(schedule.THIS_WEEK, at("2026-11-02T12:30:00"))
    assert schedule.on_time(schedule.THIS_WEEK, at("2026-11-02T23:30:00"))      # a catch-up late on Monday is still on time
    assert not schedule.on_time(schedule.THIS_WEEK, at("2026-11-03T09:00:00"))   # Tuesday: late
    assert schedule.on_time(schedule.NEXT_WEEK, at("2026-11-02T22:00:00"))      # next week's forecast is still on time late on Monday
    assert not schedule.on_time(schedule.NEXT_WEEK, at("2026-11-03T09:00:00"))
    assert schedule.on_time(schedule.TOMORROW, at("2026-11-04T23:30:00"))


def test_triggers_map_to_forecast_types():
    assert schedule.forecast_types_for("monday") == ("NW", "TW")
    assert schedule.forecast_types_for("evening") == ("TD",)
    with pytest.raises(ValueError):
        schedule.forecast_types_for("tuesday")


def test_hydro_thursdays_beyond_the_list_are_unknown_not_empty():
    thursdays = [date(2026, 12, 24), date(2026, 12, 31), date(2027, 1, 7)]
    assert inputs.unknown_hydro_thursdays(thursdays + [date(2027, 1, 8)], date(2026, 12, 31)) == {date(2027, 1, 7)}  # Friday is never unknown
    assert inputs.unknown_hydro_thursdays([date(2026, 12, 30)], date(2026, 12, 1)) == set()  # a Wednesday is not a Thursday


def test_the_weather_label_says_where_each_day_came_from():
    weather = pd.DataFrame({"date": pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"])})
    issue = date(2026, 11, 3)
    assert predict.weather_label(date(2026, 11, 2), weather, False, issue) == "observed"   # already happened when the forecast was made
    assert predict.weather_label(date(2026, 11, 4), weather, False, issue) == "forecast"
    assert predict.weather_label(date(2026, 11, 30), weather, False, issue) == "historical_average"  # beyond the service's range
    assert predict.weather_label(date(2026, 11, 5), weather.iloc[:0], True, issue) == "historical_average_api_error"  # in range, but the service failed


def test_the_version_decides_which_correction_history_is_read():
    from src.forecast_log import v1

    assert v1.correction_history_path("v2").name == "v2_correction_history.csv"
    assert v1.CORRECTION_HISTORY.name == "v1_correction_history.csv"


def _row(**over):
    base = {"run_id": "r", "issue_time": "t", "target_date": "2026-11-10", "lead_days": 8, "forecast_type": "NW", "model": "post", "model_version": "v1", "variant": "no_manager", "forecast": 1.0,
            "manager_forecast_used": 1.0, "manager_arrived_at": "", "data_through": "2026-11-01", "weather_source": "forecast", "hydro_list_through": "2026-12-31", "input_hash": "x",
            "model_fingerprint": "y", "code_commit": "z", "on_time": "on_time"}
    return {**base, **over}


def test_a_forecast_is_logged_once_and_never_repeated(tmp_path):
    path = tmp_path / "forecasts.csv"
    logio.ensure_header(path, logio.FORECAST_COLUMNS)
    first = logio.new_rows_only([_row(), _row(model="hydro"), _row()], path)  # the third repeats the first within the same batch
    assert [r["model"] for r in first] == ["post", "hydro"]
    logio._append(path, logio.FORECAST_COLUMNS, first)
    again = logio.new_rows_only([_row(), _row(model="portfolio"), _row(target_date="2026-11-11")], path)
    assert [(r["model"], r["target_date"]) for r in again] == [("portfolio", "2026-11-10"), ("post", "2026-11-11")]


def test_a_log_file_with_a_different_header_is_refused(tmp_path):
    path = tmp_path / "forecasts.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different header"):
        logio.ensure_header(path, logio.FORECAST_COLUMNS)


def test_lags_known_needs_every_lag_asked_for():
    days = pd.date_range("2026-10-01", "2026-11-20")
    sales = pd.Series(np.where(days <= pd.Timestamp("2026-11-08"), 1000.0, np.nan), index=days)  # sales known through Sunday 8 Nov
    targets = pd.Series(pd.to_datetime(["2026-11-13", "2026-11-20"]))  # Fridays, 5 and 12 days after the last known day
    assert list(predict.lags_known(sales, targets, (7, 14, 21, 28))) == [True, False]  # 13 Nov: 6 Nov is known; 20 Nov: 13 Nov is not
    assert list(predict.lags_known(sales, targets, (14, 21))) == [True, True]


# --------------------------------------------------------------------------- scoring


def _scored():
    from src.forecast_log import score_forecasts as sf  # noqa: F401  (imported lazily: it reads settings)

    rows = []
    for i in range(20):
        d = (pd.Timestamp("2026-11-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        actual = 1000.0 + 10 * i
        for model, err in (("portfolio", 100.0), ("post", 60.0), ("hydro", 40.0), ("manager", 120.0)):
            rows.append({"forecast_type": "NW", "target_date": d, "model": model, "variant": "" if model == "manager" else "no_manager", "on_time": "on_time",
                         "forecast": actual + err, "actual_sales": actual, "miss": err})
    return pd.DataFrame(rows)


def test_paired_differences_use_matched_on_time_days_only():
    from src.forecast_log import score_forecasts as sf

    frame = _scored()
    late = frame.iloc[:4].copy()
    late["on_time"] = "late"
    late["miss"] = 9999.0
    frame = pd.concat([frame, late], ignore_index=True)
    q2 = sf.paired_difference(frame, ("post", "no_manager"), ("portfolio", "no_manager"), "NW")
    assert q2["days"] == 20 and q2["change"] == pytest.approx(-40.0) and q2["better_days"] == 20  # late rows do not count
    q1 = sf.paired_difference(frame, ("portfolio", "no_manager"), ("manager", ""), "NW")
    assert q1["change"] == pytest.approx(-20.0)
    assert not np.isnan(q2["lo"])  # 20 days: enough for an interval
    assert np.isnan(sf.paired_difference(frame, ("post", "no_manager"), ("portfolio", "no_manager"), "NW", days={"2026-11-02"})["lo"])  # one day: no interval


def test_the_sign_test_is_one_sided_binomial():
    from src.forecast_log import score_forecasts as sf

    assert sf.sign_test_p(14, 17) == pytest.approx(0.0064, abs=1e-4)
    assert sf.sign_test_p(9, 17) == pytest.approx(0.5, abs=0.1)
    assert np.isnan(sf.sign_test_p(0, 0))


def test_the_accuracy_table_reports_error_and_bias_by_model():
    from src.forecast_log import score_forecasts as sf

    table = sf.accuracy_table(_scored()).set_index("model")
    assert table.loc["post", "MAE"] == pytest.approx(60.0) and table.loc["post", "bias"] == pytest.approx(60.0)
    assert table.loc["hydro", "days"] == 20


def test_the_weekly_drop_copies_only_new_files_and_finds_gaps(tmp_path):
    from src.forecast_log import weekly_drop as wd

    incoming, dest = tmp_path / "in", tmp_path / "private"
    incoming.mkdir()
    dest.mkdir()
    for n in (1, 2):
        (incoming / f"Rotaready-CostControlRawData{n}.csv").write_text("x", encoding="utf-8")
    (dest / "Rotaready-CostControlRawData1.csv").write_text("already here", encoding="utf-8")
    assert wd.copy_new_exports(incoming, dest) == ["Rotaready-CostControlRawData2.csv"]
    assert (dest / "Rotaready-CostControlRawData1.csv").read_text(encoding="utf-8") == "already here"  # never overwritten
    feats = tmp_path / "f.csv"
    pd.DataFrame({"date": ["2026-11-01", "2026-11-02", "2026-11-04"], "total_sales": [1, 2, 4]}).to_csv(feats, index=False)
    assert [d.strftime("%Y-%m-%d") for d in wd.find_gaps(feats)] == ["2026-11-03"]


def test_the_weekly_drop_finds_exports_placed_straight_in_the_private_folder(tmp_path):
    import hashlib
    import json

    from src.forecast_log import weekly_drop as wd

    private = tmp_path / "private"
    assert wd.pending_exports(private) == []  # no folder yet: nothing to do
    private.mkdir()
    done, changed, new = (private / f"Rotaready-CostControlRawData{n}.csv" for n in (1, 2, 3))
    done.write_text("processed", encoding="utf-8")
    changed.write_text("corrected re-export", encoding="utf-8")
    new.write_text("dropped in by hand", encoding="utf-8")
    (private / "~$open_in_excel.xlsx").write_text("", encoding="utf-8")  # Excel's lock file is not an export
    manifest = {done.name: {"sha256": hashlib.sha256(done.read_bytes()).hexdigest()}, changed.name: {"sha256": "an older version"}}
    (private / ".anonymize_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert wd.pending_exports(private) == [changed.name, new.name]


# --------------------------------------------------------------------------- live weather and the manager benchmark


def test_live_weather_is_laid_over_the_days_after_the_last_known_one_only():
    from src.forecast_log.run_forecasts import FEATURES

    if not FEATURES.exists():
        pytest.skip("needs data/features/engineered_features.csv (python main.py)")
    from src.experiments import run_series35_holdout as h

    full = h.load_engineered(FEATURES)
    last = pd.Timestamp("2026-08-02")
    targets = [pd.Timestamp("2026-08-04"), pd.Timestamp("2026-08-25")]
    plain = predict.served_rows(full, last, targets, "v2", pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"]))
    weather = pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04"]), "max_temp": [40.0, 30.0, 31.0], "rain_mm": [0.0, 0.0, 0.0], "sun_hours": [12.0, 12.0, 12.0]})
    live = predict.served_rows(full, last, targets, "v2", weather)
    assert live["max_temp"].iloc[0] == 31.0 and live["is_hot_for_scotland"].iloc[0] == 1  # the target day takes the service's figure
    assert live["temp_anomaly_14d"].iloc[0] != plain["temp_anomaly_14d"].iloc[0]  # and so do the features built from the days before it
    assert live["max_temp"].iloc[1] == plain["max_temp"].iloc[1]  # a day the service does not cover keeps its average
    weather_known_day = weather.copy()
    weather_known_day.loc[0, "max_temp"] = -50.0  # 1 August is a known day: its real weather must never be replaced
    assert predict.served_rows(full, last, targets, "v2", weather_known_day)["temp_anomaly_14d"].iloc[0] == live["temp_anomaly_14d"].iloc[0]


def test_the_manager_benchmark_is_matched_to_the_portfolio_forecasts():
    from src.forecast_log import score_forecasts as sf

    forecasts = pd.DataFrame({"forecast_type": ["NW", "NW", "TD", "NW"], "target_date": ["2026-11-09", "2026-11-10", "2026-11-09", "2026-11-09"],
                              "model": ["portfolio", "portfolio", "portfolio", "post"], "variant": "no_manager", "on_time": "on_time", "forecast": 1.0})
    manager = pd.DataFrame({"target_date": ["2026-11-09", "2026-11-11"], "manager_forecast": [5000.0, 6000.0]})
    rows = sf.manager_benchmark_rows(forecasts, manager)
    assert sorted(zip(rows["forecast_type"], rows["target_date"])) == [("NW", "2026-11-09"), ("TD", "2026-11-09")]  # 10 Nov has no number yet; 11 Nov was not forecast
    assert set(rows["model"]) == {"manager"} and set(rows["forecast"]) == {5000.0}


def test_the_manager_number_is_recorded_once_when_the_export_brings_it(tmp_path):
    from src.forecast_log import score_forecasts as sf

    (tmp_path / "log").mkdir()
    logio.ensure_header(tmp_path / "log" / "forecasts.csv", logio.FORECAST_COLUMNS)
    logio._append(tmp_path / "log" / "forecasts.csv", logio.FORECAST_COLUMNS, [_row(target_date="2026-11-09")])
    features = tmp_path / "features.csv"
    pd.DataFrame({"date": ["2026-11-08", "2026-11-09", "2026-11-10"], "total_sales": [1.0, 2.0, np.nan], "forecast_sales": [900.0, 1000.0, np.nan]}).to_csv(features, index=False)
    assert sf.update_manager(tmp_path, features) == 1  # 8 Nov is before the first forecast; 10 Nov has no number yet
    assert sf.update_manager(tmp_path, features) == 0  # never added twice
    recorded = pd.read_csv(tmp_path / "log" / "manager_forecasts.csv")
    assert list(recorded["target_date"]) == ["2026-11-09"] and list(recorded["manager_forecast"]) == [1000.0]


def test_a_reissue_counts_its_days_from_the_withdrawn_run_and_is_late_when_made_after_the_deadline():
    from src.forecast_log import run_forecasts
    monday = at("2026-11-16T12:30:00")
    assert run_forecasts.label(schedule.THIS_WEEK, monday, monday) == "on_time"
    assert run_forecasts.label(schedule.THIS_WEEK, monday, at("2026-11-16T23:00:00")) == "on_time"  # the same day, before the deadline
    assert run_forecasts.label(schedule.THIS_WEEK, monday, at("2026-11-17T21:30:00")) == "late"
    assert run_forecasts.label(schedule.NEXT_WEEK, monday, at("2026-11-17T21:30:00")) == "late"


def test_only_a_withdrawn_run_is_reissued_with_the_weather_it_saved(tmp_path):
    from src.forecast_log import run_forecasts
    (tmp_path / "log").mkdir()
    pd.DataFrame({"run_id": ["w", "k"], "issue_time": ["2026-11-16T12:30:31+01:00", "2026-11-09T12:51:20+01:00"], "trigger": ["monday"] * 2, "status": ["withdrawn", "ok"], "reason": ["", ""],
                  "data_through": ["2026-11-08"] * 2, "forecast_types_made": ["NW+TW"] * 2, "code_commit": ["c"] * 2}).to_csv(tmp_path / "log" / "runs.csv", index=False)
    pd.DataFrame({"run_id": ["w", "w", "k"], "date": ["2026-11-16", "2026-11-17", "2026-11-16"], "max_temp": [3.7, 6.4, 99.0], "rain_mm": [0.0, 7.1, 0.0], "sun_hours": [8.0, 0.0, 0.0],
                  "fetched_at": ["2026-11-16T12:31:04+01:00"] * 2 + ["x"]}).to_csv(tmp_path / "log" / "weather_used.csv", index=False)
    issue, weather, fetched = run_forecasts.withdrawn_run(tmp_path, "w")
    assert issue == datetime.fromisoformat("2026-11-16T12:30:31+01:00") and fetched == "2026-11-16T12:31:04+01:00"
    assert weather["max_temp"].tolist() == [3.7, 6.4] and list(weather.columns) == ["date", "max_temp", "rain_mm", "sun_hours"]
    with pytest.raises(RuntimeError, match="only a withdrawn run"):
        run_forecasts.withdrawn_run(tmp_path, "k")


# --- where the log repository is ------------------------------------------------

def _settings_in(tmp_path, monkeypatch, tracked: str, local: str | None = None) -> dict:
    from src.forecast_log import run_forecasts

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "forecast_log.yaml").write_text(tracked, encoding="utf-8")
    if local is not None:
        (project / "config" / "forecast_log.local.yaml").write_text(local, encoding="utf-8")
    monkeypatch.setattr(run_forecasts, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)  # a run started from another folder must not change the answer
    return run_forecasts.load_settings()


def test_a_relative_log_repo_is_read_from_the_project_folder(tmp_path, monkeypatch):
    settings = _settings_in(tmp_path, monkeypatch, "log_repo: ../forecasting-log\nmodel_version: v1\n")
    assert settings["log_repo"] == str((tmp_path / "forecasting-log").resolve())


def test_the_local_file_overrides_the_log_repo(tmp_path, monkeypatch):
    elsewhere = (tmp_path / "somewhere" / "forecasting-log").resolve()
    settings = _settings_in(tmp_path, monkeypatch, "log_repo: ../forecasting-log\n", f"log_repo: {elsewhere.as_posix()}\n")
    assert settings["log_repo"] == str(elsewhere)


def test_the_tracked_settings_name_no_computer_path():
    from src.config import PROJECT_ROOT as ROOT

    text = (ROOT / "config" / "forecast_log.yaml").read_text(encoding="utf-8")
    assert ":/" not in text and ":\\" not in text
