"""
Tests for the forecast log's run logic (src/forecast_log: schedule, inputs, logio, predict.lags_known). The models and the full run are exercised by a dry run in the runner folder;
these guard the rules that make the log honest: which days are forecast, when a run is late, that unknown Hydro Thursdays are skipped, that a number is only used if it arrived in time,
and that rows are appended once and never repeated.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.forecast_log import inputs, logio, predict, schedule

TZ = timezone.utc


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


def test_a_monday_morning_run_forecasts_next_week_and_this_week():
    issue = at("2026-11-02T08:00:00")  # a Monday
    nw = schedule.target_dates(schedule.NEXT_WEEK, issue)
    tw = schedule.target_dates(schedule.THIS_WEEK, issue)
    assert nw == [date(2026, 11, 9) + timedelta(days=i) for i in range(7)]
    assert tw == [date(2026, 11, 2) + timedelta(days=i) for i in range(7)]  # including Monday itself: it opens at 10:00


def test_an_evening_run_forecasts_tomorrow_only():
    assert schedule.target_dates(schedule.TOMORROW, at("2026-11-04T18:00:00")) == [date(2026, 11, 5)]


def test_a_late_this_week_run_forecasts_only_days_that_have_not_started():
    issue = at("2026-11-04T09:00:00")  # a Wednesday, before opening
    assert schedule.target_dates(schedule.THIS_WEEK, issue)[0] == date(2026, 11, 4)
    issue = at("2026-11-04T12:00:00")  # a Wednesday after opening: Wednesday has started
    assert schedule.target_dates(schedule.THIS_WEEK, issue) == [date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 8)]


def test_deadlines_and_lateness():
    assert schedule.on_time(schedule.THIS_WEEK, at("2026-11-02T08:00:00"))
    assert schedule.on_time(schedule.THIS_WEEK, at("2026-11-02T10:00:00"))
    assert not schedule.on_time(schedule.THIS_WEEK, at("2026-11-02T10:01:00"))   # two hours late after the computer was off
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


def test_manager_numbers_are_read_by_date_and_only_if_they_arrived_in_time(tmp_path):
    first = tmp_path / "week1.csv"
    first.write_text("date,forecast\n2026-11-09,5000\n09/11/2026,5100\n2026-11-10,6000\n", encoding="utf-8")
    second = tmp_path / "week2.csv"
    second.write_text("Date,Forecast\n2026-11-10,6200\n2026-11-11,7000\n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("something,else\n1,2\n", encoding="utf-8")  # not a forecast file: ignored
    old, new = datetime(2026, 11, 1).timestamp(), datetime(2026, 11, 5).timestamp()
    os.utime(first, (old, old))
    os.utime(second, (new, new))
    frame = inputs.read_manager_forecasts(tmp_path)
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2026-11-09", "2026-11-10", "2026-11-11"]
    assert frame.set_index(frame["date"].dt.strftime("%Y-%m-%d"))["forecast"].to_dict() == {"2026-11-09": 5100.0, "2026-11-10": 6200.0, "2026-11-11": 7000.0}  # the newest file wins
    cutoff = datetime(2026, 11, 3).astimezone()
    early = inputs.manager_for(frame, [date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)], not_after=cutoff)
    assert list(early["date"].dt.strftime("%Y-%m-%d")) == ["2026-11-09"]  # the other numbers arrived after the cut-off
    assert inputs.read_manager_forecasts(None).empty and inputs.read_manager_forecasts(tmp_path / "missing").empty


def _row(**over):
    base = {"run_id": "r", "issue_time": "t", "target_date": "2026-11-10", "lead_days": 8, "forecast_type": "NW", "model": "post", "model_version": "v1", "variant": "no_manager", "forecast": 1.0,
            "manager_forecast_used": 1.0, "manager_arrived_at": "", "data_through": "2026-11-01", "weather_source": "fallback_average", "hydro_list_through": "2026-12-31", "input_hash": "x",
            "model_fingerprint": "y", "code_commit": "z", "on_time": "on_time"}
    return {**base, **over}


def test_a_forecast_is_logged_once_and_never_repeated(tmp_path):
    path = tmp_path / "forecasts.csv"
    logio.ensure_header(path, logio.FORECAST_COLUMNS)
    first = logio.new_rows_only([_row(), _row(model="hydro"), _row()], path)  # the third repeats the first within the same batch
    assert [r["model"] for r in first] == ["post", "hydro"]
    logio._append(path, logio.FORECAST_COLUMNS, first)
    again = logio.new_rows_only([_row(), _row(variant="with_manager"), _row(target_date="2026-11-11")], path)
    assert [(r["variant"], r["target_date"]) for r in again] == [("with_manager", "2026-11-10"), ("no_manager", "2026-11-11")]


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
