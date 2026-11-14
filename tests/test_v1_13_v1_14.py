"""
Tests for forecast-log versions v1.13 (the year-end specialist models) and v1.14 (the year-ago sales input): the year-ago column looks back exactly 364 days and never forward, the two versions' pieces and
arm names are as decided, the specialists are used only inside their windows (the masked ones with the ten inputs missing), the registration refuses what the log's rules forbid, and Q7 and Q8 appear in the
scores only once their rows exist.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.experiments import run_series35_holdout as h  # noqa: E402
from src.features import year_ago  # noqa: E402
from src.forecast_log import predict, score_forecasts, v1, v1_12, v1_13, v1_14  # noqa: E402
from src.models.final_model import FINAL_FEATURES  # noqa: E402

# --------------------------------------------------------------------------- the year-ago input


def test_the_year_ago_input_is_the_sales_364_days_earlier_and_never_a_later_day():
    days = pd.date_range("2025-01-01", "2026-12-31")
    sales = pd.Series(np.arange(len(days), dtype=float), index=days)  # sales = the day's number, so the value shows which day was read
    out = year_ago.year_ago_sales([pd.Timestamp("2026-03-04"), pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01")], sales)
    assert out.iloc[0] == sales[pd.Timestamp("2026-03-04") - pd.Timedelta(days=364)]
    assert pd.isna(out.iloc[1])  # no day 364 days before 1 June 2025 in the series
    assert out.iloc[2] == sales[pd.Timestamp("2025-01-02")]
    assert pd.Timestamp("2026-03-04").dayofweek == (pd.Timestamp("2026-03-04") - pd.Timedelta(days=364)).dayofweek  # 52 weeks: the same weekday


def test_the_year_ago_input_keeps_closed_days_at_zero_and_stays_missing_where_the_sales_are_unknown():
    days = pd.date_range("2025-12-01", "2026-12-31")
    sales = pd.Series(100.0, index=days)
    sales[pd.Timestamp("2025-12-25")] = 0.0  # a closed day keeps its zero
    sales[pd.Timestamp("2026-06-10"):] = np.nan  # the days after the last known one
    out = year_ago.year_ago_sales([pd.Timestamp("2026-12-24"), pd.Timestamp("2026-12-25"), pd.Timestamp("2027-06-12")], sales)
    assert out.iloc[0] == 0.0  # 24 December 2026 is a Thursday, as 25 December 2025 was
    assert out.iloc[1] == 100.0  # 26 December 2025
    assert pd.isna(out.iloc[2])  # 13 June 2026 is not known yet


def test_adding_the_year_ago_input_appends_it_last_and_leaves_the_frame_alone():
    frame = pd.DataFrame({"date": pd.date_range("2026-02-01", periods=4), "x": 1})
    sales = pd.Series(5.0, index=pd.date_range("2025-01-01", "2026-12-31"))
    out = year_ago.add_year_ago_sales(frame, sales)
    assert list(out.columns) == ["date", "x", year_ago.COLUMN] and list(frame.columns) == ["date", "x"]
    assert (out[year_ago.COLUMN] == 5.0).all()


# --------------------------------------------------------------------------- v1.13


def test_v1_13_has_the_five_main_pieces_of_v1_12_and_ten_specialists_and_names_its_arm():
    assert v1_13.PARENT == "v1.12" and v1_13.ARM == predict.extra_arm_name("v1.13") == "hydro_v1.13"
    assert v1_13.MAIN_PIECES == v1_12.HYDRO_ARM_PIECES
    specialists = v1_13.specialist_names()
    assert len(specialists) == 10 and {k.split("_", 1)[0] for k in specialists} == {"comp", "mask"}
    assert {v[1] for v in specialists.values()} == set(v1_13.MAIN_PIECES)
    assert v1_13.MODELS_DIR.name == "v1.13"


def test_the_companion_has_none_of_the_ten_inputs_and_the_masked_pieces_keep_all_of_them():
    for name in v1_13.MAIN_PIECES:
        base = v1_13.base_features(name)
        assert base == list(FINAL_FEATURES) + v1_12.PIECES[name][0]  # the main piece is v1.12's own
        companion, masked = v1_13.piece_features("companion", name), v1_13.piece_features("masked", name)
        assert not set(companion) & set(v1_13.INPUTS)
        assert companion == [f for f in base if f not in v1_13.INPUTS]  # the same order, only the ten taken out
        assert masked == base
    assert set(v1_13.INPUTS) & set(FINAL_FEATURES) == {"lag_7_sales", "rolling_7_sales", "rolling_14_sales", "lag_1_fe", "lag_7_fe", "rolling_7_fe"}  # the other four are the Friday and Saturday extras


def test_the_windows_are_26_to_31_december_and_1_to_8_january():
    d = pd.Series(pd.to_datetime(["2025-12-25", "2025-12-26", "2025-12-31", "2026-01-01", "2026-01-08", "2026-01-09", "2026-07-01"]))
    assert v1_13.in_window(d, *v1_13.COMPANION_WINDOW).tolist() == [False, True, True, False, False, False, False]
    assert v1_13.in_window(d, *v1_13.MASK_WINDOW).tolist() == [False, False, False, True, True, False, False]
    assert not (v1_13.in_window(d, *v1_13.COMPANION_WINDOW) & v1_13.in_window(d, *v1_13.MASK_WINDOW)).any()


def test_the_training_mask_sets_the_inputs_to_missing_on_the_rows_of_1_to_8_january_only():
    frame = pd.DataFrame({"date": pd.date_range("2025-12-20", "2026-01-12"), "lag_7_sales": 1.0, "same_wd_max4": 2.0, "temp": 3.0})
    out = v1_13.masked_copy(frame)
    inside = v1_13.in_window(frame["date"], *v1_13.MASK_WINDOW)
    assert out.loc[inside, ["lag_7_sales", "same_wd_max4"]].isna().all().all() and (out.loc[~inside, ["lag_7_sales", "same_wd_max4"]] == [1.0, 2.0]).all().all()
    assert (out["temp"] == 3.0).all() and frame["lag_7_sales"].notna().all()  # other columns and the input frame are untouched


class _Stub:
    """A piece that returns its own number and records the frame it was given."""

    def __init__(self, value: float) -> None:
        self.value, self.seen = value, None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen = frame.copy()
        return np.full(len(frame), self.value)


def _models(with_specialists: bool) -> predict.Models:
    pieces = {name: _Stub(100.0) for name in v1_13.MAIN_PIECES}
    if with_specialists:
        pieces |= {file: _Stub(200.0 if kind == "companion" else 300.0) for file, (kind, _) in v1_13.specialist_names().items()}
    return predict.Models("v1.13", None, pieces, None, {})


def _rows() -> pd.DataFrame:
    dates = pd.date_range("2026-12-21", "2027-01-11")  # a Monday to a Monday three weeks later
    return pd.DataFrame({"date": dates, "lag_7_sales": 1.0, "rolling_7_sales": 1.0, "lag_14_sales": 1.0, "lag_21_sales": 1.0, "same_wd_mean4": 1.0, "same_wd_max4": 1.0})


def test_the_specialists_forecast_only_inside_their_windows_and_the_masked_ones_see_the_inputs_missing(monkeypatch):
    def with_weekend(rows, sales_known):  # the weekend inputs are not what is under test: every lag is known
        return rows.copy()

    monkeypatch.setattr(h, "with_weekend", with_weekend)
    monkeypatch.setattr(predict, "lags_known", lambda sales_known, dates, lags: np.ones(len(dates), dtype=bool))
    rows = _rows()
    models = _models(True)
    raw = predict.route_post(models, rows, pd.Series(dtype=float), hydro=True)
    assert (raw == 100.0).all()  # the main pieces on every day
    out = predict.with_year_end_specialists(models, rows, pd.Series(dtype=float), raw)
    dates = rows["date"]
    companion, masked = v1_13.in_window(dates, *v1_13.COMPANION_WINDOW), v1_13.in_window(dates, *v1_13.MASK_WINDOW)
    assert (out[companion] == 200.0).all() and (out[masked] == 300.0).all() and (out[~companion & ~masked] == 100.0).all()
    assert companion.sum() == 6 and masked.sum() == 8
    seen = models.pieces["mask_log_hydro_thu"].seen
    assert seen is not None and seen[["lag_7_sales", "rolling_7_sales"]].isna().all().all()  # the ten inputs are missing on the masked days
    assert rows["lag_7_sales"].notna().all()  # the caller's frame is never changed
    assert models.pieces["log_hydro_thu"].seen["lag_7_sales"].notna().all()  # the main piece saw the real inputs


def test_a_version_without_the_specialist_pieces_is_returned_unchanged():
    raw = np.arange(5.0)
    assert predict.with_year_end_specialists(_models(False), _rows().head(5), pd.Series(dtype=float), raw) is raw


# --------------------------------------------------------------------------- v1.14


def test_v1_14_adds_the_year_ago_input_last_to_three_pieces_only_and_names_its_arm():
    assert v1_14.PARENT == "v1.13" and v1_14.ARM == predict.extra_arm_name("v1.14") == "hydro_v1.14"
    assert v1_14.YEAR_AGO_PIECES == ("log_hydro_thu", "fri_A_all", "sat_A_lags")  # Monday to Thursday (hydro), Friday, Saturday
    for name in v1_13.MAIN_PIECES:
        inputs = v1_14.piece_inputs(name)
        if name in v1_14.YEAR_AGO_PIECES:
            assert inputs == v1_13.base_features(name) + [year_ago.COLUMN]
        else:
            assert inputs == v1_13.base_features(name)  # the Sunday piece and the raw Saturday fallback keep v1.13's inputs
    assert v1_14.MODELS_DIR.name == "v1.14"


# --------------------------------------------------------------------------- registration


@pytest.mark.parametrize("module,fit,version,parent", [(v1_13, "fit_v1_13", "v1.13", "v1.12"), (v1_14, "fit_v1_14", "v1.14", "v1.13")])
def test_registration_refuses_what_the_log_rules_forbid(tmp_path, monkeypatch, module, fit, version, parent):
    log_repo = tmp_path / "log"
    (log_repo / "models").mkdir(parents=True)
    versions = log_repo / "models" / "versions.json"
    run = getattr(module, fit)
    versions.write_text(json.dumps({"versions": [{"version": version}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already registered"):
        run(log_repo)
    versions.write_text(json.dumps({"versions": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"parent version {parent} is not registered"):
        run(log_repo)
    versions.write_text(json.dumps({"versions": [{"version": parent, "correction_history_sha256": "x", "pieces": {}}]}), encoding="utf-8")
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path / "models_new")
    (tmp_path / "models_new").mkdir()
    (tmp_path / "models_new" / "seed_00.cbm").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="never overwritten"):
        run(log_repo)
    (tmp_path / "models_new" / "seed_00.cbm").unlink()
    monkeypatch.setattr(v1, "correction_history_path", lambda v: tmp_path / f"{v}_missing.csv")
    with pytest.raises(RuntimeError, match="is missing"):
        run(log_repo)
    history = tmp_path / "history.csv"
    history.write_text("not the file the parent recorded", encoding="utf-8")
    monkeypatch.setattr(v1, "correction_history_path", lambda v: history)
    with pytest.raises(RuntimeError, match=f"is not the correction history {parent} recorded"):
        run(log_repo)


# --------------------------------------------------------------------------- scoring


def _scored(models: list[str]) -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2026-12-21", "2027-01-17"):
        for m in models:
            window = (d.month == 12 and d.day >= 26) or (d.month == 1 and d.day <= 8)
            bias = {"hydro_v1.12": 200.0, "hydro_v1.13": 100.0 if window else 200.0, "hydro_v1.14": 100.0 if window else (200.0 if d.dayofweek == 6 else 150.0), "hydro_v1.11": 200.0,
                    "hydro": 200.0, "post": 200.0, "portfolio": 200.0}[m]
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": m, "variant": "no_manager", "on_time": "on_time", "forecast": 1000.0 + bias, "actual_sales": 1000.0})
    frame = pd.DataFrame(rows)
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def test_q7_and_q8_appear_only_once_their_rows_exist_and_compare_with_the_parent():
    base = ["portfolio", "post", "hydro", "hydro_v1.11", "hydro_v1.12"]
    without, _ = score_forecasts.build_report(_scored(base))
    assert "Q7 v1.13 hydro" not in without and "Q8 v1.14 hydro" not in without
    with_13, _ = score_forecasts.build_report(_scored([*base, "hydro_v1.13"]))
    assert "Q7 v1.13 hydro vs v1.12 hydro, 26 to 31 December" in with_13 and "Q7 v1.13 hydro vs v1.12 hydro, 1 to 8 January" in with_13 and "Q8 v1.14 hydro" not in with_13
    with_14, _ = score_forecasts.build_report(_scored([*base, "hydro_v1.13", "hydro_v1.14"]))
    assert "Q8 v1.14 hydro vs v1.13 hydro, Monday to Saturday" in with_14 and "Q8 v1.14 hydro vs v1.13 hydro, Sundays" in with_14


def test_q7_is_read_on_the_year_end_days_and_q8_on_monday_to_saturday():
    frame = _scored(["hydro_v1.12", "hydro_v1.13", "hydro_v1.14"])
    groups = dict(score_forecasts.comparison_groups("Q7", sorted(frame["target_date"].unique())))
    december, january = groups["26 to 31 December"], groups["1 to 8 January"]
    assert len(december) == 6 and len(january) == 8
    q7 = score_forecasts.paired_difference(frame, ("hydro_v1.13", "no_manager"), ("hydro_v1.12", "no_manager"), "TW", december | january)
    assert q7["days"] == 14 and q7["change"] == pytest.approx(-100.0) and q7["better_days"] == 14
    other = score_forecasts.paired_difference(frame, ("hydro_v1.13", "no_manager"), ("hydro_v1.12", "no_manager"), "TW", groups["all other days (identical forecasts, before the weekend correction)"])
    assert other["change"] == pytest.approx(0.0)  # v1.13 forecasts what v1.12 does on every other day
    q8_groups = dict(score_forecasts.comparison_groups("Q8", sorted(frame["target_date"].unique())))
    q8 = score_forecasts.paired_difference(frame, ("hydro_v1.14", "no_manager"), ("hydro_v1.13", "no_manager"), "TW", q8_groups["Monday to Saturday"])
    assert q8["change"] < 0 and q8["better_days"] > 0
    sundays = score_forecasts.paired_difference(frame, ("hydro_v1.14", "no_manager"), ("hydro_v1.13", "no_manager"), "TW", q8_groups["Sundays (the same piece)"])
    assert sundays["change"] == pytest.approx(0.0)
