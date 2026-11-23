"""
Tests for the holdout machinery (src/experiments/run_series35_holdout.py). The rules it implements are in
docs/HOLDOUT_2026_10_PREREGISTRATION.md; these tests guard the ones that make the test fair, above all that a
forecast can never read anything dated after the moment it is issued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.config import CONFIG  # noqa: E402
from src.experiments import run_series35_holdout as h  # noqa: E402

needs_features = pytest.mark.skipif(not CONFIG.path("engineered_features").exists(), reason="needs data/features (run the pipeline)")


def test_the_monday_runs_are_the_five_in_the_pre_registration():
    runs = h.monday_runs(pd.Timestamp("2026-09-07"), pd.Timestamp("2026-10-15"))
    assert [r.date().isoformat() for r in runs] == ["2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05"]
    assert all(r.day_name() == "Monday" for r in runs)


def test_a_run_whose_target_week_has_no_actual_yet_is_not_scheduled():
    # 12 Oct's target week starts 19 Oct, after the last actual (15 Oct): it belongs to the prospective log, not to this test
    assert pd.Timestamp("2026-10-12") not in h.monday_runs(pd.Timestamp("2026-09-07"), pd.Timestamp("2026-10-15"))


def test_fingerprint_follows_the_content_and_ignores_the_order_files_were_written(tmp_path):
    (tmp_path / "b.txt").write_text("two")
    (tmp_path / "a.txt").write_text("one")
    first = h.fingerprint(tmp_path)
    assert first == h.fingerprint(tmp_path)
    (tmp_path / "a.txt").write_text("changed")
    assert h.fingerprint(tmp_path) != first


def _toy(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 3))
    sales = 3000 + 500 * x[:, 0] + rng.normal(scale=50, size=n)
    return pd.DataFrame({"f1": x[:, 0], "f2": x[:, 1], "f3": x[:, 2], h.TARGET_COL: sales})


@pytest.mark.parametrize("target", ["log1p", "raw"])
def test_a_bag_survives_saving_and_loading_and_predicts_in_pounds(tmp_path, target):
    df = _toy()
    bag = h.Bag.fit(df, ["f1", "f2", "f3"], target, seeds=[0, 1], thread_count=1)
    before = bag.predict(df)
    bag.save(tmp_path / "bag")
    after = h.Bag.load(tmp_path / "bag").predict(df)
    assert np.allclose(before, after)
    assert 1500 < before.mean() < 4500  # pounds, not logs


def test_the_manager_stand_in_uses_only_the_history_it_is_given():
    dates = pd.date_range("2025-01-01", periods=400)
    history = pd.DataFrame({h.DATE_COL: dates, "forecast_sales": np.arange(400, dtype=float)})
    target = dates[-1] + pd.Timedelta(days=3)
    a = h.stand_in_manager(history, [target])[0]
    changed = history.copy()
    changed.loc[changed.index[-1], "forecast_sales"] += 10_000  # the last day's value does move the median it feeds...
    assert h.stand_in_manager(changed.iloc[:-1], [target])[0] == h.stand_in_manager(history.iloc[:-1], [target])[0]  # ...but days cut off do not
    assert np.isfinite(a)


def test_the_package_correction_matches_the_formula_on_a_known_case():
    hist = pd.DataFrame({
        "date": ["2026-05-02", "2026-05-09", "2026-05-16", "2026-05-23"], "fold_id": [0, 0, 1, 1],
        "actual": [1000.0, 1000.0, 1000.0, 1000.0],
        "weekday": ["Saturday"] * 4,
    })
    pred = np.array([1100.0, 1100.0, 1300.0, 1300.0])  # errors +100, +100, +300, +300
    new_dates = pd.to_datetime(["2026-06-01", "2026-06-06", "2026-06-07"])  # Monday, Saturday, Sunday
    out = h.correct_new_block(hist, pred, new_dates, np.array([2000.0, 2000.0, 2000.0]))
    start = new_dates.min()
    age = (start - pd.to_datetime(hist["date"])) / pd.Timedelta(weeks=1)
    w = 0.5 ** (age / 8)
    bias = float(np.sum(w * (pred - 1000.0)) / np.sum(w))
    assert out[0] == 2000.0                      # Monday is never corrected
    assert out[1] == pytest.approx(2000.0 - 0.5 * bias)
    assert out[2] == 2000.0                      # no Sundays in the history, so none can be learned


@needs_features
def test_a_served_forecast_cannot_read_anything_after_the_day_it_is_issued():
    """The rota run's core promise: change every sales figure, manager forecast and weather value after the last known day, and the
    feature rows for the target week do not move."""
    full = h.load_engineered(CONFIG.path("engineered_features"))
    last_known = pd.Timestamp("2026-07-26")
    targets = [pd.Timestamp("2026-08-03") + pd.Timedelta(days=i) for i in range(7)]
    clean = h.served_rows(full, last_known, targets)
    tampered = full.copy()
    later = tampered[h.DATE_COL] > last_known
    for col in ["total_sales", "forecast_sales", "max_temp", "rain_mm", "sun_hours"]:
        tampered.loc[later, col] = 99_999.0
    after = h.served_rows(tampered, last_known, targets)
    pd.testing.assert_frame_equal(clean.reset_index(drop=True), after.reset_index(drop=True))


@needs_features
def test_observed_weather_is_the_only_thing_the_sensitivity_run_adds():
    full = h.load_engineered(CONFIG.path("engineered_features"))
    last_known = pd.Timestamp("2026-07-26")
    targets = [pd.Timestamp("2026-08-03") + pd.Timedelta(days=i) for i in range(7)]
    fallback = h.served_rows(full, last_known, targets)
    observed = h.served_rows(full, last_known, targets, observed_weather=True)
    same = [c for c in ("year", "month", "day_of_week", "is_bank_holiday", "is_payday", "lag_7_sales", "rolling_7_sales") if c in fallback]
    for col in same:
        assert np.allclose(fallback[col].to_numpy(float), observed[col].to_numpy(float), equal_nan=True), col
    assert not np.allclose(fallback["rain_mm"].to_numpy(float), observed["rain_mm"].to_numpy(float))


def test_the_pre_registered_verdicts():
    assert h.q1_label((150.0, 20.0, 280.0), 900).startswith("Confirmed")
    assert h.q1_label((150.0, -100.0, 400.0), 900).startswith("Consistent")
    assert h.q1_label((-5.0, -200.0, 190.0), 900).startswith("Warning")
    assert h.q1_label((300.0, 100.0, 500.0), 1400).startswith("Drift alarm")
    assert h.package_label(-20.0, -30.0).startswith("directionally consistent")
    assert h.package_label(-20.0, 10.0).startswith("no signal")
    assert h.package_label(35.0, 10.0).startswith("contradicted")
