"""
Tests for the Series 23 helpers (log versus raw target on busy days): the segment masks pick
the days they say they do, the masked block bootstrap returns the segment's own mean and a
sensible interval, the slope measures shrinkage, and the raw-target loop matches a plain fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments.splits import generate_rolling_origin_folds
from src.models.final_model import DATE_COL, TARGET_COL


def _days(n: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-06", periods=n)  # starts on a Monday
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "fold_id": np.arange(n) // 14,
        "actual": rng.uniform(4000, 16000, n), "manual": rng.uniform(4000, 16000, n),
        "weekday": dates.day_name(),
    })


def test_segments_pick_the_saturdays_and_the_busiest_days_by_the_right_measure():
    days = _days()
    seg = s23.segments(days)
    assert seg["Saturdays"].sum() == 20 and (days.loc[seg["Saturdays"], "weekday"] == "Saturday").all()
    top_actual = seg["Top 10% of days by ACTUAL sales"]
    top_manager = seg["Top 10% of days by MANAGER's forecast"]
    assert top_actual.sum() == pytest.approx(14, abs=1) and top_manager.sum() == pytest.approx(14, abs=1)
    assert days.loc[top_actual, "actual"].min() >= days.loc[~top_actual, "actual"].max()
    assert days.loc[top_manager, "manual"].min() >= days.loc[~top_manager, "manual"].max()
    top_sat = seg["Top 20% of Saturdays by actual sales"]
    assert top_sat.sum() == 4 and seg["Saturdays"][top_sat].all()


def test_masked_block_bootstrap_returns_the_segment_mean_and_an_interval_around_it():
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 100, 300)
    mask = np.zeros(300)
    mask[::5] = 1.0
    diff = np.where(mask == 1, diff + 50, diff)  # a real +50 shift inside the segment only
    mean, lo, hi = s23.masked_block_bootstrap(diff, mask, block=10, reps=800)
    assert mean == pytest.approx(diff[mask == 1].mean())
    assert lo < mean < hi and lo > 0  # the shift is detected (interval excludes zero)
    zero_mean, zero_lo, zero_hi = s23.masked_block_bootstrap(rng.normal(0, 100, 300), mask, block=10, reps=800)
    assert zero_lo < 0 < zero_hi  # and no shift gives an interval that includes zero


def test_slope_is_one_for_unbiased_forecasts_and_above_one_when_forecasts_are_compressed():
    rng = np.random.default_rng(2)
    actual = rng.uniform(5000, 15000, 400)
    assert s23.slope(actual, actual + rng.normal(0, 1, 400)) == pytest.approx(1.0, abs=0.01)
    compressed = actual.mean() + 0.5 * (actual - actual.mean())
    assert s23.slope(actual, compressed) == pytest.approx(2.0, abs=0.01)
    assert np.isnan(s23.slope(actual[:2], actual[:2]))


def test_decile_table_has_ten_equal_groups_and_the_top_decile_is_the_busiest():
    days = _days(200)
    preds = {"log1p_final": days["actual"].to_numpy() * 0.9, "raw_same_params": days["actual"].to_numpy() * 1.0,
             "raw_series8_params": days["actual"].to_numpy() * 1.0}
    t = s23.decile_table(days, preds)
    assert list(t["decile_of_actual"]) == list(range(1, 11)) and (t["days"] == 20).all()
    assert t["mean_actual"].is_monotonic_increasing
    assert (t["log1p_final_bias"] < 0).all() and t["raw_same_params_bias"].abs().max() < 1e-9


def test_raw_predictions_equal_a_plain_catboost_fit_on_each_fold():
    from catboost import CatBoostRegressor

    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({DATE_COL: pd.date_range("2025-01-01", periods=n), "x": rng.normal(size=n)})
    df[TARGET_COL] = 5000 + 800 * df["x"] + rng.normal(0, 50, n)
    folds = generate_rolling_origin_folds(df[DATE_COL], initial_train_days=80, test_days=20, step_days=20)
    params = {"iterations": 20, "learning_rate": 0.1, "depth": 2}
    out = s23.raw_predictions(df, folds, ["x"], params, [7])
    assert out.shape == (1, 40)
    fold = folds[0]
    train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
    test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
    model = CatBoostRegressor(random_state=7, verbose=False, allow_writing_files=False, thread_count=1, **params)
    model.fit(train[["x"]], train[TARGET_COL])
    assert np.allclose(out[0, :20], model.predict(test[["x"]]))


def test_the_three_variants_and_their_seed_chunks_cover_25_seeds_each():
    todo = s23.tasks()
    assert len({t["id"] for t in todo}) == len(todo) == 15
    for variant in s23.VARIANTS:
        seeds = sorted(s for t in todo if t["variant"] == variant for s in t["seeds"])
        assert seeds == list(range(25))
    assert s23.VARIANTS["raw_series8_params"]["params"]["learning_rate"] == 0.03
    assert s23.VARIANTS["log1p_final"]["target"] == "log1p"
