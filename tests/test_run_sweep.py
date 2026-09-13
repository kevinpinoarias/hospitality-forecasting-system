"""
Tests for src/experiments/run_sweep.py's per-fit bookkeeping.

Regression test for a real bug: run_one_fit originally logged the fold's
nominal train_start (always the dataset's absolute start date) instead of
the actual training data date range after a sliding train_window_days was
applied - the models themselves trained on the correctly restricted data,
but the results CSV misreported what period that was, which would mislead
anyone reading the report (e.g. "what period was the winning config
trained on?") even though the metrics themselves were unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.run_sweep import run_one_fit
from src.experiments.splits import generate_rolling_origin_folds


def make_model_df(n_days: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": dates,
        "total_sales": rng.random(n_days) * 1000,
        "day_of_week": dates.dayofweek,
    })
    return df


def test_train_start_reflects_actual_sliding_window_not_fold_nominal_start():
    model_df = make_model_df()
    folds = generate_rolling_origin_folds(model_df["date"], initial_train_days=200, test_days=30)
    fold = folds[2]  # a later fold with plenty of accumulated history

    row = run_one_fit("naive", {}, fold, train_window_days=60, model_df=model_df, eng_df=model_df)

    expected_train_start = (fold.train_end - pd.Timedelta(days=60)).date()
    assert row["train_start"] == expected_train_start
    assert row["train_start"] != fold.train_start.date()


def test_train_start_matches_fold_start_for_expanding_window():
    model_df = make_model_df()
    folds = generate_rolling_origin_folds(model_df["date"], initial_train_days=200, test_days=30)
    fold = folds[2]

    row = run_one_fit("naive", {}, fold, train_window_days=None, model_df=model_df, eng_df=model_df)

    assert row["train_start"] == fold.train_start.date()


def test_successful_fit_reports_ok_status_and_metrics():
    model_df = make_model_df()
    folds = generate_rolling_origin_folds(model_df["date"], initial_train_days=200, test_days=30)
    fold = folds[0]

    row = run_one_fit("naive", {}, fold, train_window_days=None, model_df=model_df, eng_df=model_df)

    assert row["status"] == "OK"
    assert row["n"] > 0
    assert "MAE" in row
