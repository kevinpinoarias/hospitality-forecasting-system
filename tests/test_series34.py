"""
Tests for Series 34: the weights halve every half-life and respect the floor, the effective sample size shrinks as the weighting
sharpens, the variant names decode, every fit uses weights computed only from its own training days, and the bar behaves as stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series34_recency_weights as s34


def test_the_weight_halves_every_half_life_and_the_last_training_day_counts_fully():
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-04-02", "2025-07-02"]))  # 0, 91 and 182 days before 2025-07-02
    last = pd.Timestamp("2025-07-02")
    w = s34.recency_weights(dates, last, half_life=91)
    assert np.allclose(w, [0.25, 0.5, 1.0])
    assert s34.recency_weights(dates, last, half_life=91)[-1] == 1.0


def test_the_floor_keeps_the_oldest_days_from_vanishing():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2025-07-02"]))
    w = s34.recency_weights(dates, pd.Timestamp("2025-07-02"), half_life=45, floor=0.2)
    assert w[0] == 0.2 and w[1] == 1.0
    assert s34.recency_weights(dates, pd.Timestamp("2025-07-02"), half_life=45)[0] < 1e-3


def test_effective_sample_size_is_the_row_count_unweighted_and_shrinks_with_a_shorter_half_life():
    assert np.isclose(s34.effective_sample_size(np.ones(100)), 100.0)
    dates = pd.Series(pd.date_range("2024-01-01", periods=500))
    last = dates.max()
    sizes = [s34.effective_sample_size(s34.recency_weights(dates, last, h)) for h in (45, 90, 180, 365, 730)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1] < 500
    assert s34.effective_sample_size(s34.recency_weights(dates, last, 45, floor=0.2)) > s34.effective_sample_size(s34.recency_weights(dates, last, 45))


def test_variant_names_decode_to_half_life_and_floor():
    assert s34.variant_spec("hl45") == (45, 0.0)
    assert s34.variant_spec("hl180_floor") == (180, 0.2)


def test_the_plan_matches_the_header():
    assert s34.HALF_LIVES == (45, 90, 180, 365, 730) and s34.FLOOR == 0.2
    tasks = s34.make_tasks([f"hl{h}" for h in s34.HALF_LIVES], list(range(25)))
    assert len(tasks) == 50 and len({t["id"] for t in tasks}) == 50


def test_weights_in_a_fold_use_only_that_folds_training_days(monkeypatch):
    seen = []

    class Spy:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, sample_weight=None):
            seen.append((len(X), np.asarray(sample_weight).copy()))

        def predict(self, X):
            return np.zeros(len(X))

    import catboost

    from src.experiments.splits import generate_rolling_origin_folds
    from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

    monkeypatch.setattr(catboost, "CatBoostRegressor", Spy)
    n = 120
    rng = np.random.default_rng(0)
    df = pd.DataFrame({DATE_COL: pd.date_range("2025-01-01", periods=n)})
    for c in FINAL_FEATURES:
        df[c] = rng.normal(size=n)
    df[TARGET_COL] = rng.uniform(5000, 9000, n)
    folds = generate_rolling_origin_folds(df[DATE_COL], initial_train_days=80, test_days=20, step_days=20)
    out, ess = s34.fit_variant(df, folds, "hl30", "raw", [1])
    assert out.shape == (1, 40) and len(ess) == len(folds)
    first_rows, first_w = seen[0]
    assert first_rows == 80 and first_w[-1] == 1.0 and first_w[0] < first_w[-1]  # the last training day has weight 1, older days less


def test_the_bar_needs_an_overall_interval_below_zero_and_no_segment_significantly_worse():
    def table(overall_hi, segment_lo):
        return pd.DataFrame([{"segment": "All 585 days", "lo": -20.0, "hi": overall_hi}, {"segment": "Saturday", "lo": segment_lo, "hi": 30.0}])

    assert s34.passes_bar(table(-1.0, -10.0)) == (True, True)
    assert s34.passes_bar(table(2.0, -10.0))[0] is False
    assert s34.passes_bar(table(-1.0, 5.0))[1] is False
