"""
Tests for the Series 25 weekend-training variants: weights and row subsets do what they say, a separate model
predicts only the days it covers, and the assembly, winner report and bar behave as the plan states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series25_weekend_training as s25
from src.experiments.splits import generate_rolling_origin_folds
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL


def _frame(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({DATE_COL: pd.date_range("2025-01-06", periods=n)})  # starts on a Monday
    for i, col in enumerate(FINAL_FEATURES):
        df[col] = rng.normal(size=n) + (df[DATE_COL].dt.dayofweek.to_numpy() >= 4) * (i == 0)
    df[TARGET_COL] = 6000 + 1500 * df[FINAL_FEATURES[0]] + (df[DATE_COL].dt.dayofweek >= 4) * 4000 + rng.normal(0, 100, n)
    return df


def _folds(df: pd.DataFrame) -> list:
    return generate_rolling_origin_folds(df[DATE_COL], initial_train_days=180, test_days=40, step_days=40)


def _fast(monkeypatch) -> None:
    monkeypatch.setattr(s25, "FINAL_PARAMS", {"iterations": 15, "learning_rate": 0.2, "depth": 2})


def test_a_separate_model_predicts_only_its_days_and_leaves_the_rest_nan(monkeypatch):
    _fast(monkeypatch)
    df = _frame()
    folds = _folds(df)
    out = s25.fit_variant(df, folds, "sepSat", "raw", [1])
    test_dates = pd.concat([df[(df[DATE_COL] >= f.test_start) & (df[DATE_COL] < f.test_end)][DATE_COL] for f in folds])
    saturday = (test_dates.dt.dayofweek == 5).to_numpy()
    assert out.shape == (1, len(test_dates))
    assert not np.isnan(out[0, saturday]).any() and np.isnan(out[0, ~saturday]).all()
    fs = s25.fit_variant(df, folds, "sepFS", "log1p", [1])
    weekend = np.isin(test_dates.dt.dayofweek.to_numpy(), (4, 5, 6))
    assert not np.isnan(fs[0, weekend]).any() and np.isnan(fs[0, ~weekend]).all()


def test_weighted_variants_predict_every_day_and_change_the_fit(monkeypatch):
    _fast(monkeypatch)
    df = _frame()
    folds = _folds(df)
    plain = s25.fit_variant(df, folds, "w1.5", "raw", [3])
    heavy = s25.fit_variant(df, folds, "w3", "raw", [3])
    assert not np.isnan(plain).any() and not np.isnan(heavy).any()
    assert not np.allclose(plain, heavy)


def test_weights_count_friday_to_sunday_rows_only(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, sample_weight=None):
            seen["w"], seen["n"] = sample_weight, len(X)

        def predict(self, X):
            return np.zeros(len(X))

    import catboost

    monkeypatch.setattr(catboost, "CatBoostRegressor", Spy)
    df = _frame()
    s25.fit_variant(df, _folds(df)[:1], "w2", "raw", [0])
    dow = df.loc[df[DATE_COL] < _folds(df)[0].train_end, DATE_COL].dt.dayofweek.to_numpy()
    assert set(np.unique(seen["w"])) == {1.0, 2.0}
    assert np.array_equal(seen["w"] == 2.0, np.isin(dow, (4, 5, 6)))
    s25.fit_variant(df, _folds(df)[:1], "sepSat", "raw", [0])
    assert seen["w"] is None and seen["n"] == int((dow == 5).sum())


def test_assemble_fills_the_gaps_from_the_hybrid():
    var = np.array([1.0, np.nan, 3.0, np.nan])
    hyb = np.array([10.0, 20.0, 30.0, 40.0])
    assert s25.assemble(var, hyb).tolist() == [1.0, 20.0, 3.0, 40.0]


def test_fresh_bags_round_trip_none_as_nan_and_average_over_seeds(tmp_path):
    import json

    path = tmp_path / "r.jsonl"
    rows = [{"id": "a", "variant": "sepSat", "target": "raw", "per_seed": [[1.0, None], [3.0, None]]},
            {"id": "b", "variant": "sepSat", "target": "raw", "per_seed": [[5.0, None]]}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bag = s25.load_variant_bags(path, 3, [("sepSat", "raw")])[("sepSat", "raw")]
    assert bag[0] == 3.0 and np.isnan(bag[1])


def test_the_bar_needs_an_interval_below_zero_and_no_significantly_worse_weekend_day():
    def rep(fs, fri, sat, sun, mon=None):
        rows = [("Friday to Sunday", *fs), ("Friday", *fri), ("Saturday", *sat), ("Sunday", *sun)]
        if mon is not None:
            rows.append(("Monday to Thursday (vs log model)", *mon))
        return pd.DataFrame(rows, columns=["segment", "change", "lo", "hi"])

    good = rep((-10, -20, -1), (5, -20, 30), (-30, -60, 0), (-5, -40, 25), (2, -10, 14))
    assert s25.passes_bar(good) == (True, True, True)
    assert s25.passes_bar(rep((-10, -20, 2), (5, -20, 30), (-30, -60, 0), (-5, -40, 25)))[0] is False
    assert s25.passes_bar(rep((-10, -20, -1), (15, 4, 30), (-30, -60, 0), (-5, -40, 25)))[1] is False
    assert s25.passes_bar(rep((-10, -20, -1), (5, -20, 30), (-30, -60, 0), (-5, -40, 25), (20, 5, 40)))[2] is False


def test_there_are_ten_runs_of_five_seed_chunks_each():
    vt = s25.all_variant_targets()
    assert len(vt) == 10 and len(s25.make_tasks(vt, list(range(25)))) == 50
    assert {v for v, _ in vt} == {"w1.5", "w2", "w3", "sepFS", "sepSat"} and {t for _, t in vt} == {"log1p", "raw"}
