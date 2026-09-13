"""
Tests for src/models/final_model.py - the model the API serves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.series_common import ADD_ALL_CANDIDATES, LOG1P_PARAMS
from src.models.final_model import FINAL_FEATURES, FINAL_PARAMS, BaggedCatBoost


def test_final_configuration_matches_the_experiment_log():
    """Series 16's winner: the add_all_candidates features, in the same
    order (column order changes seeded CatBoost fits), with the Series 14
    log1p hyperparameters."""
    assert FINAL_FEATURES == ADD_ALL_CANDIDATES
    assert FINAL_PARAMS == LOG1P_PARAMS


def make_training_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.uniform(0, 1, (rows, 3)), columns=["a", "b", "c"])
    df["total_sales"] = 5000 + 8000 * df["a"] + rng.normal(0, 200, rows)
    return df


def test_bagged_prediction_is_the_mean_of_each_seed_on_the_sales_scale():
    df = make_training_frame()
    model = BaggedCatBoost.fit(df, features=["a", "b", "c"], params={"iterations": 20, "depth": 2}, seeds=[0, 1, 2])
    manual = np.mean([np.expm1(m.predict(df[["a", "b", "c"]])) for m in model.models], axis=0)
    assert np.allclose(model.predict(df), manual)


def test_save_and_load_round_trip(tmp_path):
    df = make_training_frame()
    model = BaggedCatBoost.fit(df, features=["a", "b", "c"], params={"iterations": 20, "depth": 2}, seeds=[0, 1])
    model.save(tmp_path, metadata={"note": "test"})

    loaded, metadata = BaggedCatBoost.load(tmp_path)
    assert metadata["note"] == "test"
    assert metadata["features"] == ["a", "b", "c"]
    assert np.allclose(loaded.predict(df), model.predict(df))
    # A numpy array in feature order gives the same answer as a DataFrame.
    assert np.allclose(loaded.predict(df[["a", "b", "c"]].to_numpy()), model.predict(df))
