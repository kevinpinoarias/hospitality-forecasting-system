"""
The final forecasting model: CatBoost on the 38-feature `add_all_candidates`
set, trained on log1p(sales) and averaged over 25 random seeds.

This is the configuration the experimentation programme settled on (see
docs/EXPERIMENT_LOG.md, Series 14 for the log1p target and Series 16 for
seed bagging). It lives in its own lightweight module, separate from the
training script, so the API can load and run the model without importing
the training pipeline's plotting and tracking stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "final_model"
METADATA_FILENAME = "metadata.json"
BACKTEST_PREDICTIONS_FILENAME = "backtest_predictions.csv"

DATE_COL = "date"
TARGET_COL = "total_sales"

# The 15 original production features followed by the 23 candidates added
# in Series 2, in the same order as ADD_ALL_CANDIDATES in
# src/experiments/series_common.py - column order changes CatBoost's
# seeded fits, so it has to match for the backtest to reproduce the log.
# tests/test_final_model.py checks the two stay identical.
FINAL_FEATURES = [
    "forecast_sales",
    "month_sin",
    "day_of_week",
    "day_of_year_cos",
    "day_of_year_sin",
    "day_of_year",
    "month_cos",
    "month",
    "day_of_week_sin",
    "is_bank_holiday",
    "days_to_bank_holiday",
    "days_since_payday",
    "is_payday_window_pm3",
    "is_long_weekend",
    "is_heavy_rain",
    "lag_7_sales",
    "rolling_7_sales",
    "rolling_14_sales",
    "lag_1_fe",
    "lag_7_fe",
    "rolling_7_fe",
    "year",
    "is_weekend",
    "day_of_week_cos",
    "is_payday",
    "days_to_payday",
    "days_since_bank_holiday",
    "is_christmas_break",
    "is_summer_break",
    "is_easter_break",
    "is_school_holiday",
    "max_temp",
    "rain_mm",
    "sun_hours",
    "is_hot_for_scotland",
    "is_dry_day",
    "temp_anomaly_14d",
    "warm_streak_len",
]

FINAL_PARAMS = {"iterations": 300, "learning_rate": 0.05, "depth": 4}
FINAL_SEEDS = list(range(25))


class BaggedCatBoost:
    """CatBoost models that differ only in random seed. Each model predicts
    on the log1p scale; predictions are converted back to sales and then
    averaged, exactly as in Series 16."""

    def __init__(self, models: list, features: list[str]) -> None:
        self.models = models
        self.features = features

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        features: list[str] = FINAL_FEATURES,
        params: dict = FINAL_PARAMS,
        seeds: list[int] = FINAL_SEEDS,
    ) -> "BaggedCatBoost":
        from catboost import CatBoostRegressor

        target = np.log1p(train_df[TARGET_COL])
        models = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, **params)
            model.fit(train_df[features], target)
            models.append(model)
        return cls(models, list(features))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.features]
        return np.mean([np.expm1(model.predict(X)) for model in self.models], axis=0)

    def save(self, directory: Path, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("seed_*.cbm"):
            old.unlink()
        filenames = []
        for i, model in enumerate(self.models):
            filename = f"seed_{i:02d}.cbm"
            model.save_model(str(directory / filename))
            filenames.append(filename)
        with open(directory / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump({**metadata, "features": self.features, "model_files": filenames}, f, indent=2)

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> tuple["BaggedCatBoost", dict]:
        from catboost import CatBoostRegressor

        with open(directory / METADATA_FILENAME, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        models = []
        for filename in metadata["model_files"]:
            model = CatBoostRegressor()
            model.load_model(str(directory / filename))
            models.append(model)
        return cls(models, metadata["features"]), metadata
