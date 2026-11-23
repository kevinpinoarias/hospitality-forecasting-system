"""A CatBoost bag over seeds, and the file fingerprints the forecast log registers (moved from src/experiments/run_series35_holdout.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.final_model import FINAL_PARAMS

DATE_COL, TARGET_COL = "date", "total_sales"
SEEDS = list(range(25))


class Bag:
    """A CatBoost bag over seeds on one target (log1p or raw pounds); the final model's own recipe."""

    def __init__(self, models: list, features: list[str], target: str) -> None:
        self.models, self.features, self.target = models, list(features), target

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str], target: str, seeds: list[int] = SEEDS, thread_count: int = 4) -> "Bag":
        from catboost import CatBoostRegressor

        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        models = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=thread_count, **FINAL_PARAMS)
            model.fit(train[list(features)], y)
            models.append(model)
        return cls(models, features, target)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X[self.features]) for m in self.models]
        if self.target == "log1p":
            preds = [np.expm1(p) for p in preds]
        return np.mean(preds, axis=0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("seed_*.cbm"):
            old.unlink()
        for i, m in enumerate(self.models):
            m.save_model(str(directory / f"seed_{i:02d}.cbm"))
        (directory / "meta.json").write_text(json.dumps({"features": self.features, "target": self.target, "seeds": len(self.models)}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Bag":
        from catboost import CatBoostRegressor

        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        models = []
        for i in range(meta["seeds"]):
            m = CatBoostRegressor()
            m.load_model(str(directory / f"seed_{i:02d}.cbm"))
            models.append(m)
        return cls(models, meta["features"], meta["target"])


def fingerprint(directory: Path) -> str:
    """sha256 of every file in a folder, sorted by name and concatenated (the pre-registration's convention)."""
    h = hashlib.sha256()
    for f in sorted(p for p in directory.iterdir() if p.is_file()):
        h.update(f.read_bytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engineered(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)
