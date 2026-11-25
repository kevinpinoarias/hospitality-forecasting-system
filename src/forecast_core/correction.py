"""
The Saturday and Sunday correction from recent errors (Series 24, as validated in the package and fixed in
the holdout's pre-registration). Moved unchanged from src/experiments/run_series24_weekend_correction.py
(correct), run_package_validation.py (CORRECTION), run_series35_holdout.py (correct_new_block) and
src/forecast_log/v1.py (the correction history files).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.forecast_core.rows import weekday_names

SCOPE = ("Friday", "Saturday", "Sunday")
MIN_HISTORY = 4
CORRECTION = {"half_life_weeks": 8, "strength": 0.5, "scope": ("Saturday", "Sunday")}


def correct(days: pd.DataFrame, pred: np.ndarray, half_life_weeks: float, strength: float, scope: tuple[str, ...] = SCOPE) -> np.ndarray:
    """The adaptive weekday correction. Block k's forecasts for each weekday in `scope` are shifted by
    -strength x (the recency-weighted mean of forecast minus actual for that weekday in blocks BEFORE k).
    Block 0, and weekdays outside `scope`, are unchanged."""
    out = pred.astype(float).copy()
    fold = days["fold_id"].to_numpy()
    weekday = days["weekday"].to_numpy()
    dates = pd.to_datetime(days["date"]).to_numpy()
    error = pred - days["actual"].to_numpy()
    for k in sorted(set(fold.tolist()))[1:]:
        block = fold == k
        start = dates[block].min()
        for day in scope:
            history = (fold < k) & (weekday == day)
            if history.sum() < MIN_HISTORY:
                continue
            age_weeks = (start - dates[history]) / np.timedelta64(7, "D")
            weights = 0.5 ** (age_weeks / half_life_weeks)
            bias = float(np.sum(weights * error[history]) / np.sum(weights))
            out[block & (weekday == day)] = pred[block & (weekday == day)] - strength * bias
    return out


def correct_new_block(hist_days: pd.DataFrame, hist_pred: np.ndarray, new_dates, new_pred: np.ndarray) -> np.ndarray:
    """The Series 24 correction for a new test block: shift each Saturday and Sunday forecast by -strength x the recency-weighted bias of the
    package's errors in the history, exactly as the protocol does for a block after the history."""
    new_days = pd.DataFrame({"date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in new_dates],
                             "fold_id": int(hist_days["fold_id"].max()) + 1, "actual": np.nan, "weekday": weekday_names(new_dates)})
    days = pd.concat([hist_days[["date", "fold_id", "actual", "weekday"]], new_days], ignore_index=True)
    pred = np.concatenate([hist_pred, new_pred])
    return correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])[len(hist_days):]


def correction_history_path(version: str) -> Path:
    """The Saturday and Sunday correction history a version uses (frozen at its cut-off, kept beside the runner's data)."""
    return PROJECT_ROOT / "data" / "training" / f"{version}_correction_history.csv"


def load_correction_history(path: Path) -> tuple[pd.DataFrame, "pd.Series"]:
    frame = pd.read_csv(path)
    return frame[["date", "fold_id", "actual", "weekday"]], frame["package_weekend_forecast"].to_numpy()
