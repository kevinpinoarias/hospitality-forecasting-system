"""
Shared building blocks for the Series 5-17 runners (see
docs/EXPERIMENT_LOG.md).

Series 1-4 each got a dedicated script as they were run. Series 5-16 were
originally run as one-off computations during the same working session;
their runners were reconstructed afterwards from the exact code that
produced each logged result, so every figure in the experiment log can be
regenerated from committed code rather than trusted on the log's word.

These helpers exist only where several series share literally the same
loop - each runner still states its own configuration explicitly, so a
series can be read and understood without chasing indirection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.adapters import DATE_COL, FEATURES as BASELINE_FEATURES, MODEL_ADAPTERS, TARGET_COL
from src.experiments.feature_sets import ALL_CANDIDATES
from src.experiments.grids import TRAIN_WINDOW_OPTIONS
from src.experiments.run_feature_sweep import build_folds, load_data
from src.experiments.splits import slice_train_window, slice_test_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "reports" / "experiments"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# The 38-feature winning set from Series 2 (15 production + 23 candidates).
ADD_ALL_CANDIDATES = list(BASELINE_FEATURES) + list(ALL_CANDIDATES)

# Series 1/2/8 winner on the raw sales scale.
REFERENCE_PARAMS = {"iterations": 300, "learning_rate": 0.03, "depth": 4}

# Series 14 winner, trained on log1p(sales).
LOG1P_PARAMS = {"iterations": 300, "learning_rate": 0.05, "depth": 4}

__all__ = [
    "ADD_ALL_CANDIDATES", "DATE_COL", "FIGURES_DIR", "LOG1P_PARAMS", "REFERENCE_PARAMS",
    "RESULTS_DIR", "TARGET_COL", "TRAIN_WINDOW_OPTIONS", "apply_december_ratio", "build_folds",
    "collect_fold_predictions", "fit_predict_catboost_seeded", "fold_mean_metrics", "leaderboard",
    "load_data", "pooled_mae", "require_output", "results_path", "run_catboost_trials",
    "slice_test_window", "slice_train_window",
]


def results_path(filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename


def require_output(filename: str, producing_script: str) -> Path:
    """Series 9-17 chain off earlier series' saved predictions; fail with a
    pointer to the right script instead of a bare FileNotFoundError."""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.experiments.{producing_script}` first "
            f"(or `python -m src.experiments.run_all_series` to run every series in order)."
        )
    return path


def run_catboost_trials(
    trials: dict[str, tuple[dict, list[str]]],
    label_col: str,
    windows: list[int | None] = TRAIN_WINDOW_OPTIONS,
) -> pd.DataFrame:
    """Score each named (params, features) trial on every rolling-origin
    fold and training window, one row of metrics per fold - the loop shared
    verbatim by Series 5, 6, 7, 8 and 13."""
    df = load_data()
    folds = build_folds(df)

    rows = []
    for name, (params, features) in trials.items():
        for window in windows:
            for fold in folds:
                train_df = slice_train_window(df, DATE_COL, fold, window)
                test_df = slice_test_window(df, DATE_COL, fold)
                result = MODEL_ADAPTERS["catboost"](train_df, test_df, params, features=features)
                metrics = regression_metrics(result["actual"], result["pred"])
                rows.append({
                    label_col: name,
                    "train_window_days": window if window else "expanding",
                    "fold_id": fold.fold_id,
                    **metrics,
                })
    return pd.DataFrame(rows)


def collect_fold_predictions(params: dict, features: list[str], window: int | None = None) -> pd.DataFrame:
    """Per-day out-of-fold predictions (date, actual, pred, fold_id) from
    the CatBoost adapter across every rolling-origin fold - the base that
    Series 9-12's post-hoc corrections are applied on top of."""
    df = load_data()
    folds = build_folds(df)
    parts = []
    for fold in folds:
        train_df = slice_train_window(df, DATE_COL, fold, window)
        test_df = slice_test_window(df, DATE_COL, fold)
        result = MODEL_ADAPTERS["catboost"](train_df, test_df, params, features=features)
        result["fold_id"] = fold.fold_id
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def apply_december_ratio(preds: pd.DataFrame, ratio_by_day: dict[int, float], out_col: str) -> pd.DataFrame:
    """Multiply December predictions by a per-day-of-month ratio; every
    other month passes through unchanged."""
    out = preds.copy()
    out["date"] = pd.to_datetime(out["date"])
    is_dec = out["date"].dt.month == 12
    ratio = out["date"].dt.day.map(ratio_by_day)
    out[out_col] = out["pred"]
    out.loc[is_dec, out_col] = out.loc[is_dec, "pred"] * ratio[is_dec]
    return out


def pooled_mae(df: pd.DataFrame, pred_col: str, december_only: bool = False) -> float:
    """MAE pooled over individual days (not averaged per fold) - Series 9-12
    report corrections this way because December days are concentrated in
    a couple of folds, so a per-fold average would dilute the effect."""
    sub = df[pd.to_datetime(df["date"]).dt.month == 12] if december_only else df
    return float((sub["actual"] - sub[pred_col]).abs().mean())


def leaderboard(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        results.groupby(group_cols)
        .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
        .reset_index()
        .sort_values("MAE_mean")
    )


def fit_predict_catboost_seeded(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    features: list[str],
    seed: int = 42,
    log1p_target: bool = False,
) -> np.ndarray:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, **params)
    target = np.log1p(train_df[TARGET_COL]) if log1p_target else train_df[TARGET_COL]
    model.fit(train_df[features], target)
    pred = model.predict(test_df[features])
    return np.expm1(pred) if log1p_target else pred


def fold_mean_metrics(df: pd.DataFrame, pred_col: str, actual_col: str = "actual") -> dict:
    """Metrics averaged across folds - the convention every series in the
    log reports, as opposed to pooling all test days into one sample."""
    per_fold = pd.DataFrame([
        regression_metrics(g[actual_col], g[pred_col]) for _, g in df.groupby("fold_id")
    ])
    return {
        "MAE_mean": per_fold["MAE"].mean(),
        "MAE_std": per_fold["MAE"].std(),
        "RMSE_mean": per_fold["RMSE"].mean(),
        "MAPE_mean": per_fold["MAPE"].mean(),
        "Bias_mean": per_fold["Bias"].mean(),
    }
