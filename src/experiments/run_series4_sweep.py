"""
Series 4: short-lag/rolling feature test, on top of Series 2's winning
`add_all_candidates` configuration (38 features) - not the raw 15-feature
production baseline.

Tests the 6 features engineered on 2026-09-10 (`lag_1_sales`,
`lag_2_sales`, `lag_3_sales`, `lag_2_fe`, `lag_3_fe`, `rolling_28_sales` -
see docs/EXPERIMENT_LOG.md's appendix), which fill a gap Series 2's own
conclusions identified: plain "yesterday's sales" wasn't in the engineered
pool at all, despite `lag_7_sales` and `lag_1_fe` being the two strongest
individual features found in that series.

Two questions:
1. Do all 6 together, added on top of `add_all_candidates`, improve
   further, or has that signal already been captured by the existing
   lag_7_sales/lag_1_fe/rolling_7_fe features?
2. Individually, does each one help or hurt on that same 38-feature base?

Deliberately uses Series 2's original (non-recursive) rolling-origin
evaluation, not Series 3's recursive/no-fresh-data methodology - this
keeps the result directly comparable to the £907.77 `add_all_candidates`
figure. CatBoost only, fixed at its Series 1/2 winning hyperparameters
(iterations=300, learning_rate=0.03, depth=4), the same 10 folds, and the
same two training-window modes as Series 2, for full methodological
parity. `add_all_candidates` itself is re-run here too (not just cited)
so it sits in the same directly-comparable table as everything else.

Outputs
-------
- reports/experiments/series4_results_full.csv
- reports/experiments/series4_leaderboard.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.adapters import DATE_COL, FEATURES as BASELINE_FEATURES, MODEL_ADAPTERS
from src.experiments.feature_sets import ALL_CANDIDATES
from src.experiments.grids import TRAIN_WINDOW_OPTIONS
from src.experiments.run_feature_sweep import build_folds, load_data
from src.experiments.splits import Fold, slice_test_window, slice_train_window

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments"

CATBOOST_PARAMS = {"iterations": 300, "learning_rate": 0.03, "depth": 4}

# Series 2's winning 38-feature configuration.
ADD_ALL_CANDIDATES = list(BASELINE_FEATURES) + list(ALL_CANDIDATES)

NEW_FEATURES = ["lag_1_sales", "lag_2_sales", "lag_3_sales", "lag_2_fe", "lag_3_fe", "rolling_28_sales"]

FEATURE_SETS: dict[str, list[str]] = {
    "add_all_candidates (Series 2 reference, re-run here)": ADD_ALL_CANDIDATES,
    "add_all_candidates_plus_all_6_new": ADD_ALL_CANDIDATES + NEW_FEATURES,
}
for feat in NEW_FEATURES:
    FEATURE_SETS[f"add_all_candidates_plus__{feat}"] = ADD_ALL_CANDIDATES + [feat]


def run_one_fit(feature_set_name: str, features: list[str], fold: Fold, train_window_days: int | None, df: pd.DataFrame) -> dict:
    train_df = slice_train_window(df, DATE_COL, fold, train_window_days)
    test_df = slice_test_window(df, DATE_COL, fold)

    row = {
        "feature_set": feature_set_name,
        "n_features": len(features),
        "train_window_days": train_window_days if train_window_days is not None else "expanding",
        "fold_id": fold.fold_id,
        "train_start": train_df[DATE_COL].min().date() if len(train_df) else None,
        "train_end": fold.train_end.date(),
        "test_start": fold.test_start.date(),
        "test_end": fold.test_end.date(),
    }

    try:
        result = MODEL_ADAPTERS["catboost"](train_df, test_df, CATBOOST_PARAMS, features=features)
        result = result.dropna(subset=["actual", "pred"])
        if len(result) == 0:
            raise ValueError("no non-null predictions produced")
        metrics = regression_metrics(result["actual"], result["pred"])
        row.update({"status": "OK", "n": len(result), **metrics})
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "ERROR", "n": 0, "error": f"{type(exc).__name__}: {exc}"})

    return row


def run_series4() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    folds = build_folds(df)

    total = len(FEATURE_SETS) * len(TRAIN_WINDOW_OPTIONS) * len(folds)
    print(f"Series 4: {len(FEATURE_SETS)} feature sets x {len(TRAIN_WINDOW_OPTIONS)} window modes x "
          f"{len(folds)} folds = {total} fits")

    rows = []
    for name, features in FEATURE_SETS.items():
        for window in TRAIN_WINDOW_OPTIONS:
            for fold in folds:
                rows.append(run_one_fit(name, features, fold, window, df))
        print(f"  done: {name}")

    results = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "series4_results_full.csv"
    results.to_csv(results_path, index=False)

    n_errors = int((results["status"] == "ERROR").sum())
    ok = results[results["status"] == "OK"]
    leaderboard = (
        ok.groupby(["feature_set", "n_features", "train_window_days"])
        .agg(
            n_folds=("fold_id", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAPE_mean=("MAPE", "mean"),
            Bias_mean=("Bias", "mean"),
        )
        .reset_index()
        .sort_values("MAE_mean")
        .reset_index(drop=True)
    )
    leaderboard_path = RESULTS_DIR / "series4_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"\n{len(results)} fits total, {n_errors} failed")
    print(f"Full results: {results_path}")
    print(f"Leaderboard: {leaderboard_path}\n")
    print(leaderboard.to_string(index=False))

    return {"results": results, "leaderboard": leaderboard}


if __name__ == "__main__":
    run_series4()
