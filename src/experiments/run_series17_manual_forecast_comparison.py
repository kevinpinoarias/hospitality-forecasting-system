"""
Series 17 - The final model against the venue's manual forecast.

Every earlier series compares models with each other. This one answers
the question that matters commercially: how does the final model (Series
16's 25-seed bagged log1p CatBoost) compare with the forecast the venue
actually uses - the manager's manual sales forecast?

All metrics here are pooled over individual days rather than averaged per
fold, because the comparison is over a set of matched calendar days rather
than over folds. This is why the final model reads £919 here but ~£910 in
Series 16: same model, different averaging and a different set of days.

Section A - the headline figure (reproduced exactly):
  1. Re-generate the Series 1 CatBoost winner's out-of-fold predictions
     (15 production features, sliding 365-day window) on Series 1's folds.
  2. Join them to labour data. Closure days have no labour records and
     drop out - 596 days.
  3. Intersect with the final model's test days - 569 days,
     2025-01-15 to 2026-08-23.

      |                     | MAE    | MAPE  |
      | Manual forecast     | £1,086 | 14.9% |
      | Series 1 CatBoost   | £967   |       |
      | Final bagged model  | £919   | 12.0% |

  About 15% more accurate than the manual process.

  Caveat: the 569-day window is the overlap of two different fold setups
  (Series 1's and Series 2-16's), not a window chosen on its own merits.

Section B - robustness of that window (new measurement): the same final-
vs-manual comparison across every final-model test day that has labour
data, not only those overlapping Series 1.

Section C - like-for-like model history (new measurement): the original
portfolio XGBoost (15 features, its production hyperparameters) retrained
on the same rolling-origin folds and scored on the same 569 days, so the
portfolio model can be compared with the final model under one protocol
rather than a single-split score against a backtested one.

Requires: series16_bagging_raw_predictions.csv (Series 16)
Outputs (reports/experiments/):
- catboost_winner_predictions.csv
- wage_comparison_ai_vs_manual_vs_actual.csv
- series17_manual_comparison_matched_days.csv
- series17_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.adapters import MODEL_ADAPTERS, fit_predict_catboost
from src.experiments.run_sweep import (
    DEFAULT_INITIAL_TRAIN_DAYS, DEFAULT_STEP_DAYS, DEFAULT_TEST_DAYS,
    build_folds as build_series1_folds, load_data as load_series1_data,
)
from src.experiments.series_common import (
    DATE_COL, build_folds, load_data, require_output, results_path, slice_test_window, slice_train_window,
)
from src.models.evaluate_human_forecast import load_processed_data, prepare_merged_dataset

SERIES1_CATBOOST_PARAMS = {"iterations": 300, "learning_rate": 0.03, "depth": 4}
SERIES1_WINDOW_DAYS = 365

# Identical to src/models/train_xgboost.py's build_model(); the sweep's
# XGBoost adapter already fixes objective and random_state.
PORTFOLIO_XGBOOST_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
}


def _mae(actual, pred) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _mape(actual, pred) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def series1_winner_predictions() -> pd.DataFrame:
    model_df, _ = load_series1_data()
    folds = build_series1_folds(model_df, DEFAULT_INITIAL_TRAIN_DAYS, DEFAULT_TEST_DAYS, DEFAULT_STEP_DAYS)
    parts = []
    for fold in folds:
        train_df = slice_train_window(model_df, DATE_COL, fold, SERIES1_WINDOW_DAYS)
        test_df = slice_test_window(model_df, DATE_COL, fold)
        parts.append(fit_predict_catboost(train_df, test_df, SERIES1_CATBOOST_PARAMS))
    return pd.concat(parts, ignore_index=True).rename(columns={"pred": "ai_prediction", "actual": "total_sales"})


def labour_dataset() -> pd.DataFrame:
    sales, labour = load_processed_data()
    return prepare_merged_dataset(sales, labour)


def portfolio_xgboost_predictions() -> pd.DataFrame:
    df = load_data()
    parts = []
    for fold in build_folds(df):
        train_df = slice_train_window(df, DATE_COL, fold, None)
        test_df = slice_test_window(df, DATE_COL, fold)
        parts.append(MODEL_ADAPTERS["xgboost"](train_df, test_df, PORTFOLIO_XGBOOST_PARAMS))
    return pd.concat(parts, ignore_index=True).rename(columns={"pred": "portfolio_xgboost_pred"})


def run() -> dict:
    bag = pd.read_csv(require_output("series16_bagging_raw_predictions.csv", "run_series16_seed_bagging"), parse_dates=["date"])
    final = bag.groupby(["date", "actual"], as_index=False)["pred"].mean().rename(columns={"pred": "final_pred"})

    # --- Section A: the headline 569-day comparison ---
    series1 = series1_winner_predictions()
    series1.to_csv(results_path("catboost_winner_predictions.csv"), index=False)

    labour = labour_dataset()
    wage = series1.merge(labour[["date", "forecast_sales", "total_wages", "forecast_wages"]], on="date", how="inner")
    wage.to_csv(results_path("wage_comparison_ai_vs_manual_vs_actual.csv"), index=False)

    matched = final.merge(wage[["date", "total_sales", "forecast_sales", "ai_prediction"]], on="date", how="inner")
    assert np.allclose(matched["actual"], matched["total_sales"]), "actual sales disagree between sources"

    xgb = portfolio_xgboost_predictions()
    matched = matched.merge(xgb[["date", "portfolio_xgboost_pred"]], on="date", how="left")
    matched.to_csv(results_path("series17_manual_comparison_matched_days.csv"), index=False)

    manual_mae = _mae(matched["actual"], matched["forecast_sales"])
    rows = [
        {"section": "A", "model": "Manual forecast", "days": len(matched),
         "MAE": manual_mae, "MAPE": _mape(matched["actual"], matched["forecast_sales"])},
        {"section": "A", "model": "Series 1 CatBoost winner", "days": len(matched),
         "MAE": _mae(matched["actual"], matched["ai_prediction"]), "MAPE": _mape(matched["actual"], matched["ai_prediction"])},
        {"section": "A", "model": "Final bagged log1p CatBoost", "days": len(matched),
         "MAE": _mae(matched["actual"], matched["final_pred"]), "MAPE": _mape(matched["actual"], matched["final_pred"])},
        {"section": "C", "model": "Portfolio XGBoost, same folds", "days": int(matched["portfolio_xgboost_pred"].notna().sum()),
         "MAE": _mae(matched["actual"], matched["portfolio_xgboost_pred"]), "MAPE": _mape(matched["actual"], matched["portfolio_xgboost_pred"])},
    ]

    # --- Section B: every final-model test day that has labour data ---
    full = final.merge(labour[["date", "forecast_sales"]], on="date", how="inner")
    rows += [
        {"section": "B", "model": "Manual forecast", "days": len(full),
         "MAE": _mae(full["actual"], full["forecast_sales"]), "MAPE": _mape(full["actual"], full["forecast_sales"])},
        {"section": "B", "model": "Final bagged log1p CatBoost", "days": len(full),
         "MAE": _mae(full["actual"], full["final_pred"]), "MAPE": _mape(full["actual"], full["final_pred"])},
    ]

    summary = pd.DataFrame(rows)
    summary["pct_better_than_manual"] = np.nan
    for section, g in summary.groupby("section"):
        manual = summary.loc[(summary["section"] == section) & (summary["model"] == "Manual forecast"), "MAE"]
        baseline = manual.iloc[0] if len(manual) else manual_mae
        summary.loc[g.index, "pct_better_than_manual"] = (baseline - g["MAE"]) / baseline * 100
    summary.to_csv(results_path("series17_summary.csv"), index=False)

    return {
        "summary": summary,
        "matched_window": (matched["date"].min().date(), matched["date"].max().date()),
        "full_window": (full["date"].min().date(), full["date"].max().date()),
    }


if __name__ == "__main__":
    out = run()
    print(f"Section A/C window: {out['matched_window'][0]} to {out['matched_window'][1]}")
    print(f"Section B window:   {out['full_window'][0]} to {out['full_window'][1]}")
    print()
    print(out["summary"].round(2).to_string(index=False))
