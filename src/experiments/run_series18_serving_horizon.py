"""
Series 18 - Serving the final model beyond the end of the data.

Every earlier series scores a day with its real recent history available:
the lag/rolling features read the actual sales of the days before it, and
the weather features read the weather that actually happened. A live
forecast has neither for a date weeks or months ahead - and the public
API's data stops at a fixed date, so almost every request is for a date
beyond it. Series 3 tested recursive lag-filling for single-seed CatBoost
out to 30 days; this series tests the final model as the API would serve
it, out to 180 days, using the serving code itself
(src/features/serving_features.py).

Method, for each rolling-origin fold whose origin has 180 days of data
after it (8 origins):
1. Train the final model (25-seed log1p CatBoost, 38 features) and the
   original 15-feature XGBoost on every day before the origin.
2. Lay out the days after the origin as the API sees them on the origin
   date: weather for the first 16 days is the weather that happened (a
   stand-in for a live forecast, so slightly optimistic); after that, the
   weekday/season median from training history. Days without a manager's
   forecast get the same historical median as forecast_sales.
3. Fill the history features for the days after the origin two ways:
   - recursive: each day's sales is the model's own prediction, fed
     forward (serving_features.fill_future_sales_recursively);
   - seasonal: each day's sales is the historical weekday/season median.
4. Score each day with the manager's real forecast supplied, and without
   it (historical median in its place). The final model with every real
   input available (Series 16's conditions) is included as the ceiling.

Scored days: the days in each 180-day window that are in the experiment
dataset (same rows as every other series) - 1,425 day-scores.

Result (MAE £, all 180 days):

    |                                  | Manager forecast supplied | Not supplied |
    | Manual forecast                  | 1,082                     |              |
    | Final model, every real input    | 933                       |              |
    | Final model, seasonal history    | 978                       | 1,078        |
    | Final model, recursive history   | 972                       | 1,082        |
    | Original XGBoost                 | 1,078                     | 1,132        |

With the manager's forecast, the served final model stays about 10%
better than the manual forecast; without it, it is level with the manual
forecast and about 5% better than XGBoost. Recursive and seasonal filling
are within £6 of each other; the API uses the seasonal fill, which was
better at the short horizons most requests fall in and needs no
prediction loop. See docs/EXPERIMENT_LOG.md, Series 18, for the breakdown
by horizon and the caveats.

Requires: data/features/engineered_features.csv (python main.py)
Outputs (reports/experiments/):
- series18_serving_horizon_predictions.csv
- series18_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.series_common import DATE_COL, TARGET_COL, build_folds, load_data, results_path
from src.experiments.run_feature_sweep import ENGINEERED_FEATURES_PATH
from src.features.serving_features import (
    WEATHER_COLS,
    build_feature_frame,
    extend_daily_frame,
    fill_future_sales_recursively,
    fill_future_sales_seasonally,
    with_history_features,
)
from src.models.final_model import FINAL_FEATURES, BaggedCatBoost
from src.models.train_xgboost import FEATURES as XGBOOST_FEATURES, build_model as build_xgboost

HORIZON_DAYS = 180
LIVE_WEATHER_DAYS = 16  # Open-Meteo's forecast range, as in src/api/weather.py
HORIZON_BUCKETS = [(1, 7), (8, 16), (17, 30), (31, 60), (61, 90), (91, 180)]

PREDICTION_COLUMNS = {
    "manual_forecast": "Manual forecast",
    "final_real_inputs": "Final model, every real input (Series 16 conditions)",
    "final_recursive_manager": "Final model, recursive history, manager forecast supplied",
    "final_seasonal_manager": "Final model, seasonal history, manager forecast supplied",
    "xgboost_manager": "Original XGBoost, manager forecast supplied",
    "final_recursive_no_manager": "Final model, recursive history, no manager forecast",
    "final_seasonal_no_manager": "Final model, seasonal history, no manager forecast",
    "xgboost_no_manager": "Original XGBoost, no manager forecast",
}


def load_full_history() -> pd.DataFrame:
    df = pd.read_csv(ENGINEERED_FEATURES_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def served_daily_frame(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """History plus the future days as the API would fill them on the origin
    date: real weather for the first LIVE_WEATHER_DAYS, historical medians
    after that, and historical-median forecast_sales throughout."""
    daily = extend_daily_frame(history, future[DATE_COL].max().date())
    live = slice(len(history), len(history) + LIVE_WEATHER_DAYS)
    daily.loc[daily.index[live], WEATHER_COLS] = future[WEATHER_COLS].to_numpy()[:LIVE_WEATHER_DAYS]
    return daily


def run_origin(fold, experiment_df: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    origin = fold.test_start
    end = origin + pd.Timedelta(days=HORIZON_DAYS)
    train_df = experiment_df[experiment_df[DATE_COL] < origin]
    history = full[full[DATE_COL] < origin]
    future = full[(full[DATE_COL] >= origin) & (full[DATE_COL] < end)]

    final_model = BaggedCatBoost.fit(train_df)
    xgboost = build_xgboost().fit(train_df[XGBOOST_FEATURES], train_df[TARGET_COL])

    frame = build_feature_frame(served_daily_frame(history, future))
    first_future = len(history)
    rows = list(range(first_future, len(frame)))
    manager_forecast = future["forecast_sales"].to_numpy()

    recursive_sales = fill_future_sales_recursively(frame, first_future, FINAL_FEATURES, final_model.predict)
    seasonal_sales = fill_future_sales_seasonally(frame, first_future, history)

    recursive = with_history_features(frame, recursive_sales, rows)
    seasonal = with_history_features(frame, seasonal_sales, rows)

    def with_manager(features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out["forecast_sales"] = manager_forecast
        return out

    real_inputs = future.set_index(DATE_COL)
    out = pd.DataFrame({
        "origin": origin,
        DATE_COL: future[DATE_COL].to_numpy(),
        "horizon_day": np.arange(1, len(future) + 1),
        "actual": future[TARGET_COL].to_numpy(),
        "manual_forecast": manager_forecast,
        "final_real_inputs": final_model.predict(real_inputs[FINAL_FEATURES]),
        "final_recursive_manager": final_model.predict(with_manager(recursive)[FINAL_FEATURES]),
        "final_seasonal_manager": final_model.predict(with_manager(seasonal)[FINAL_FEATURES]),
        "xgboost_manager": xgboost.predict(with_manager(recursive)[XGBOOST_FEATURES]),
        "final_recursive_no_manager": recursive_sales[rows],
        "final_seasonal_no_manager": final_model.predict(seasonal[FINAL_FEATURES]),
        "xgboost_no_manager": xgboost.predict(recursive[XGBOOST_FEATURES]),
    })
    scored_dates = set(experiment_df[DATE_COL])
    return out[out[DATE_COL].isin(scored_dates)]


def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = HORIZON_BUCKETS + [(1, HORIZON_DAYS)]
    for low, high in buckets:
        sub = predictions[predictions["horizon_day"].between(low, high)]
        for col, label in PREDICTION_COLUMNS.items():
            metrics = regression_metrics(sub["actual"], sub[col])
            rows.append({"horizon": f"days {low}-{high}", "model": label, "days": len(sub),
                         "MAE": metrics["MAE"], "MAPE": metrics["MAPE"], "Bias": metrics["Bias"]})
    return pd.DataFrame(rows)


def main() -> None:
    experiment_df = load_data()
    full = load_full_history()
    last_date = full[DATE_COL].max()
    folds = [f for f in build_folds(experiment_df)
             if f.test_start + pd.Timedelta(days=HORIZON_DAYS) <= last_date + pd.Timedelta(days=1)]
    print(f"{len(folds)} origins with {HORIZON_DAYS} days of data after them: "
          f"{[f.test_start.date().isoformat() for f in folds]}")

    predictions = []
    for fold in folds:
        predictions.append(run_origin(fold, experiment_df, full))
        print(f"  origin {fold.test_start.date()} done")
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(results_path("series18_serving_horizon_predictions.csv"), index=False)

    summary = summarise(predictions)
    summary.to_csv(results_path("series18_summary.csv"), index=False)

    table = summary.pivot(index="model", columns="horizon", values="MAE")
    table = table[[f"days {lo}-{hi}" for lo, hi in HORIZON_BUCKETS] + [f"days 1-{HORIZON_DAYS}"]]
    table = table.loc[list(PREDICTION_COLUMNS.values())]
    print("\nMAE (£) by horizon")
    print(table.round(0).to_string())


if __name__ == "__main__":
    main()
