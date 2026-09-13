"""
Trains and evaluates the final forecasting model - the one the live API
serves.

Purpose
-------
Promotes the configuration the experimentation programme settled on
(docs/EXPERIMENT_LOG.md, Series 14 and 16) into the pipeline: CatBoost on
the 38-feature set, trained on log1p(sales), averaged over 25 random seeds
(see src/models/final_model.py).

Evaluation uses the experiment log's 10-fold rolling-origin backtest
rather than one fixed split, and scores the original 15-feature XGBoost
and the venue's manual forecast on exactly the same days, so all three
figures are comparable. The backtest's out-of-sample predictions are also
saved next to the model: for a date inside the historical data the API
returns what the model predicted before seeing that day, not a prediction
from a model that was trained on it.

The model the API loads is then refit on every available day.

Outputs
-------
- reports/results/final_model_metrics.csv
- reports/results/final_model_backtest_predictions.csv
- reports/figures/final_model_backtest_vs_actual.png
- models/final_model/seed_00.cbm ... seed_24.cbm
- models/final_model/metadata.json
- models/final_model/backtest_predictions.csv
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.evaluation.experiment_tracking import log_run
from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_forecast_vs_actual
from src.experiments.splits import generate_rolling_origin_folds
from src.models.final_model import (
    BACKTEST_PREDICTIONS_FILENAME,
    DATE_COL,
    FINAL_FEATURES,
    FINAL_PARAMS,
    FINAL_SEEDS,
    MODEL_DIR,
    TARGET_COL,
    BaggedCatBoost,
)
from src.models.train_xgboost import FEATURES as XGBOOST_FEATURES, build_model as build_xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# The experiment log's backtest: a year of initial history, then
# back-to-back 60-day test blocks with an expanding training window.
INITIAL_TRAIN_DAYS = 365
TEST_DAYS = 60


def load_final_model_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Every day with all 38 features and the target present - the same rows
    as the experiment log (src/experiments/run_feature_sweep.load_data)."""
    if not path.exists():
        raise FileNotFoundError(f"Engineered feature file not found: {path}")

    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    required = sorted(set(FINAL_FEATURES) | {DATE_COL, TARGET_COL})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required final-model columns: {missing}")
    return df.dropna(subset=required).sort_values(DATE_COL).reset_index(drop=True)


def rolling_origin_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions from the final model and the original
    XGBoost on every backtest test day."""
    folds = generate_rolling_origin_folds(
        df[DATE_COL], initial_train_days=INITIAL_TRAIN_DAYS, test_days=TEST_DAYS, step_days=TEST_DAYS,
    )
    frames = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]

        final_model = BaggedCatBoost.fit(train)
        xgboost = build_xgboost().fit(train[XGBOOST_FEATURES], train[TARGET_COL])

        frames.append(pd.DataFrame({
            DATE_COL: test[DATE_COL].to_numpy(),
            "fold_id": fold.fold_id,
            TARGET_COL: test[TARGET_COL].to_numpy(),
            "forecast_sales": test["forecast_sales"].to_numpy(),
            "final_model_prediction": final_model.predict(test),
            "xgboost_prediction": xgboost.predict(test[XGBOOST_FEATURES]),
        }))
        print(f"  fold {fold.fold_id}: {fold.test_start.date()} to {(fold.test_end - pd.Timedelta(days=1)).date()}")
    return pd.concat(frames, ignore_index=True)


def summarise_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    actual = predictions[TARGET_COL]
    rows = []
    for label, col in [
        ("final_model", "final_model_prediction"),
        ("xgboost", "xgboost_prediction"),
        ("manual_forecast", "forecast_sales"),
    ]:
        pooled = regression_metrics(actual, predictions[col])
        fold_mae = [regression_metrics(g[TARGET_COL], g[col])["MAE"] for _, g in predictions.groupby("fold_id")]
        rows.append({"model": label, "days": len(predictions), **pooled, "MAE_fold_mean": sum(fold_mae) / len(fold_mae)})

    metrics = pd.DataFrame(rows)
    manual_mae = metrics.loc[metrics["model"] == "manual_forecast", "MAE"].iloc[0]
    metrics["pct_better_than_manual"] = (manual_mae - metrics["MAE"]) / manual_mae * 100
    return metrics


def run_final_model() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_final_model_data()

    print("Rolling-origin backtest (25 seeds x each fold)...")
    predictions = rolling_origin_backtest(df)
    metrics = summarise_backtest(predictions)
    final = metrics.set_index("model").loc["final_model"]
    manual = metrics.set_index("model").loc["manual_forecast"]
    xgb = metrics.set_index("model").loc["xgboost"]

    predictions.to_csv(RESULTS_DIR / "final_model_backtest_predictions.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "final_model_metrics.csv", index=False)

    figure_path = FIGURES_DIR / "final_model_backtest_vs_actual.png"
    plot_forecast_vs_actual(
        df=predictions,
        date_col=DATE_COL,
        actual_col=TARGET_COL,
        pred_col="final_model_prediction",
        title="Final model vs actual sales (rolling-origin backtest, out-of-sample)",
        output_path=figure_path,
        pred_label="Final model",
    )

    print("Refitting on every available day...")
    production_model = BaggedCatBoost.fit(df)
    production_model.save(MODEL_DIR, metadata={
        "model": "bagged_log1p_catboost",
        "params": FINAL_PARAMS,
        "seeds": FINAL_SEEDS,
        "target_transform": "log1p",
        "trained_on": {
            "start_date": df[DATE_COL].min().date().isoformat(),
            "end_date": df[DATE_COL].max().date().isoformat(),
            "rows": len(df),
        },
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "backtest": {
            "folds": int(predictions["fold_id"].nunique()),
            "days": len(predictions),
            "start_date": predictions[DATE_COL].min().date().isoformat(),
            "end_date": predictions[DATE_COL].max().date().isoformat(),
            "mae": round(float(final["MAE"]), 2),
            "mape_pct": round(float(final["MAPE"]), 2),
            "manual_forecast_mae": round(float(manual["MAE"]), 2),
            "pct_better_than_manual": round(float(final["pct_better_than_manual"]), 2),
        },
    })
    predictions[[DATE_COL, "forecast_sales", "final_model_prediction"]].to_csv(
        MODEL_DIR / BACKTEST_PREDICTIONS_FILENAME, index=False,
    )

    log_run(
        model_name="catboost_final_ensemble",
        config={
            "model": "CatBoost, log1p target, mean of 25 seeds",
            "n_features": len(FINAL_FEATURES),
            "seeds": len(FINAL_SEEDS),
            "evaluation": f"rolling-origin backtest, {INITIAL_TRAIN_DAYS}-day initial window, "
                          f"{TEST_DAYS}-day test blocks, expanding training window",
            "backtest_folds": int(predictions["fold_id"].nunique()),
            "backtest_days": len(predictions),
            **FINAL_PARAMS,
        },
        metrics={
            "MAE": final["MAE"],
            "RMSE": final["RMSE"],
            "MAPE": final["MAPE"],
            "Bias": final["Bias"],
            "MAE_fold_mean": final["MAE_fold_mean"],
            "manual_forecast_MAE_same_days": manual["MAE"],
            "xgboost_MAE_same_backtest": xgb["MAE"],
            "pct_better_than_manual": final["pct_better_than_manual"],
        },
        figure_path=figure_path,
    )

    print(metrics.round(2).to_string(index=False))
    print(f"Final model saved to: {MODEL_DIR}")
    return predictions, metrics


if __name__ == "__main__":
    run_final_model()
