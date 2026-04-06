"""
Trains and evaluates the final XGBoost model for hospitality sales forecasting.

Purpose
-------
Uses the final selected feature set to train an XGBoost regressor and compare
predictions against realised sales.

Outputs
-------
- reports/results/xgboost_metrics.csv
- reports/results/xgboost_predictions.csv
- reports/results/best_8_week_window_metrics.csv
- reports/figures/xgboost_forecast_vs_actual.png
- reports/figures/best_8_week_window.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from xgboost import XGBRegressor

from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_forecast_vs_actual, plot_best_window_comparison


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "model_features.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATE_COL = "date"
TARGET_COL = "total_sales"
DEFAULT_SPLIT_DATE = "2024-12-31"

FEATURES = [
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
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_model_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load final model-ready feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Model feature file not found: {path}")

    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    required = [DATE_COL, TARGET_COL] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required XGBoost columns: {missing}")

    return df


def temporal_train_valid_split(
    df: pd.DataFrame,
    split_date: str = DEFAULT_SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and validation sets."""
    split_ts = pd.to_datetime(split_date)

    train = df[df[DATE_COL] < split_ts].copy()
    valid = df[df[DATE_COL] >= split_ts].copy()

    if len(train) == 0 or len(valid) == 0:
        raise ValueError("Train/validation split produced an empty set.")

    return train, valid


def build_model() -> XGBRegressor:
    """Instantiate the final XGBoost regressor."""
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )


def compute_best_window(
    valid_df: pd.DataFrame,
    window: int = 56,
) -> tuple[pd.DataFrame, dict]:
    """
    Find the best rolling window where AI beats human forecast most strongly.

    Window default:
    56 days = 8 weeks
    """
    df_plot = valid_df[[DATE_COL, "total_sales", "forecast_sales", "ai_prediction"]].dropna().copy()
    df_plot = df_plot.sort_values(DATE_COL).reset_index(drop=True)

    df_plot["human_abs_error"] = (df_plot["total_sales"] - df_plot["forecast_sales"]).abs()
    df_plot["ai_abs_error"] = (df_plot["total_sales"] - df_plot["ai_prediction"]).abs()

    df_plot["human_rolling_mae"] = df_plot["human_abs_error"].rolling(window=window).mean()
    df_plot["ai_rolling_mae"] = df_plot["ai_abs_error"].rolling(window=window).mean()
    df_plot["rolling_mae_improvement"] = df_plot["human_rolling_mae"] - df_plot["ai_rolling_mae"]

    valid_idx = df_plot["rolling_mae_improvement"].dropna().idxmax()

    best_end_date = df_plot.loc[valid_idx, DATE_COL]
    best_start_date = df_plot.loc[valid_idx - window + 1, DATE_COL]

    best_window = df_plot[
        (df_plot[DATE_COL] >= best_start_date) &
        (df_plot[DATE_COL] <= best_end_date)
    ].copy()

    human_mae_best = best_window["human_abs_error"].mean()
    ai_mae_best = best_window["ai_abs_error"].mean()
    improvement_best = human_mae_best - ai_mae_best

    summary = {
        "start_date": best_start_date.date(),
        "end_date": best_end_date.date(),
        "window_days": window,
        "human_mae": float(human_mae_best),
        "ai_mae": float(ai_mae_best),
        "mae_improvement": float(improvement_best),
    }

    return best_window, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_xgboost(split_date: str = DEFAULT_SPLIT_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate XGBoost model."""
    df = load_model_data()
    train, valid = temporal_train_valid_split(df, split_date=split_date)

    X_train = train[FEATURES]
    y_train = train[TARGET_COL]

    X_valid = valid[FEATURES]
    y_valid = valid[TARGET_COL]

    model = build_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    preds = model.predict(X_valid)
    preds = pd.Series(preds, index=valid.index)

    metrics = regression_metrics(y_valid, preds)
    metrics_df = pd.DataFrame([{
        "model": "xgboost",
        **metrics,
    }])

    valid = valid.copy()
    valid["ai_prediction"] = preds

    predictions_df = valid[[DATE_COL, TARGET_COL, "forecast_sales", "ai_prediction"]].copy()

    metrics_path = RESULTS_DIR / "xgboost_metrics.csv"
    predictions_path = RESULTS_DIR / "xgboost_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    plot_forecast_vs_actual(
        df=predictions_df,
        date_col=DATE_COL,
        actual_col=TARGET_COL,
        pred_col="ai_prediction",
        title="AI Forecast vs Actual Sales",
        output_path=FIGURES_DIR / "xgboost_forecast_vs_actual.png",
        pred_label="AI Forecast",
    )

    best_window_df, best_window_summary = compute_best_window(valid, window=56)
    best_window_metrics_df = pd.DataFrame([best_window_summary])
    best_window_metrics_df.to_csv(
        RESULTS_DIR / "best_8_week_window_metrics.csv",
        index=False,
    )

    plot_best_window_comparison(
        df=best_window_df,
        date_col=DATE_COL,
        actual_col="total_sales",
        human_col="forecast_sales",
        ai_col="ai_prediction",
        title="Selected 8-week period: AI forecast tracked actual demand more closely than the manual forecast",
        output_path=FIGURES_DIR / "best_8_week_window.png",
    )

    print(f"AI Model MAPE: {metrics['MAPE']:.2f}%")
    print(f"XGBoost MAE: {metrics['MAE']:.2f}")
    print(f"XGBoost RMSE: {metrics['RMSE']:.2f}")

    print("\n=== BEST 8-WEEK WINDOW ===")
    print(f"Start date: {best_window_summary['start_date']}")
    print(f"End date:   {best_window_summary['end_date']}")
    print(f"Human MAE:  {best_window_summary['human_mae']:.2f}")
    print(f"AI MAE:     {best_window_summary['ai_mae']:.2f}")
    print(f"Improvement:{best_window_summary['mae_improvement']:.2f}")

    return predictions_df, metrics_df


if __name__ == "__main__":
    run_xgboost()
