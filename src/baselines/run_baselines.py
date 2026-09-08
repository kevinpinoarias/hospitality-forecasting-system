"""
Runs baseline forecasting benchmarks for the hospitality forecasting project.

Purpose
-------
Evaluates simple baseline models against daily sales data:
- naive (previous day)
- seasonal naive (same day last week)
- rolling 7-day mean
- rolling 14-day mean

These benchmarks provide a reference point before training more advanced models
such as SARIMAX or XGBoost.

Outputs
-------
- reports/results/baseline_metrics.csv
- reports/results/baseline_predictions.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.experiment_tracking import log_run


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sales" / "daily_sales_totals_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATE_COL = "date"
TARGET_COL = "total_sales"
DEFAULT_SPLIT_DATE = "2024-11-01"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_sales_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load daily sales master dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{DATE_COL}' and '{TARGET_COL}' in input data."
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a minimal set of calendar and lag features used in the baseline analysis.

    This keeps the baseline script self-contained and makes it easier to inspect
    data behaviour before using the full feature engineering pipeline.
    """
    out = df.copy()

    out["year"] = out[DATE_COL].dt.year
    out["month"] = out[DATE_COL].dt.month
    out["day_of_week"] = out[DATE_COL].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 4).astype(int)
    out["day_of_year"] = out[DATE_COL].dt.dayofyear

    out["lag_7_sales"] = out[TARGET_COL].shift(7)
    out["rolling_7_sales"] = out[TARGET_COL].shift(1).rolling(window=7).mean()
    out["rolling_14_sales"] = out[TARGET_COL].shift(1).rolling(window=14).mean()

    return out


def add_cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Add cyclical sine and cosine encoding."""
    angle = 2 * np.pi * (df[col] - 1) / period
    df[f"{col}_sin"] = np.sin(angle)
    df[f"{col}_cos"] = np.cos(angle)
    return df


def add_exploratory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exploratory features that were used during baseline-stage analysis.

    These are not strictly required for baseline predictions themselves, but
    they preserve the spirit of the original project and support EDA.
    """
    out = add_basic_time_features(df)
    out = add_cyclical(out, "month", 12)
    out = add_cyclical(out, "day_of_week", 7)
    out = add_cyclical(out, "day_of_year", 365)

    return out


def add_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Create baseline forecast columns."""
    out = df.copy()

    # Naive baseline: yesterday's sales
    out["naive_pred"] = out[TARGET_COL].shift(1)

    # Seasonal naive: same day last week
    out["seasonal_naive_pred"] = out[TARGET_COL].shift(7)

    # Rolling means
    out["roll7_pred"] = out[TARGET_COL].shift(1).rolling(7).mean()
    out["roll14_pred"] = out[TARGET_COL].shift(1).rolling(14).mean()

    # Optional extra benchmark: rolling 28-day mean
    out["roll28_pred"] = out[TARGET_COL].shift(1).rolling(28).mean()

    return out


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def eval_metrics(df: pd.DataFrame, y_col: str, pred_col: str) -> dict:
    """Evaluate one prediction column against the target."""
    tmp = df[[y_col, pred_col]].dropna()

    y = tmp[y_col].to_numpy()
    p = tmp[pred_col].to_numpy()

    return {
        "n": int(len(tmp)),
        "MAE": mae(y, p),
        "RMSE": rmse(y, p),
        "MAPE": mape(y, p),
    }


def add_weekday_average_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Predict test-set sales using the historical average for each weekday
    computed from the training set.
    """
    weekday_means = train_df.groupby("day_of_week")[target_col].mean()

    out = test_df.copy()
    out["weekday_avg_pred"] = out["day_of_week"].map(weekday_means)

    return out

def run_baselines(split_date: str = DEFAULT_SPLIT_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_df = load_sales_data()
    sales_df = add_exploratory_features(sales_df)
    sales_df = add_baseline_predictions(sales_df)

    split_date = pd.to_datetime(split_date)

    train_df = sales_df[sales_df[DATE_COL] < split_date].copy()
    eval_df = sales_df[sales_df[DATE_COL] >= split_date].copy()

    # Add weekday-average baseline using training data only
    eval_df = add_weekday_average_baseline(train_df, eval_df)

    metrics = {
        "naive": eval_metrics(eval_df, TARGET_COL, "naive_pred"),
        "seasonal_naive": eval_metrics(eval_df, TARGET_COL, "seasonal_naive_pred"),
        "roll7": eval_metrics(eval_df, TARGET_COL, "roll7_pred"),
        "roll14": eval_metrics(eval_df, TARGET_COL, "roll14_pred"),
        "roll28": eval_metrics(eval_df, TARGET_COL, "roll28_pred"),
        "weekday_average": eval_metrics(eval_df, TARGET_COL, "weekday_avg_pred"),
    }

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})

    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Each baseline gets logged as its own run, so it's directly comparable
    # to SARIMAX/XGBoost runs in the same W&B project rather than bundled
    # into one row.
    baseline_configs = {
        "naive": {"lag_days": 1},
        "seasonal_naive": {"lag_days": 7},
        "roll7": {"window_days": 7},
        "roll14": {"window_days": 14},
        "roll28": {"window_days": 28},
        "weekday_average": {},
    }
    for model_name, model_metrics in metrics.items():
        log_run(
            model_name=model_name,
            config={"split_date": str(split_date.date()), **baseline_configs[model_name]},
            metrics={k: v for k, v in model_metrics.items() if k != "n"},
            job_type="baseline",
        )

    prediction_cols = [
        DATE_COL,
        TARGET_COL,
        "naive_pred",
        "seasonal_naive_pred",
        "roll7_pred",
        "roll14_pred",
        "roll28_pred",
        "weekday_avg_pred",
    ]
    predictions_df = eval_df[prediction_cols].copy()

    predictions_path = OUTPUT_DIR / "baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Baseline metrics written to: {metrics_path}")
    print(f"Baseline predictions written to: {predictions_path}")

    return predictions_df, metrics_df


if __name__ == "__main__":
    _, baseline_metrics = run_baselines()
    print("\nBaseline performance summary:")
    print(baseline_metrics)
    
