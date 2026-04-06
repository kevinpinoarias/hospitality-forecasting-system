"""
Trains and evaluates a SARIMAX model for daily hospitality sales forecasting.

Purpose
-------
Uses the engineered feature dataset to train a SARIMAX model with exogenous
variables and compare forecasts against realised sales.

Outputs
-------
- reports/results/sarimax_metrics.csv
- reports/results/sarimax_predictions.csv
- reports/figures/sarimax_forecast_vs_actual.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATE_COL = "date"
TARGET_COL = "total_sales"
DEFAULT_SPLIT_DATE = "2024-11-01"

# Keep SARIMAX modest and interpretable
DEFAULT_EXOG_FEATURES = [
    "forecast_sales",
    "lag_7_sales",
]

SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (0, 1, 1, 7)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_feature_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load engineered feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Expected '{DATE_COL}' column in feature dataset.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def prepare_sarimax_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    exog_features: list[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Prepare target and exogenous matrices for SARIMAX.

    Uses only rows where both target and exogenous variables are present.
    """
    exog_features = exog_features or DEFAULT_EXOG_FEATURES

    missing = [col for col in [target_col] + exog_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required SARIMAX columns: {missing}")

    data = df[[DATE_COL, target_col] + exog_features].copy()
    data = data.dropna().sort_values(DATE_COL)

    y = data.set_index(DATE_COL)[target_col].asfreq("D")
    exog = data.set_index(DATE_COL)[exog_features].asfreq("D")

    # After asfreq, gaps may introduce NaN rows again
    mask = y.notna() & exog.notna().all(axis=1)
    y = y.loc[mask]
    exog = exog.loc[mask]

    return y, exog


def temporal_train_test_split(
    y: pd.Series,
    exog: pd.DataFrame,
    split_date: str = DEFAULT_SPLIT_DATE,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train and test sets without overlap.

    Train: dates strictly before split_date
    Test:  dates on or after split_date
    """
    split_ts = pd.to_datetime(split_date)

    y_train = y.loc[y.index < split_ts]
    y_test = y.loc[y.index >= split_ts]

    exog_train = exog.loc[exog.index < split_ts]
    exog_test = exog.loc[exog.index >= split_ts]

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Train/test split produced an empty set. Check split_date.")

    return y_train, y_test, exog_train, exog_test


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape) if pd.notnull(mape) else np.nan,
    }


def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    order: tuple[int, int, int] = SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
):
    """Fit SARIMAX model."""
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    results = model.fit(disp=False)
    return results


def save_predictions(
    y_test: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Save SARIMAX predictions."""
    pred_df = pd.DataFrame({
        "date": y_test.index,
        "actual_sales": y_test.values,
        "sarimax_pred": y_pred.values,
    })
    pred_df.to_csv(output_path, index=False)
    return pred_df


def plot_predictions(
    y_test: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str = "SARIMAX Forecast vs Actual",
) -> None:
    """Save SARIMAX forecast plot."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, y_pred, label="SARIMAX Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_sarimax(
    split_date: str = DEFAULT_SPLIT_DATE,
    exog_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate SARIMAX.

    Returns
    -------
    predictions_df : pd.DataFrame
    metrics_df : pd.DataFrame
    """
    df = load_feature_data()
    y, exog = prepare_sarimax_data(df, exog_features=exog_features)

    y_train, y_test, exog_train, exog_test = temporal_train_test_split(
        y, exog, split_date=split_date
    )

    print(f"Train period: {y_train.index.min().date()} -> {y_train.index.max().date()}")
    print(f"Test period:  {y_test.index.min().date()} -> {y_test.index.max().date()}")
    print(f"Using exogenous features: {list(exog_train.columns)}")

    results = fit_sarimax(y_train, exog_train)
    print(results.summary())

    y_pred = results.forecast(steps=len(y_test), exog=exog_test)
    y_pred.index = y_test.index

    metrics = compute_metrics(y_test, y_pred)
    metrics_df = pd.DataFrame([{
        "model": "sarimax",
        "order": str(SARIMAX_ORDER),
        "seasonal_order": str(SARIMAX_SEASONAL_ORDER),
        "exog_features": ", ".join(exog_train.columns),
        **metrics,
    }])

    metrics_path = RESULTS_DIR / "sarimax_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = RESULTS_DIR / "sarimax_predictions.csv"
    predictions_df = save_predictions(y_test, y_pred, predictions_path)

    figure_path = FIGURES_DIR / "sarimax_forecast_vs_actual.png"
    plot_predictions(y_test, y_pred, figure_path)

    print(f"\nSARIMAX MAE:  {metrics['MAE']:.2f}")
    print(f"SARIMAX RMSE: {metrics['RMSE']:.2f}")
    print(f"SARIMAX MAPE: {metrics['MAPE']:.2f}%")

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved figure to: {figure_path}")

    return predictions_df, metrics_df


if __name__ == "__main__":
    run_sarimax()