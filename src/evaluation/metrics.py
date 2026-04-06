"""
Shared evaluation metrics for forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error in percent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def bias(y_true, y_pred) -> float:
    """Mean signed forecast bias."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred - y_true))


def regression_metrics(y_true, y_pred) -> dict:
    """Return standard regression metrics as a dictionary."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_prediction_column(df: pd.DataFrame, y_col: str, pred_col: str) -> dict:
    """Evaluate one prediction column from a dataframe."""
    tmp = df[[y_col, pred_col]].dropna()
    return {
        "n": int(len(tmp)),
        **regression_metrics(tmp[y_col], tmp[pred_col]),
    }