"""
plots.py

Shared plotting utilities for forecasting evaluation.

Purpose
-------
Centralises reusable plotting functions for:
- actual vs predicted forecasts
- actual vs human forecast
- best-window AI vs human comparison
- rolling MAE comparison
- feature importance
- residual distributions
- residuals over time
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_forecast_vs_actual(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    title: str,
    output_path: Path,
    pred_label: str = "Prediction",
) -> None:
    """Plot actual vs predicted time series and save figure."""
    plot_df = df[[date_col, actual_col, pred_col]].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df[date_col], plot_df[actual_col], label="Actual")
    plt.plot(plot_df[date_col], plot_df[pred_col], label=pred_label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_actual_vs_human_forecast(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    human_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot actual vs human forecast and save figure."""
    plot_df = df[[date_col, actual_col, human_col]].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df[date_col], plot_df[actual_col], label="Actual Sales")
    plt.plot(plot_df[date_col], plot_df[human_col], label="Human Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales (£)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_best_window_comparison(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    human_col: str,
    ai_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot selected best-performing time window comparing actual, human, and AI forecast."""
    plot_df = df[[date_col, actual_col, human_col, ai_col]].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col)

    plt.figure(figsize=(16, 7))
    plt.plot(plot_df[date_col], plot_df[actual_col], label="Actual Sales", linewidth=2.8)
    plt.plot(plot_df[date_col], plot_df[human_col], label="Human Forecast", linewidth=2.4)
    plt.plot(plot_df[date_col], plot_df[ai_col], label="AI Forecast", linewidth=2.4)

    plt.title(title, fontsize=18, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales (£)", fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=35, fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend(fontsize=11, frameon=True)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_mae_comparison(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    human_col: str,
    ai_col: str,
    window: int,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    """
    Plot rolling MAE comparison between human and AI forecast.

    Returns the dataframe with rolling MAE columns included.
    """
    plot_df = df[[date_col, actual_col, human_col, ai_col]].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col).reset_index(drop=True)

    plot_df["human_abs_error"] = (plot_df[actual_col] - plot_df[human_col]).abs()
    plot_df["ai_abs_error"] = (plot_df[actual_col] - plot_df[ai_col]).abs()

    plot_df["human_rolling_mae"] = plot_df["human_abs_error"].rolling(window=window).mean()
    plot_df["ai_rolling_mae"] = plot_df["ai_abs_error"].rolling(window=window).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df[date_col], plot_df["human_rolling_mae"], label="Human rolling MAE")
    plt.plot(plot_df[date_col], plot_df["ai_rolling_mae"], label="AI rolling MAE")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return plot_df


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: Path,
    title: str = "Feature Importance",
    top_n: int | None = None,
) -> None:
    """
    Plot model feature importance.

    Works well for tree-based models such as XGBoost.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distribution(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    output_path: Path,
    title: str = "Residual Distribution",
) -> None:
    """Plot histogram of residuals."""
    plot_df = df[[actual_col, pred_col]].dropna().copy()
    residuals = plot_df[pred_col] - plot_df[actual_col]

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)
    plt.title(title)
    plt.xlabel("Residual (Prediction - Actual)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals_over_time(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    output_path: Path,
    title: str = "Residuals Over Time",
) -> None:
    """Plot residuals over time."""
    plot_df = df[[date_col, actual_col, pred_col]].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col)
    plot_df["residual"] = plot_df[pred_col] - plot_df[actual_col]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df[date_col], plot_df["residual"])
    plt.axhline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_actual_vs_multiple_predictions(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    prediction_cols: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot actual values against multiple prediction series.

    prediction_cols example:
    {
        "forecast_sales": "Human Forecast",
        "ai_prediction": "AI Forecast",
        "sarimax_pred": "SARIMAX"
    }
    """
    needed_cols = [date_col, actual_col] + list(prediction_cols.keys())
    plot_df = df[needed_cols].dropna().copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.sort_values(date_col)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df[date_col], plot_df[actual_col], label="Actual", linewidth=2.8)

    for col, label in prediction_cols.items():
        plt.plot(plot_df[date_col], plot_df[col], label=label, linewidth=2.0)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()