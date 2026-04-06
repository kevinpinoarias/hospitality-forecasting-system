"""
Error analysis utilities for forecasting results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_error_columns(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Add signed and absolute error columns."""
    out = df.copy()
    out["signed_error"] = out[pred_col] - out[actual_col]
    out["absolute_error"] = out["signed_error"].abs()
    out["overprediction"] = out["signed_error"].clip(lower=0)
    out["underprediction"] = (-out["signed_error"]).clip(lower=0)
    return out


def top_worst_days(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    n: int = 20,
) -> pd.DataFrame:
    """Return top n worst days by absolute error."""
    out = add_error_columns(df, actual_col, pred_col)
    return out.sort_values("absolute_error", ascending=False).head(n)


def top_overpredictions(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    n: int = 20,
) -> pd.DataFrame:
    """Return top n overprediction days."""
    out = add_error_columns(df, actual_col, pred_col)
    return out.sort_values("signed_error", ascending=False).head(n)


def top_underpredictions(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    n: int = 20,
) -> pd.DataFrame:
    """Return top n underprediction days."""
    out = add_error_columns(df, actual_col, pred_col)
    return out.sort_values("signed_error", ascending=True).head(n)


def summarise_error_by_group(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Aggregate mean absolute and signed error by grouping column."""
    out = add_error_columns(df, actual_col, pred_col)

    summary = (
        out.groupby(group_col)
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            n=("absolute_error", "size"),
        )
        .reset_index()
        .sort_values("mean_absolute_error", ascending=False)
    )

    return summary


def save_error_analysis_tables(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save standard error analysis tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top_worst_days(df, actual_col, pred_col).to_csv(
        output_dir / f"{prefix}_top_worst_days.csv",
        index=False,
    )

    top_overpredictions(df, actual_col, pred_col).to_csv(
        output_dir / f"{prefix}_top_overpredictions.csv",
        index=False,
    )

    top_underpredictions(df, actual_col, pred_col).to_csv(
        output_dir / f"{prefix}_top_underpredictions.csv",
        index=False,
    )