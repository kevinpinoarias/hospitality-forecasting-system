"""
Evaluates human sales and labour forecasts against realised outcomes.

Purpose
-------
Compares manual forecasts to actual sales and labour outcomes, reporting:
- sales forecasting accuracy
- forecast bias
- labour-related business impact
- annual sales growth comparison
- plots of actual vs human forecast

Outputs
-------
- reports/results/human_forecast_metrics_all.csv
- reports/results/human_forecast_metrics_2025.csv
- reports/results/actual_sales_growth_comparison.csv
- reports/figures/human_forecast_vs_actual_all.png
- reports/figures/human_forecast_vs_actual_2025.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sales" / "daily_sales_totals_master.csv"
LABOUR_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "labour" / "daily_labour_totals_master.csv"

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------

def load_processed_data(
    sales_path: Path = SALES_INPUT_PATH,
    labour_path: Path = LABOUR_INPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed daily sales and labour datasets."""
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales input file not found: {sales_path}")
    if not labour_path.exists():
        raise FileNotFoundError(f"Labour input file not found: {labour_path}")

    sales = pd.read_csv(sales_path)
    labour = pd.read_csv(labour_path)

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    labour["date"] = pd.to_datetime(labour["date"], errors="coerce")

    sales = sales.sort_values("date").dropna().reset_index(drop=True)
    labour = labour.sort_values("date").dropna().reset_index(drop=True)

    return sales, labour


def prepare_merged_dataset(sales: pd.DataFrame, labour: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sales and labour datasets.

    Portfolio-safe version:
    - merge on date and department if available
    - otherwise merge on date only
    """
    sales_cols = ["date", "total_sales", "forecast_sales"]
    labour_cols = ["date", "total_hours", "total_wages", "forecast_hours", "forecast_wages"]

    if "department" in sales.columns:
        sales_cols.append("department")
    if "department" in labour.columns:
        labour_cols.append("department")

    sales = sales[sales_cols].copy()
    labour = labour[labour_cols].copy()

    merge_keys = ["date"]
    if "department" in sales.columns and "department" in labour.columns:
        merge_keys.append("department")

    merged = pd.merge(
        sales,
        labour,
        on=merge_keys,
        how="inner"
    ).sort_values(merge_keys).reset_index(drop=True)

    return merged


# ---------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------

def calculate_sales_error_metrics(df: pd.DataFrame) -> dict:
    """Calculate sales forecast performance metrics."""
    y_true = df["total_sales"]
    y_pred = df["forecast_sales"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    else:
        mape = np.nan

    bias = np.mean(y_pred - y_true)

    daily_sales = df.groupby("date", as_index=False)["total_sales"].sum()
    avg_daily_sales = daily_sales["total_sales"].mean() if len(daily_sales) > 0 else np.nan

    return {
        "Average daily sales (£)": round(avg_daily_sales, 2) if pd.notnull(avg_daily_sales) else np.nan,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2) if pd.notnull(mape) else np.nan,
        "Bias (£)": round(bias, 2),
    }


def calculate_labour_business_metrics(df: pd.DataFrame) -> dict:
    """Calculate labour-related business impact metrics."""
    results = df.copy()

    results["sales_error"] = results["forecast_sales"] - results["total_sales"]
    results["overforecast_sales"] = results["sales_error"].clip(lower=0)

    results["wage_error"] = results["forecast_wages"] - results["total_wages"]
    results["overforecast_wages"] = results["wage_error"].clip(lower=0)
    results["underforecast_wages"] = (-results["wage_error"]).clip(lower=0)

    results["is_overstaffed_day"] = results["wage_error"] > 0
    results["is_understaffed_day"] = results["wage_error"] < 0

    total_overforecasted_wages = results["overforecast_wages"].sum()
    total_underforecasted_wages = results["underforecast_wages"].sum()
    total_overforecasted_sales = results["overforecast_sales"].sum()

    overstaffed_days = int(results["is_overstaffed_day"].sum())
    understaffed_days = int(results["is_understaffed_day"].sum())

    avg_wasted_wages_overstaffed_days = (
        results.loc[results["is_overstaffed_day"], "overforecast_wages"].mean()
        if overstaffed_days > 0 else 0
    )

    wage_bias = results["wage_error"].mean()

    pounds_wasted_per_pound_overforecast_sales = (
        total_overforecasted_wages / total_overforecasted_sales
        if total_overforecasted_sales > 0 else 0
    )

    return {
        "Total overforecasted wages (£)": round(total_overforecasted_wages, 2),
        "Total underforecasted wages (£)": round(total_underforecasted_wages, 2),
        "Total overforecasted sales (£)": round(total_overforecasted_sales, 2),
        "£ wasted per £1 overforecasted sales": round(pounds_wasted_per_pound_overforecast_sales, 4),
        "Overstaffed days": overstaffed_days,
        "Understaffed days": understaffed_days,
        "Average wasted wages on overstaffed days (£)": round(avg_wasted_wages_overstaffed_days, 2),
        "Wage bias (£)": round(wage_bias, 2),
    }


def build_metrics_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Create one metrics table combining sales and labour metrics."""
    sales_metrics = calculate_sales_error_metrics(df)
    labour_metrics = calculate_labour_business_metrics(df)

    combined = {"dataset": label, **sales_metrics, **labour_metrics}
    return pd.DataFrame([combined])


def print_metrics_block(df: pd.DataFrame, label: str = "DATASET") -> None:
    """Print a readable metrics summary."""
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")

    print("\n--- SALES ERROR METRICS ---")
    sales_metrics = calculate_sales_error_metrics(df)
    for key, value in sales_metrics.items():
        print(f"{key}: {value}")

    print("\n--- LABOUR / BUSINESS METRICS ---")
    labour_metrics = calculate_labour_business_metrics(df)
    for key, value in labour_metrics.items():
        print(f"{key}: {value}")

    print(f"\nRows: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


# ---------------------------------------------------------------------
# Growth comparison
# ---------------------------------------------------------------------

def calculate_actual_sales_growth(sales: pd.DataFrame) -> pd.DataFrame:
    """Compare total actual sales between 2024 and 2025."""
    sales_2024 = sales[sales["date"].dt.year == 2024].copy()
    sales_2025 = sales[sales["date"].dt.year == 2025].copy()

    total_sales_2024 = sales_2024["total_sales"].sum()
    total_sales_2025 = sales_2025["total_sales"].sum()

    growth_pct = (
        ((total_sales_2025 - total_sales_2024) / total_sales_2024) * 100
        if total_sales_2024 != 0 else np.nan
    )

    result = pd.DataFrame([{
        "actual_total_sales_2024": round(total_sales_2024, 2),
        "actual_total_sales_2025": round(total_sales_2025, 2),
        "growth_pct_2024_to_2025": round(growth_pct, 2) if pd.notnull(growth_pct) else np.nan,
    }])

    return result


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_actual_vs_human_forecast(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot actual sales vs human forecast and save figure."""
    plot_df = (
        df.groupby("date", as_index=False)[["total_sales", "forecast_sales"]]
        .sum()
        .sort_values("date")
    )

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["date"], plot_df["total_sales"], label="Actual Sales")
    plt.plot(plot_df["date"], plot_df["forecast_sales"], label="Human Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales (£)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def evaluate_human_forecast() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full human forecast evaluation workflow."""
    sales, labour = load_processed_data()
    merged_df = prepare_merged_dataset(sales, labour)

    df_all = merged_df.copy()
    df_2025 = merged_df[merged_df["date"].dt.year == 2025].copy()

    print_metrics_block(df_all, label="WHOLE DATASET")
    print_metrics_block(df_2025, label="2025 ONLY")

    # Save metrics
    metrics_all = build_metrics_table(df_all, label="whole_dataset")
    metrics_2025 = build_metrics_table(df_2025, label="2025_only")

    metrics_all_path = RESULTS_DIR / "human_forecast_metrics_all.csv"
    metrics_2025_path = RESULTS_DIR / "human_forecast_metrics_2025.csv"

    metrics_all.to_csv(metrics_all_path, index=False)
    metrics_2025.to_csv(metrics_2025_path, index=False)

    # Growth comparison
    growth_df = calculate_actual_sales_growth(sales)
    growth_path = RESULTS_DIR / "actual_sales_growth_comparison.csv"
    growth_df.to_csv(growth_path, index=False)

    print(f"\n{'=' * 60}")
    print("ACTUAL TOTAL SALES COMPARISON")
    print(f"{'=' * 60}")
    print(f"Actual total sales 2024: £{growth_df['actual_total_sales_2024'].iloc[0]:,.2f}")
    print(f"Actual total sales 2025: £{growth_df['actual_total_sales_2025'].iloc[0]:,.2f}")
    print(f"Growth from 2024 to 2025: {growth_df['growth_pct_2024_to_2025'].iloc[0]:.2f}%")

    # Plots
    plot_actual_vs_human_forecast(
        df_all,
        "Daily Total Actual Sales vs Human Forecast - Whole Dataset",
        FIGURES_DIR / "human_forecast_vs_actual_all.png",
    )

    plot_actual_vs_human_forecast(
        df_2025,
        "Daily Total Actual Sales vs Human Forecast - 2025 Only",
        FIGURES_DIR / "human_forecast_vs_actual_2025.png",
    )

    print(f"\nSaved metrics to: {metrics_all_path}")
    print(f"Saved metrics to: {metrics_2025_path}")
    print(f"Saved growth comparison to: {growth_path}")
    print(f"Saved figures to: {FIGURES_DIR}")

    return metrics_all, metrics_2025


if __name__ == "__main__":
    evaluate_human_forecast()