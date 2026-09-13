"""
Business impact - what forecast error costs in labour.

Model accuracy only matters to the venue through staffing: rotas are
planned from the sales forecast, so an over-forecast day means paying
for staff the demand never needed. This analysis translates forecast
error into wages and hours.

Why forecast-side ratios rather than actual wages: actual wages already
reflect on-the-day correction (staff sent home once a quiet shift is
obvious), so any measure built on them mixes the forecast's own error
with how well the venue corrected for it - the more aggressive the
correction, the less a forecast's mistakes show up in what was actually
paid. Instead, a wage-per-£-sale
and hours-per-£-sale ratio is derived purely from planned figures (what
the planning process itself believes it needs), then applied to the
sales over-forecast. See calculate_implied_labour_metrics in
src/models/evaluate_human_forecast.py.

Section 1 - the manual forecast's own labour planning, by year.

Section 2 - wage simulation for the Series 1 CatBoost winner, reproduced
from the original analysis over the 596 days it covers:
  - wages implied by staffing to each forecast, and the actual wage bill
  - overstaffing cost: wages attributable to days a forecast exceeded
    actual sales
  - the correction factor (actual wages / planned wages): how far below
    plan the venue historically ran after on-the-day correction. Applying
    it to the model's plan simulates "staff to the model, then correct on
    site as usual" - the realistic deployment, since a better forecast
    would not stop managers correcting in real time.

Section 3 - the same simulation for the final model (Series 16's bagged
log1p CatBoost) on Series 17's 569 matched days (new measurement).

Requires: catboost_winner_predictions.csv,
          wage_comparison_ai_vs_manual_vs_actual.csv,
          series17_manual_comparison_matched_days.csv (all Series 17)
Output:   reports/experiments/business_impact_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.series_common import require_output, results_path
from src.models.evaluate_human_forecast import load_processed_data, prepare_merged_dataset

YEARS = [2024, 2025, 2026]


def manual_labour_planning_by_year(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [(str(y), merged[merged["date"].dt.year == y]) for y in YEARS] + [("Total", merged)]:
        df = df.copy()
        overforecast_sales = (df["forecast_sales"] - df["total_sales"]).clip(lower=0).sum()
        valid = df[df["forecast_sales"] > 0]  # closures have zero forecast sales
        wage_ratio = valid["forecast_wages"].sum() / valid["forecast_sales"].sum()
        wage_ratio_daily = (valid["forecast_wages"] / valid["forecast_sales"]).mean()
        hours_ratio = valid["forecast_hours"].sum() / valid["forecast_sales"].sum()
        rows.append({
            "period": label,
            "days": len(df),
            "overforecast_sales": overforecast_sales,
            "wage_per_sale_ratio_pooled": wage_ratio,
            "wage_per_sale_ratio_daily_avg": wage_ratio_daily,
            "implied_wasted_wages_pooled_ratio": overforecast_sales * wage_ratio,
            "implied_wasted_wages_daily_avg_ratio": overforecast_sales * wage_ratio_daily,
            "overforecast_wages_net_of_correction": (df["forecast_wages"] - df["total_wages"]).clip(lower=0).sum(),
            "forecast_hours_per_pound_sale": hours_ratio,
            "actual_hours_per_pound_sale": valid["total_hours"].sum() / valid["total_sales"].sum(),
            "implied_overforecast_hours": overforecast_sales * hours_ratio,
            "overforecast_hours_net_of_correction": (df["forecast_hours"] - df["total_hours"]).clip(lower=0).sum(),
            "days_actual_hours_exceeded_plan": int((df["total_hours"] > df["forecast_hours"]).sum()),
            "total_hours_worked_beyond_plan": (df["total_hours"] - df["forecast_hours"]).clip(lower=0).sum(),
        })
    return pd.DataFrame(rows)


def wage_simulation(df: pd.DataFrame, actual_col: str, model_col: str, label: str, wage_bill_2025: float, revenue_2025: float) -> dict:
    """Plan-vs-actual wages for one model against the manual forecast,
    over whatever days `df` covers."""
    ratio = df["forecast_wages"].sum() / df["forecast_sales"].sum()
    model_plan = (df[model_col] * ratio).sum()
    manual_plan = (df["forecast_sales"] * ratio).sum()
    actual_spent = df["total_wages"].sum()

    model_overstaff = ((df[model_col] - df[actual_col]).clip(lower=0) * ratio).sum()
    manual_overstaff = ((df["forecast_sales"] - df[actual_col]).clip(lower=0) * ratio).sum()

    correction_pooled = actual_spent / df["forecast_wages"].sum()
    per_day_correction = (df["total_wages"] / df["forecast_wages"]).replace([np.inf, -np.inf], np.nan)
    simulated_pooled = model_plan * correction_pooled
    simulated_per_day = (df[model_col] * ratio * per_day_correction).sum()

    days = len(df)
    # Headline uses the per-day correction factor, as the original analysis
    # did: it respects that correction varies day to day rather than
    # assuming one average rate applies uniformly.
    savings_per_day = actual_spent - simulated_per_day
    annualised = savings_per_day / (days / 365)
    return {
        "model": label,
        "days": days,
        "wage_per_sale_ratio": ratio,
        "manual_plan_wages": manual_plan,
        "model_plan_wages": model_plan,
        "actual_wages_spent": actual_spent,
        "manual_overstaffing_cost": manual_overstaff,
        "model_overstaffing_cost": model_overstaff,
        "manual_overforecast_days": int((df["forecast_sales"] > df[actual_col]).sum()),
        "model_overforecast_days": int((df[model_col] > df[actual_col]).sum()),
        "plan_savings_vs_manual": manual_plan - model_plan,
        "overstaffing_savings_vs_manual": manual_overstaff - model_overstaff,
        "correction_factor_pooled": correction_pooled,
        "correction_factor_daily_avg": (df["total_wages"] / df["forecast_wages"]).mean(),
        "simulated_model_plus_correction_pooled": simulated_pooled,
        "simulated_model_plus_correction_per_day": simulated_per_day,
        "simulated_savings_vs_actual_pooled": actual_spent - simulated_pooled,
        "simulated_savings_vs_actual_per_day": savings_per_day,
        "annualised_savings_per_day_factor": annualised,
        "annualised_savings_pct_of_2025_wage_bill": annualised / wage_bill_2025 * 100,
        "annualised_savings_pct_of_2025_revenue": annualised / revenue_2025 * 100,
    }


def run() -> dict:
    sales, labour = load_processed_data()
    merged = prepare_merged_dataset(sales, labour)
    y2025 = merged[merged["date"].dt.year == 2025]
    wage_bill_2025, revenue_2025 = y2025["total_wages"].sum(), y2025["total_sales"].sum()

    planning = manual_labour_planning_by_year(merged)

    series1 = pd.read_csv(
        require_output("wage_comparison_ai_vs_manual_vs_actual.csv", "run_series17_manual_forecast_comparison"),
        parse_dates=["date"],
    )
    historical = wage_simulation(series1, "total_sales", "ai_prediction", "Series 1 CatBoost winner", wage_bill_2025, revenue_2025)

    matched = pd.read_csv(
        require_output("series17_manual_comparison_matched_days.csv", "run_series17_manual_forecast_comparison"),
        parse_dates=["date"],
    )
    matched = matched.merge(merged[["date", "total_wages", "forecast_wages"]], on="date", how="inner")
    final = wage_simulation(matched, "actual", "final_pred", "Final bagged log1p CatBoost", wage_bill_2025, revenue_2025)

    simulations = pd.DataFrame([historical, final])
    planning.to_csv(results_path("business_impact_manual_planning_by_year.csv"), index=False)
    simulations.to_csv(results_path("business_impact_summary.csv"), index=False)

    return {
        "manual_planning_by_year": planning,
        "wage_simulations": simulations,
        "revenue_and_wages_by_year": {
            y: (merged.loc[merged["date"].dt.year == y, "total_sales"].sum(),
                merged.loc[merged["date"].dt.year == y, "total_wages"].sum())
            for y in [2024, 2025]
        },
    }


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 220)
    print("=== Section 1: manual forecast labour planning by year ===")
    print(out["manual_planning_by_year"].round(4).T.to_string())
    print()
    print("=== Sections 2-3: wage simulations ===")
    print(out["wage_simulations"].round(2).T.to_string())
    print()
    for year, (revenue, wages) in out["revenue_and_wages_by_year"].items():
        print(f"{year}: revenue £{revenue:,.0f} | wage bill £{wages:,.0f} | wages {wages / revenue * 100:.1f}% of revenue")
