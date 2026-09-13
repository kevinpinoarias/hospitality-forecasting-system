"""
Real, already-computed model results, exposed as a grounded tool so the
assistant can answer questions about how the models this project tested
actually performed - not just serve /predict forecasts. Without this, the
chat layer only ever talks about forecasting individual dates, and the
substantial ML engineering work behind it never surfaces unless a visitor
happens to read the README or notebook themselves.

Two groups of results:

- The final model - the one the live API serves (25-seed log1p CatBoost,
  chosen by the rolling-origin experimentation programme in
  docs/EXPERIMENT_LOG.md): its comparison with the venue's manual forecast
  and the original XGBoost (Series 17), the pipeline's own backtest, how
  accuracy holds up further ahead (Series 18), and the business-impact
  simulation.
- The earlier single-split comparison of every model family (Track A step
  3: five baselines, SARIMAX, XGBoost, an LSTM and a small Transformer),
  which picked XGBoost as the starting point.

Hardcoded rather than read from reports/ at runtime: these are finalized
results, not something that changes per request, and hardcoding avoids
shipping reports/ into the assistant's deliberately lean Docker image. The
final-model figures come from `python -m src.evaluation.assistant_grounding`
(tests/test_assistant_grounding.py fails if they drift from it); the
single-split figures were copied from reports/results/model_comparison_*.csv
on 2026-09-08.
"""

from __future__ import annotations

# Headline figures quoted in the assistant's greeting and system prompt
# (query_forecast.py) - kept here, not duplicated, so both always agree.
# From Series 17, on 569 matched days.
FINAL_MODEL_MAPE_PCT = 11.97
FINAL_MODEL_PCT_BETTER_THAN_MANUAL = 15.36

FINAL_MODEL = {
    "description": "CatBoost on 38 features, trained on log-transformed sales, averaged over 25 random seeds",
    "chosen_by": "18 series of rolling-origin backtesting experiments (10 folds of 60-day test windows)",
    "trained_on": "every day from 2024-01-16 to 2026-09-06",
}

VS_MANUAL_FORECAST = {
    "days": 569,
    "start_date": "2025-01-15",
    "end_date": "2026-08-23",
    "manual_forecast": {"mae": 1085.97, "mape_pct": 14.86},
    "final_model": {"mae": 919.17, "mape_pct": 11.97},
    "original_xgboost_same_backtest": {"mae": 1034.82, "mape_pct": 13.94},
    "final_model_pct_better_than_manual": 15.36,
    "original_xgboost_pct_better_than_manual": 4.71,
}

PIPELINE_BACKTEST = {
    "days": 585,
    "start_date": "2025-01-15",
    "end_date": "2026-09-06",
    "final_model_mae": 909.57,
    "final_model_mape_pct": 11.89,
    "original_xgboost_mae": 1026.64,
    "manual_forecast_mae": 1074.46,
}

# MAE (GBP) by how many days past the last day of real sales data the
# forecast date is - Series 18, 8 forecast origins. "With manager forecast"
# means the manager's own sales estimate was supplied as an input.
ACCURACY_BY_DAYS_AHEAD = [
    {"days_ahead": "1-7", "manual_forecast": 1103.59, "final_model_with_manager_forecast": 862.52,
     "original_xgboost_with_manager_forecast": 952.94, "final_model_without_manager_forecast": 884.52,
     "original_xgboost_without_manager_forecast": 912.46},
    {"days_ahead": "8-16", "manual_forecast": 948.75, "final_model_with_manager_forecast": 972.66,
     "original_xgboost_with_manager_forecast": 979.24, "final_model_without_manager_forecast": 1041.39,
     "original_xgboost_without_manager_forecast": 1030.68},
    {"days_ahead": "17-30", "manual_forecast": 1037.85, "final_model_with_manager_forecast": 1031.37,
     "original_xgboost_with_manager_forecast": 1118.68, "final_model_without_manager_forecast": 1083.29,
     "original_xgboost_without_manager_forecast": 1132.42},
    {"days_ahead": "31-60", "manual_forecast": 1117.61, "final_model_with_manager_forecast": 1014.94,
     "original_xgboost_with_manager_forecast": 1077.87, "final_model_without_manager_forecast": 1198.69,
     "original_xgboost_without_manager_forecast": 1169.34},
    {"days_ahead": "61-90", "manual_forecast": 1021.67, "final_model_with_manager_forecast": 910.68,
     "original_xgboost_with_manager_forecast": 1078.38, "final_model_without_manager_forecast": 952.98,
     "original_xgboost_without_manager_forecast": 1093.59},
    {"days_ahead": "91-180", "manual_forecast": 1107.88, "final_model_with_manager_forecast": 989.41,
     "original_xgboost_with_manager_forecast": 1090.51, "final_model_without_manager_forecast": 1097.27,
     "original_xgboost_without_manager_forecast": 1159.36},
    {"days_ahead": "1-180", "manual_forecast": 1082.46, "final_model_with_manager_forecast": 977.76,
     "original_xgboost_with_manager_forecast": 1077.91, "final_model_without_manager_forecast": 1078.12,
     "original_xgboost_without_manager_forecast": 1132.42},
]

BUSINESS_IMPACT_SIMULATION = {
    "days": 569,
    "planned_wages_lower_than_manual_gbp": 96651.77,
    "simulated_saving_vs_actual_wages_gbp": 90194.29,
    "annualised_saving_gbp": 57857.5,
    "annualised_saving_pct_of_2025_wage_bill": 5.55,
}

FINAL_MODEL_NOTES = [
    "The 15% lead over the manual forecast is for forecasts made with recent sales available. "
    "The live demo's data ends on 2026-09-06, so its forecasts run further ahead - see accuracy_by_days_ahead.",
    "Seed selection bias: a log-transformed target first looked £15.39 better than the previous best. "
    "Re-running across 25 random seeds showed the seed used until then was an unusually lucky draw; "
    "the real gain was £3.35, and that is the figure reported.",
    "The business-impact figure is a simulation over historical days that counts wage savings only: "
    "it does not cost understaffing on days the model under-forecasts, and is not a measured outcome.",
]

# ---------------------------------------------------------------------------
# Earlier single-split comparison (Track A step 3)
# ---------------------------------------------------------------------------

EVALUATION_WINDOW = {
    "start_date": "2024-12-31",
    "end_date": "2026-01-04",
    "total_days": 339,
    "spike_days": 34,
    "spike_day_definition": "top 9% of days by actual sales",
}

OVERALL_ACCURACY = [
    {"model": "xgboost", "mae": 1049.94, "rmse": 1373.08, "mape_pct": 13.24, "bias": 43.89},
    {"model": "transformer", "mae": 1280.83, "rmse": 1849.46, "mape_pct": 15.73, "bias": 106.81},
    {"model": "seasonal_naive", "mae": 1353.24, "rmse": 1999.81, "mape_pct": 18.47, "bias": 41.56},
    {"model": "lstm", "mae": 1357.35, "rmse": 1886.74, "mape_pct": 16.19, "bias": -144.28},
    {"model": "weekday_average", "mae": 1419.01, "rmse": 1982.10, "mape_pct": 16.58, "bias": -610.26},
    {"model": "roll7", "mae": 2877.24, "rmse": 3621.23, "mape_pct": 34.93, "bias": 77.17},
    {"model": "roll14", "mae": 2884.31, "rmse": 3650.09, "mape_pct": 35.36, "bias": 116.48},
    {"model": "roll28", "mae": 2942.73, "rmse": 3731.41, "mape_pct": 36.96, "bias": 176.21},
    {"model": "sarimax", "mae": 3080.70, "rmse": 3977.42, "mape_pct": 42.52, "bias": 885.72},
    {"model": "naive", "mae": 3153.29, "rmse": 4114.98, "mape_pct": 37.27, "bias": 22.01},
]

WINS_BY_INDIVIDUAL_DAY = [
    {"model": "seasonal_naive", "wins": 71, "win_pct": 20.9},
    {"model": "xgboost", "wins": 64, "win_pct": 18.9},
    {"model": "transformer", "wins": 47, "win_pct": 13.9},
    {"model": "weekday_average", "wins": 42, "win_pct": 12.4},
    {"model": "lstm", "wins": 39, "win_pct": 11.5},
    {"model": "naive", "wins": 27, "win_pct": 8.0},
    {"model": "roll7", "wins": 14, "win_pct": 4.1},
    {"model": "roll28", "wins": 13, "win_pct": 3.8},
    {"model": "sarimax", "wins": 13, "win_pct": 3.8},
    {"model": "roll14", "wins": 9, "win_pct": 2.7},
]

SPIKE_DAY_ACCURACY = [
    {"model": "xgboost", "mae_on_spike_days": 1285.01, "mae_on_normal_days": 1023.74},
    {"model": "lstm", "mae_on_spike_days": 1963.81, "mae_on_normal_days": 1289.74},
    {"model": "transformer", "mae_on_spike_days": 1618.65, "mae_on_normal_days": 1243.18},
    {"model": "seasonal_naive", "mae_on_spike_days": 1300.94, "mae_on_normal_days": 1359.07},
    {"model": "weekday_average", "mae_on_spike_days": 2273.20, "mae_on_normal_days": 1323.79},
    {"model": "naive", "mae_on_spike_days": 4837.83, "mae_on_normal_days": 2965.50},
    {"model": "roll7", "mae_on_spike_days": 7169.36, "mae_on_normal_days": 2398.78},
    {"model": "roll14", "mae_on_spike_days": 7325.33, "mae_on_normal_days": 2389.25},
    {"model": "roll28", "mae_on_spike_days": 7488.95, "mae_on_normal_days": 2435.94},
    {"model": "sarimax", "mae_on_spike_days": 5497.65, "mae_on_normal_days": 2811.27},
]

BUSINESS_IMPACT_IMPLIED_WAGES = [
    {"model": "weekday_average", "overforecasted_wages_gbp": 43537.3, "underforecasted_wages_gbp": 109241.0, "overstaffed_days": 122, "understaffed_days": 217},
    {"model": "xgboost", "overforecasted_wages_gbp": 58883.9, "underforecasted_wages_gbp": 54158.4, "overstaffed_days": 167, "understaffed_days": 172},
    {"model": "lstm", "overforecasted_wages_gbp": 65302.3, "underforecasted_wages_gbp": 80836.6, "overstaffed_days": 165, "understaffed_days": 174},
    {"model": "transformer", "overforecasted_wages_gbp": 74700.4, "underforecasted_wages_gbp": 63200.7, "overstaffed_days": 175, "understaffed_days": 164},
    {"model": "seasonal_naive", "overforecasted_wages_gbp": 75085.9, "underforecasted_wages_gbp": 70611.1, "overstaffed_days": 156, "understaffed_days": 181},
    {"model": "roll7", "overforecasted_wages_gbp": 159044.0, "underforecasted_wages_gbp": 150735.0, "overstaffed_days": 214, "understaffed_days": 125},
    {"model": "roll14", "overforecasted_wages_gbp": 161540.0, "underforecasted_wages_gbp": 149000.0, "overstaffed_days": 216, "understaffed_days": 123},
    {"model": "roll28", "overforecasted_wages_gbp": 167900.0, "underforecasted_wages_gbp": 148929.0, "overstaffed_days": 210, "understaffed_days": 129},
    {"model": "naive", "overforecasted_wages_gbp": 170934.0, "underforecasted_wages_gbp": 168565.0, "overstaffed_days": 144, "understaffed_days": 195},
    {"model": "sarimax", "overforecasted_wages_gbp": 213522.0, "underforecasted_wages_gbp": 118161.0, "overstaffed_days": 215, "understaffed_days": 124},
]

BUSINESS_IMPACT_NOTE = (
    "Business-impact figures are an approximation: no model here predicts wages directly, so "
    "each model's predicted sales is converted to an implied wage figure using the training-period "
    "wage-to-sales ratio (about 0.32) - useful for comparing models against each other, not a "
    "measured wage relationship."
)


def get_model_comparison() -> dict:
    """Real, already-computed results for the forecasting models this
    project tested. `final_model` is the model the forecasts come from: how
    it compares with the venue's manual forecast and the original XGBoost,
    how its accuracy holds up for dates further ahead, and the simulated
    labour-cost impact. `earlier_model_comparison` is the first,
    single-split comparison of every model family - XGBoost, a small
    Transformer, an LSTM, SARIMAX and five simple baselines - including
    spike-day accuracy. Use this whenever asked about the project's model
    comparison, the neural network results, which model performed best,
    how accurate the forecasts are further ahead, spike-day performance, or
    the business/labour-cost impact of forecast error - these figures only
    come from here, never invent or estimate them.
    """
    return {
        "final_model": {
            **FINAL_MODEL,
            "vs_manual_forecast": VS_MANUAL_FORECAST,
            "pipeline_backtest": PIPELINE_BACKTEST,
            "accuracy_by_days_ahead": ACCURACY_BY_DAYS_AHEAD,
            "business_impact_simulation": BUSINESS_IMPACT_SIMULATION,
            "notes": FINAL_MODEL_NOTES,
        },
        "earlier_model_comparison": {
            "evaluation_window": EVALUATION_WINDOW,
            "overall_accuracy": OVERALL_ACCURACY,
            "wins_by_individual_day": WINS_BY_INDIVIDUAL_DAY,
            "spike_day_accuracy": SPIKE_DAY_ACCURACY,
            "business_impact_implied_wages": BUSINESS_IMPACT_IMPLIED_WAGES,
            "business_impact_note": BUSINESS_IMPACT_NOTE,
        },
    }
