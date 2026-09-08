"""
Real, already-computed Track A model-comparison results, exposed as a
grounded tool so the assistant can answer questions about how the models
this project tested actually performed - not just serve /predict
forecasts. Without this, the chat layer only ever talks about forecasting
individual dates, and the substantial ML engineering work behind it (five
baselines, SARIMAX, XGBoost, an LSTM, and a small Transformer, evaluated
and compared properly) never surfaces unless a visitor happens to read the
notebook or README themselves.

Hardcoded rather than read from reports/results/*.csv at runtime: these
are finalized, one-off evaluation results from completed work (Track A
step 3), not something that changes per request, and hardcoding avoids
needing to ship the reports/ directory into the assistant's own
deliberately lean Docker image. Every figure below was copied directly
from reports/results/model_comparison_*.csv on 2026-09-08 - re-run
src/evaluation/model_comparison.py and update this file if the models are
ever retrained.
"""

from __future__ import annotations

EVALUATION_WINDOW = {
    "start_date": "2024-12-31",
    "end_date": "2026-01-04",
    "total_days": 339,
    "spike_days": 34,
    "spike_day_definition": "top 9% of days by actual sales",
}

# The single figure quoted in the assistant's own greeting/system prompt
# (query_forecast.py) - kept here, not duplicated, so both always agree.
XGBOOST_MAPE_PCT = 13.24

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
    """Real, already-computed results comparing every forecasting model this
    project tested - XGBoost, a small Transformer, an LSTM, SARIMAX, and
    five simple baselines - against 339 days of real held-out historical
    data. Use this whenever asked about the project's model comparison,
    the neural network results, which model performed best, spike-day
    performance, or the business/labour-cost impact of forecast error -
    these figures only come from here, never invent or estimate them.
    """
    return {
        "evaluation_window": EVALUATION_WINDOW,
        "overall_accuracy": OVERALL_ACCURACY,
        "wins_by_individual_day": WINS_BY_INDIVIDUAL_DAY,
        "spike_day_accuracy": SPIKE_DAY_ACCURACY,
        "business_impact_implied_wages": BUSINESS_IMPACT_IMPLIED_WAGES,
        "business_impact_note": BUSINESS_IMPACT_NOTE,
    }
