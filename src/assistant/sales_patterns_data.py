"""
Real, already-computed sales and forecast-accuracy patterns for the
deployed XGBoost model - which single day/week/weekend had the highest or
lowest sales, which was forecast best or worst, and the average pattern by
day-of-week and by calendar month. Exposed as a grounded tool so the
assistant can answer real business questions ("which day of the week
earns the most", "was December a strong month") that neither get_forecast
(one date at a time) nor get_model_comparison (model-vs-model, not
day/week/month patterns) can answer.

Hardcoded rather than computed at runtime, for the same reason as
model_comparison_data.py: this is a fixed, already-computed analysis of
the historical evaluation window, not something that changes per request,
and hardcoding avoids shipping reports/ or a pandas groupby step into the
assistant's own lean Docker image. Every figure below was computed
directly from reports/results/xgboost_predictions.csv on 2026-09-08 (see
scratch computation in that day's session - not re-derivable from a
committed script, since this file wasn't itself part of the checked-in
evaluation pipeline). Re-run against a fresh xgboost_predictions.csv and
update this file if the model is ever retrained.

Known closure days (Christmas Day, New Year's Day - see
KNOWN_CLOSURE_MONTH_DAYS in src/preprocessing/build_daily_sales.py) are
excluded from every single-day record and every day-of-week/month
average below, for the same reason the live API now flags them
separately: a real £0 closure day would trivially "win" every worst-case
category and distort every average it touches, which isn't a genuine
sales or forecasting signal.
"""

from __future__ import annotations

DATA_WINDOW = {
    "start_date": "2024-12-31",
    "end_date": "2026-01-04",
    "total_days": 369,
    "note": "Known closure days are excluded from the records/averages below, not from this total.",
}

BEST_WORST_SINGLE_DAY = {
    "highest_sales_day": {"date": "2025-06-28", "day_of_week": "Saturday", "actual_sales": 20067.04},
    "lowest_sales_day": {"date": "2025-01-13", "day_of_week": "Monday", "actual_sales": 2329.48},
    "best_forecast_day": {
        "date": "2025-12-04", "day_of_week": "Thursday",
        "actual_sales": 10080.62, "forecast": 10075.96, "abs_error": 4.67,
    },
    "worst_forecast_day": {
        "date": "2025-06-28", "day_of_week": "Saturday",
        "actual_sales": 20067.04, "forecast": 15660.49, "abs_error": 4406.55,
        "note": "Same day as the highest-sales day - a real example of the project's own spike-day finding: the hardest days to forecast are the busiest ones.",
    },
}

BEST_WORST_SINGLE_WEEK = {
    "definition": "Monday-Sunday, full 7-day weeks only (partial weeks at the start/end of the data excluded for a fair comparison)",
    "highest_sales_week": {"week_starting": "2025-12-08", "total_sales": 79736.17},
    "lowest_sales_week": {"week_starting": "2025-01-06", "total_sales": 43336.59},
    "best_forecast_week": {"week_starting": "2025-07-28", "mean_daily_abs_error": 376.73},
    "worst_forecast_week": {
        "week_starting": "2025-12-22", "mean_daily_abs_error": 2412.85,
        "note": "This week contains Christmas Day, a known closure day - part of this week's high error is the model not knowing the venue was shut that day, not a genuine forecasting weakness.",
    },
}

BEST_WORST_SINGLE_WEEKEND = {
    "definition": "Saturday+Sunday pairs, referenced by the Saturday's date",
    "highest_sales_weekend": {"saturday_date": "2025-03-29", "total_sales": 30309.46},
    "lowest_sales_weekend": {"saturday_date": "2026-01-03", "total_sales": 15676.47},
    "best_forecast_weekend": {"saturday_date": "2025-08-02", "mean_daily_abs_error": 107.43},
    "worst_forecast_weekend": {"saturday_date": "2025-11-29", "mean_daily_abs_error": 3549.81},
}

BY_DAY_OF_WEEK = [
    {"day_of_week": "Monday", "avg_sales": 5609.92, "avg_forecast_abs_error": 1080.19},
    {"day_of_week": "Tuesday", "avg_sales": 6626.63, "avg_forecast_abs_error": 848.37},
    {"day_of_week": "Wednesday", "avg_sales": 6763.34, "avg_forecast_abs_error": 756.17},
    {"day_of_week": "Thursday", "avg_sales": 7285.73, "avg_forecast_abs_error": 825.47},
    {"day_of_week": "Friday", "avg_sales": 10925.53, "avg_forecast_abs_error": 1245.94},
    {"day_of_week": "Saturday", "avg_sales": 15380.25, "avg_forecast_abs_error": 1244.62},
    {"day_of_week": "Sunday", "avg_sales": 8552.57, "avg_forecast_abs_error": 1191.45},
]

BY_MONTH = [
    {"month": "January", "avg_daily_sales": 7158.85, "avg_forecast_abs_error": 1223.41},
    {"month": "February", "avg_daily_sales": 8245.74, "avg_forecast_abs_error": 901.32},
    {"month": "March", "avg_daily_sales": 8620.16, "avg_forecast_abs_error": 899.65},
    {"month": "April", "avg_daily_sales": 8974.95, "avg_forecast_abs_error": 877.70},
    {"month": "May", "avg_daily_sales": 9145.75, "avg_forecast_abs_error": 819.48},
    {"month": "June", "avg_daily_sales": 9003.13, "avg_forecast_abs_error": 1120.95},
    {"month": "July", "avg_daily_sales": 8869.16, "avg_forecast_abs_error": 990.70},
    {"month": "August", "avg_daily_sales": 9186.45, "avg_forecast_abs_error": 1066.13},
    {"month": "September", "avg_daily_sales": 7684.95, "avg_forecast_abs_error": 870.78},
    {"month": "October", "avg_daily_sales": 8170.51, "avg_forecast_abs_error": 954.33},
    {"month": "November", "avg_daily_sales": 8971.96, "avg_forecast_abs_error": 1022.41},
    {"month": "December", "avg_daily_sales": 10967.54, "avg_forecast_abs_error": 1570.10},
]


def get_sales_and_forecast_patterns() -> dict:
    """Real, already-computed patterns from the historical evaluation
    window: which single day/week/weekend had the highest or lowest actual
    sales, which was forecast best or worst (smallest/largest error), and
    the average sales level and forecast accuracy broken down by
    day-of-week and by calendar month. Use this for any question about
    which day/week/weekend/month was busiest, quietest, best-forecast, or
    worst-forecast, or about typical patterns by weekday or month (e.g.
    "which day of the week earns the most", "is December a strong
    month", "what was our best week"). These figures only come from here
    - never invent or estimate them, and never average multiple
    get_forecast calls yourself to answer this kind of question.
    """
    return {
        "data_window": DATA_WINDOW,
        "best_worst_single_day": BEST_WORST_SINGLE_DAY,
        "best_worst_single_week": BEST_WORST_SINGLE_WEEK,
        "best_worst_single_weekend": BEST_WORST_SINGLE_WEEKEND,
        "by_day_of_week": BY_DAY_OF_WEEK,
        "by_month": BY_MONTH,
    }
