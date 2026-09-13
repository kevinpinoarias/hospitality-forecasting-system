"""
Real, already-computed sales and forecast-accuracy patterns for the
deployed final model - which single day/week/weekend had the highest or
lowest sales, which was forecast best or worst, and the average pattern by
day-of-week and by calendar month. Exposed as a grounded tool so the
assistant can answer real business questions ("which day of the week
earns the most", "was December a strong month") that neither get_forecast
(one date at a time) nor get_model_comparison (model-vs-model, not
day/week/month patterns) can answer.

Every forecast figure is the final model's out-of-sample backtest
prediction (reports/results/final_model_backtest_predictions.csv): what it
predicted for each day before it had seen that day.

Hardcoded rather than computed at runtime, for the same reason as
model_comparison_data.py: this is a fixed analysis of the historical
backtest window, not something that changes per request, and hardcoding
avoids shipping reports/ or a pandas groupby step into the assistant's own
lean Docker image. The figures come from
`python -m src.evaluation.assistant_grounding`, and
tests/test_assistant_grounding.py fails if they drift from it.

Known closure days (Christmas Day, New Year's Day - see
KNOWN_CLOSURE_MONTH_DAYS in src/preprocessing/build_daily_sales.py) are
excluded from every record and average below, and so is any week or
weekend containing one: a real £0 closure day would trivially "win" every
worst-case category and distort every average it touches, which isn't a
genuine sales or forecasting signal. Weeks and weekends must also be
complete in the data, so the 15 days missing from the backtest
(2025-01-24 to 2025-02-07) break the weeks around them.
"""

from __future__ import annotations

DATA_WINDOW = {
    "start_date": "2025-01-15",
    "end_date": "2026-09-06",
    "total_days": 585,
    "note": "January to September appear in two years of this window, October to December in one. "
            "Known closure days are excluded from the records and averages below, not from this total.",
}

BEST_WORST_SINGLE_DAY = {
    "highest_sales_day": {"date": "2025-06-28", "day_of_week": "Saturday", "actual_sales": 20067.04},
    "lowest_sales_day": {"date": "2026-01-05", "day_of_week": "Monday", "actual_sales": 2605.0},
    "best_forecast_day": {
        "date": "2025-03-11", "day_of_week": "Tuesday",
        "actual_sales": 5379.06, "forecast": 5383.04, "abs_error": 3.98,
    },
    "worst_forecast_day": {
        "date": "2025-06-28", "day_of_week": "Saturday",
        "actual_sales": 20067.04, "forecast": 13910.87, "abs_error": 6156.17,
        "note": "Same day as the highest-sales day - a real example of the project's own spike-day finding: the hardest days to forecast are the busiest ones.",
    },
}

BEST_WORST_SINGLE_WEEK = {
    "definition": "Monday-Sunday, complete weeks with no known closure day",
    "highest_sales_week": {"week_starting": "2025-12-08", "total_sales": 79736.18},
    "lowest_sales_week": {"week_starting": "2026-01-05", "total_sales": 36230.56},
    "best_forecast_week": {"week_starting": "2025-07-14", "mean_daily_abs_error": 372.58},
    "worst_forecast_week": {"week_starting": "2025-06-30", "mean_daily_abs_error": 2088.05},
}

BEST_WORST_SINGLE_WEEKEND = {
    "definition": "Saturday+Sunday pairs, referenced by the Saturday's date",
    "highest_sales_weekend": {"saturday_date": "2025-03-29", "total_sales": 30309.46},
    "lowest_sales_weekend": {"saturday_date": "2026-01-03", "total_sales": 15676.47},
    "best_forecast_weekend": {"saturday_date": "2025-09-20", "mean_daily_abs_error": 140.8},
    "worst_forecast_weekend": {"saturday_date": "2025-06-28", "mean_daily_abs_error": 4051.37},
}

BY_DAY_OF_WEEK = [
    {"day_of_week": "Monday", "avg_sales": 5328.11, "avg_forecast_abs_error": 725.27},
    {"day_of_week": "Tuesday", "avg_sales": 6150.42, "avg_forecast_abs_error": 755.47},
    {"day_of_week": "Wednesday", "avg_sales": 6664.76, "avg_forecast_abs_error": 703.95},
    {"day_of_week": "Thursday", "avg_sales": 7065.82, "avg_forecast_abs_error": 711.9},
    {"day_of_week": "Friday", "avg_sales": 10564.45, "avg_forecast_abs_error": 1087.65},
    {"day_of_week": "Saturday", "avg_sales": 15142.35, "avg_forecast_abs_error": 1290.03},
    {"day_of_week": "Sunday", "avg_sales": 8214.83, "avg_forecast_abs_error": 1107.32},
]

BY_MONTH = [
    {"month": "January", "avg_daily_sales": 7112.09, "avg_forecast_abs_error": 1237.16, "days": 39},
    {"month": "February", "avg_daily_sales": 8133.36, "avg_forecast_abs_error": 891.86, "days": 49},
    {"month": "March", "avg_daily_sales": 8007.27, "avg_forecast_abs_error": 789.89, "days": 62},
    {"month": "April", "avg_daily_sales": 8567.54, "avg_forecast_abs_error": 837.85, "days": 60},
    {"month": "May", "avg_daily_sales": 8582.71, "avg_forecast_abs_error": 896.59, "days": 62},
    {"month": "June", "avg_daily_sales": 8550.07, "avg_forecast_abs_error": 997.14, "days": 60},
    {"month": "July", "avg_daily_sales": 8589.98, "avg_forecast_abs_error": 1003.38, "days": 62},
    {"month": "August", "avg_daily_sales": 8640.85, "avg_forecast_abs_error": 785.81, "days": 62},
    {"month": "September", "avg_daily_sales": 7731.78, "avg_forecast_abs_error": 782.86, "days": 36},
    {"month": "October", "avg_daily_sales": 8170.51, "avg_forecast_abs_error": 754.14, "days": 31},
    {"month": "November", "avg_daily_sales": 8971.96, "avg_forecast_abs_error": 880.92, "days": 30},
    {"month": "December", "avg_daily_sales": 11005.5, "avg_forecast_abs_error": 1217.71, "days": 30},
]


def get_sales_and_forecast_patterns() -> dict:
    """Real, already-computed patterns from the final model's historical
    backtest window (2025-01-15 to 2026-09-06): which single
    day/week/weekend had the highest or lowest actual sales, which was
    forecast best or worst (smallest/largest error), and
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
