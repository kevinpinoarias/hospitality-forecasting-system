"""
Computes the fixed figures the chat assistant quotes, from the pipeline's
and experiment log's own result files.

The assistant keeps these figures hardcoded (src/assistant/
model_comparison_data.py and sales_patterns_data.py) so its Docker image
doesn't need pandas or the reports/ directory. This script is where those
numbers come from: run it after `python main.py` and
`python -m src.experiments.run_all_series`, and copy its output into the
two files. tests/test_assistant_grounding.py fails if the hardcoded values
drift from what this script computes.

Usage:
    python -m src.evaluation.assistant_grounding
"""

from __future__ import annotations

import pprint
from pathlib import Path

import pandas as pd

from src.preprocessing.build_daily_sales import KNOWN_CLOSURE_MONTH_DAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _money(x: float) -> float:
    return round(float(x), 2)


def _is_closure(dates: pd.Series) -> pd.Series:
    return pd.Series([(d.month, d.day) in KNOWN_CLOSURE_MONTH_DAYS for d in dates], index=dates.index)


def sales_patterns(predictions_path: Path = RESULTS_DIR / "final_model_backtest_predictions.csv") -> dict:
    """Best/worst days, weeks and weekends, and weekday/month averages, from
    the final model's out-of-sample backtest predictions. Known closure days
    are left out of every record and average, and so is any week or weekend
    containing one: a £0 closure day is not a sales or forecasting signal."""
    df = pd.read_csv(predictions_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["abs_error"] = (df["total_sales"] - df["final_model_prediction"]).abs()
    df["closure"] = _is_closure(df["date"])
    open_days = df[~df["closure"]]

    def day_record(row: pd.Series, with_forecast: bool) -> dict:
        record = {"date": row["date"].date().isoformat(), "day_of_week": WEEKDAYS[row["date"].dayofweek],
                  "actual_sales": _money(row["total_sales"])}
        if with_forecast:
            record.update(forecast=_money(row["final_model_prediction"]), abs_error=_money(row["abs_error"]))
        return record

    single_day = {
        "highest_sales_day": day_record(open_days.loc[open_days["total_sales"].idxmax()], False),
        "lowest_sales_day": day_record(open_days.loc[open_days["total_sales"].idxmin()], False),
        "best_forecast_day": day_record(open_days.loc[open_days["abs_error"].idxmin()], True),
        "worst_forecast_day": day_record(open_days.loc[open_days["abs_error"].idxmax()], True),
    }

    indexed = df.set_index("date")
    weeks = []
    for monday in pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON"):
        days = pd.date_range(monday, periods=7)
        if all(d in indexed.index for d in days) and not indexed.loc[days, "closure"].any():
            block = indexed.loc[days]
            weeks.append({"week_starting": monday.date().isoformat(), "total_sales": block["total_sales"].sum(),
                          "mean_daily_abs_error": block["abs_error"].mean()})
    weeks = pd.DataFrame(weeks)

    weekends = []
    for saturday in pd.date_range(df["date"].min(), df["date"].max(), freq="W-SAT"):
        days = pd.date_range(saturday, periods=2)
        if all(d in indexed.index for d in days) and not indexed.loc[days, "closure"].any():
            block = indexed.loc[days]
            weekends.append({"saturday_date": saturday.date().isoformat(), "total_sales": block["total_sales"].sum(),
                             "mean_daily_abs_error": block["abs_error"].mean()})
    weekends = pd.DataFrame(weekends)

    def block_records(blocks: pd.DataFrame, key: str, noun: str) -> dict:
        def rec(i, field):
            return {key: blocks.at[i, key], field: _money(blocks.at[i, field])}
        return {
            f"highest_sales_{noun}": rec(blocks["total_sales"].idxmax(), "total_sales"),
            f"lowest_sales_{noun}": rec(blocks["total_sales"].idxmin(), "total_sales"),
            f"best_forecast_{noun}": rec(blocks["mean_daily_abs_error"].idxmin(), "mean_daily_abs_error"),
            f"worst_forecast_{noun}": rec(blocks["mean_daily_abs_error"].idxmax(), "mean_daily_abs_error"),
        }

    by_weekday = open_days.groupby(open_days["date"].dt.dayofweek).agg(
        avg_sales=("total_sales", "mean"), avg_forecast_abs_error=("abs_error", "mean"))
    by_month = open_days.groupby(open_days["date"].dt.month).agg(
        avg_daily_sales=("total_sales", "mean"), avg_forecast_abs_error=("abs_error", "mean"),
        days=("total_sales", "size"))

    return {
        "DATA_WINDOW": {
            "start_date": df["date"].min().date().isoformat(),
            "end_date": df["date"].max().date().isoformat(),
            "total_days": len(df),
        },
        "BEST_WORST_SINGLE_DAY": single_day,
        "BEST_WORST_SINGLE_WEEK": block_records(weeks, "week_starting", "week"),
        "BEST_WORST_SINGLE_WEEKEND": block_records(weekends, "saturday_date", "weekend"),
        "BY_DAY_OF_WEEK": [
            {"day_of_week": WEEKDAYS[i], "avg_sales": _money(r["avg_sales"]),
             "avg_forecast_abs_error": _money(r["avg_forecast_abs_error"])}
            for i, r in by_weekday.iterrows()
        ],
        "BY_MONTH": [
            {"month": MONTHS[i - 1], "avg_daily_sales": _money(r["avg_daily_sales"]),
             "avg_forecast_abs_error": _money(r["avg_forecast_abs_error"]), "days": int(r["days"])}
            for i, r in by_month.iterrows()
        ],
    }


def final_model_results() -> dict:
    """The final model's headline results: the like-for-like comparison with
    the manual forecast (Series 17), the pipeline's own backtest, how
    accuracy holds up further ahead (Series 18), and the business-impact
    simulation."""
    s17 = pd.read_csv(EXPERIMENTS_DIR / "series17_summary.csv")
    a = s17[s17["section"].isin(["A", "C"])].set_index("model")
    manual, final, xgb = (a.loc["Manual forecast"], a.loc["Final bagged log1p CatBoost"],
                          a.loc["Portfolio XGBoost, same folds"])
    matched = pd.read_csv(EXPERIMENTS_DIR / "series17_manual_comparison_matched_days.csv", parse_dates=["date"])

    pipeline = pd.read_csv(RESULTS_DIR / "final_model_metrics.csv").set_index("model")
    backtest = pd.read_csv(RESULTS_DIR / "final_model_backtest_predictions.csv", parse_dates=["date"])

    s18 = pd.read_csv(EXPERIMENTS_DIR / "series18_summary.csv")
    labels = {
        "Manual forecast": "manual_forecast",
        "Final model, seasonal history, manager forecast supplied": "final_model_with_manager_forecast",
        "Final model, seasonal history, no manager forecast": "final_model_without_manager_forecast",
        "Original XGBoost, manager forecast supplied": "original_xgboost_with_manager_forecast",
        "Original XGBoost, no manager forecast": "original_xgboost_without_manager_forecast",
    }
    s18 = s18[s18["model"].isin(labels)]
    horizon = [
        {"days_ahead": h.replace("days ", ""), **{labels[m]: _money(v) for m, v in zip(g["model"], g["MAE"])}}
        for h, g in s18.groupby("horizon", sort=False)
    ]

    impact = pd.read_csv(EXPERIMENTS_DIR / "business_impact_summary.csv").set_index("model").loc[
        "Final bagged log1p CatBoost"]

    return {
        "VS_MANUAL_FORECAST": {
            "days": int(manual["days"]),
            "start_date": matched["date"].min().date().isoformat(),
            "end_date": matched["date"].max().date().isoformat(),
            "manual_forecast": {"mae": _money(manual["MAE"]), "mape_pct": _money(manual["MAPE"])},
            "final_model": {"mae": _money(final["MAE"]), "mape_pct": _money(final["MAPE"])},
            "original_xgboost_same_backtest": {"mae": _money(xgb["MAE"]), "mape_pct": _money(xgb["MAPE"])},
            "final_model_pct_better_than_manual": _money(final["pct_better_than_manual"]),
            "original_xgboost_pct_better_than_manual": _money(xgb["pct_better_than_manual"]),
        },
        "PIPELINE_BACKTEST": {
            "days": int(pipeline.at["final_model", "days"]),
            "start_date": backtest["date"].min().date().isoformat(),
            "end_date": backtest["date"].max().date().isoformat(),
            "final_model_mae": _money(pipeline.at["final_model", "MAE"]),
            "final_model_mape_pct": _money(pipeline.at["final_model", "MAPE"]),
            "original_xgboost_mae": _money(pipeline.at["xgboost", "MAE"]),
            "manual_forecast_mae": _money(pipeline.at["manual_forecast", "MAE"]),
        },
        "ACCURACY_BY_DAYS_AHEAD": horizon,
        "BUSINESS_IMPACT_SIMULATION": {
            "days": int(impact["days"]),
            "planned_wages_lower_than_manual_gbp": _money(impact["plan_savings_vs_manual"]),
            "simulated_saving_vs_actual_wages_gbp": _money(impact["simulated_savings_vs_actual_per_day"]),
            "annualised_saving_gbp": _money(impact["annualised_savings_per_day_factor"]),
            "annualised_saving_pct_of_2025_wage_bill": _money(impact["annualised_savings_pct_of_2025_wage_bill"]),
        },
    }


def main() -> None:
    for name, block in {**final_model_results(), **sales_patterns()}.items():
        print(f"{name} = ", end="")
        pprint.pprint(block, sort_dicts=False, width=110)
        print()


if __name__ == "__main__":
    main()
