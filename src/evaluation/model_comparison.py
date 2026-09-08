"""
Formal cross-model evaluation framework (Track A, step 3).

Loads every model's saved predictions (baselines, SARIMAX, XGBoost, LSTM,
Transformer) and produces a structured, documented comparison: not just one
aggregate MAE per model, but who wins, on which days, and why - reusing
src/evaluation/error_analysis.py's tools across all models consistently,
adding a spike-day breakdown, and extending the project's existing
labour-cost business-impact analysis to the sales-forecasting models.

Outputs
-------
- reports/results/model_comparison_overall.csv
- reports/results/model_comparison_by_weekday.csv
- reports/results/model_comparison_spike_vs_normal.csv
- reports/results/model_comparison_daily_winner.csv
- reports/results/model_comparison_win_counts.csv
- reports/results/model_comparison_business_impact.csv
- reports/results/model_comparison_summary.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.evaluation.error_analysis import add_error_columns, summarise_error_by_group

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
SALES_PATH = PROJECT_ROOT / "data" / "processed" / "sales" / "daily_sales_totals_master.csv"
LABOUR_PATH = PROJECT_ROOT / "data" / "processed" / "labour" / "daily_labour_totals_master.csv"

DATE_COL = "date"
ACTUAL_COL = "total_sales"
SPIKE_PERCENTILE = 0.90  # top 10% of actual sales days, within the comparison window

# (source file, actual-sales column in that file, prediction column, output name)
MODEL_SOURCES = [
    ("baseline_predictions.csv", "total_sales", "naive_pred", "naive"),
    ("baseline_predictions.csv", "total_sales", "seasonal_naive_pred", "seasonal_naive"),
    ("baseline_predictions.csv", "total_sales", "roll7_pred", "roll7"),
    ("baseline_predictions.csv", "total_sales", "roll14_pred", "roll14"),
    ("baseline_predictions.csv", "total_sales", "roll28_pred", "roll28"),
    ("baseline_predictions.csv", "total_sales", "weekday_avg_pred", "weekday_average"),
    ("sarimax_predictions.csv", "actual_sales", "sarimax_pred", "sarimax"),
    ("xgboost_predictions.csv", "total_sales", "ai_prediction", "xgboost"),
    ("lstm_predictions.csv", "total_sales", "lstm_pred", "lstm"),
    ("transformer_predictions.csv", "total_sales", "transformer_pred", "transformer"),
]


def load_all_predictions() -> pd.DataFrame:
    """
    Load every model's predictions and merge them into one wide table,
    keyed by date, with one column per model plus a single `actual` column.

    Only dates where every model has a prediction are kept - baselines and
    SARIMAX were validated over a wider date range than XGBoost/LSTM/
    Transformer, so this restricts the comparison to the range they all
    genuinely share, rather than comparing models over different periods.
    """
    wide = None

    for filename, actual_col, pred_col, model_name in MODEL_SOURCES:
        df = pd.read_csv(RESULTS_DIR / filename)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        model_df = df[[DATE_COL, actual_col, pred_col]].rename(
            columns={actual_col: ACTUAL_COL, pred_col: model_name}
        )

        if wide is None:
            wide = model_df
        else:
            wide = wide.merge(model_df[[DATE_COL, model_name]], on=DATE_COL, how="inner")

    return wide.sort_values(DATE_COL).reset_index(drop=True)


def add_spike_flag(df: pd.DataFrame, percentile: float = SPIKE_PERCENTILE) -> pd.DataFrame:
    """Flag the top `percentile` of actual-sales days, within this
    comparison window, as spike days."""
    out = df.copy()
    threshold = out[ACTUAL_COL].quantile(percentile)
    out["is_spike_day"] = out[ACTUAL_COL] > threshold
    out["day_of_week"] = out[DATE_COL].dt.day_name()
    return out


def model_names(df: pd.DataFrame) -> list[str]:
    return [name for _, _, _, name in MODEL_SOURCES]


def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in model_names(df):
        m = regression_metrics(df[ACTUAL_COL], df[name])
        rows.append({"model": name, **m, "n": len(df)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def compute_grouped_comparison(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Cross-model version of error_analysis.summarise_error_by_group -
    same underlying tool, run once per model and stacked together."""
    frames = []
    for name in model_names(df):
        summary = summarise_error_by_group(df, ACTUAL_COL, name, group_col)
        summary.insert(0, "model", name)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def compute_daily_winner(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each day, which model had the smallest absolute error."""
    names = model_names(df)
    errors = pd.DataFrame({name: (df[name] - df[ACTUAL_COL]).abs() for name in names})

    winner_df = df[[DATE_COL, ACTUAL_COL, "is_spike_day"]].copy()
    winner_df["winning_model"] = errors.idxmin(axis=1)
    winner_df["winning_model_error"] = errors.min(axis=1)

    win_counts = (
        winner_df["winning_model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="wins")
    )
    win_counts["win_pct"] = (win_counts["wins"] / len(winner_df) * 100).round(1)

    return winner_df, win_counts.sort_values("wins", ascending=False).reset_index(drop=True)


def compute_business_impact(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Extends the project's existing wage-forecast-vs-actual business metrics
    (see src/models/evaluate_human_forecast.py) to the sales-forecasting
    models compared here.

    Important assumption, stated explicitly rather than hidden: none of
    these models predict wages or staffing directly, only sales. To make
    them comparable on business terms, each model's predicted sales is
    converted to an "implied wage forecast" using the average wage-to-sales
    ratio observed in the training period (before this comparison's date
    range). This is a real approximation - actual staffing decisions are
    not purely proportional to sales - not a measured quantity.
    """
    sales = pd.read_csv(SALES_PATH)
    labour = pd.read_csv(LABOUR_PATH)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL])
    labour[DATE_COL] = pd.to_datetime(labour[DATE_COL])

    merged = pd.merge(
        sales[[DATE_COL, "total_sales"]],
        labour[[DATE_COL, "total_wages"]],
        on=DATE_COL,
        how="inner",
    )

    comparison_start = df[DATE_COL].min()
    training_period = merged[merged[DATE_COL] < comparison_start]
    wage_to_sales_ratio = training_period["total_wages"].sum() / training_period["total_sales"].sum()

    rows = []
    for name in model_names(df):
        implied_wages_forecast = df[name] * wage_to_sales_ratio
        implied_wages_actual = df[ACTUAL_COL] * wage_to_sales_ratio  # same ratio, actual sales

        wage_error = implied_wages_forecast - implied_wages_actual
        overforecast_wages = wage_error.clip(lower=0)
        underforecast_wages = (-wage_error).clip(lower=0)

        rows.append({
            "model": name,
            "implied total overforecasted wages (£)": round(overforecast_wages.sum(), 2),
            "implied total underforecasted wages (£)": round(underforecast_wages.sum(), 2),
            "implied overstaffed days": int((wage_error > 0).sum()),
            "implied understaffed days": int((wage_error < 0).sum()),
        })

    return pd.DataFrame(rows).sort_values(
        "implied total overforecasted wages (£)"
    ).reset_index(drop=True), wage_to_sales_ratio


def write_summary_markdown(
    overall: pd.DataFrame,
    win_counts: pd.DataFrame,
    spike_comparison: pd.DataFrame,
    business_impact: pd.DataFrame,
    wage_ratio: float,
    n_days: int,
    n_spike_days: int,
    date_range: tuple,
    output_path: Path,
) -> None:
    best_overall = overall.iloc[0]
    best_wins = win_counts.iloc[0]

    spike_rows = spike_comparison[spike_comparison["is_spike_day"] == True].sort_values("mean_absolute_error")
    best_on_spikes = spike_rows.iloc[0] if len(spike_rows) else None

    lines = [
        "# Model Comparison Summary",
        "",
        f"Comparison window: {date_range[0].date()} to {date_range[1].date()} "
        f"({n_days} days, {n_spike_days} flagged as spike days, top {int((1 - SPIKE_PERCENTILE) * 100)}% by actual sales).",
        "",
        "## Overall accuracy",
        "",
        overall.to_markdown(index=False),
        "",
        f"**{best_overall['model']}** has the lowest overall MAE (£{best_overall['MAE']:.2f}).",
        "",
        "## Which model wins on the most individual days",
        "",
        win_counts.to_markdown(index=False),
        "",
        f"**{best_wins['model']}** wins the most individual days ({best_wins['wins']} of {n_days}, {best_wins['win_pct']}%). "
        "Note this can differ from the overall-MAE winner - a model can be consistently decent "
        "without ever being the single best on any given day, and vice versa.",
        "",
        "## Spike days specifically",
        "",
        "The project's error analysis has repeatedly found that the hardest days to forecast - "
        "unusually strong peaks - are where forecasting value matters most. This breaks that out explicitly:",
        "",
    ]

    if best_on_spikes is not None:
        lines.append(
            f"On spike days specifically, **{best_on_spikes['model']}** has the lowest mean absolute error "
            f"(£{best_on_spikes['mean_absolute_error']:.2f}), compared to £{best_overall['MAE']:.2f} overall for the best all-round model."
        )
        lines.append("")

    lines += [
        "## Business impact (implied wages)",
        "",
        f"*Assumption: no model here predicts wages directly. Each model's predicted sales is converted to an "
        f"implied wage forecast using the training-period wage-to-sales ratio of "
        f"{wage_ratio:.4f} (i.e. roughly £{wage_ratio:.2f} of wages per £1 of sales, on average). "
        "This is an approximation for comparison purposes, not a measured relationship.*",
        "",
        business_impact.to_markdown(index=False),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model_comparison() -> dict:
    df = load_all_predictions()
    df = add_spike_flag(df)

    overall = compute_overall_metrics(df)
    by_weekday = compute_grouped_comparison(df, "day_of_week")
    by_spike = compute_grouped_comparison(df, "is_spike_day")
    daily_winner, win_counts = compute_daily_winner(df)
    business_impact, wage_ratio = compute_business_impact(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(RESULTS_DIR / "model_comparison_overall.csv", index=False)
    by_weekday.to_csv(RESULTS_DIR / "model_comparison_by_weekday.csv", index=False)
    by_spike.to_csv(RESULTS_DIR / "model_comparison_spike_vs_normal.csv", index=False)
    daily_winner.to_csv(RESULTS_DIR / "model_comparison_daily_winner.csv", index=False)
    win_counts.to_csv(RESULTS_DIR / "model_comparison_win_counts.csv", index=False)
    business_impact.to_csv(RESULTS_DIR / "model_comparison_business_impact.csv", index=False)

    write_summary_markdown(
        overall=overall,
        win_counts=win_counts,
        spike_comparison=by_spike,
        business_impact=business_impact,
        wage_ratio=wage_ratio,
        n_days=len(df),
        n_spike_days=int(df["is_spike_day"].sum()),
        date_range=(df[DATE_COL].min(), df[DATE_COL].max()),
        output_path=RESULTS_DIR / "model_comparison_summary.md",
    )

    print(f"Comparison window: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()} ({len(df)} days)")
    print(f"\nOverall MAE ranking:\n{overall[['model', 'MAE', 'RMSE', 'MAPE']].to_string(index=False)}")
    print(f"\nDaily win counts:\n{win_counts.to_string(index=False)}")
    print(f"\nSaved outputs to: {RESULTS_DIR}")

    return {
        "overall": overall,
        "by_weekday": by_weekday,
        "by_spike": by_spike,
        "daily_winner": daily_winner,
        "win_counts": win_counts,
        "business_impact": business_impact,
    }


if __name__ == "__main__":
    run_model_comparison()
