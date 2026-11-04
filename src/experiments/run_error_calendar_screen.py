"""
Screen - does the post model's error follow day of month, week of month or the ends of the month, before building features for them?

The calendar candidates not yet in the model (day of month, week of month, week of year, month start and end flags) can only help if the post model's
errors move with them. This reads the backtest's out-of-sample errors of the post model as validated (Monday-Thursday from the log model, the
Friday, Saturday and Sunday pieces, then the Series 24 correction; the 585 days, seeds 200-299, closed days left out) as a percentage of the
forecast (+ = under-forecast) and groups them by those calendar facts. Week of year is not screened: with about 11 days per week it is day of year
under another name, and day of year is already in the model. A second table lists the named event days in the test period.

A screen is a necessary condition, not a proof: a feature could still help through an interaction (with the weekday, say) that a one-way table
does not show, and with about a dozen comparisons one |t| near 2 is expected by chance.

Run:  python -m src.experiments.run_error_calendar_screen
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series50_drop_year_in_pieces as s50
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import load_protocol, test_days

EVENT_DAYS = {
    "Mother's Day": ["2025-03-30", "2026-03-15"], "Father's Day": ["2025-06-15", "2026-06-21"],
    "Valentine's Day": ["2025-02-14", "2026-02-14"], "Burns Night": ["2025-01-25", "2026-01-25"],
    "Halloween": ["2025-10-31"], "Bonfire Night": ["2025-11-05"],
    "Summer break starts": ["2025-06-27", "2025-06-28", "2026-06-26", "2026-06-27", "2026-06-28"],
}


def post_model_errors() -> pd.DataFrame:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    date, dow = pd.to_datetime(days["date"]), pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    kept = s50.load_kept("backtest")
    pred = np.select([dow == 4, dow == 5, dow == 6], [kept["Friday"], kept["Saturday"], kept["Sunday"]], default=kept["Monday-Thursday"])
    post = s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
    out = pd.DataFrame({"date": date, "actual": days["actual"].to_numpy(), "post": post, "dow": dow})
    out = out[out["actual"] > 0].copy()
    out["pct"] = (out["actual"] / out["post"] - 1) * 100
    out["dom"], out["dim"] = out["date"].dt.day, out["date"].dt.days_in_month
    out["week_of_month"] = (out["dom"] - 1) // 7 + 1
    out["day_of_month"] = pd.cut(out["dom"], [0, 5, 10, 15, 20, 25, 31], labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-31"])
    out["month_edge"] = np.where(out["dom"] <= 3, "first 3 days", np.where(out["dim"] - out["dom"] <= 2, "last 3 days", "middle"))
    return out


def summarise(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    g = frame.groupby(by, observed=True)["pct"].agg(["count", "mean", "std"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["t"] = g["mean"] / g["se"]
    return g[["count", "mean", "se", "t"]]


def main() -> None:
    errors = post_model_errors()
    pd.set_option("display.width", 200)
    print(f"{len(errors)} open days; mean % error {errors['pct'].mean():+.2f}; typical miss (median absolute) {errors['pct'].abs().median():.1f}%\n")
    for title, by in (("Day of month", "day_of_month"), ("Week of the month (days 1-7, 8-14, ...)", "week_of_month"), ("Month start and end", "month_edge")):
        print(f"{title}: mean % error of the post model (+ = under-forecast)")
        print(summarise(errors, by).round(2).to_string(), "\n")
    last3 = errors[errors["month_edge"] == "last 3 days"]
    for name, sub in {"last 3 days, December": last3[last3["date"].dt.month == 12], "last 3 days, not December": last3[last3["date"].dt.month != 12]}.items():
        print(f"{name}: n={len(sub)}, mean {sub['pct'].mean():+.2f}%, t {sub['pct'].mean() / (sub['pct'].std() / np.sqrt(len(sub))):.2f}")
    print("\nLargest under-forecasts among the last 3 days outside December:")
    print(last3[last3["date"].dt.month != 12].sort_values("pct", ascending=False).head(6)[["date", "actual", "post", "pct"]]
          .assign(weekday=lambda t: t["date"].dt.day_name()).round(1).to_string(index=False))
    print("\nNamed event days (% error, + = under-forecast):")
    indexed = errors.set_index("date")
    for name, dates in EVENT_DAYS.items():
        print(f"  {name}: " + " | ".join(f"{d[2:]} {pd.Timestamp(d).day_name()[:3]} {indexed.loc[d, 'pct']:+.0f}%" for d in dates if pd.Timestamp(d) in indexed.index))


if __name__ == "__main__":
    main()
