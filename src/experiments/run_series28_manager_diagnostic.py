"""
Series 28 - does the manager's forecast know something the weekend package does not?

Track A item 5 of the accuracy roadmap, run after items 1-4. The "package" is everything Series 23-27 built: the log
model on Monday to Thursday, the raw model on Saturday, the per-weekday weekend features (Friday the mean and highest
of the last 4 Fridays, Saturday sales 2 and 3 weeks ago, Sunday the weekend interactions, as chosen on blocks 0-4 in
Series 27) and the Saturday and Sunday correction (8 weeks, strength 0.5), on seeds 0-24 and the 585 test days.

A diagnostic, not an experiment with a bar. It reports, for each weekday:
  * the manager's MAE against the package's, and on what share of days the package is closer
  * whether the manager's forecast adds information: the correlation between the package's miss (actual minus package)
    and the manager's disagreement with it (manager minus package). Near zero means the manager knows nothing extra;
    clearly positive means that when the manager forecasts above the package, sales tend to land above it too.
    95% interval from a block bootstrap of that weekday's days (blocks of 4 consecutive weeks).
  * what happens to the MAE if some of the manager's forecast is blended in (0, 10, 20, 30%)
and writes the 20 weekend days (Friday to Sunday) and the 20 days overall on which the manager beat the package by the
most, with context and a blank "What happened?" column for the venue manager (CSV and a printable page).

Run:  python -m src.experiments.run_series28_manager_diagnostic
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_error_autopsy as autopsy
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series25_weekend_training as s25
from src.experiments import run_series26_weekend_features as s26
from src.experiments import run_series27_weekend_per_weekday as s27
from src.experiments.report_series19_21 import CSS
from src.experiments.series19_21_common import test_days

REPORT_DIR = CONFIG.path("experiments_dir") / "series28_manager_diagnostic"
CHOSEN = {"Friday": "A_summary", "Saturday": "A_lags", "Sunday": "B"}  # Series 27, chosen on blocks 0-4
WEEKDAYS = s23.WEEKDAYS
BLENDS = (0.0, 0.1, 0.2, 0.3)
LIST_LENGTH = 20


def build_package(days: pd.DataFrame, b0: dict, bags: dict) -> np.ndarray:
    wd = days["weekday"].to_numpy()
    hybrid = s24.hybrid(wd, b0["log1p_final"], b0["raw_same_params"])
    options = {d: s27.option_predictions(d, bags, b0) for d in CHOSEN}
    weekend = s27.assemble_weekend(days, hybrid, options, CHOSEN)
    return s24.correct(days, weekend, 8, 0.5, scope=("Saturday", "Sunday"))


def corr_with_interval(x: np.ndarray, y: np.ndarray, block: int = 4, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Pearson correlation of x and y (in time order) with a 95% moving-block bootstrap interval."""
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    vals = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    vals = vals[~np.isnan(vals)]
    return float(np.corrcoef(x, y)[0, 1]), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def weekday_table(days: pd.DataFrame, package: np.ndarray) -> pd.DataFrame:
    a, man, wd = days["actual"].to_numpy(), days["manual"].to_numpy(), days["weekday"].to_numpy()
    rows = []
    for d in WEEKDAYS:
        m = wd == d
        r, lo, hi = corr_with_interval((a - package)[m], (man - package)[m])
        rows.append({"weekday": d, "days": int(m.sum()), "manager_MAE": float(np.abs(man - a)[m].mean()),
                     "package_MAE": float(np.abs(package - a)[m].mean()),
                     "package_closer_%": float(100 * (np.abs(package - a)[m] < np.abs(man - a)[m]).mean()),
                     "correlation": r, "corr_lo": lo, "corr_hi": hi,
                     "manager_bias": float((man - a)[m].mean()), "package_bias": float((package - a)[m].mean())})
    return pd.DataFrame(rows)


def blend_table(days: pd.DataFrame, package: np.ndarray) -> pd.DataFrame:
    a, man, wd = days["actual"].to_numpy(), days["manual"].to_numpy(), days["weekday"].to_numpy()
    fs = s25.fri_sun_mask(wd)
    rows = []
    for w in BLENDS:
        p = (1 - w) * package + w * man
        rows.append({"manager_share_%": int(w * 100), "all_days_MAE": float(np.abs(p - a).mean()),
                     "Fri_to_Sun_MAE": float(np.abs(p - a)[fs].mean()), "Mon_to_Thu_MAE": float(np.abs(p - a)[~fs].mean())})
    return pd.DataFrame(rows)


def manager_wins(frame: pd.DataFrame, package: np.ndarray, weekend_only: bool, n: int = LIST_LENGTH) -> pd.DataFrame:
    """The days the manager's forecast was closer than the package's by the most (closure days excluded)."""
    df = frame.copy()
    df["package"] = package
    df["package_miss"] = (df["package"] - df["actual"]).abs()
    df["manager_miss"] = (df["manual"] - df["actual"]).abs()
    df["advantage"] = df["package_miss"] - df["manager_miss"]
    df = df[~df["closure"] & (df["advantage"] > 0)]
    if weekend_only:
        df = df[df["weekday"].isin(s25.FRI_SUN)]
    top = df.sort_values("advantage", ascending=False).head(n)
    return pd.DataFrame({
        "Date": top["date"].dt.strftime("%a %d %b %Y"), "Actual sales": top["actual"], "Package forecast": top["package"],
        "Manager's forecast": top["manual"], "Package miss": top["package_miss"], "Manager miss": top["manager_miss"],
        "Manager advantage": top["advantage"],
        "Sales vs package": np.where(top["actual"] > top["package"], "busier than the package forecast", "quieter than the package forecast"),
        "Usual for this weekday (4 wk)": top["same_weekday_4wk_avg"], "Holiday context": top["holiday"].replace("", "-"),
        "School": np.where(top["is_school_holiday"] == 1, "holidays", "term"), "Rain mm": top["rain_mm"], "Max temp C": top["max_temp"],
        "What happened?": "",
    }).reset_index(drop=True)


def page(tables: dict[str, tuple[str, pd.DataFrame]]) -> str:
    def block(title: str, note: str, df: pd.DataFrame) -> str:
        return f"<h2>{html.escape(title)}</h2><p>{note}</p>" + df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:,.0f}", escape=True)

    body = "".join(block(t, note, df) for t, (note, df) in tables.items())
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Manager versus package</title><style>{CSS}</style></head><body><main><h1>Does the manager's forecast know more?</h1>{body}</main></body></html>")


def main() -> None:
    df, folds = s26.load_frame()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    b0 = s23.load_bags()
    bags = s26.load_bags(s26.RESULTS, 25, s26.all_variant_targets())
    package = build_package(days, b0, bags)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    wt, bt = weekday_table(days, package), blend_table(days, package)
    wt.to_csv(REPORT_DIR / "by_weekday.csv", index=False)
    bt.to_csv(REPORT_DIR / "blend.csv", index=False)
    print("\nBy weekday (package vs the manager's forecast, 585 days):\n", wt.round(2).to_string(index=False))
    print("\nBlending in some of the manager's forecast:\n", bt.round(1).to_string(index=False))

    ctx = autopsy.build_frame()[["date", "closure", "holiday", "is_school_holiday", "rain_mm", "max_temp", "same_weekday_4wk_avg"]]
    frame = days.assign(date=pd.to_datetime(days["date"])).merge(ctx, on="date", how="left")
    wk, allday = manager_wins(frame, package, True), manager_wins(frame, package, False)
    wk.to_csv(REPORT_DIR / "manager_wins_weekend.csv", index=False)
    allday.to_csv(REPORT_DIR / "manager_wins_all_days.csv", index=False)
    print(f"\nThe {LIST_LENGTH} weekend days the manager beat the package by the most:\n", wk.drop(columns=["What happened?"]).round(0).to_string(index=False))

    note_w = ("Per weekday, the manager's forecast against the package's. <b>Correlation</b>: of the package's miss with the manager's "
              "disagreement; about zero means the manager knows nothing extra, clearly positive means some information. Interval from a block bootstrap.")
    (REPORT_DIR / "manager_diagnostic.html").write_text(page({
        "By weekday": (note_w, wt.rename(columns={"corr_lo": "corr 2.5%", "corr_hi": "corr 97.5%"})),
        "Blending in the manager's forecast": ("Share of the manager's forecast mixed into the package.", bt),
        f"The {LIST_LENGTH} weekend days the manager beat the package by the most": (
            "Friday to Sunday, 585-day backtest. 'Manager advantage' is how much closer the manager's forecast was. The last column is for you.", wk),
        f"The {LIST_LENGTH} days overall the manager beat the package by the most": ("All weekdays.", allday),
    }), encoding="utf-8")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
