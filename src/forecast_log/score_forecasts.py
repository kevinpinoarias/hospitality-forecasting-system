"""
Scoring the forecast log: add the actual sales that have arrived, then compare the forecasts with them (rules: section 7 and 8 of the log's PREREGISTRATION.md).

  python -m src.forecast_log.score_forecasts [--dry-run]

* Actuals are appended to `log/actuals.csv` for days from the first forecast's target date onward, once each, never edited.
* The manager's forecast for those days reaches us in the same weekly export, after the day; it is appended to `log/manager_forecasts.csv` the same way and scored as a benchmark
  (it was never an input to any forecast in the log).
* Scores are written to `log/scores/latest.md` and `log/scores/tables.csv` (derived outputs, rewritten each time); forecasts are never touched.
* Every table is on matched days: the two things compared are scored on the same target dates, same forecast type and same variant.
* Q1 portfolio vs manager; Q2 post vs portfolio; Q3 hydro vs post on Hydro Thursdays (sign test); Q4 (once v1.11 rows exist) the v1.11 hydro arm vs the v1 hydro arm, on all days and on
  Saturdays (sign test). Before the pre-registered reading times (26 weeks; 20 Hydro Thursdays or 52 weeks; Q4 as the amendment says) the tallies are descriptive only, and the report says so.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.features.events import concert_calendar
from src.forecast_log import logio
from src.forecast_log.run_forecasts import FEATURES, load_settings
from src.models.final_model import DATE_COL, TARGET_COL

MANAGER_COL = "forecast_sales"  # the manager's forecast column of the engineered features

ACTUAL_COLUMNS = ["target_date", "actual_sales", "recorded_at", "source"]
MANAGER_COLUMNS = ["target_date", "manager_forecast", "recorded_at", "source"]
MIN_DAYS_FOR_INTERVAL = 14
FORMAL_READING = "26 weeks of forecasts (Q1, Q2); 20 Hydro Thursdays or 52 weeks (Q3); 52 Saturdays (Q4, descriptive until then)"


def update_actuals(log_repo: Path, features_path: Path = FEATURES) -> int:
    """Append the actual sales of days from the first forecast target onward that are not in actuals.csv yet. Returns the number of rows added."""
    forecasts = log_repo / "log" / "forecasts.csv"
    if not forecasts.exists() or pd.read_csv(forecasts).empty:
        return 0
    first = pd.read_csv(forecasts)["target_date"].min()
    path = log_repo / "log" / "actuals.csv"
    logio.ensure_header(path, ACTUAL_COLUMNS)
    have = set(pd.read_csv(path, dtype=str)["target_date"]) if path.stat().st_size else set()
    sales = pd.read_csv(features_path, parse_dates=[DATE_COL]).dropna(subset=[TARGET_COL])
    new = sales[(sales[DATE_COL] >= pd.Timestamp(first)) & ~sales[DATE_COL].dt.strftime("%Y-%m-%d").isin(have)]
    rows = [{"target_date": d.strftime("%Y-%m-%d"), "actual_sales": round(float(v), 2), "recorded_at": logio.stamp(), "source": "weekly export"}
            for d, v in zip(new[DATE_COL], new[TARGET_COL])]
    logio._append(path, ACTUAL_COLUMNS, rows)
    return len(rows)


def update_manager(log_repo: Path, features_path: Path = FEATURES) -> int:
    """Append the manager's forecast (as it arrives in the weekly export) for days from the first forecast target onward that are not in manager_forecasts.csv yet. Returns the
    number of rows added."""
    forecasts = log_repo / "log" / "forecasts.csv"
    if not forecasts.exists() or pd.read_csv(forecasts).empty:
        return 0
    first = pd.read_csv(forecasts)["target_date"].min()
    path = log_repo / "log" / "manager_forecasts.csv"
    logio.ensure_header(path, MANAGER_COLUMNS)
    have = set(pd.read_csv(path, dtype=str)["target_date"]) if path.stat().st_size else set()
    known = pd.read_csv(features_path, parse_dates=[DATE_COL]).dropna(subset=[MANAGER_COL])
    new = known[(known[DATE_COL] >= pd.Timestamp(first)) & ~known[DATE_COL].dt.strftime("%Y-%m-%d").isin(have)]
    rows = [{"target_date": d.strftime("%Y-%m-%d"), "manager_forecast": round(float(v), 2), "recorded_at": logio.stamp(), "source": "weekly export"}
            for d, v in zip(new[DATE_COL], new[MANAGER_COL])]
    logio._append(path, MANAGER_COLUMNS, rows)
    return len(rows)


def manager_benchmark_rows(forecasts: pd.DataFrame, manager: pd.DataFrame) -> pd.DataFrame:
    """The manager's number as a benchmark row for every forecast type and target date the portfolio model forecast (so it is compared on the same days)."""
    keys = forecasts.loc[forecasts["model"] == "portfolio", ["forecast_type", "target_date"]].drop_duplicates()
    m = keys.merge(manager[["target_date", "manager_forecast"]], on="target_date", how="inner")
    return pd.DataFrame({"forecast_type": m["forecast_type"], "target_date": m["target_date"], "model": "manager", "variant": "", "on_time": "on_time",
                         "forecast": m["manager_forecast"].astype(float)})


def scored_frame(log_repo: Path) -> pd.DataFrame:
    forecasts = pd.read_csv(log_repo / "log" / "forecasts.csv", dtype={"variant": str, "target_date": str}).fillna({"variant": ""})
    manager_path = log_repo / "log" / "manager_forecasts.csv"
    if manager_path.exists() and manager_path.stat().st_size:
        manager = pd.read_csv(manager_path, dtype={"target_date": str})
        forecasts = pd.concat([forecasts, manager_benchmark_rows(forecasts, manager)], ignore_index=True)
    actuals = pd.read_csv(log_repo / "log" / "actuals.csv")
    frame = forecasts.merge(actuals[["target_date", "actual_sales"]], on="target_date", how="inner")
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def accuracy_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Days, mean absolute error and bias (forecast minus actual) for every forecast type, model, variant and on-time label."""
    keys = ["forecast_type", "model", "variant", "on_time"]
    g = frame.groupby(keys)
    return pd.DataFrame({"days": g.size(), "MAE": g["miss"].apply(lambda s: s.abs().mean()), "bias": g["miss"].mean()}).reset_index()


def paired_difference(frame: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], forecast_type: str, days: set[str] | None = None) -> dict:
    """Error of a minus error of b (negative: a is better) on the days both have a forecast, with a 95% moving-block bootstrap interval once there are enough days.
    a and b are (model, variant)."""
    sel = frame[(frame["forecast_type"] == forecast_type) & (frame["on_time"] == "on_time")]
    left = sel[(sel["model"] == a[0]) & (sel["variant"] == a[1])].set_index("target_date")
    right = sel[(sel["model"] == b[0]) & (sel["variant"] == b[1])].set_index("target_date")
    common = sorted(set(left.index) & set(right.index) & (days if days is not None else set(left.index)))
    if not common:
        return {"days": 0, "change": np.nan, "lo": np.nan, "hi": np.nan, "better_days": 0}
    diff = (left.loc[common, "miss"].abs() - right.loc[common, "miss"].abs()).to_numpy()
    out = {"days": len(common), "change": float(diff.mean()), "lo": np.nan, "hi": np.nan, "better_days": int((diff < 0).sum())}
    if len(common) >= MIN_DAYS_FOR_INTERVAL:
        out["change"], out["lo"], out["hi"] = masked_block_bootstrap(diff, np.ones(len(diff)))
    return out


def sign_test_p(better: int, n: int) -> float:
    """One-sided binomial p that at least `better` of `n` days favour the first model if each day were a coin flip."""
    return sum(comb(n, k) for k in range(better, n + 1)) / 2 ** n if n else float("nan")


def hydro_thursdays(dates: list[str]) -> set[str]:
    if not dates:
        return set()
    cal = concert_calendar(min(dates), max(dates))
    flagged = cal[(cal["hydro_show"] == 1) & (cal[DATE_COL].dt.dayofweek == 3)][DATE_COL]
    return set(flagged.dt.strftime("%Y-%m-%d")) & set(dates)


def build_report(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if frame.empty:
        return "# Forecast log: scores\n\nNo forecast has a recorded actual yet.\n", pd.DataFrame()
    table = accuracy_table(frame)
    n_days = frame["target_date"].nunique()
    lines = ["# Forecast log: scores", "", f"Days with an actual: {n_days} (first {frame['target_date'].min()}, last {frame['target_date'].max()}). "
             f"Descriptive until the pre-registered readings: {FORMAL_READING}.", "", "## Accuracy (mean absolute error and bias, £ a day; on-time and late forecasts apart)", "",
             table.round(1).to_markdown(index=False), "", "## The decisions (on-time forecasts, matched days; negative change means the first is better)", ""]
    rows = []
    for ftype in sorted(frame["forecast_type"].unique()):
        for question, a, b, variant_label in (
            ("Q1 portfolio vs manager (no-manager variant)", ("portfolio", "no_manager"), ("manager", ""), ""),
            ("Q2 post vs portfolio (no-manager variant)", ("post", "no_manager"), ("portfolio", "no_manager"), ""),
            ("Q3 hydro vs post, all days (no-manager variant)", ("hydro", "no_manager"), ("post", "no_manager"), ""),
        ):
            rows.append({"type": ftype, "question": question, **paired_difference(frame, a, b, ftype)})
        hy = paired_difference(frame, ("hydro", "no_manager"), ("post", "no_manager"), ftype, hydro_thursdays(sorted(frame["target_date"].unique())))
        rows.append({"type": ftype, "question": "Q3 hydro vs post on Hydro Thursdays (no-manager variant)", **hy,
                     "sign_test_p": sign_test_p(hy["better_days"], hy["days"])})
        if "hydro_v1.11" in set(frame["model"]):  # Q4, added by the pre-registration amendment for v1.11: the same hydro arm with the payday intensity in the Saturday pieces
            saturdays = {d for d in frame["target_date"].unique() if pd.Timestamp(d).dayofweek == 5}
            for label, days in (("all days", None), ("Saturdays", saturdays)):
                q4 = paired_difference(frame, ("hydro_v1.11", "no_manager"), ("hydro", "no_manager"), ftype, days)
                rows.append({"type": ftype, "question": f"Q4 v1.11 hydro vs v1 hydro, {label} (no-manager variant)", **q4,
                             **({"sign_test_p": sign_test_p(q4["better_days"], q4["days"])} if label == "Saturdays" else {})})
    result = pd.DataFrame(rows)
    lines.append(result.round(2).to_markdown(index=False))
    lines += ["", "Change = mean of (|first's error| - |second's error|) in £ a day; the interval is a 95% moving-block bootstrap and is shown once there are at least "
              f"{MIN_DAYS_FOR_INTERVAL} matched days. Q3's sign test is one-sided on days where the hydro forecast is closer.", ""]
    return "\n".join(lines), table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log_repo = Path(load_settings()["log_repo"])
    added = 0 if args.dry_run else update_actuals(log_repo)
    managers = 0 if args.dry_run else update_manager(log_repo)
    frame = scored_frame(log_repo)
    report, table = build_report(frame)
    print(report)
    if args.dry_run:
        return 0
    (log_repo / "log" / "scores").mkdir(parents=True, exist_ok=True)
    (log_repo / "log" / "scores" / "latest.md").write_text(report, encoding="utf-8")
    if not table.empty:
        table.to_csv(log_repo / "log" / "scores" / "tables.csv", index=False)
    outcome = logio.commit_and_push(log_repo, f"scores: {added} new actuals, {managers} manager forecasts, {frame['target_date'].nunique() if not frame.empty else 0} scored days")
    print(f"[score] {added} actuals and {managers} manager forecasts added; {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
