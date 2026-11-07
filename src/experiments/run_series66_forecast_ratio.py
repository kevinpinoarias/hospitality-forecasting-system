"""
Series 66 - the manager's forecast against the recent level of sales, as a ready-made ratio or gap.

The model already has the manager's forecast (`forecast_sales`) and the recent level of sales (lag_7_sales, the rolling means, and, in the
weekend pieces, sales 2 and 3 weeks ago and the mean and highest of the last four same weekdays). A tree model splits on one column at a time,
so it can only build "the forecast is well above what this weekday has been selling" by combining several cuts. Writing the comparison down as one
column gives the trees a direct handle on it. Series 26 tested the recent level of the same weekday (not its ratio to the forecast) and Series 28 found
the manager's forecast adds little for Saturday and Sunday, so little is expected. This is not a payday idea: it asks whether the trees need help
computing a ratio.

All three columns use only sales from 14, 21 and 28 days before the day, so they are known when a forecast is made up to two weeks ahead (sales 7 days
back are not known for a next-week forecast, as Series 53 found for Fridays). Closure days with zero sales are left out of the mean.

Variants, each ONE column added on top of the piece's inputs and run on its own:

  fc_ratio_3wk     forecast_sales divided by the mean of the sales 14, 21 and 28 days earlier
  fc_gap_3wk       forecast_sales minus that mean, in pounds (not a monotone change of the ratio, so it is a different split for a tree)
  fc_ratio_lag14   forecast_sales divided by the sales 14 days earlier alone

(A logarithm of the ratio is not tried: a tree only asks "above or below a cut", so a monotone change of one column changes nothing.)

Series 54's design: every variant in every piece of the weekly log (the post model's four pieces, the hydro Monday-to-Thursday piece, the raw
Saturday fallback), 100 seeds (200-299), 585 days and the 56 new days, the two models assembled and corrected, read by judgment on the net effect.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at    for each variant, the change in the post and hydro models' MAE after the correction with a 95% moving-block bootstrap interval; by weekday and
             piece; folds better; the new days
  expected   little or nothing: the information is already in the columns, and Series 26 and 28 found weekend memory and the manager's view add little
  would matter  an interval below zero in both models, more than half the folds better, and the new days not worse
  limits     no pass/fail gate; nothing in production changes; a new column would change the serving code too

Run:  python -m src.experiments.run_series66_forecast_ratio list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, TARGET_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series66"
RESULTS = OUT_DIR / "series66_forecast_ratio.jsonl"
VARIANTS = ["none", "fc_ratio_3wk", "fc_gap_3wk", "fc_ratio_lag14"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = VARIANTS[1:]


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return cols + ([variant] if variant != "none" else [])


def add_ratio_columns(frame: pd.DataFrame, daily_sales: pd.Series) -> pd.DataFrame:
    """The three columns. `daily_sales` is sales by calendar date over all days of the data; a lag is the sales on that calendar date, and a
    closure day (zero sales) counts as missing."""
    s = daily_sales.asfreq("D").where(lambda x: x > 0)
    lags = pd.DataFrame({k: s.shift(k) for k in (14, 21, 28)})
    mean3 = lags.mean(axis=1)  # NaN only when all three are missing
    index = pd.DatetimeIndex(frame[DATE_COL])
    mean3, lag14 = mean3.reindex(index).to_numpy(), lags[14].reindex(index).to_numpy()
    forecast = frame["forecast_sales"].to_numpy(dtype=float)
    out = frame.copy()
    out["fc_ratio_3wk"] = np.where(mean3 > 0, forecast / mean3, np.nan)
    out["fc_gap_3wk"] = forecast - mean3
    out["fc_ratio_lag14"] = np.where(lag14 > 0, forecast / lag14, np.nan)
    return out


def load_daily_sales() -> pd.Series:
    full = pd.read_csv(CONFIG.path("features_dir") / "engineered_features.csv", parse_dates=[DATE_COL])
    return full.set_index(DATE_COL)[TARGET_COL].sort_index()


def init_worker66() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_ratio_columns(add_hydro_thu(_STATE["df"]), load_daily_sales())


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANTS:
            if variant == "none":
                continue  # every baseline exists already: Series 46-47 for the post pieces, Series 54's `none` for the other two
            features = piece_features(piece, variant)
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), size):
                    chunk = SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, each variant {len(piece_features(piece, VARIANTS[1]))}")
        df, _ = load_protocol()
        sales = load_daily_sales()
        df = add_ratio_columns(df, sales)
        for name in NEW_COLUMNS:
            x = df[name]
            print(f"{name:16s} missing {int(x.isna().sum()):3d} of {len(df)}; min {x.min():10.2f}  median {x.median():9.2f}  max {x.max():10.2f}")
        # a lag is the sales on that calendar date, never a neighbouring row, and uses only days at least 14 days earlier
        row = df.iloc[400]
        day = row[DATE_COL]
        manual = np.mean([sales[day - pd.Timedelta(days=k)] for k in (14, 21, 28)])
        assert abs(row["fc_gap_3wk"] - (row["forecast_sales"] - manual)) < 1e-6, "the 3-week mean is not the mean of the sales 14, 21 and 28 days earlier"
        print(f"checked {day.date()}: forecast {row['forecast_sales']:.0f}, mean of sales 14/21/28 days earlier {manual:.0f}, ratio {row['fc_ratio_3wk']:.3f}")
        for name in NEW_COLUMNS:
            print(f"  correlation of {name} with forecast_sales {df[name].corr(df['forecast_sales']):.3f}, with the day's sales {df[name].corr(df[TARGET_COL]):.3f}")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker66, "series66")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
