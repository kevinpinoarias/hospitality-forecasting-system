"""
Series 55 - does which week of the month it is help the weekly log's models?

Series 54 found nothing in day of month. Week of month is day of month with the detail removed (the model can already cut day of month at 7, 14
and 21), so the reason to try it is the one the weekday flags showed: the trees are four splits deep, and a ready-made "first week" or "last week"
column saves splits. Written four ways, appended after each piece's own inputs (so the baseline is a prefix of every variant):

  none                    today's piece, nothing added
  week_of_month           1 to 5 from the 1st (days 1-7, 8-14, ..., 29-31); also "which occurrence of its weekday it is" (the first Saturday, ...)
  week_and_weeks_to_end   week_of_month and weeks_to_month_end ((days in the month - day) // 7: 0 is the last seven days, "the last Saturday")
  calendar_week           Monday-to-Sunday weeks of the month, 1 to 6, week 1 being the week that holds the 1st
  all                     the three columns together

No circle: week of month does not wrap round the way a month does. Every variant goes into every piece of the weekly log, exactly as in Series 54
(its docstring has the pieces, the post and hydro models, the correction and the baselines; the hydro and Saturday-fallback baselines are Series 54's
`none` fits, same seeds and data).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece; in how many folds it is better; the new days
  expected   little or nothing: Series 54 and the error screen found no day-of-month pattern, and week of month is a coarser version of it
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version

Run:  python -m src.experiments.run_series55_week_of_month list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series55"
RESULTS = OUT_DIR / "series55_week_of_month.jsonl"
VARIANT_COLUMNS = {
    "none": [],
    "week_of_month": ["week_of_month"],
    "week_and_weeks_to_end": ["week_of_month", "weeks_to_month_end"],
    "calendar_week": ["calendar_week_of_month"],
    "all": ["week_of_month", "weeks_to_month_end", "calendar_week_of_month"],
}


def piece_features(piece: str, variant: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[piece][0] + VARIANT_COLUMNS[variant]


def add_week_of_month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    day, length = out[DATE_COL].dt.day, out[DATE_COL].dt.days_in_month
    first_weekday = (out[DATE_COL] - pd.to_timedelta(day - 1, unit="D")).dt.dayofweek  # Monday is 0, for the 1st of this month
    out["week_of_month"] = (day - 1) // 7 + 1
    out["weeks_to_month_end"] = (length - day) // 7
    out["calendar_week_of_month"] = (day - 1 + first_weekday) // 7 + 1
    return out


def init_worker55() -> None:
    s47.init_worker47()
    clash = {"week_of_month", "weeks_to_month_end", "calendar_week_of_month"} & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_week_of_month_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANT_COLUMNS:
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
        print(f"{len(s54.PIECES)} pieces x {len(VARIANT_COLUMNS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} baseline {len(piece_features(piece, 'none'))} columns, 'all' {len(piece_features(piece, 'all'))}")
        frame = pd.DataFrame({DATE_COL: pd.to_datetime(["2026-02-01", "2026-02-28", "2026-03-01", "2026-03-31", "2026-05-01", "2026-05-31"])})
        print(add_week_of_month_columns(frame).drop(columns=[]).assign(weekday=lambda t: t[DATE_COL].dt.day_name().str[:3]).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker55, "series55")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
